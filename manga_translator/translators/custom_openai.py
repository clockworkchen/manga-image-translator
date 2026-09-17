import re

from ..config import TranslatorConfig
from .config_gpt import ConfigGPT  # Import the `gpt_config` parsing parent class

try:
    import openai
except ImportError:
    openai = None
import asyncio
import time
from typing import List
from .common import CommonTranslator, InvalidServerResponse
from .keys import CUSTOM_OPENAI_API_KEY, CUSTOM_OPENAI_API_BASE, CUSTOM_OPENAI_MODEL, CUSTOM_OPENAI_MODEL_CONF


class CustomOpenAiTranslator(ConfigGPT, CommonTranslator):
    _INVALID_REPEAT_COUNT = 2  # 如果检测到"无效"翻译，最多重复 2 次
    _STRICT_TRANSLATION_COUNT = True
    _NUMBERED_MARKER = re.compile(r'<\|([1-9][0-9]*)\|>')
    _PROTOCOL_INSTRUCTION = (
        '\n## Required response protocol\n'
        'Each <|N|> input is a separate image region, not a sentence to regroup. '
        'Return every input ID exactly once, in the same order, with the exact <|N|> marker. '
        'Translate only the text belonging to that ID. Never move words, clauses, '
        'dialogue, or sound effects to another ID, even to improve fluency. '
        'Context may clarify meaning but must not change region ownership. '
        'Do not merge, split, omit, duplicate, or renumber regions. '
        'Keep already-target-language text, numbers, units, names, and ambiguous '
        'fragments when appropriate; never invent missing text. '
        'Output only final translations after their markers, with no reasoning, '
        'headings, source/translation labels, Markdown fences, or commentary. '
        'Every region must have a nonempty translation.\n'
    )
    _MAX_REQUESTS_PER_MINUTE = 40  # 每分钟最大请求次数
    # 本地 30B 级模型在冷启动/高负载时首 token 常超过 40 秒；过早取消会让
    # 同一提示反复从头生成，最终制造 4 次无效请求并把前端任务判失败。
    _TIMEOUT = 120  # 在重试之前等待服务器响应的时间（秒）
    _RETRY_ATTEMPTS = 3  # 在放弃之前重试错误请求的次数
    _TIMEOUT_RETRY_ATTEMPTS = 1  # 超时后只重试一次，避免取消风暴
    _RATELIMIT_RETRY_ATTEMPTS = 3  # 在放弃之前重试速率限制请求的次数

    # 最大令牌数量，用于控制处理的文本长度
    _MAX_TOKENS = 4096

    # 是否包含模板，用于决定是否使用预设的提示模板
    _INCLUDE_TEMPLATE = False

    def __init__(self, model=None, api_base=None, api_key=None, check_openai_key=False):
        # If the user has specified a nested key to use for the model, append the key
        #   Otherwise: Use the `ollama` defaults.
        _CONFIG_KEY='ollama'
        if CUSTOM_OPENAI_MODEL_CONF:
            _CONFIG_KEY+=f".{CUSTOM_OPENAI_MODEL_CONF}"

        ConfigGPT.__init__(self, config_key=_CONFIG_KEY)
        self.model = model
        CommonTranslator.__init__(self)
        self.client = openai.AsyncOpenAI(api_key=api_key or CUSTOM_OPENAI_API_KEY or "ollama", max_retries=0) # required, but unused for ollama
        self.client.base_url = api_base or CUSTOM_OPENAI_API_BASE
        self.token_count = 0
        self.token_count_last = 0

    def parse_args(self, args: TranslatorConfig):
        self.config = args.chatgpt_config
        # Translator instances are cached; do not leak manga samples into the
        # next product request's explicitly empty sample configuration.
        self.langSamples = None
        override = getattr(args, "llm_model", None)
        if override:
            self.model = str(override).strip()
        api_base = getattr(args, "llm_api_base", None)
        api_key = getattr(args, "llm_api_key", None)
        if api_base:
            self.client.base_url = str(api_base).rstrip("/")
        if api_key:
            self.client.api_key = str(api_key)


    def extract_capture_groups(self, text, regex=r"(.*)"):
        """
        Extracts all capture groups from matches and concatenates them into a single string.
        
        :param text: The multi-line text to search.
        :param regex: The regex pattern with capture groups.
        :return: A concatenated string of all matched groups.
        """
        pattern = re.compile(regex, re.DOTALL)  # DOTALL to match across multiple lines
        matches = pattern.findall(text)  # Find all matches
        
        # Ensure matches are concatonated (handles multiple groups per match)
        extracted_text = "\n".join(
            "\n".join(m) if isinstance(m, tuple) else m for m in matches
        )
        
        return extracted_text.strip() if extracted_text else None

    @staticmethod
    def _strip_response_envelope(response: str) -> str:
        """Remove only unambiguous whole-response protocol wrappers, not dialogue."""
        if not isinstance(response, str) or not response.strip():
            raise InvalidServerResponse('custom_openai protocol: empty or non-text response')
        text = response.strip()
        # Reasoning is removable only before the first translated region. Never
        # regex-delete tags or quoted phrases inside an individual translation.
        if text.startswith('<think>'):
            end = text.find('</think>')
            if end < 0:
                raise InvalidServerResponse('custom_openai protocol: unclosed reasoning envelope')
            text = text[end + len('</think>'):].strip()
        fence = re.fullmatch(r'```(?:text|plaintext)?[ \t]*\r?\n([\s\S]*?)\r?\n```', text)
        if fence:
            text = fence.group(1).strip()
        # These labels are metadata only BEFORE the first numbered region.
        # A dialogue such as <|1|>Translation: ... must remain untouched.
        text = re.sub(
            r'\A(?:Translation|Translations|Translated text|译文|翻译结果)[：:]\s*(?=<\|[1-9][0-9]*\|>)',
            '', text, count=1, flags=re.IGNORECASE,
        )
        return text

    def _parse_numbered_response(self, response: str, query_size: int) -> List[str]:
        text = self._strip_response_envelope(response)
        # Legacy capture regexes must not silently erase IDs or body text. Only
        # identity/whitespace captures remain compatible with strict numbering.
        if self.rgx_capture != self._RGX_REMOVE:
            try:
                captured = self.extract_capture_groups(text, self.rgx_capture)
            except re.error as exc:
                raise InvalidServerResponse('custom_openai protocol: invalid rgx_capture') from exc
            if captured != text:
                raise InvalidServerResponse(
                    'custom_openai protocol: rgx_capture changes numbered payload; remove this override'
                )
        markers = list(self._NUMBERED_MARKER.finditer(text))
        ids = [int(marker.group(1)) for marker in markers]
        expected = list(range(1, query_size + 1))
        if ids != expected:
            raise InvalidServerResponse(
                f'custom_openai protocol: expected ordered IDs {expected}, received {ids}; '
                'missing, duplicate, reordered, malformed, or extra IDs are not accepted'
            )
        if not markers or text[:markers[0].start()].strip():
            raise InvalidServerResponse('custom_openai protocol: unexpected text before first ID')
        translations = []
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            translation = text[marker.end():end].strip()
            if not translation:
                raise InvalidServerResponse(f'custom_openai protocol: empty translation for ID {index + 1}')
            # Never repair malformed markers: doing so can silently change IDs.
            # Reject residual protocol tokens rather than render leaked envelopes.
            if re.search(r'<\||\|>|<\d+\s*\|?>|</?think>|^```', translation, re.MULTILINE):
                raise InvalidServerResponse(f'custom_openai protocol: residual wrapper/marker in ID {index + 1}')
            translations.append(translation)
        return translations

    def _clean_translation_output(self, query: str, trans: str, to_lang: str) -> str:
        # CommonTranslator's punctuation/repetition heuristics can rewrite
        # dialogue and product decimals. Protocol parsing already removed wrappers.
        return trans.strip()

    def _system_prompt(self, to_lang: str) -> str:
        # Also applies to simple_prompt/custom templates without replacing their
        # product-specific instructions or importing manga few-shot samples.
        return self.chat_system_template.format(to_lang=to_lang) + self._PROTOCOL_INSTRUCTION

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        prefix = self.prompt_template.format(to_lang=to_lang) if self._INCLUDE_TEMPLATE else ''
        lines = []
        size = len(prefix)
        for query in queries:
            if re.search(r'<\|.*?\|>', query):
                raise InvalidServerResponse('custom_openai protocol: source contains reserved ID marker')
            line = f'<|{len(lines) + 1}|>{query}'
            if lines and size + len(line) + 1 > self._MAX_TOKENS * 2:
                yield (prefix + '\n' + '\n'.join(lines)).strip(), len(lines)
                lines = []
                size = len(prefix)
                line = f'<|1|>{query}'
            lines.append(line)
            size += len(line) + 1
        if lines:
            yield (prefix + '\n' + '\n'.join(lines)).strip(), len(lines)

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        if to_lang in self.chat_sample:
            return '\n'.join([
                'System:',
                self._system_prompt(to_lang),
                'User:',
                self.chat_sample[to_lang][0],
                'Assistant:',
                self.chat_sample[to_lang][1],
                'User:',
                prompt,
            ])
        else:
            return '\n'.join([
                'System:',
                self._system_prompt(to_lang),
                'User:',
                prompt,
            ])

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')

        for prompt, query_size in self._assemble_prompts(from_lang, to_lang, queries):
            self.logger.debug('-- GPT Prompt --\n' + self._format_prompt_log(to_lang, prompt))

            ratelimit_attempt = 0
            server_error_attempt = 0
            timeout_attempt = 0
            while True:
                request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                started = time.time()
                while not request_task.done():
                    await asyncio.sleep(0.1)
                    if time.time() - started > self._TIMEOUT + (timeout_attempt * self._TIMEOUT / 2):
                        # Server takes too long to respond
                        if timeout_attempt >= self._TIMEOUT_RETRY_ATTEMPTS:
                            raise Exception('ollama servers did not respond quickly enough.')
                        timeout_attempt += 1
                        self.logger.warning(f'Restarting request due to timeout. Attempt: {timeout_attempt}')
                        request_task.cancel()
                        request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                        started = time.time()
                try:
                    response = await request_task
                    break
                except openai.RateLimitError as exc:  # Honor provider cooldown; no retry storm.
                    if any(marker in str(exc).lower() for marker in (
                            'model_cooldown', 'safety_check_type_csam', 'content violates usage guidelines')):
                        raise InvalidServerResponse('upstream_request_blocked: safety refusal or model cooldown') from exc
                    ratelimit_attempt += 1
                    if ratelimit_attempt >= self._RATELIMIT_RETRY_ATTEMPTS:
                        raise
                    self.logger.warning(
                        f'Restarting request due to ratelimiting by Ollama servers. Attempt: {ratelimit_attempt}')
                    await asyncio.sleep(2)
                except openai.APIError as exc:
                    if any(marker in str(exc).lower() for marker in (
                            'safety_check_type_csam', 'content_policy_violation',
                            'content violates usage guidelines', 'content_filter', 'content_moderated')):
                        raise InvalidServerResponse('upstream_safety_refusal: translation stopped') from exc
                    server_error_attempt += 1
                    if server_error_attempt >= self._RETRY_ATTEMPTS:
                        self.logger.error(
                            'Ollama encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                        raise
                    self.logger.warning(f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                    await asyncio.sleep(1)

            # Validate the entire batch before exposing any translated regions.
            

            try:
                new_translations = self._parse_numbered_response(response, query_size)
            except InvalidServerResponse as exc:
                self.logger.error('Translation batch rejected (offset=%s): %s', len(translations), exc)
                raise


            

            # No positional fallback, truncation, or padding.

            translations.extend([t.strip() for t in new_translations])

        if len(translations) != len(queries):
            raise InvalidServerResponse(
                f'custom_openai protocol: expected {len(queries)} translations, received {len(translations)}'
            )
        self.logger.debug(translations)
        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        messages = [{'role': 'system', 'content': self._system_prompt(to_lang)}]

        # Add chat samples if available
        lang_chat_samples = self.get_chat_sample(to_lang)
        if lang_chat_samples:
            messages.append({'role': 'user', 'content': lang_chat_samples[0]})
            messages.append({'role': 'assistant', 'content': lang_chat_samples[1]})

        messages.append({'role': 'user', 'content': prompt})

        response = await self.client.chat.completions.create(
            model=self.model or CUSTOM_OPENAI_MODEL,
            messages=messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        if not response.choices:
            raise InvalidServerResponse('custom_openai protocol: response has no choices')
        choice = response.choices[0]
        # A truncated final item can contain every ID but still lose body text.
        if choice.finish_reason not in ('stop', None):
            raise InvalidServerResponse(
                f'custom_openai protocol: incomplete response (finish_reason={choice.finish_reason})'
            )
        if getattr(choice.message, 'refusal', None):
            raise InvalidServerResponse('custom_openai protocol: model refused translation')


        self.token_count_last = getattr(getattr(response, 'usage', None), 'total_tokens', 0) or 0
        self.token_count += self.token_count_last

        return response.choices[0].message.content
