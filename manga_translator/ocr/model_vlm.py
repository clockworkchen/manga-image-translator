"""VLM OCR backend (Ocr.vlm): read detected boxes with a vision LLM.

Why this exists
---------------
On stylized comic lettering the local recognizers are the weakest link in the
pipeline, and they fail in a way that is invisible downstream:

* A detector legitimately returns a box covering a WHOLE multi-line bubble.
  `48px_ctc` can only read a single line, so it returns junk, the junk is
  filtered as "not valuable", and the entire bubble silently disappears from
  the output. On the reference page two bubbles vanished exactly this way.
* Wide lines come back truncated ("AFTER ALL, I'M ABOL") or garbled
  ("JUST THE FEELINC", "TLIST THEEEELIN"), which then gets translated
  faithfully into nonsense.
* Phone-screenshot furniture (clock, battery, page counter) is recognized as
  text and translated, e.g. "38%" -> "389".

A VLM reads a tight crop far more reliably, reports the line breaks it sees,
and can be asked to reject non-story text. Line breaks matter beyond accuracy:
each returned line becomes its own Quadrilateral, so `textline_merge` regroups
them into one region with a correct line count, and the renderer can keep the
translation on at least as many lines as the original instead of collapsing a
three-line bubble onto one overflowing line.

Configuration (env, all optional except the base/key):
    VLM_OCR_API_BASE / CUSTOM_OPENAI_API_BASE
    VLM_OCR_API_KEY  / CUSTOM_OPENAI_API_KEY
    VLM_OCR_MODEL    / CUSTOM_OPENAI_MODEL
    VLM_OCR_BATCH    (default 12 crops per request)
    VLM_OCR_TIMEOUT  (default 120 seconds)

If the VLM is unreachable or answers unparseably, this falls back to the
regular 48px_ctc recognizer so the pipeline still produces something.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import List, Optional

import cv2
import numpy as np

from .common import CommonOCR
from ..config import OcrConfig
from ..utils import Quadrilateral

_PROMPT = """You are an OCR engine for comic/manga pages.

The image is one text area cropped from a comic page, {hint}. It is usually a
speech bubble or a caption. Transcribe its text EXACTLY as printed.

Rules:
1. Return one array entry per printed line. If the crop shows three lines of
   text, return three entries, in reading order.
2. Transcribe verbatim. Do not translate, correct spelling, expand
   abbreviations, or add punctuation that is not there.
3. Keep the original capitalisation.
4. Return an empty array if this is not story text. Where it sits on the page is
   how you tell these apart:
   - phone screenshot furniture: a clock, a battery percentage or signal bars
     sitting in the top few percent of a very tall page
   - a page counter like "6/26", or a bare page number
   - a watermark, site name or uploader credit
   - no readable text at all
   These must NOT be transcribed, even though they are legible.
5. Sound effects and hand-lettered text ARE story text: transcribe them.

Answer with JSON only, no prose and no code fence:
{{"lines": ["first line", "second line"]}}"""


class VlmRequestBlocked(RuntimeError):
    """Upstream safety/rate-limit refusal: never downgrade to local OCR."""


class ModelVlmOCR(CommonOCR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fallback = None

    # ------------------------------------------------------------------ #
    async def _recognize(self, image: np.ndarray, textlines: List[Quadrilateral],
                         config: OcrConfig, verbose: bool = False) -> List[Quadrilateral]:
        if not textlines:
            return []

        base = os.environ.get("VLM_OCR_API_BASE") or os.environ.get("CUSTOM_OPENAI_API_BASE")
        key = os.environ.get("VLM_OCR_API_KEY") or os.environ.get("CUSTOM_OPENAI_API_KEY")
        model = os.environ.get("VLM_OCR_MODEL") or os.environ.get("CUSTOM_OPENAI_MODEL")
        if not base or not model:
            self.logger.warning("vlm ocr: no API base/model configured, using 48px_ctc")
            return await self._run_fallback(image, textlines, config, verbose)

        # Read at BUBBLE level, not per detection box. Detector boxes overlap and
        # under-cover multi-line text, so a per-box crop shows fragments of its
        # neighbours and every shared line gets transcribed two or three times -
        # which then renders on top of itself. One crop per bubble contains each
        # line exactly once, and the VLM reports the line breaks, which is what
        # the renderer needs to avoid collapsing the translation onto one line.
        blocks = self._group_boxes(textlines)

        # One request per block, concurrently. Batching many crops into a single
        # request and matching answers by index looked cheaper but is unsafe: on
        # the reference page one crop held more text than the model expected, it
        # answered under the next index, and every block after it received its
        # neighbour's dialogue. A crop per request has nothing to misalign.
        sem = asyncio.Semaphore(max(1, int(os.environ.get("VLM_OCR_CONCURRENCY", "6"))))
        failures = 0
        blocked = asyncio.Event()

        async def one(rect):
            nonlocal failures
            async with sem:
                if blocked.is_set():
                    raise VlmRequestBlocked('VLM batch stopped after upstream refusal')
                try:
                    return await self._ask(base, key, model,
                                           self._crop(image, rect),
                                           self._where(image, rect))
                except VlmRequestBlocked:
                    blocked.set()
                    raise
                except Exception as e:
                    failures += 1
                    self.logger.warning(f"vlm ocr: block failed ({e})")
                    return None

        tasks = [asyncio.create_task(one(rect)) for rect in blocks]
        try:
            texts = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        if failures >= max(1, len(blocks) // 2):
            self.logger.warning("vlm ocr: too many failures, falling back")
            return await self._run_fallback(image, textlines, config, verbose)

        out: List[Quadrilateral] = []
        for rect, crop_text in zip(blocks, texts):
            if crop_text is None:
                # A transport/parse failure is not evidence of an empty bubble.
                # Never silently discard it while returning a successful page.
                x1, y1, x2, y2 = rect
                affected = []
                for detected in textlines:
                    pts = np.asarray(detected.pts).reshape(-1, 2)
                    if (pts[:, 0].min() >= x1 and pts[:, 0].max() <= x2
                            and pts[:, 1].min() >= y1 and pts[:, 1].max() <= y2):
                        affected.append(detected)
                recovered = await self._run_fallback(image, affected, config, verbose)
                if not recovered:
                    raise RuntimeError('VLM OCR block failed and fallback recovered no text')
                self.logger.warning('vlm ocr: recovered failed block with local OCR; review required')
                out.extend(recovered)
                continue
            lines = [ln.strip() for ln in str(crop_text).splitlines() if ln.strip()]
            if not lines:
                continue
            if self._is_chrome(image, rect, lines):
                continue
            out.extend(self._split_into_lines(image, rect, lines))
        self.logger.info(f"vlm ocr: {len(textlines)} boxes -> {len(blocks)} blocks "
                         f"-> {len(out)} lines")
        return out

    # ------------------------------------------------------------------ #
    async def _run_fallback(self, image, textlines, config, verbose):
        if self._fallback is None:
            from .model_48px_ctc import Model48pxCTCOCR
            self._fallback = Model48pxCTCOCR()
            await self._fallback.load("cpu")
        return await self._fallback.recognize(image, textlines, config, verbose)

    @staticmethod
    def _group_boxes(textlines: List[Quadrilateral]) -> List[List[float]]:
        """Union detection boxes that belong to the same block of text.

        Grouping is on generously grown rects rather than raw overlap because
        the boxes for consecutive lines of one bubble frequently do not touch at
        all - the detector clips them tight to the ink - while still overlapping
        each other's content once padded for the crop.
        """
        rects = []
        for t in textlines:
            p = np.asarray(t.pts).reshape(-1, 2).astype(float)
            rects.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
        n = len(rects)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def same_block(a, b):
            # Consecutive lines of one bubble sit in the same column with only a
            # leading gap between them. Requiring the horizontal overlap as well
            # as a small vertical gap is what keeps a short bubble from
            # swallowing the caption underneath it: growing the boxes by a fixed
            # margin alone merged "IT'S WEIRD" into the three-line caption 55px
            # below, and the two then shared one crop.
            ow = min(a[2], b[2]) - max(a[0], b[0])
            if ow <= 0.25 * min(a[2] - a[0], b[2] - b[0]):
                return False
            oh = min(a[3], b[3]) - max(a[1], b[1])
            if oh > 0:
                return True  # boxes actually overlap
            # Half a line height: the leading inside a paragraph is much smaller
            # than the space between two separate balloons. Erring tight is
            # deliberate - splitting one paragraph into two crops costs nothing
            # because textline_merge regroups the lines afterwards, whereas two
            # balloons sharing a crop come back as one merged block.
            return -oh <= 0.5 * min(a[3] - a[1], b[3] - b[1])

        for i in range(n):
            for j in range(i + 1, n):
                if same_block(rects[i], rects[j]):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(rects[i])
        out = []
        for members in groups.values():
            out.append([min(r[0] for r in members), min(r[1] for r in members),
                        max(r[2] for r in members), max(r[3] for r in members)])
        # Reading order, so batches stay locally coherent and the returned JSON
        # is easy to line up against the page when debugging.
        out.sort(key=lambda r: (r[1], r[0]))
        return out

    @staticmethod
    def _is_chrome(image: np.ndarray, rect: List[float], lines: List[str]) -> bool:
        """Reject phone-screenshot furniture and page counters.

        Asking the VLM to skip these works only sometimes - a legible "38%" is
        hard to refuse - so the actual decision is made here, where it can be
        narrow enough to be safe: an exact clock/battery/counter string, and for
        the status bar also the requirement that it sits in the top sliver of a
        page far taller than it is wide. Real dialogue never matches both.
        """
        if len(lines) != 1:
            return False
        text = lines[0].strip()
        if re.fullmatch(r"\d{1,3}\s*/\s*\d{1,3}", text):
            return True  # page counter, e.g. "6/26"
        h, w = image.shape[:2]
        in_status_bar = (rect[3] / max(1, h) < 0.04) and (h > w * 1.6)
        if in_status_bar and re.fullmatch(r"\d{1,2}:\d{2}|\d{1,3}\s*%", text):
            return True  # clock or battery
        return False

    @staticmethod
    def _where(image: np.ndarray, rect: List[float]) -> str:
        """Describe where the crop sits, so screenshot furniture is recognisable.

        A crop of "02:52" or "38%" is indistinguishable from dialogue once it is
        cut out of the page; knowing it sits in the top 2% of a very tall image
        is what makes it identifiable as a phone status bar.
        """
        h, w = image.shape[:2]
        return (f"at {rect[1] / max(1, h) * 100:.0f}% down, "
                f"{rect[0] / max(1, w) * 100:.0f}% across a {w}x{h} page")

    @staticmethod
    def _crop(image: np.ndarray, rect: List[float], pad: int = 8) -> np.ndarray:
        h, w = image.shape[:2]
        x1 = max(0, int(rect[0]) - pad)
        y1 = max(0, int(rect[1]) - pad)
        x2 = min(w, int(rect[2]) + pad)
        y2 = min(h, int(rect[3]) + pad)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((8, 8, 3), np.uint8)
        # Upscale small crops: the models read 12px lettering much better at 3x,
        # and these crops are tiny so the payload stays small either way.
        scale = max(1.0, min(4.0, 96.0 / max(1, crop.shape[0])))
        if scale > 1.01:
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        return crop

    async def _ask(self, base: str, key: Optional[str], model: str,
                   crop: np.ndarray, hint: str) -> Optional[str]:
        import httpx

        ok, buf = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        content = [
            {"type": "text", "text": _PROMPT.format(hint=hint)},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]

        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        timeout = float(os.environ.get("VLM_OCR_TIMEOUT", "120"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                base.rstrip("/") + "/chat/completions",
                headers=headers,
                json={"model": model, "temperature": 0,
                      "messages": [{"role": "user", "content": content}]})
        body = resp.text.lower()
        if any(marker in body for marker in (
                'safety_check_type_csam', 'content_policy_violation',
                'content violates usage guidelines', 'content_moderated',
                'content_filter', 'safety_violation')):
            raise VlmRequestBlocked('upstream_safety_refusal: no OCR fallback permitted')
        if resp.status_code == 429 or 'model_cooldown' in body:
            raise VlmRequestBlocked('upstream_rate_limited: stop batch and respect cooldown')
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        choice = resp.json()["choices"][0]
        if choice.get('finish_reason') == 'content_filter' or choice['message'].get('refusal'):
            raise VlmRequestBlocked('upstream_safety_refusal: no OCR fallback permitted')
        return self._parse(choice["message"]["content"])

    @staticmethod
    def _parse(answer: str) -> Optional[str]:
        m = re.search(r"\{.*\}", answer or "", re.S)
        if not m:
            raise RuntimeError(f"no JSON in answer: {str(answer)[:200]!r}")
        data = json.loads(m.group(0))
        lines = data.get("lines")
        if isinstance(lines, list):
            return "\n".join(str(v) for v in lines)
        text = data.get("text")
        return text if isinstance(text, str) else None

    # ------------------------------------------------------------------ #
    def _split_into_lines(self, image: np.ndarray, rect: List[float],
                          lines: List[str]) -> List[Quadrilateral]:
        """Turn one block holding `len(lines)` printed lines into one quad per line.

        Downstream everything assumes a Quadrilateral is a single line: the merge
        step groups lines into blocks and the renderer counts them to decide how
        many lines the translation may occupy. Splitting here is what lets a
        three-line bubble stay three lines.
        """
        style_heights = []
        bands = self._ink_bands(image, rect, len(lines), style_heights=style_heights)
        out = []
        for i, (text, (bx1, by1, bx2, by2)) in enumerate(zip(lines, bands)):
            box = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
                           dtype=np.float32)
            line = Quadrilateral(box, text, 1.0)
            # Equal-slice fallback has no measured style evidence. Do not label
            # its guessed height as a real font boundary.
            if style_heights:
                line.ocr_ink_height = float(by2 - by1)
                line.ocr_style_height = style_heights[i]
            out.append(line)
        return out

    @staticmethod
    def _ink_bands(image, rect, count, pad: int = 2, style_heights=None):
        """Locate the `count` lines of ink, as (x1, y1, x2, y2) each.

        Deliberately measured almost exactly inside `rect`. Padding this out to
        the crop the model was shown (8px) looked like a way to recover lines the
        detector had clipped, but it reaches the balloon outline: that ink merges
        into the first or last row-run and the band comes out about twice as
        tall. Since box height IS the font size downstream, a one-line "HMM~"
        went from 19px to 41px. Detector thresholds are the right place to fix
        clipped boxes, not this.

        Falls back to equal slices of `rect` when the projection disagrees with
        the line count, which happens on textured art or very tight leading.
        """
        x1, y1, x2, y2 = rect
        equal = [(x1, y1 + (y2 - y1) * i / count, x2, y1 + (y2 - y1) * (i + 1) / count)
                 for i in range(count)]
        h, w = image.shape[:2]
        px1, py1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        px2, py2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
        crop = image[py1:py2, px1:px2]
        if crop.size == 0:
            return equal
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if th.mean() > 127:
            th = 255 - th
        ink = th > 0
        row_ink = ink.sum(axis=1) > max(2, th.shape[1] * 0.02)
        runs, start = [], None
        for i, v in enumerate(row_ink):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(row_ink)))
        # Drop hairlines (bubble outline nicked by the box, antialiasing) so the
        # run count has a chance of matching the real line count.
        if runs:
            tallest = max(b - a for a, b in runs)
            runs = [r for r in runs if (r[1] - r[0]) >= max(3, tallest * 0.4)]
        if len(runs) != count:
            return equal
        # A shared OCR crop does NOT imply a shared font. Keep the real run
        # bounds, including the full width measured over the original height:
        # median-height replacement erased shouting/body hierarchy and could
        # also truncate the bottom (and hence width) of the larger lettering.
        if style_heights is not None:
            style_heights.extend(ModelVlmOCR._glyph_style_height(ink[a:b])
                                 for a, b in runs)
        out = []
        for a, b in runs:
            cols = np.where(ink[a:b].sum(axis=0) > 0)[0]
            if cols.size:
                bx1, bx2 = px1 + int(cols[0]), px1 + int(cols[-1]) + 1
            else:
                bx1, bx2 = x1, x2
            out.append((bx1, py1 + a, bx2, py1 + b))
        return out

    @staticmethod
    def _glyph_style_height(ink):
        """Robust evidence of letter size, not the extrema of a whole ink run.

        Punctuation, speckles and long balloon edges must not turn a normal
        dialogue line into a large-font style. Require several similarly tall
        glyph components; ambiguous/connected lettering stays unlabelled and
        retains the existing geometry-based merge policy.
        """
        height = ink.shape[0]
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            ink.astype(np.uint8), connectivity=8)
        sizes = [int(h) for x, y, w, h, area in stats[1:]
                 if max(3, height * 0.45) <= h
                 and 2 <= w <= height * 2
                 and area >= max(4, w * h * 0.08)]
        if len(sizes) < 3:
            return None
        q25, q75 = np.percentile(sizes, [25, 75])
        if q75 > q25 * 1.25:
            return None
        return float(np.median(sizes))
