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
speech bubble or a caption. The detector found {expected_lines} separate line
band(s) in this crop. Transcribe its text EXACTLY as printed.

Rules:
1. Return one array entry per printed line, in reading order. The detector line
   count is geometric evidence: normally return exactly {expected_lines}
   entries. Do not join a short middle line into either neighbour and do not
   omit a line merely because it overlaps another detector box.
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

        # Request-level Panel selection wins, so changing the OCR model/channel
        # applies to the next task without recreating the MT container. Dedicated
        # env vars and legacy CUSTOM_OPENAI_* remain compatibility fallbacks.
        base = (config.vlm_api_base or os.environ.get("VLM_OCR_API_BASE")
                or os.environ.get("CUSTOM_OPENAI_API_BASE"))
        key = (config.vlm_api_key or os.environ.get("VLM_OCR_API_KEY")
               or os.environ.get("CUSTOM_OPENAI_API_KEY"))
        model = (config.vlm_model or os.environ.get("VLM_OCR_MODEL")
                 or os.environ.get("CUSTOM_OPENAI_MODEL"))
        if not base or not model:
            self.logger.warning("vlm ocr: no API base/model configured, using 48px_ctc")
            return await self._run_fallback(image, textlines, config, verbose)
        source = "request" if config.vlm_model else (
            "VLM_OCR env" if os.environ.get("VLM_OCR_MODEL") else "CUSTOM_OPENAI env")
        self.logger.info("vlm ocr: model=%s source=%s", model, source)

        # Read at BUBBLE level, not per detection box. Detector boxes overlap and
        # under-cover multi-line text, so a per-box crop shows fragments of its
        # neighbours and every shared line gets transcribed two or three times -
        # which then renders on top of itself. One crop per bubble contains each
        # line exactly once, and the VLM reports the line breaks, which is what
        # the renderer needs to avoid collapsing the translation onto one line.
        block_members = self._group_box_members(textlines)
        blocks = [block[0] for block in block_members]

        # One request per block, concurrently. Batching many crops into a single
        # request and matching answers by index looked cheaper but is unsafe: on
        # the reference page one crop held more text than the model expected, it
        # answered under the next index, and every block after it received its
        # neighbour's dialogue. A crop per request has nothing to misalign.
        concurrency = config.vlm_concurrency or int(os.environ.get("VLM_OCR_CONCURRENCY", "6"))
        sem = asyncio.Semaphore(max(1, min(int(concurrency), 32)))
        failures = 0
        blocked = asyncio.Event()

        async def one(block):
            nonlocal failures
            rect, members = block
            async with sem:
                if blocked.is_set():
                    raise VlmRequestBlocked('VLM batch stopped after upstream refusal')
                try:
                    member_heights = []
                    for index in members:
                        points = np.asarray(textlines[index].pts).reshape(-1, 2)
                        member_heights.append(float(points[:, 1].max() - points[:, 1].min()))
                    line_height = float(np.median(member_heights)) if member_heights else 16.0
                    crop = self._crop(
                        image, rect,
                        pad=(max(8, int(round(line_height * 0.65))),
                             max(8, int(round(line_height * 0.30)))))
                    answer = await self._ask(
                        base, key, model, crop, self._where(image, rect),
                        len(members), config.vlm_timeout)
                    # A short answer is the characteristic VLM truncation mode:
                    # it reads the first visible row and silently drops the rest.
                    # Retry that block once with more context instead of widening
                    # detector/merge thresholds for the whole page.
                    returned = [line for line in str(answer or '').splitlines() if line.strip()]
                    if len(members) > 1 and len(returned) != len(members):
                        retry = await self._ask(
                            base, key, model,
                            self._crop(
                                image, rect,
                                pad=(max(12, int(round(line_height * 1.0))),
                                     max(10, int(round(line_height * 0.50))))),
                            self._where(image, rect) +
                            f'; previous OCR returned {len(returned)} rows, re-check every printed row',
                            len(members), config.vlm_timeout)
                        retry_lines = [line for line in str(retry or '').splitlines() if line.strip()]
                        if abs(len(retry_lines) - len(members)) < abs(len(returned) - len(members)):
                            answer = retry
                    return answer
                except VlmRequestBlocked:
                    blocked.set()
                    raise
                except Exception as e:
                    failures += 1
                    self.logger.warning(f"vlm ocr: block failed ({e})")
                    return None

        tasks = [asyncio.create_task(one(block)) for block in block_members]
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
        for block_id, ((rect, members), crop_text) in enumerate(zip(block_members, texts)):
            if crop_text is None:
                # A transport/parse failure is not evidence of an empty bubble.
                # Never silently discard it while returning a successful page.
                affected = [textlines[index] for index in members]
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
            member_boxes = [textlines[index] for index in members]
            # Geometry is detector-owned. When the VLM returns fewer strings than
            # detector rows, assign text to row groups rather than stretching one
            # line over the union rectangle. This keeps OCR crop context separate
            # from render geometry and prevents duplicated/nested fragments.
            bands, line_groups = self._align_lines_to_detector(image, rect, lines, member_boxes)
            generated = self._split_into_lines(
                image, rect, lines, self._block_rotation(member_boxes, rect),
                detected_lines=None, aligned_bands=bands)
            for line, group in zip(generated, line_groups):
                line.ocr_detector_rows = int(group)
                line.ocr_source_rows = len(member_boxes)
                line.ocr_complete_block = len(lines) == len(member_boxes)
                line.ocr_block_rect = tuple(float(value) for value in rect)
            for line in generated:
                line.ocr_block_id = block_id
            out.extend(generated)
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

    @classmethod
    def _group_boxes(cls, textlines: List[Quadrilateral]) -> List[List[float]]:
        """Backward-compatible geometry-only view used by detector diagnostics."""
        return [rect for rect, _ in cls._group_box_members(textlines)]

    @staticmethod
    def _group_box_members(textlines: List[Quadrilateral]):
        """Union detector lines and retain exact membership for OCR alignment.

        Keeping the member indices is essential: containment against a union
        rectangle also catches unrelated nested/overlapping detections. That was
        how a three-line bubble became a two-line OCR region plus a duplicated
        one-line fragment, producing the production R0/R1 overlap.
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
            groups.setdefault(find(i), []).append(i)
        out = []
        for member_indices in groups.values():
            member_indices.sort(key=lambda index: (rects[index][1], rects[index][0]))
            members = [rects[index] for index in member_indices]
            rect = [min(r[0] for r in members), min(r[1] for r in members),
                    max(r[2] for r in members), max(r[3] for r in members)]
            out.append((rect, member_indices))
        # Reading order, so batches stay locally coherent and the returned JSON
        # is easy to line up against the page when debugging.
        out.sort(key=lambda item: (item[0][1], item[0][0]))
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
    def _crop(image: np.ndarray, rect: List[float], pad=8) -> np.ndarray:
        h, w = image.shape[:2]
        pad_x, pad_y = pad if isinstance(pad, tuple) else (pad, pad)
        x1 = max(0, int(rect[0]) - int(pad_x))
        y1 = max(0, int(rect[1]) - int(pad_y))
        x2 = min(w, int(rect[2]) + int(pad_x))
        y2 = min(h, int(rect[3]) + int(pad_y))
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
                   crop: np.ndarray, hint: str, expected_lines: int,
                   configured_timeout: Optional[float] = None) -> Optional[str]:
        import httpx

        ok, buf = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        content = [
            {"type": "text", "text": _PROMPT.format(
                hint=hint, expected_lines=max(1, int(expected_lines)))},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]

        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        timeout = float(configured_timeout or os.environ.get("VLM_OCR_TIMEOUT", "120"))
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
    @staticmethod
    def _block_rotation(textlines, rect):
        """Robust signed detector rotation in degrees for one OCR crop."""
        x1, y1, x2, y2 = rect
        values = []
        for detected in textlines:
            pts = np.asarray(detected.pts).reshape(-1, 2).astype(float)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            # Choose the longer adjacent edge as the text baseline, then normalize
            # to [-90, 90). Point order can reverse, so 180 degrees is equivalent.
            edges = [(pts[(i+1) % 4] - pts[i]) for i in range(4)]
            vector = max(edges, key=lambda edge: np.linalg.norm(edge))
            angle = np.degrees(np.arctan2(vector[1], vector[0]))
            angle = (angle + 90) % 180 - 90
            if abs(angle) <= 45:
                values.append(angle)
        return float(np.median(values)) if values else 0.0

    def _split_into_lines(self, image: np.ndarray, rect: List[float],
                          lines: List[str], rotation: float = 0.0,
                          detected_lines=None, aligned_bands=None) -> List[Quadrilateral]:
        """Turn one OCR block into one geometrically grounded quad per line.

        Detector lines are the primary geometry. Projection remains a fallback
        only when OCR and detector counts disagree. This avoids assigning the
        full crop width to every VLM line and prevents a short middle line from
        becoming a separate nested region that overlaps its own bubble.
        """
        style_heights = []
        bands = aligned_bands or self._detected_bands(detected_lines, len(lines))
        if bands is None:
            bands = self._ink_bands(image, rect, len(lines), style_heights=style_heights)
        out = []
        for i, (text, (bx1, by1, bx2, by2)) in enumerate(zip(lines, bands)):
            box = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
                           dtype=np.float32)
            if abs(rotation) >= 3:
                center = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
                radians = np.radians(rotation)
                matrix = np.array([[np.cos(radians), -np.sin(radians)],
                                   [np.sin(radians), np.cos(radians)]])
                box = (box - center) @ matrix.T + center
            line = Quadrilateral(box, text, 1.0)
            fg, bg = self._extract_line_colors(image, box)
            line.fg_r, line.fg_g, line.fg_b = (int(value) for value in fg)
            line.bg_r, line.bg_g, line.bg_b = (int(value) for value in bg)
            # Equal-slice fallback has no measured style evidence. Do not label
            # its guessed height as a real font boundary.
            line.ocr_ink_height = float(by2 - by1)
            if style_heights:
                line.ocr_style_height = style_heights[i]
            out.append(line)
        return out

    @staticmethod
    def _detected_bands(detected_lines, count):
        """Return exact detector boxes when they map one-to-one to VLM lines."""
        if not detected_lines or len(detected_lines) != count:
            return None
        bands = []
        for detected in detected_lines:
            pts = np.asarray(detected.pts).reshape(-1, 2).astype(float)
            bands.append((float(pts[:, 0].min()), float(pts[:, 1].min()),
                          float(pts[:, 0].max()), float(pts[:, 1].max())))
        bands.sort(key=lambda box: (box[1], box[0]))
        return ModelVlmOCR._regularize_vertical_bands(bands)

    @classmethod
    def _align_lines_to_detector(cls, image, rect, lines, detected_lines):
        """Map VLM strings onto contiguous detector-row groups.

        Equal slicing of a union crop created synthetic 49-65px "font sizes" on
        blocks where the VLM returned one string for several detector rows. The
        groups here preserve the detector's outer geometry and expose the number
        of source rows carried by each OCR string to the bubble renderer.
        """
        bands = cls._detected_bands(detected_lines, len(detected_lines)) or []
        if not bands or len(lines) >= len(bands):
            exact = bands if len(lines) == len(bands) else cls._ink_bands(image, rect, len(lines))
            return exact, [1] * len(lines)
        count, rows = len(lines), len(bands)
        weights = [max(1, len(re.sub(r'\s+', '', line))) for line in lines]
        raw = np.asarray(weights, dtype=float) * rows / max(sum(weights), 1)
        groups = [max(1, int(np.floor(value))) for value in raw]
        while sum(groups) < rows:
            fractions = raw - np.floor(raw)
            choices = sorted(range(count), key=lambda i: (fractions[i], weights[i]), reverse=True)
            groups[choices[(sum(groups) - count) % count]] += 1
        while sum(groups) > rows:
            choices = sorted((i for i in range(count) if groups[i] > 1),
                             key=lambda i: (raw[i] - groups[i], weights[i]))
            if not choices:
                break
            groups[choices[0]] -= 1
        aligned, offset = [], 0
        for group in groups:
            owned = bands[offset:offset + group]
            aligned.append((min(box[0] for box in owned), min(box[1] for box in owned),
                            max(box[2] for box in owned), max(box[3] for box in owned)))
            offset += group
        return aligned, groups

    @staticmethod
    def _regularize_vertical_bands(bands):
        """Clip overlapping detector rows at their centroid midpoint.

        Ensemble boxes often overlap by 30-60% vertically. Keeping that overlap
        duplicates source ink in two OCR rows and later produces intersecting
        regions. Midpoint boundaries retain every row's centre while making
        ownership disjoint; horizontal extents remain detector-derived.
        """
        if len(bands) < 2:
            return bands
        result = [list(box) for box in bands]
        for i in range(len(result) - 1):
            upper, lower = result[i], result[i + 1]
            if upper[3] <= lower[1]:
                continue
            upper_center = (upper[1] + upper[3]) / 2
            lower_center = (lower[1] + lower[3]) / 2
            boundary = (upper_center + lower_center) / 2
            upper[3] = max(upper[1] + 1, boundary)
            lower[1] = min(lower[3] - 1, boundary)
        return [tuple(box) for box in result]

    @staticmethod
    def _extract_line_colors(image: np.ndarray, box: np.ndarray):
        """Estimate glyph/background RGB for VLM OCR lines.

        VLM only returns text, so the old path left every line at Quadrilateral's
        black-on-black defaults. Rendering then changed coloured lettering to
        black. Use Otsu solely for polarity separation and robust medians for the
        colours; the tight line box makes the minority cluster the glyph ink.
        """
        h, w = image.shape[:2]
        pts = np.asarray(box).reshape(-1, 2).astype(np.int32)
        x1, y1 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
        x2, y2 = min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max()))
        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0, 0]), np.array([255, 255, 255])
        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark, light = threshold == 0, threshold == 255
        text_mask, bg_mask = (dark, light) if dark.sum() <= light.sum() else (light, dark)
        if text_mask.sum() < 4 or bg_mask.sum() < 4:
            return np.array([0, 0, 0]), np.array([255, 255, 255])
        fg = np.rint(np.median(crop[text_mask], axis=0)).astype(np.int32)
        bg = np.rint(np.median(crop[bg_mask], axis=0)).astype(np.int32)
        return fg, bg

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
