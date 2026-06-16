"""
PaddleOCR-based detector for the custom manga-image-translator build.

Runs PaddleOCR (detection + recognition) and returns text lines with their
recognized text already attached, plus a stroke-level text mask for inpainting.
Pair with ``Ocr.paddle`` (a passthrough OCR) so the recognized text is kept.

Why a detector (not just an OCR): the user's problems (small note lines missed,
decorative dashed lines detected as text) are DETECTION problems. PaddleOCR's
detector handles product/document images much better than DBConvNeXt.
"""
from typing import List, Tuple
import cv2
import numpy as np

from .common import CommonDetector
from ..utils import Quadrilateral
from ..utils.paddle_engine import get_paddle_engine


class PaddleOcrDetector(CommonDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loaded = False

    # ModelWrapper-style no-ops (paddle manages its own model download/load)
    def is_downloaded(self) -> bool:
        return True

    async def download(self, force=False):
        pass

    async def load(self, device: str = 'cpu', *args, **kwargs):
        use_gpu = bool(device and str(device).startswith('cuda'))
        # Initialise the (lazy) singleton engine
        get_paddle_engine(lang='ch', use_gpu=use_gpu)
        self._loaded = True

    async def unload(self):
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def _build_stroke_mask(self, image: np.ndarray, boxes: List[np.ndarray]) -> np.ndarray:
        """Extract a stroke-level text mask (text=255) via per-box Otsu threshold.

        A stroke mask (not a filled box) is required so downstream mask
        refinement (connected-component based) keeps character strokes and
        inpainting erases only the text, avoiding ghosting.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = np.zeros((h, w), dtype=np.uint8)
        for pts in boxes:
            x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
            x2, y2 = min(x + bw, w), min(y + bh, h)
            x, y = max(x, 0), max(y, 0)
            if x2 <= x or y2 <= y:
                continue
            crop = gray[y:y2, x:x2]
            if crop.size == 0:
                continue
            _, th = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Text is the minority polarity within the box -> ensure text=255
            if th.mean() > 127:
                th = 255 - th
            poly_local = pts.astype(np.int32) - np.array([x, y])
            poly_mask = np.zeros_like(crop)
            cv2.fillPoly(poly_mask, [poly_local], 255)
            th = cv2.bitwise_and(th, poly_mask)
            mask[y:y2, x:x2] = cv2.bitwise_or(mask[y:y2, x:x2], th)
        # Slight dilation to fully cover anti-aliased edges
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask

    @staticmethod
    def _extract_colors(image: np.ndarray, pts: np.ndarray):
        """Extract dominant foreground and background RGB colours from a text box.

        Uses Otsu thresholding to separate text (minority dark/light pixels)
        from background, then computes the mean colour of each group.
        Returns (fg_r, fg_g, fg_b, bg_r, bg_g, bg_b).
        """
        h, w = image.shape[:2]
        x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        x2, y2 = min(x + bw, w), min(y + bh, h)
        x, y = max(x, 0), max(y, 0)
        if x2 <= x or y2 <= y:
            return 0, 0, 0, 0, 0, 0
        crop = image[y:y2, x:x2]
        if crop.size == 0:
            return 0, 0, 0, 0, 0, 0
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Text = minority polarity (fewer pixels)
        if th.mean() > 127:
            text_mask = (th == 0)
        else:
            text_mask = (th == 255)
        bg_mask = ~text_mask
        if text_mask.sum() < 4:
            return 0, 0, 0, 255, 255, 255
        fg_mean = crop[text_mask].mean(axis=0).astype(int)
        bg_mean = crop[bg_mask].mean(axis=0).astype(int) if bg_mask.sum() > 0 else np.array([255, 255, 255])
        return int(fg_mean[0]), int(fg_mean[1]), int(fg_mean[2]), int(bg_mean[0]), int(bg_mean[1]), int(bg_mean[2])

    async def _detect(self, image: np.ndarray, detect_size: int, text_threshold: float,
                      box_threshold: float, unclip_ratio: float, verbose: bool = False
                      ) -> Tuple[List[Quadrilateral], np.ndarray, np.ndarray]:
        engine = get_paddle_engine()
        results = engine.ocr(image)

        # text_threshold is reused as the minimum recognition confidence.
        # Paddle confidences are usually high; keep a sane floor so we don't
        # over-filter while still dropping very-low-confidence junk.
        conf_min = min(max(float(text_threshold), 0.1), 0.6)

        textlines: List[Quadrilateral] = []
        boxes = []
        for pts, text, conf in results:
            if not text or conf < conf_min:
                continue
            if pts.shape[0] != 4:
                x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
                pts = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.int64)
            fr, fg, fb, br, bg_c, bb = self._extract_colors(image, pts)
            textlines.append(Quadrilateral(pts.astype(np.int64), text, conf,
                                           fg_r=fr, fg_g=fg, fg_b=fb,
                                           bg_r=br, bg_g=bg_c, bg_b=bb))
            boxes.append(pts)

        mask_raw = self._build_stroke_mask(image, boxes) if boxes else \
            np.zeros(image.shape[:2], dtype=np.uint8)
        # Return mask=None so MT's mask refinement builds the final stroke mask.
        return textlines, mask_raw, None
