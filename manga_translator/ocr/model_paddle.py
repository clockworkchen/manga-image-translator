"""
Paddle passthrough OCR for the custom manga-image-translator build.

PaddleOcrDetector (Detector.paddle_ocr) already recognizes text during
detection and attaches it to each textline. This OCR simply keeps that text.
If a textline has no text yet (e.g. used with a different detector), it falls
back to running PaddleOCR recognition on the cropped region.
"""
from typing import List
import numpy as np

from .common import CommonOCR
from ..config import OcrConfig
from ..utils import Quadrilateral
from ..utils.paddle_engine import get_paddle_engine


class ModelPaddleOCR(CommonOCR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _recognize(self, image: np.ndarray, textlines: List[Quadrilateral],
                         config: OcrConfig, verbose: bool = False) -> List[Quadrilateral]:
        # Fast path: text already attached by PaddleOcrDetector.
        if all(getattr(t, "text", "") for t in textlines):
            return textlines

        # Fallback: recognize any textlines missing text using paddle on the crop.
        engine = get_paddle_engine()
        out: List[Quadrilateral] = []
        for t in textlines:
            if getattr(t, "text", ""):
                out.append(t)
                continue
            try:
                x1, y1, x2, y2 = t.xyxy
                crop = image[int(y1):int(y2), int(x1):int(x2)]
                res = engine.ocr(crop) if crop.size else []
                text = " ".join(r[1] for r in res) if res else ""
                prob = max((r[2] for r in res), default=0.0)
            except Exception:
                text, prob = "", 0.0
            out.append(Quadrilateral(t.pts, text, prob))
        return out
