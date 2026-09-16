"""Ensemble detector (Detector.ensemble): union of several detectors' boxes.

No single detector finds every balloon on a page, and the ones they miss differ.
Measured on one reference page:

  * `ctd`     finds a small "HMM~" and a "THAT SKINSUIT" balloon, but misses the
              "GO AHEAD~ / I'LL WAIT FOR YOU" balloon completely, at every
              threshold from 0.5/0.7 down to 0.2/0.25 (its output does not move).
  * `default` finds "GO AHEAD~", but misses the other two.

Neither is fixable by tuning, so this runs both and unions the results. Recall is
what matters at this stage: a balloon that is never detected can never be
translated, whereas a spurious box costs nothing once `Ocr.vlm` reads it -
it groups boxes into balloons, transcribes them and returns nothing for crops
that hold no story text. Precision is delegated to the recognizer on purpose.

Configure the members with DETECT_ENSEMBLE (comma separated), default
"default,ctd".
"""
from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np

from .common import CommonDetector, OfflineDetector
from ..config import Detector
from ..utils import Quadrilateral


class EnsembleDetector(CommonDetector):
    async def _detect(self, image: np.ndarray, detect_size: int, text_threshold: float,
                      box_threshold: float, unclip_ratio: float, verbose: bool = False):
        from . import get_detector

        names = [n.strip() for n in
                 os.environ.get("DETECT_ENSEMBLE", "default,ctd").split(",") if n.strip()]
        textlines: List[Quadrilateral] = []
        mask_raw = None
        mask = None
        for name in names:
            try:
                key = Detector(name)
            except ValueError:
                self.logger.warning(f"ensemble: unknown detector {name!r}, skipped")
                continue
            if key is Detector.ensemble:
                continue  # would recurse
            try:
                det = get_detector(key)
                if isinstance(det, OfflineDetector):
                    await det.load(getattr(self, "device", "cpu"))
                tl, mr, m = await det.detect(
                    image, detect_size, text_threshold, box_threshold, unclip_ratio,
                    False, False, False, False, verbose)
            except Exception as e:
                self.logger.warning(f"ensemble: {name} failed ({e}), skipped")
                continue
            self.logger.info(f"ensemble: {name} -> {len(tl)} textlines")
            textlines.extend(tl)
            # Masks say which pixels to erase, so union them: a box found by only
            # one member still needs its ink removed before the translation is
            # drawn, otherwise the original text shows through underneath.
            # Members return masks at their own working resolution (measured
            # 2048x926 vs 2388x1080 for these two), so resize before combining.
            mask_raw = self._union_mask(mask_raw, mr, image.shape[:2])
            mask = self._union_mask(mask, m, image.shape[:2])

        before = len(textlines)
        textlines = self._drop_contained(textlines)
        self.logger.info(f"ensemble: {before} textlines -> {len(textlines)} after "
                         f"dropping duplicates")
        return textlines, mask_raw, mask

    @staticmethod
    def _union_mask(acc, new, shape):
        if new is None:
            return acc
        if new.shape[:2] != tuple(shape):
            new = cv2.resize(new, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        if acc is None:
            return new
        return np.maximum(acc, new)

    @staticmethod
    def _drop_contained(textlines: List[Quadrilateral],
                        contain_ratio: float = 0.8) -> List[Quadrilateral]:
        """Drop boxes almost entirely inside a bigger one.

        Members largely agree, so most boxes arrive two or three times. Exact
        duplicates are harmless for the VLM recognizer (it groups by position
        anyway) but they inflate every later log and make the box list unreadable
        when diagnosing. Only near-containment is removed - partial overlaps are
        kept, since they may be two genuinely different lines.
        """
        def xyxy(q):
            p = np.asarray(q.pts).reshape(-1, 2).astype(float)
            return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()

        items = sorted(((xyxy(t), t) for t in textlines),
                       key=lambda it: -((it[0][2] - it[0][0]) * (it[0][3] - it[0][1])))
        kept = []
        for box, t in items:
            area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            if any((max(0.0, min(box[2], k[2]) - max(box[0], k[0]))
                    * max(0.0, min(box[3], k[3]) - max(box[1], k[1]))) / area >= contain_ratio
                   for k, _ in kept):
                continue
            kept.append((box, t))
        return [t for _, t in kept]
