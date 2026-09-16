from typing import List
import cv2
import numpy as np

from .text_mask_utils import complete_mask_fill, complete_mask
from ..utils import TextBlock, Quadrilateral
from ..utils.bubble import is_ignore

def seed_unmasked_lines(raw_mask: np.ndarray, raw_image: np.ndarray,
                        text_regions: List[TextBlock], min_coverage: float = 0.02) -> int:
    """Add stroke mask for recognized lines the detector's mask never covered.

    `complete_mask` only GROWS mask that already exists inside a textline, so a
    line the detector's segmentation missed gets no mask at all, is never
    inpainted, and the original lettering stays visible underneath the
    translation. That happens on bold outlined text: on the reference page the
    detector masked the small "THAT" but not the "SKINSUIT" below it, even though
    it had returned a box covering both and OCR read both.

    Seeding is per-line and stroke-level (Otsu inside the line box), never a
    filled box: a solid box seed makes the refinement step treat the whole area
    as text and inpaint it flat, which leaves a visible patch over the artwork.

    Returns the number of lines seeded.
    """
    seeded = 0
    h, w = raw_mask.shape[:2]
    for region in text_regions:
        lines = getattr(region, 'lines', None)
        if lines is None:
            continue
        for line in lines:
            pts = np.array(line).reshape(-1, 2).astype(np.int32)
            x1, y1 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
            x2, y2 = min(w, pts[:, 0].max()), min(h, pts[:, 1].max())
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            window = raw_mask[y1:y2, x1:x2]
            if window.size == 0 or (window > 0).mean() >= min_coverage:
                continue
            crop = raw_image[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if th.mean() > 127:      # keep the minority (the glyphs), not the paper
                th = 255 - th
            # A line box tight around the glyphs still clips the balloon outline
            # at its corners often enough to matter, and masking the outline
            # makes inpainting eat the balloon edge. Ignore a 1px frame.
            th[0, :] = 0; th[-1, :] = 0; th[:, 0] = 0; th[:, -1] = 0
            if (th > 0).mean() < 0.01:
                continue                      # nothing that looks like ink
            raw_mask[y1:y2, x1:x2] = np.maximum(window, th)
            seeded += 1
    return seeded

def clip_mask_to_lines(final_mask: np.ndarray, text_regions: List[TextBlock]) -> np.ndarray:
    """Drop mask that lies outside the recognized text lines.

    Mask components are grown from connected pixels, so once a glyph's mask
    touches the balloon outline - which it does on a small balloon, where the
    lettering nearly reaches the edge and everything gets dilated - the outline
    becomes part of the same component and is erased with the text. The balloon
    then comes back as an outline-less blob, or with a bite taken out of it.

    Nothing outside a text line needs erasing by definition, so clipping to the
    lines bounds the damage to the balloon while still erasing all of the text.

    The margin only has to cover the antialiased edge of the glyphs and a little
    slack between the line box the recognizer reported and the actual strokes, so
    it is kept to a few pixels. A generous margin (0.35 of the font size, which
    was the first attempt) is itself enough to reach the outline of a small
    balloon, which defeats the point of clipping at all.
    """
    keep = np.zeros_like(final_mask)
    any_line = False
    for region in text_regions:
        lines = getattr(region, 'lines', None)
        if lines is None:
            continue
        margin = max(3, int(round(float(getattr(region, 'font_size', 0) or 0) * 0.15)))
        for line in lines:
            pts = np.array(line).reshape(-1, 2).astype(np.int32)
            x1, y1 = pts[:, 0].min() - margin, pts[:, 1].min() - margin
            x2, y2 = pts[:, 0].max() + margin, pts[:, 1].max() + margin
            cv2.rectangle(keep, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
            any_line = True
    if not any_line:
        return final_mask
    return cv2.bitwise_and(final_mask, keep)


async def dispatch(text_regions: List[TextBlock], raw_image: np.ndarray, raw_mask: np.ndarray, method: str = 'fit_text', dilation_offset: int = 0, ignore_bubble: int = 0, verbose: bool = False,kernel_size:int=3) -> np.ndarray:
    raw_mask = raw_mask.copy()   # seeding must not mutate the caller's mask_raw
    seed_unmasked_lines(raw_mask, raw_image, text_regions)

    # Larger sized mask images will probably have crisper and thinner mask segments due to being able to fit the text pixels better
    # so we dont want to size them down as much to not lose information
    scale_factor = max(min((raw_mask.shape[0] - raw_image.shape[0] / 3) / raw_mask.shape[0], 1), 0.5)

    img_resized = cv2.resize(raw_image, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)
    mask_resized = cv2.resize(raw_mask, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)

    mask_resized[mask_resized > 0] = 255
    textlines = []
    for region in text_regions:
        for l in region.lines:
            q = Quadrilateral(l * scale_factor, '', 0)
            textlines.append(q)

    final_mask = complete_mask(img_resized, mask_resized, textlines, dilation_offset=dilation_offset,kernel_size=kernel_size) if method == 'fit_text' else complete_mask_fill([txtln.aabb.xywh for txtln in textlines])
    if final_mask is None:
        final_mask = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype = np.uint8)
    else:
        final_mask = cv2.resize(final_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation = cv2.INTER_LINEAR)
        final_mask[final_mask > 0] = 255
        final_mask = clip_mask_to_lines(final_mask, text_regions)

    if ignore_bubble < 1 or ignore_bubble > 50:
        return final_mask

    # bubble
    kernel_size = int(max(final_mask.shape) * 0.025)  # 选择一个合适的核大小
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)  # 根据需要调整迭代次数
    # border
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        temp_mask = np.zeros_like(final_mask)
        # rect min
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, -1)
        # get textblock
        textblock=cv2.bitwise_and(raw_image, raw_image, mask=temp_mask)
        if is_ignore(textblock, ignore_bubble):
            cv2.drawContours(final_mask, [cnt], -1, 0, -1)

    return final_mask
