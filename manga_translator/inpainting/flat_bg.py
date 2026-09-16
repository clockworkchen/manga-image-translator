from typing import List

import cv2
import numpy as np

from ..mask_refinement import seed_unmasked_lines
from ..utils import get_logger

logger = get_logger('flat_bg')


def _fill_holes(region: np.ndarray) -> np.ndarray:
    """Return `region` with its enclosed holes filled."""
    inv = (~region).astype(np.uint8)
    # Flood the outside in from a border known to be outside, so what is left of
    # `inv` is exactly what is unreachable from outside: the holes.
    padded = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    cv2.floodFill(padded, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2),
                                   np.uint8), (0, 0), 0)
    return region | (padded[1:-1, 1:-1] > 0)


def _line_boxes(region, w: int, h: int):
    lines = getattr(region, 'lines', None)
    if lines is None:
        return []
    out = []
    for line in lines:
        pts = np.array(line).reshape(-1, 2).astype(np.int32)
        x1, y1 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
        x2, y2 = min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max()))
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            out.append((x1, y1, x2, y2))
    return out


def restore_flat_backgrounds(img_rgb: np.ndarray, mask: np.ndarray,
                             inpainted: np.ndarray, text_regions: List = None,
                             mask_raw: np.ndarray = None, tol: int = 22,
                             min_flat_frac: float = 0.55,
                             min_samples: int = 40) -> int:
    """Erase text on flat backgrounds by repainting the background colour, instead
    of letting the inpainter rebuild the area.

    A speech balloon is a flat fill, so there is nothing to infer there - but an
    inpainter does not know that, and it is not given the chance either. The erase
    mask has to over-cover the glyphs to guarantee the text is gone, which on a
    small balloon reaches the outline, and the inpainter then rebuilds the covered
    area from the nearest content outside it - the artwork the balloon sits on. So
    the balloon comes back filled with skin and clothing, or with a bite out of
    its outline. Three balloons on the reference page were destroyed this way.

    Where the background is flat, painting its exact colour is a perfect erase and
    needs no inference. This selects the balloon interior, floods it with the fill
    colour, glyphs and all, and puts the original pixels back everywhere else the
    mask erased - so the outline and the artwork around the balloon survive.

    The three measurements this rests on, and why each is taken the way it is:

    - The fill colour: the median inside the text line boxes, which the fill wins
      because it outnumbers the glyphs. Measuring it from a window around the
      erased area instead does not work - that area reaches the outline, so the
      window spills onto the page and the page colour wins. That read white
      balloons as page beige (234,231,225), and repainting a white balloon beige
      is its own defect.

    - The interior: the fill-coloured areas that meet the region's text, as
      opposed to the page outside the balloon, which is often within tolerance of
      the fill colour too (beige page, white balloon: ~20 apart). Told apart by
      whether the area reaches the edge of the window while holding little of the
      text footprint. This is measured per REGION, not per line: a line box sees
      the gap between two lines of its own balloon as page, because that gap
      reaches the edge of the line's window, and the letters next to it were then
      taken for page-side artwork and left on the page.

    - The glyphs: the stroke segmentation of each text line, plus dark shapes that
      touch the interior and not the page - the outline separates the two, so it
      touches both. Filling the interior's enclosed holes is not enough on its own:
      tall lettering cuts the interior into pockets, and half the letters of
      "WELL, LET'S GET STARTED" belonged to no pocket.

    Requiring the samples to agree on one colour is what makes this safe to run
    unconditionally: over artwork they disagree and the inpainter's output stands.

    Returns the number of regions repainted.
    """
    if mask is None or inpainted is None or not text_regions:
        return 0
    erased = mask > 0
    if not erased.any():
        return 0
    h, w = erased.shape[:2]

    # Segment the strokes here rather than relying on the detector's mask: by the
    # time inpainting runs, `ctx.mask_raw` is None, so depending on it meant the
    # glyph rescue below and the colour sampling silently ran in their degraded
    # form on every real page - the letters leaning on a balloon outline stayed on
    # the page, while an offline replay with the mask present looked correct.
    strokes = np.zeros((h, w), np.uint8)
    seed_unmasked_lines(strokes, img_rgb, text_regions)
    if mask_raw is not None:
        strokes |= (mask_raw > 0).astype(np.uint8) * 255
    strokes = strokes > 0
    k3 = np.ones((3, 3), np.uint8)

    fixed = 0
    for region in text_regions:
        boxes = _line_boxes(region, w, h)
        if not boxes:
            continue
        font = float(getattr(region, 'font_size', 0) or 0)

        samples = np.concatenate([img_rgb[y1:y2, x1:x2][~strokes[y1:y2, x1:x2]]
                                  for x1, y1, x2, y2 in boxes])
        if samples.shape[0] < min_samples:
            continue
        colour = np.median(samples, axis=0)
        flat = np.abs(samples.astype(np.int16)
                      - colour.astype(np.int16)).max(axis=1) <= tol
        logger.debug(f'flat_bg region {boxes[0]}: colour={colour.tolist()} '
                     f'flat={flat.mean():.2f}')
        if flat.mean() < min_flat_frac:
            continue              # textured background: leave the inpainter alone
        colour = np.median(samples[flat], axis=0).astype(inpainted.dtype)

        # Wide enough that the whole balloon fits in the window: whatever falls
        # outside it reaches the window edge and is taken for the page, and if that
        # happens to the balloon's own interior then the letters beside it are taken
        # for artwork and left on the page.
        pad = max(12, int(round(font * 1.5)))
        wx1 = max(0, min(b[0] for b in boxes) - pad)
        wy1 = max(0, min(b[1] for b in boxes) - pad)
        wx2 = min(w, max(b[2] for b in boxes) + pad)
        wy2 = min(h, max(b[3] for b in boxes) + pad)
        win = img_rgb[wy1:wy2, wx1:wx2]
        near = (np.abs(win.astype(np.int16) - colour.astype(np.int16)).max(axis=2)
                <= tol).astype(np.uint8)
        num, labels = cv2.connectedComponents(near, connectivity=4)[:2]

        footprint = np.zeros(labels.shape, bool)
        for x1, y1, x2, y2 in boxes:
            footprint[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1] = True
        counts = np.bincount(labels[footprint & (near > 0)].ravel(), minlength=num)
        counts[0] = 0
        border = set(labels[0, :]) | set(labels[-1, :]) \
            | set(labels[:, 0]) | set(labels[:, -1])
        fp_area = max(1, int(footprint.sum()))
        pockets = [lb for lb in np.nonzero(counts)[0]
                   if lb not in border or counts[lb] >= 0.15 * fp_area]
        if not pockets:
            logger.debug(f'flat_bg region {boxes[0]}: no interior found')
            continue
        interior = np.isin(labels, pockets)
        outside = np.isin(labels, [lb for lb in border if lb and lb not in pockets])

        num_d, labels_d, stats_d, _ = cv2.connectedComponentsWithStats(
            (~interior & ~outside).astype(np.uint8), connectivity=8)
        paint = interior.copy()
        for label in range(1, num_d):
            x, y = stats_d[label, cv2.CC_STAT_LEFT], stats_d[label, cv2.CC_STAT_TOP]
            bw, bh = stats_d[label, cv2.CC_STAT_WIDTH], stats_d[label, cv2.CC_STAT_HEIGHT]
            sub = (labels_d[y:y + bh, x:x + bw] == label).astype(np.uint8)
            ring = (cv2.dilate(sub, k3) > 0) & (sub == 0)
            if outside[y:y + bh, x:x + bw][ring].any():
                continue
            if interior[y:y + bh, x:x + bw][ring].any():
                paint[y:y + bh, x:x + bw] |= sub > 0

        # Letters that touch the outline are one component with it - and with the
        # artwork beyond it, once the outline is crossed (measured on "WELL, LET'S
        # GET STARTED": a single 211x123 blob holding half the letters, the outline
        # and the artwork). Nothing about the shape separates them at that point, but
        # the stroke segmentation does, because it only looks inside the text lines.
        # Distance from the page does not work here - the gap between a letter and
        # the outline it touches is thinner than the outline itself, so no threshold
        # keeps one without the other.
        in_strokes = cv2.dilate(
            (strokes[wy1:wy2, wx1:wx2] & footprint).astype(np.uint8), k3) > 0
        # Grown by a pixel for the antialiased edge the segmentation stops short of,
        # which would otherwise stay as a grey ghost of the lettering.
        paint |= in_strokes
        paint = _fill_holes(paint)

        # Repaint only what needs it - what the mask erased, and the strokes - and
        # leave the rest of the interior alone even though it is the same colour by
        # definition. "The same colour" is the same only to within the tolerance
        # that made it one region: on a background with a slight gradient, flooding
        # the whole selection with one value replaces the gradient with a flat
        # rectangle, which is what this looked like on four panels of the reference
        # page.
        paint &= erased[wy1:wy2, wx1:wx2] | in_strokes

        # Put the original back wherever the mask erased something this pass is not
        # repainting: the outline the mask chewed off, and the artwork outside the
        # balloon it overran. Reverting before deciding what to paint - which is
        # what an earlier version did - undid correct erases and put whole words
        # back on the page.
        revert = erased[wy1:wy2, wx1:wx2] & ~paint
        inpainted[wy1:wy2, wx1:wx2][revert] = win[revert]
        inpainted[wy1:wy2, wx1:wx2][paint] = colour
        logger.debug(f'flat_bg {boxes[0]} colour={colour.tolist()} '
                     f'paint={int(paint.sum())} revert={int(revert.sum())}')
        fixed += 1
    return fixed
