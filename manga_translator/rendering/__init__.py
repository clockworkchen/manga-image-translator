import os
import re
import cv2
import numpy as np
from PIL import Image
from typing import List
from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

# from .ballon_extractor import extract_ballon_region
from . import text_render
from .text_render_eng import render_textblock_list_eng
from .text_render_pillow_eng import render_textblock_list_eng as render_textblock_list_eng_pillow
from ..utils import (
    BASE_PATH,
    TextBlock,
    color_difference,
    get_logger,
    rotate_polygons,
)

logger = get_logger('render')

def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    if path:
        parsed = path.split(',')
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed

def fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg

def count_text_length(text: str) -> float:
    """Calculate text length, treating っッぁぃぅぇぉ as 0.5 characters"""
    half_width_chars = 'っッぁぃぅぇぉ'  
    length = 0.0
    for char in text.strip():
        if char in half_width_chars:
            length += 0.5
        else:
            length += 1.0
    return length

def resize_regions_to_font_size(img: np.ndarray, text_regions: List['TextBlock'], font_size_fixed: int, font_size_offset: int, font_size_minimum: int, overflow_strategy: str = "cascade", max_font_shrink_ratio: float = 0.5):  
    """
    Adjust text region size to accommodate font size and translated text length.
    
    Args:  
        img: Input image
        text_regions: List of text regions to process
        font_size_fixed: Fixed font size (overrides other font parameters)
        font_size_offset: Font size offset
        font_size_minimum: Minimum font size (-1 for auto-calculation)

    Returns:  
        List of adjusted text region bounding boxes
    """    
    
    # Define minimum font size
    if font_size_minimum == -1:  
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 200)  
    # logger.debug(f'font_size_minimum {font_size_minimum}')  
    font_size_minimum = max(1, font_size_minimum)  

    dst_points_list = []  
    for region in text_regions: 
    
        # Store and validate original font size
        original_region_font_size = region.font_size  
        if original_region_font_size <= 0:  
            # logger.warning(f"Invalid original font size ({original_region_font_size}) for text '{region.translation}'. Using default value {font_size_minimum}.")  
            original_region_font_size = font_size_minimum
        # Remember the detected (pre-resize) font size for diagnostics
        region._orig_font_size = int(original_region_font_size)

        # Determine target font size
        current_base_font_size = original_region_font_size  
        if font_size_fixed is not None:  
            target_font_size = font_size_fixed  
        else:  
            target_font_size = current_base_font_size + font_size_offset  

        target_font_size = max(target_font_size, font_size_minimum, 1)  
        # print("-" * 50)
        # logger.debug(f"Calculated target font size: {target_font_size} for text '{region.translation}'")  

        # Single-axis text box expansion
        single_axis_expanded = False
        dst_points = None
        
        if region.horizontal: 
            used_rows = len(region.texts)
            # logger.debug(f"Horizontal text - used rows: {used_rows}")
            
            line_text_list, _ = text_render.calc_horizontal(
                region.font_size,
                region.translation,
                max_width=region.unrotated_size[0],
                max_height=region.unrotated_size[1],
                language=getattr(region, "target_lang", "en_US")
            )
            needed_rows = len(line_text_list)
            # logger.debug(f"Needed rows: {needed_rows}")                

            if needed_rows > used_rows:
                scale_x = ((needed_rows - used_rows) / used_rows) * 1 + 1
                try:  
                    poly = Polygon(region.unrotated_min_rect[0])
                    minx, miny, maxx, maxy = poly.bounds
                    # Use center origin for centered text to expand symmetrically
                    scale_origin = ((minx + maxx) / 2, miny) if region.alignment in ('center', 'auto') else (minx, miny)
                    poly = affinity.scale(poly, xfact=scale_x, yfact=1.0, origin=scale_origin)        
                
                    pts = np.array(poly.exterior.coords[:4])  
                    dst_points = rotate_polygons(  
                        region.center, pts.reshape(1, -1), -region.angle,  
                        to_int=False  
                    ).reshape(-1, 4, 2)  
                    # 移除边界限制，允许文本超出检测框边界
                    # dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1] - 1)  
                    # dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0] - 1)  
                    dst_points = dst_points.astype(np.int64)
                    single_axis_expanded = True
                    # logger.debug(f"Successfully expanded horizontal text width: xfact={scale_x:.2f}")  
                except Exception as e:  
                    # logger.error(f"Failed to expand horizontal text: {e}")  
                    pass
                    
        if region.vertical:
            used_cols = len(region.texts)
            # logger.debug(f"Vertical text - used columns: {used_cols}")
            
            line_text_list, _ = text_render.calc_vertical(
                region.font_size, 
                region.translation, 
                max_height=region.unrotated_size[1],
            )
            needed_cols = len(line_text_list)
            # logger.debug(f"Needed columns: {needed_cols}") 
            if needed_cols > used_cols:
                scale_x = ((needed_cols - used_cols) / used_cols) * 1 + 1
                try:  
                    poly = Polygon(region.unrotated_min_rect[0])
                    minx, miny, maxx, maxy = poly.bounds
                    # Use center origin for centered text to expand symmetrically
                    scale_origin = (minx, (miny + maxy) / 2) if region.alignment in ('center', 'auto') else (minx, miny)
                    poly = affinity.scale(poly, xfact=1.0, yfact=scale_x, origin=scale_origin)                    
                    
                    pts = np.array(poly.exterior.coords[:4])  
                    dst_points = rotate_polygons(  
                        region.center, pts.reshape(1, -1), -region.angle,  
                        to_int=False  
                    ).reshape(-1, 4, 2)  
                    # 移除边界限制，允许文本超出检测框边界
                    # dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1] - 1)  
                    # dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0] - 1)  
                    dst_points = dst_points.astype(np.int64)
                    single_axis_expanded = True
                    # logger.debug(f"Successfully expanded vertical text width: xfact={scale_x:.2f}")  
                except Exception as e:  
                    # logger.error(f"Failed to expand vertical text: {e}")  
                    pass

        # If single-axis expansion failed, use general scaling
        if not single_axis_expanded:
            # Calculate scaling factor based on text length ratio
            orig_text = getattr(region, "text_raw", region.text)
            char_count_orig = count_text_length(orig_text)
            char_count_trans = count_text_length(region.translation.strip())     
            length_ratio = 1.0

            if char_count_orig > 0 and char_count_trans > char_count_orig:  
                increase_percentage = (char_count_trans - char_count_orig) / char_count_orig
                font_increase_ratio = 1 + (increase_percentage * 0.3)
                font_increase_ratio = min(1.5, max(1.0, font_increase_ratio))
                # logger.debug(f"Translation is {increase_percentage:.2%} longer, font increase ratio: {font_increase_ratio:.2f}")
                target_font_size = int(target_font_size * font_increase_ratio)
                # logger.debug(f"Adjusted target font size: {target_font_size}")
                # Need greater bounding box scaling to accommodate larger font size and longer text
                target_scale = max(1, min(1 + increase_percentage * 0.3, 2))  # Possibly max(1, min(1 + (font_increase_ratio-1), 2))
                # logger.debug(f"Translation is longer than original and font increased, need larger bounding box scaling. Target scale factor: {target_scale:.2f}")
            # Short text box expansion is quite aggressive, in many cases short text boxes don't need expansion
            # elif char_count_orig > 0 and char_count_trans < char_count_orig:
            #     # Translation is shorter, increase font proportionally
            #     decrease_percentage = (char_count_orig - char_count_trans) / char_count_orig
            #     # Font increase ratio equals text reduction ratio
            #     font_increase_ratio = 1 + decrease_percentage
            #     # Limit font increase ratio to reasonable range, e.g., between 1.0 and 1.5
            #     font_increase_ratio = min(1.5, max(1.0, font_increase_ratio))
            #     logger.debug(f"Translation is {decrease_percentage:.2%} shorter than original, font increase ratio: {font_increase_ratio:.2f}")
            #     # Update target font size
            #     target_font_size = int(target_font_size * font_increase_ratio)
            #     logger.debug(f"Adjusted target font size: {target_font_size}")
            #     target_scale = 1.0  # No additional bounding box scaling needed
            #     logger.debug(f"Translation is shorter than original, no bounding box scaling applied, only font increase. Target scale factor: {target_scale:.2f}")            
            else:  
                target_scale = 1              
                # logger.debug(f"No length ratio scaling applied. Target scale factor: {target_scale:.2f}")   

            # Calculate final scaling factor
            font_size_scale = (((target_font_size - original_region_font_size) / original_region_font_size) * 0.4 + 1) if original_region_font_size > 0 else 1.0  
            # logger.debug(f"Font size ratio: ({target_font_size} / {original_region_font_size})")  
            final_scale = max(font_size_scale, target_scale)
            final_scale = max(1, min(final_scale, 1.1))  
            
            # logger.debug(f"Final scaling factor: {final_scale:.2f}")  

            # Scale bounding box if needed
            if final_scale > 1.001:  
                # logger.debug(f"Scaling bounding box: text='{region.translation}', scale={final_scale:.2f}")  
                try:  
                    poly = Polygon(region.unrotated_min_rect[0])  
                     # Scale from the center  
                    poly = affinity.scale(poly, xfact=final_scale, yfact=final_scale, origin='center')  
                    scaled_unrotated_points = np.array(poly.exterior.coords[:4])  

                    dst_points = rotate_polygons(region.center, scaled_unrotated_points.reshape(1, -1), -region.angle, to_int=False).reshape(-1, 4, 2)  
                    # 移除边界限制，允许文本超出检测框边界
                    # dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1] - 1)  
                    # dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0] - 1)  
                    dst_points = dst_points.astype(np.int64)  
                    dst_points = dst_points.reshape((-1, 4, 2))  
                    # logger.debug(f"Finished calculating scaled dst_points.")  

                except Exception as e:  
                    # logger.error(f"Error during scaling for text '{region.translation}': {e}. Using original min_rect.")  
                    dst_points = region.min_rect
            else:
                dst_points = region.min_rect

        # -- Overflow handling: cascade strategy --
        # Strategy: "cascade" (default) = expand > shift > wrap > shrink (priority order)
        #           "expand" = expand only (allow overflow, for comics)
        #           "expand_wrap" = left-align, expand right to boundary, auto-wrap
        #           "wrap" = no expansion, let text wrap at original font size
        #           "shrink" = no expansion, shrink font to fit in original lines
        img_h, img_w = img.shape[:2]
        
        if overflow_strategy in ("cascade", "auto", "bubble") and dst_points is not None:
            # Step 1: Check if expanded dst_points fits within image
            pts = dst_points.reshape(-1, 2) if dst_points.ndim > 2 else dst_points
            min_x, max_x = float(pts[:, 0].min()), float(pts[:, 0].max())
            overflow_right = max(0, max_x - img_w + 1)
            overflow_left = max(0, -min_x)
            
            if overflow_right > 0 or overflow_left > 0:
                # Step 2: Try shifting into bounds first
                shift_x = 0
                if overflow_right > 0 and min_x > overflow_right:
                    shift_x = -overflow_right
                elif overflow_left > 0:
                    shift_x = overflow_left
                
                if shift_x != 0:
                    dst_points = dst_points.copy()
                    if dst_points.ndim == 3:
                        dst_points[:, :, 0] = dst_points[:, :, 0] + shift_x
                    else:
                        dst_points[:, 0] = dst_points[:, 0] + shift_x
                
                # Re-check after shift
                pts2 = dst_points.reshape(-1, 2)
                still_overflow = float(pts2[:, 0].max()) > img_w - 1 or float(pts2[:, 0].min()) < 0
                
                if still_overflow and single_axis_expanded:
                    # Step 2b: Full expansion too wide - try max-fit expansion
                    # Re-expand from original polygon with clamped scale to fill
                    # available image width (with small margin)
                    try:
                        margin = 4
                        poly_orig = Polygon(region.unrotated_min_rect[0])
                        orig_minx, _, orig_maxx, _ = poly_orig.bounds
                        orig_width = orig_maxx - orig_minx
                        center_x_unrot = (orig_minx + orig_maxx) / 2
                        # Max expansion: use full image width minus margin
                        max_avail_width = img_w - margin * 2
                        if orig_width > 0 and max_avail_width > orig_width:
                            clamped_scale = max_avail_width / orig_width
                            scale_origin = ((orig_minx + orig_maxx) / 2, poly_orig.bounds[1])
                            poly_clamped = affinity.scale(poly_orig, xfact=clamped_scale, yfact=1.0, origin=scale_origin)
                            pts_c = np.array(poly_clamped.exterior.coords[:4])
                            dst_points = rotate_polygons(
                                region.center, pts_c.reshape(1, -1), -region.angle,
                                to_int=False
                            ).reshape(-1, 4, 2)
                            # Center horizontally in image
                            pts_c2 = dst_points.reshape(-1, 2)
                            cx_min, cx_max = float(pts_c2[:, 0].min()), float(pts_c2[:, 0].max())
                            shift_to_center = margin - cx_min if cx_min < margin else 0
                            if cx_max + shift_to_center > img_w - margin:
                                shift_to_center = img_w - margin - cx_max
                            if shift_to_center != 0:
                                dst_points[:, :, 0] = dst_points[:, :, 0] + shift_to_center
                            dst_points = dst_points.astype(np.int64)
                            still_overflow = False
                    except Exception:
                        pass
                
                if still_overflow:
                    # Step 3: Fall back to original box (text wraps naturally)
                    dst_points = region.min_rect
                    # Only shrink as LAST resort if original box is somehow out of bounds
                    pts3 = dst_points.reshape(-1, 2) if dst_points.ndim > 2 else dst_points
                    if float(pts3[:, 0].max()) > img_w - 1 or float(pts3[:, 0].min()) < 0:
                        min_font = max(int(original_region_font_size * max_font_shrink_ratio), font_size_minimum, 4)
                        target_font_size = max(min_font, int(target_font_size * 0.7))
        
        elif overflow_strategy == "shrink" and dst_points is not None:
            # Shrink-only: use original box and reduce font
            pts = dst_points.reshape(-1, 2) if dst_points.ndim > 2 else dst_points
            if float(pts[:, 0].max()) > img_w - 1 or float(pts[:, 0].min()) < 0:
                min_font = max(int(original_region_font_size * max_font_shrink_ratio), font_size_minimum, 4)
                target_font_size = max(min_font, int(target_font_size * 0.7))
                dst_points = region.min_rect
        
        elif overflow_strategy == "wrap":
            # Wrap-only: always use original box, no expansion, no shrink
            dst_points = region.min_rect
        
        elif overflow_strategy == "expand_wrap":
            # Expand+Wrap: let _fit_regions_cascade handle the layout
            # (expand right from left edge, wrap at boundary).
            # Start with original box; cascade will expand to boundary.
            dst_points = region.min_rect
        
        # "expand" strategy: keep expanded dst_points as-is (allow overflow)

        # Store results and update font size
        dst_points_list.append(dst_points)  
        region.font_size = int(target_font_size)

    return dst_points_list

def _resolve_overlaps(dst_points_list, text_regions):
    """Detect and resolve overlapping dst_points boxes.
    
    Strategy: When two expanded boxes overlap, only shrink the EXPANDED
    portion. Never shrink below the original min_rect. Only shrink by the
    minimum amount needed to eliminate the overlap.
    """
    if len(dst_points_list) <= 1:
        return dst_points_list
    
    def _get_xyxy(pts):
        """Get axis-aligned bounding box (x1, y1, x2, y2) from polygon points."""
        flat = pts.reshape(-1, 2)
        return [float(flat[:, 0].min()), float(flat[:, 1].min()),
                float(flat[:, 0].max()), float(flat[:, 1].max())]
    
    def _boxes_overlap(b1, b2):
        """Check if two AABB boxes overlap."""
        return not (b1[2] <= b2[0] or b1[0] >= b2[2] or
                    b1[3] <= b2[1] or b1[1] >= b2[3])
    
    # Get current boxes and original (min_rect) boxes
    boxes = [_get_xyxy(pts) for pts in dst_points_list]
    orig_boxes = [_get_xyxy(text_regions[i].min_rect) for i in range(len(text_regions))]

    def _expansion(box, orig, axis):
        """How much the box grew vs its original on the given axis (0=x width, 1=y height)."""
        if axis == 0:
            return (box[2] - box[0]) - (orig[2] - orig[0])
        return (box[3] - box[1]) - (orig[3] - orig[1])

    # For each overlapping pair, resolve along the axis of MINIMUM penetration
    # by blending the more-expanded box back toward its original min_rect.
    # Handles both horizontal (side-by-side) and vertical (stacked) overlaps.
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if not _boxes_overlap(boxes[i], boxes[j]):
                continue
            # Original (unexpanded) boxes already overlap - nothing we can do
            if _boxes_overlap(orig_boxes[i], orig_boxes[j]):
                continue

            overlap_x = min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])
            overlap_y = min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1])
            if overlap_x <= 0 or overlap_y <= 0:
                continue

            # Resolve along the axis with the smaller penetration (least disruptive)
            axis = 0 if overlap_x <= overlap_y else 1
            overlap = overlap_x if axis == 0 else overlap_y

            expanded_i = _expansion(boxes[i], orig_boxes[i], axis)
            expanded_j = _expansion(boxes[j], orig_boxes[j], axis)
            total_expansion = max(expanded_i + expanded_j, 1)
            clip_i = overlap * max(expanded_i, 0) / total_expansion
            clip_j = overlap * max(expanded_j, 0) / total_expansion

            # Blend each box back toward its min_rect only enough to remove the
            # overlap on the chosen axis. Never below min_rect (blend capped at 1).
            if clip_i > 1 and expanded_i > 0:
                current = dst_points_list[i].astype(np.float64)
                original = text_regions[i].min_rect.astype(np.float64)
                blend = min(clip_i / max(expanded_i, 1), 1.0)
                if current.shape == original.shape:
                    dst_points_list[i] = (current * (1 - blend) + original * blend).astype(np.int64)
                    boxes[i] = _get_xyxy(dst_points_list[i])

            if clip_j > 1 and expanded_j > 0:
                current = dst_points_list[j].astype(np.float64)
                original = text_regions[j].min_rect.astype(np.float64)
                blend = min(clip_j / max(expanded_j, 1), 1.0)
                if current.shape == original.shape:
                    dst_points_list[j] = (current * (1 - blend) + original * blend).astype(np.int64)
                    boxes[j] = _get_xyxy(dst_points_list[j])

    return dst_points_list

def _normalize_font_sizes(dst_points_list, text_regions):
    """Normalize uneven font sizes among body text (product-image mode).

    Detection often assigns inconsistent font sizes to spec lines that should
    look uniform (e.g. 24,25,35,27,31 px), making some lines appear "bold"/
    larger than their neighbours. For center/auto-aligned horizontal regions we
    clamp oversized body lines down toward the body median and scale their box
    proportionally so the rendered text is not upscaled (which would thicken
    strokes). Large titles (font >> median) are left untouched.

    Gated to product-style layouts: >= 3 horizontal center/auto regions.
    """
    idxs = [i for i, r in enumerate(text_regions)
            if getattr(r, "horizontal", False)
            and getattr(r, "alignment", None) in ("center", "auto")
            and getattr(r, "font_size", 0) > 0]
    if len(idxs) < 3:
        return dst_points_list

    fonts = sorted(text_regions[i].font_size for i in idxs)
    median = fonts[len(fonts) // 2]
    if median <= 0:
        return dst_points_list

    # Body = lines that are not much larger than the median (exclude titles)
    body_idxs = [i for i in idxs if text_regions[i].font_size <= median * 1.8]
    if len(body_idxs) < 3:
        return dst_points_list
    body_fonts = sorted(text_regions[i].font_size for i in body_idxs)
    target = body_fonts[len(body_fonts) // 2]
    if target <= 0:
        return dst_points_list

    for i in body_idxs:
        cur = text_regions[i].font_size
        # Only shrink lines that stick out as larger than the group
        if cur > target * 1.15:
            ratio = target / cur
            pts = dst_points_list[i].astype(np.float64)
            center = pts.reshape(-1, 2).mean(axis=0)
            pts = (pts - center) * ratio + center
            dst_points_list[i] = pts.astype(np.int64)
            text_regions[i].font_size = int(target)
    return dst_points_list

def _is_primarily_cjk(text):
    """True if text is primarily CJK (Chinese/Japanese/Korean) characters."""
    if not text:
        return False
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf'
              or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    return cjk > len(text) * 0.3

def _is_primarily_latin(text):
    """True if text is primarily Latin (ASCII letters/digits)."""
    if not text:
        return False
    lat = sum(1 for c in text if ('A' <= c <= 'Z') or ('a' <= c <= 'z') or ('0' <= c <= '9'))
    return lat > len(text) * 0.3

def _aabb_of(pts):
    a = np.array(pts).reshape(-1, 2).astype(np.float64)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]

def _measure_ink_height(original_img, box):
    """Measure the ACTUAL **per-line** text ink height (px) inside a box.

    The detection box is often taller than the real glyphs (especially PaddleOCR
    boxes), which would make the re-rendered text larger than the original, so we
    threshold the original crop and look at rows that actually contain glyph ink.

    IMPORTANT: a detection box may cover SEVERAL original text lines. Returning
    the total inked-row count would then be a multi-line height, and the caller
    uses this as the height of ONE line (and multiplies by the line count again),
    which blew the rendered font up to roughly line-count times too big.
    So we split the inked rows into runs — one run per original line — and return
    a single line's height (the median run), which is what the caller needs.

    Returns 0 if it can't be measured.
    """
    if original_img is None:
        return 0
    try:
        h, w = original_img.shape[:2]
        x1 = max(int(box[0]), 0); y1 = max(int(box[1]), 0)
        x2 = min(int(box[2]), w); y2 = min(int(box[3]), h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return 0
        crop = original_img[y1:y2, x1:x2]
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, th = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if th.mean() > 127:      # ensure text (minority) = 255
            th = 255 - th
        row_has_ink = (th > 0).sum(axis=1) > max(2, th.shape[1] * 0.02)
        total = int(row_has_ink.sum())
        if total <= 0:
            return 0

        # Split into runs of consecutive inked rows: one run ≈ one original line.
        runs = []
        run = 0
        for inked in row_has_ink:
            if inked:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)

        # Drop hairline runs (underlines, bubble outline clipped into the box,
        # the crossbar of a tall letter separated by antialiasing) so they don't
        # skew the estimate; keep them if that would leave nothing.
        #
        # This threshold MUST be relative to the tallest run, not an absolute
        # pixel count. With the old `>= 4` rule a 4px sliver survived and, since
        # the median below is deliberately biased low, it was then returned as
        # "the height of one line": measured runs [4, 14, 2] -> 4, [4, 19] -> 19
        # became 4, [4, 13, 2] -> 4. The caller treats anything under 6 as
        # unmeasurable and falls back to the DETECTION BOX height, which includes
        # padding, so the text was re-rendered about twice its original size -
        # the "some lines are way too big" complaint.
        tallest = max(runs)
        real = [r for r in runs if r >= max(4, tallest * 0.4)] or runs
        if not real:
            return total

        # Use the LOWER median. Two reasons to bias small:
        #  - with an even number of lines the upper median overshoots (runs of
        #    88 and 124 px must yield 88, not 124, to match the original size);
        #  - touching lines can merge into one tall run, and a too-large basis
        #    scales the font up, which is the visually worst failure.
        real.sort()
        return int(real[(len(real) - 1) // 2])
    except Exception:
        return 0

def _orig_line_count(region) -> int:
    """How many printed lines the original text occupied."""
    lines = getattr(region, "lines", None)
    try:
        return max(1, len(lines)) if lines is not None else 1
    except TypeError:
        return 1

def _line_box_height(region) -> float:
    """Median height of the region's own per-line boxes, or 0 if unavailable.

    A recognizer that reports one box per printed line hands us the per-line
    height directly, which is strictly better than inferring it from pixels.
    Regions whose lines were never split (one quad for the whole block) return 0
    so the caller falls back to ink measurement.
    """
    lines = getattr(region, "lines", None)
    if lines is None:
        return 0.0
    try:
        heights = []
        for ln in lines:
            pts = np.array(ln).reshape(-1, 2).astype(np.float64)
            heights.append(float(pts[:, 1].max() - pts[:, 1].min()))
    except Exception:
        return 0.0
    if len(heights) < 1:
        return 0.0
    heights.sort()
    return heights[(len(heights) - 1) // 2]

def _per_line_height(region, box, original_img) -> float:
    """Height of ONE original line, the basis for the rendered font size.

    Two independent estimates are available and each fails in the same
    direction - too big - so the smaller one is taken:

      * the region's own per-line boxes, which are exact when the recognizer
        splits lines but include padding around the glyphs;
      * ink-row measurement, which is tight on the glyphs but merges lines into
        one run when the leading is small, and then reports the height of the
        whole block (measured 56px for three 15px lines).

    Over-estimating is the failure users notice - the text renders far larger
    than the original - so a low estimate is always preferred, and the result is
    finally capped by the share of the detection box one line can occupy.
    """
    det_h = box[3] - box[1]
    n_lines = max(1, _orig_line_count(region))
    candidates = []
    per_line = _line_box_height(region)
    if per_line >= 6:
        candidates.append(per_line * 1.1)
    ink = _measure_ink_height(original_img, box)
    if ink >= 6:
        candidates.append(ink * 1.1)
    h = min(candidates) if candidates else det_h / n_lines
    if det_h > 0:
        h = min(h, det_h / n_lines * 1.2)
    return h

def _estimate_bubble_bounds(original_img, box, ink_h=0):
    """Estimate the enclosing speech-bubble bounds (x1,y1,x2,y2) around a text box.

    Why this exists: the cascade layout used to bound each region only by the
    MIDPOINT GAP to neighbouring regions. On a product image (spec lines packed
    in columns) that is the right constraint, but a comic bubble usually has no
    neighbour in its row, so the "available width" became almost the whole image.
    A short CJK translation of a wrapped English line then fit on ONE line and
    the box was re-sized to that single line: the rendered text ended up ~1/4 the
    original height and up to 1.6x wider than the bubble. Measured on a real
    moderated page: 5/15 regions collapsed from 2-4 lines to 1, one going from a
    131x72 box to 187x17.

    The bubble interior is a near-uniform blob enclosing the glyphs, so we
    threshold "close to the local background level", morphologically close the
    gaps between strokes (so the blob is not cut apart by the text itself), and
    take the connected component containing the text box.

    Returns None when no plausible bubble is found (borderless text over
    artwork, product images, gradients) - callers must then keep their previous
    behaviour, which makes this self-gating: comics get bubble constraints,
    product images are untouched.
    """
    if original_img is None:
        return None
    try:
        h, w = original_img.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        bw, bh = x2 - x1, y2 - y1
        if bw < 6 or bh < 6:
            return None

        # Search window: a bubble is rarely more than ~2x the text block wide and
        # ~3x tall. Bounding the window also bounds the cost of this per region.
        px, py = int(bw * 1.2), int(bh * 2.0 + 24)
        sx1, sy1 = max(0, x1 - px), max(0, y1 - py)
        sx2, sy2 = min(w, x2 + px), min(h, y2 + py)
        crop = original_img[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop

        # Local background level = the majority class inside the text box (the
        # glyphs are the minority). Sampling outside the box would pick up the
        # artwork instead of the bubble fill.
        tb = gray[y1 - sy1:y2 - sy1, x1 - sx1:x2 - sx1]
        if tb.size < 16:
            return None
        _, th = cv2.threshold(tb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_is_high = (th > 0).mean() <= 0.5
        bg_pixels = tb[th == 0] if fg_is_high else tb[th > 0]
        if bg_pixels.size == 0:
            return None
        bg_level = int(round(float(np.median(bg_pixels))))

        # Flood fill the bubble interior. floodFill is the right tool here because
        # it STOPS at the bubble's dark outline. (A global "close to bg" mask plus
        # connected components does not: on a manga page the paper outside the
        # bubble is white too, and the morphological closing needed to bridge the
        # glyph strokes also bridges straight over the thin outline, so the
        # component swallowed the whole panel - measured 513x128 for a 151x16
        # text box. The glyphs stay as holes inside the filled area, which does
        # not affect its bounding box, so no closing is needed at all.)
        seed = None
        cy_mid = (y1 + y2) // 2 - sy1
        for yy in (cy_mid, (y1 - sy1 + cy_mid) // 2, min(gray.shape[0] - 1, y2 - sy1 + 1)):
            if not (0 <= yy < gray.shape[0]):
                continue
            row = gray[yy]
            xs = np.where(np.abs(row.astype(np.int16) - bg_level) <= 12)[0]
            xs = xs[(xs >= x1 - sx1) & (xs < x2 - sx1)]
            if xs.size:
                seed = (int(xs[xs.size // 2]), int(yy))
                break
        if seed is None:
            return None

        ff = gray.copy()
        ff_mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)
        area, _, _, rect = cv2.floodFill(
            ff, ff_mask, seed, 255, loDiff=40, upDiff=40,
            flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        bx1, by1 = sx1 + int(rect[0]), sy1 + int(rect[1])
        bx2, by2 = bx1 + int(rect[2]), by1 + int(rect[3])

        # Sanity: must actually enclose the text box, and must not be the open
        # artwork background (which would swallow the whole search window).
        if bx1 > x1 + 2 or by1 > y1 + 2 or bx2 < x2 - 2 or by2 < y2 - 2:
            return None
        win_area = max(1, (sx2 - sx1) * (sy2 - sy1))
        if area > win_area * 0.85:
            return None
        # A real bubble has visible margin around its text. When the fill barely
        # exceeds the text box the flood got stuck (seed inside a glyph counter,
        # text almost filling a small bubble), and using it as a hard bound
        # squeezes the translation into a couple of tiny lines - measured: a
        # 44x26 box reported a 48x33 "bubble", and a single character ended up
        # rendered as two 8px lines. Reject and let the caller stay unclamped.
        if (bx2 - bx1) < bw * 1.15 or (by2 - by1) < bh * 1.15:
            return None
        return [float(bx1), float(by1), float(bx2), float(by2)]
    except Exception:
        return None

def _fit_regions_cascade(img, text_regions, dst_points_list, font_size_minimum, max_font_shrink_ratio, original_img=None, strategy="cascade"):
    """Neighbor-aware cascade layout for horizontal text (product-image mode).

    Implements the intended cascade priority while keeping the displayed font as
    close to the ORIGINAL detected size as possible:

      1. EXPAND  : grow the box width up to the midpoint gap to row-neighbours
                   (and image edge) - never crossing into a neighbour column.
      2. WRAP    : if the translation still doesn't fit on the available width,
                   it wraps to more lines at the SAME font size (box grows down).
      3. SHRINK+WRAP : only if the wrapped block is taller than the vertical
                   space available (gap to the row below / image edge) do we
                   shrink the font - and we keep it wrapped (never collapse to a
                   single squished line). This preserves size fidelity.

    The box is sized to the text's NATURAL rendered dimensions at the chosen
    font, so warpPerspective maps ~1:1 and the displayed size == chosen font.
    """
    if font_size_minimum == -1:
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 200)
    font_size_minimum = max(1, font_size_minimum)

    img_h, img_w = img.shape[:2]
    margin = 4
    bubble_mode = strategy == "bubble"
    boxes = [_aabb_of(r.min_rect) for r in text_regions]

    # Per-line height basis = the ACTUAL original ink height (more faithful than
    # the detection box, which is often padded). Fall back to detection box
    # height when ink can't be measured. A small 1.1x gives a little headroom.
    h_idxs = [i for i, r in enumerate(text_regions) if getattr(r, "horizontal", False) and r.translation]
    line_h = {}
    for i in h_idxs:
        line_h[i] = _per_line_height(text_regions[i], boxes[i], original_img)

    # Uniform body line-height: snap body lines to the group median so spec lines
    # render at a consistent size. Large titles (>> median) keep their own size.
    body_median = None
    if len(h_idxs) >= 3:
        hs = sorted(line_h[i] for i in h_idxs)
        body_median = hs[len(hs) // 2]

    # Common top per ROW: regions that share a row (strong vertical overlap, e.g.
    # the left+right columns of a spec line) must start at the SAME y so their
    # tops align. Anchor each region to the topmost detection box in its row.
    row_top = {}
    row_mates = {}
    for i in h_idxs:
        yi1, yi2 = boxes[i][1], boxes[i][3]
        tops = [yi1]
        mates = 0
        for j in h_idxs:
            if j == i:
                continue
            yj1, yj2 = boxes[j][1], boxes[j][3]
            ov = min(yi2, yj2) - max(yi1, yj1)
            if ov > 0.5 * min(yi2 - yi1, yj2 - yj1):
                tops.append(yj1)
                mates += 1
        row_top[i] = min(tops)
        row_mates[i] = mates

    for i, region in enumerate(text_regions):
        if not getattr(region, "horizontal", False) or not region.translation:
            continue  # vertical / empty -> keep existing dst_points

        x1, y1, x2, y2 = boxes[i]
        cx = (x1 + x2) / 2
        h_i, w_i = line_h.get(i, y2 - y1), x2 - x1
        # Snap body line-height to the group median for a uniform look,
        # but ONLY if this line is genuinely close to the median (within a
        # narrow band). Lines that are clearly larger (titles) or clearly
        # smaller (notes / disclaimers) keep their own measured height so
        # the visual hierarchy is preserved.
        if body_median and body_median * 0.8 <= h_i <= body_median * 1.3:
            h_i = body_median

        # ── Available bounds via midpoints to nearest neighbours ──
        L, R, T, B = float(margin), float(img_w - margin), float(margin), float(img_h - margin)
        if bubble_mode:
            # Comics: the hard bound is the ORIGINAL text's own footprint.
            # The letterer fitted that text inside the balloon, so anything that
            # fits in the same rectangle is inside the balloon too - a guarantee
            # no bubble-outline detection can give. _estimate_bubble_bounds is
            # skipped here for exactly that reason: its flood fill leaks through
            # balloon tails and hand-drawn gaps and then reports the whole panel,
            # which is worse than no bound at all.
            # The inset matters because CJK glyphs fill their em box while Latin
            # ones carry side bearings: rendering CJK at the measured Latin ink
            # width puts ink where the original had whitespace, right on the
            # outline.
            inset = max(1.0, h_i * 0.12)
            L, R = x1 + inset, x2 - inset
            T, B = float(y1), float(y2)
            if R - L < h_i * 1.5:      # tiny balloon: keep it usable
                L, R = float(x1), float(x2)
        for j, ob in enumerate(boxes):
            if j == i or bubble_mode:
                continue
            ox1, oy1, ox2, oy2 = ob
            v_ov = min(y2, oy2) - max(y1, oy1)        # vertical overlap (same row?)
            h_ov = min(x2, ox2) - max(x1, ox1)        # horizontal overlap (same column?)
            if v_ov > 0.3 * min(h_i, oy2 - oy1):
                if ox2 <= x1:                          # neighbour to the left
                    L = max(L, (x1 + ox2) / 2)
                elif ox1 >= x2:                        # neighbour to the right
                    R = min(R, (x2 + ox1) / 2)
            if h_ov > 0.3 * min(w_i, ox2 - ox1):
                if oy2 <= y1:                          # neighbour above
                    T = max(T, (y1 + oy2) / 2)
                elif oy1 >= y2:                        # neighbour below
                    B = min(B, (y2 + oy1) / 2)

        # Per-line height = the ORIGINAL detected box height. The displayed text
        # size is governed by render_box_height / num_lines (warpPerspective
        # scales the rendered text to fill the box), so to restore the original
        # size we make each rendered line exactly as tall as the detected line.
        per_line = max(h_i, font_size_minimum)

        # "wrap" = wrap-only: NEVER shrink the font (min == per_line). Other modes
        # may shrink as a last resort.
        if strategy in ("wrap", "expand_wrap"):
            min_per_line = per_line   # never shrink font
        elif bubble_mode:
            # Fitting inside the balloon outranks size fidelity here: a balloon has
            # no free space to grow into, so shrinking is the only way out and
            # max_font_shrink_ratio must not stop it short of fitting.
            min_per_line = max(float(font_size_minimum), 6.0)
        else:
            min_per_line = max(per_line * max_font_shrink_ratio, font_size_minimum, 6)
        target_lang = getattr(region, "target_lang", "en_US")

        # Wrap target width:
        #  - left-align OR wrap-only = highest fidelity: wrap at the ORIGINAL
        #    occupied width (only widen if a single token can't fit), keep left edge.
        #  - expand_wrap = left-align, expand right to boundary, then wrap.
        #  - center/auto cascade = start from the ORIGINAL block width, widen only
        #    if the translation cannot fit in the original number of lines.
        alignment = getattr(region, "alignment", "center")
        left_mode = (alignment == "left") or strategy in ("wrap", "expand_wrap")
        orig_w = max(x2 - x1, per_line * 2.0)
        # region.lines is a numpy array; `x or []` would evaluate its truth value.
        _rlines = getattr(region, "lines", None)
        orig_lines = max(1, len(_rlines) if _rlines is not None else 1)
        avail_w = max(R - L, per_line * 2.0)
        # wrap-only / expand_wrap may grow downward freely, so allow generous height
        avail_h = max(B - T, per_line)
        if strategy in ("wrap", "expand_wrap"):
            avail_h = max(avail_h, per_line * 12)  # effectively unlimited -> never shrink

        # Bubble bound (comics): a speech bubble usually has NO neighbour in its
        # row, so R-L below is almost the whole image width. Clamp to the bubble
        # when one is found; returns None on product images, leaving them as-is.
        bubble = None if bubble_mode else _estimate_bubble_bounds(
            original_img, [x1, y1, x2, y2], h_i)
        if bubble:
            pad = max(2.0, per_line * 0.25)
            L = max(L, bubble[0] + pad)
            R = min(R, bubble[2] - pad)
            T = max(T, bubble[1] + pad)
            B = min(B, bubble[3] - pad)
            if R - L < per_line * 2.0 or B - T < per_line:
                L, R, T, B = (float(margin), float(img_w - margin),
                              float(margin), float(img_h - margin))
            else:
                avail_w = max(R - L, per_line * 2.0)
                avail_h = max(B - T, per_line)

        if strategy == "expand_wrap":
            # Expand+Wrap: use ALL available space to the right, then wrap
            wrap_target = max(R - x1, orig_w)
        elif left_mode:
            wrap_target = min(max(orig_w, per_line * 2.0), R - x1)
        else:
            # Start from the ORIGINAL block width so the translation keeps the
            # original footprint - and therefore roughly the original number of
            # lines. Using the neighbour-column width here (the old behaviour)
            # meant a 11-character CJK translation of a 4-line English bubble was
            # laid out on ONE 165px-wide line inside a 131px-wide block: the text
            # spilled out of the bubble and the block height collapsed to a
            # quarter of the original. Widen toward the neighbour/bubble bound
            # only when the text genuinely does not fit in orig_lines lines,
            # which is what product images (longer target text) need.
            wrap_target = min(orig_w, avail_w)
            # Cap the widening. Without a cap this grows to `avail_w`, which on a
            # comic page is most of the image: a balloon has no neighbouring
            # column, so the neighbour-derived bound says "the whole row is free"
            # and a long line is preferred over an extra line. The line then runs
            # straight out through the sides of the balloon. Doubling the original
            # footprint is enough for the longer-target case product images need.
            widen_limit = wrap_target if bubble_mode else min(avail_w, orig_w * 2.0)
            f0 = max(int(round(per_line)), 6)
            guard = 0
            while wrap_target < widen_limit - 1 and guard < 12:
                guard += 1
                probe_lines, _ = text_render.calc_horizontal(
                    f0, region.translation, max(int(wrap_target), 2 * f0),
                    int(max(avail_h, per_line * 12)), language=target_lang)
                if len(probe_lines) <= orig_lines:
                    break
                wrap_target = min(widen_limit, wrap_target * 1.25)

        # Cascade: keep original per-line height; wrap within the target width
        # (EXPAND bounded by neighbours -> WRAP). Only if the wrapped block is
        # taller than the available height do we shrink the per-line height
        # (SHRINK+WRAP), never collapsing to a squished single line.
        chosen = None
        plh = per_line
        while True:
            f = max(int(round(plh)), 6)
            wrap_w = max(int(wrap_target), int(2 * f))
            lines, widths = text_render.calc_horizontal(f, region.translation, wrap_w,
                                                        int(avail_h), language=target_lang)
            n = max(len(lines), 1)
            maxw = max(widths) if widths else wrap_w
            box_h = plh * n
            # The box width IS the wrap width at render time (render() passes the
            # box width to put_text_horizontal), so it must not exceed the width
            # these n lines were measured at. Widening it back to the original
            # detection width - which is what used to happen - made render() wrap
            # the text again on a wider canvas and produce FEWER lines than n,
            # and those few lines were then stretched to fill an n-line-tall box:
            # a three-line balloon came out as two oversized lines running past
            # the balloon outline.
            box_w = min(max(maxw, 2.0 * f), wrap_w, avail_w)
            chosen = (f, box_w, box_h, n)
            if box_h <= avail_h or plh <= min_per_line:
                break
            # shrink per-line height to try to fit vertically
            plh = max(min_per_line, plh * min(avail_h / box_h, 0.92))

        f, box_w, box_h, n = chosen
        region.font_size = int(f)
        region._render_lines = int(n)

        # Horizontal position: left-align anchors at the original left edge;
        # center/auto centers within [L, R]. Both clamped to [L, R].
        if left_mode:
            nx1 = max(L, x1)
            nx2 = nx1 + box_w
            if nx2 > R:
                nx2 = R
                nx1 = max(L, R - box_w)
        else:
            half = box_w / 2.0
            bxc = min(max(cx, L + half), R - half) if (R - L) >= box_w else (L + R) / 2.0
            nx1, nx2 = bxc - half, bxc + half
        # Vertical: anchor top to the ROW's common top (same-row columns align).
        ny1 = max(T, row_top.get(i, y1))
        # A standalone block that came out shorter than the original (the
        # translation needed fewer lines) is centred in the original footprint
        # instead. Balloons are rounded, so top-anchoring a short block puts it
        # where the balloon is narrowest and the line ends poke through the
        # outline; the vertical middle is the widest part. Regions that share a
        # row keep the common top - that alignment is the point on product images.
        if (bubble_mode or row_mates.get(i, 0) == 0) and box_h < (y2 - y1) - 1:
            ny1 = max(T, y1 + ((y2 - y1) - box_h) / 2.0)
        if ny1 + box_h > B:
            ny1 = max(T, B - box_h)
        ny2 = ny1 + box_h

        dst_points_list[i] = np.array(
            [[[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]]], dtype=np.int64
        ).reshape(-1, 4, 2)

    return dst_points_list

def _bubble_ink_runs(mask):
    """Visible row runs; an estimate, not OCR ground truth (touching lines merge)."""
    rows = np.any(mask > 32, axis=1)
    edges = np.diff(np.pad(rows.astype(np.int8), (1, 1)))
    return (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist()


def _bubble_components(original_img, boxes):
    """Find conservative closed balloon interiors and their true pixel masks.

    The light/dark fill itself is the component. We deliberately do not dilate or
    close it: either operation can jump a thin balloon outline, a panel border or
    a character edge. A component is accepted only when its fill owns a material
    part of the source-text box, encloses the box centre, stays away from the page
    edge, and is small enough to be one balloon rather than a panel/background.
    """
    found = {}
    if original_img is None:
        return found
    gray = original_img if original_img.ndim == 2 else cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    image_area = width * height
    for polarity, background in enumerate((gray >= 180, gray <= 80)):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            background.astype(np.uint8), connectivity=4)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if i in found:
                continue
            ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
            ix2, iy2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
            crop = labels[iy1:iy2, ix1:ix2]
            if not crop.size:
                continue
            ids, counts = np.unique(crop, return_counts=True)
            candidates = sorted(((int(count), int(label)) for label, count in zip(ids, counts)
                                 if label), reverse=True)
            if not candidates:
                continue
            count, label = candidates[0]
            x, y, w, h, area = (int(v) for v in stats[label])
            cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
            owns_centre = x <= cx < x + w and y <= cy < y + h
            # The whole source AABB need not be fill: glyphs and outline pixels
            # are holes. Requiring its centre inside the component envelope plus
            # 30% fill is robust to a centre pixel landing on a glyph.
            if (count < crop.size * 0.30 or not owns_centre or
                    x <= 0 or y <= 0 or x + w >= width or y + h >= height or
                    area > image_area * 0.08 or w > width * 0.55 or h > height * 0.35):
                continue
            if w < (x2 - x1) * 0.65 or h < (y2 - y1) * 0.65:
                continue
            mask = (labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255
            # Text is a set of holes in the balloon fill. Fill only enclosed
            # holes; unlike morphological closing this cannot cross the outline.
            padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            exterior = padded.copy()
            flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
            cv2.floodFill(exterior, flood_mask, (0, 0), 255)
            holes = cv2.bitwise_not(exterior)[1:-1, 1:-1]
            mask = cv2.bitwise_or(mask, holes)
            found[i] = {
                'key': (polarity, label), 'bounds': (x, y, x + w, y + h),
                'mask': mask, 'area': area,
            }
    return found


def _bubble_groups(original_img, boxes):
    """Return stable group ids for text blocks sharing one closed balloon."""
    groups = list(range(len(boxes)))
    components = _bubble_components(original_img, boxes)
    first = {}
    for i, component in components.items():
        groups[i] = first.setdefault(component['key'], i)
    return groups


def _bubble_safe_rect(component, source_box, ink_height):
    """Conservative layout bounds from one closed balloon component.

    The component is eroded by a glyph-dependent safety margin. Its resulting
    envelope may include rounded corners outside the fill, so placement also keeps
    a generous outline inset; unlike source AABB fitting this still recovers the
    balloon's real central width and height without crossing panels or neighbours.
    """
    if not component:
        return None
    bx1, by1, bx2, by2 = component['bounds']
    mask = component['mask']
    pad = max(2, int(round(max(ink_height, 6.0) * 0.18)))
    safe = cv2.erode(mask, np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8))
    if not np.any(safe):
        return None
    x1, y1, x2, y2 = source_box
    cx = int(round((x1 + x2) / 2)) - bx1
    cy = int(round((y1 + y2) / 2)) - by1
    if not (0 <= cx < safe.shape[1] and 0 <= cy < safe.shape[0]):
        return None

    if safe[cy, cx] == 0:
        # The exact centre can still coincide with residual antialiasing. Use the
        # nearest safe pixel inside one source-line radius, never another lobe.
        ys, xs = np.where(safe > 0)
        near = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
        if (xs[near] - cx) ** 2 + (ys[near] - cy) ** 2 > max(ink_height, 6.0) ** 2:
            return None
    sx, sy, sw, sh = cv2.boundingRect(safe)
    result = (float(bx1 + sx), float(by1 + sy),
              float(bx1 + sx + sw), float(by1 + sy + sh))
    # Accept a rectangle when it materially improves either line width or total
    # area. Rounded balloons can trade unused source-box height for the wider
    # centre band that a short translation actually needs.
    result_w, result_h = result[2] - result[0], result[3] - result[1]
    source_w, source_h = x2 - x1, y2 - y1
    if result_w < source_w * 1.05 and result_w * result_h < source_w * source_h * 1.10:
        return None
    return result


def _bubble_balanced_lines(text, requested, font):
    """Split at words/CJK characters, never at Latin letters or empty lines.

    Explicit lines bypass calc_horizontal's 2-em minimum, which otherwise makes
    e.g. four CJK characters unable to occupy four real lines at ANY font size.
    Punctuation-only tokens attach to the preceding unit rather than forming a
    line of their own. This is a modest tokenizer, not full Unicode line breaking.
    """
    cjk = '\\u3400-\\u9fff\\u3040-\\u30ff\\uac00-\\ud7af'
    matches = list(re.finditer('[' + cjk + ']|[^\\s' + cjk + ']+', text))
    starts = [m.start() for m in matches if any(c.isalnum() for c in m.group())]
    if len(starts) < 2:
        return None
    starts[0] = 0  # retain any leading punctuation
    count = min(requested, len(starts))
    if count < 2:
        return None
    starts.append(len(text))
    units = [text[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    weights = [max(1, text_render.get_string_width(font, unit)) for unit in units]
    lines, begin = [], 0
    for remaining in range(count, 1, -1):
        target = sum(weights[begin:]) / remaining
        total, end = 0, begin
        # Leave >=1 real unit for every remaining line.
        limit = len(units) - remaining + 1
        while end < limit:
            next_total = total + weights[end]
            if end > begin and abs(total - target) <= abs(next_total - target):
                break
            total = next_total
            end += 1
        end = max(begin + 1, end)
        lines.append(''.join(units[begin:end]).strip())
        begin = end
    lines.append(''.join(units[begin:]).strip())
    return tuple(line for line in lines if line)


def _fit_regions_bubble(img, text_regions, original_img, hyphenate, line_spacing, disable_font_border):
    """Comic-only raster fitting inside each ORIGINAL axis-aligned footprint.

    Select real nonempty lines, then measure the rendered alpha rather than
    stretching an n*font-size box. No product-page median, row-top or expansion.
    Line retention is best-effort: legal wrapping, finite width probes and the
    source-line estimate limit it; never insert blank lines to meet a quota.
    """
    boxes = [_aabb_of(r.min_rect) for r in text_regions]
    components = _bubble_components(original_img, boxes)
    groups = list(range(len(boxes)))
    first = {}
    for i, component in components.items():
        groups[i] = first.setdefault(component['key'], i)
    physical_groups = groups.copy()
    heights, original_counts = {}, {}
    for i, region in enumerate(text_regions):
        box = boxes[i]
        heights[i] = max(1.0, _per_line_height(region, box, original_img) / 1.1)
        count = max(_orig_line_count(region), int(getattr(region, 'ocr_detector_rows', 0)))
        if not region.horizontal:
            # Vertical source size is ink COLUMN width, not row height.
            measured = _measure_ink_height(
                np.swapaxes(original_img, 0, 1) if original_img is not None else None,
                [box[1], box[0], box[3], box[2]])
            heights[i] = max(1.0, min(measured if measured > 0 else float('inf'),
                                     (box[2] - box[0]) / count))
        # A VLM may return one quad for several printed rows. Use separated ink
        # runs as an additional estimate, not len(region.lines) alone.
        if original_img is not None and region.horizontal:
            x1, y1, x2, y2 = box
            crop = original_img[max(0, int(y1)):max(0, int(y2)),
                                max(0, int(x1)):max(0, int(x2))]
            if crop.size:
                gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if ink.mean() > 127:
                    ink = 255 - ink
                runs = _bubble_ink_runs(ink)
                if runs:
                    real = [run for run in runs if run >= max(3, max(runs) * 0.4)]
                    count = max(count, len(real))
        original_counts[i] = count

    targets = heights.copy()
    # A page made of many parallel one-line labels is typographic UI/caption
    # content, not independent speech balloons. Detection/VLM box padding made
    # one row 36px while its peers were 26-28px. Normalize the obvious body rows
    # before fitting, while preserving genuine title/body hierarchy elsewhere.
    horizontal = [i for i, region in enumerate(text_regions) if region.horizontal]
    if len(horizontal) >= 5 and all(original_counts[i] == 1 for i in horizontal):
        median = float(np.median([heights[i] for i in horizontal]))
        for i in horizontal:
            if median * 0.75 <= heights[i] <= median * 1.40:
                targets[i] = median

    for i, region in enumerate(text_regions):
        block_ids = getattr(region, 'ocr_block_ids', set())
        if len(block_ids) == 1:
            groups[i] = ('vlm', next(iter(block_ids)))

    for group in set(groups):
        members = [i for i, g in enumerate(groups) if g == group and text_regions[i].horizontal]
        if len(members) < 2:
            continue
        median = float(np.median([heights[i] for i in members]))
        # Only near-equal body sizes are snapped. Shouts/titles/small asides keep
        # their relative size, even when they live in the very same balloon.
        for i in members:
            if median * 0.87 <= heights[i] <= median * 1.15:
                targets[i] = min(median, heights[i] * 1.10)

    plans = {}
    for i, region in enumerate(text_regions):
        x1, y1, x2, y2 = boxes[i]
        safe_rect = _bubble_safe_rect(components.get(i), boxes[i], heights[i])
        # A shared component may cover multiple separately styled blocks. Keep
        # each block in its own source band; widening is safe, but moving both to
        # the component centre would overlap them (page1 title/body regression).
        shared_component = any(j != i and physical_groups[j] == physical_groups[i]
                               for j in range(len(physical_groups)))
        boundary_kind = 'closed_component' if safe_rect is not None else 'source_footprint'
        if safe_rect is not None:
            left, top, right, bottom = safe_rect
            if shared_component:
                top, bottom = max(top, y1), min(bottom, y2)
                if bottom - top < max(heights[i], 6.0):
                    top, bottom = y1, y2
        else:
            inset = max(1.0, heights[i] * 0.12)
            if x2 - x1 <= inset * 2 + 2:
                inset = 0
            left, top, right, bottom = x1 + inset, y1, x2 - inset, y2
        left, top = max(0, int(np.ceil(left))), max(0, int(np.ceil(top)))
        right, bottom = min(img.shape[1], int(np.floor(right))), min(img.shape[0], int(np.floor(bottom)))
        avail_w, avail_h = right - left, bottom - top
        # Diagnostics and ratio floors must use measured source ink, not the
        # detector's padded box-height font estimate (observed false 58/65px).
        source_ink = float(max(heights[i], 1.0))
        # A single-line translation can be width-limited by its original label
        # footprint even when the detector's padded height is much larger. Cap
        # diagnostics to the largest source-equivalent glyph size that the same
        # safe rectangle can physically support.
        if original_counts[i] == 1 and region.horizontal:
            source_text = text_render.compact_special_symbols(str(region.text or '').strip())
            if source_text:
                try:
                    source_font = max(6, int(np.ceil(source_ink)))
                    source_raster = text_render.put_text_horizontal(
                        source_font, source_text, max(avail_w, source_font * 2), avail_h,
                        region.alignment, region.direction == 'hl', (255, 255, 255), None,
                        region.target_lang, hyphenate, line_spacing,
                        prewrapped_lines=(source_text,))
                    if source_raster is not None and source_raster.size:
                        source_runs = _bubble_ink_runs(source_raster[:, :, 3])
                        source_visible = max(source_runs, default=source_raster.shape[0])
                        source_scale = min(1.0, avail_w / source_raster.shape[1],
                                           avail_h / source_raster.shape[0])
                        source_ink = min(source_ink, source_visible * source_scale)
                except Exception:
                    pass
        heights[i] = max(source_ink, 1.0)
        targets[i] = min(targets[i], heights[i])
        region._orig_font_size = int(round(heights[i]))
        region._bubble_source_ink_height = float(heights[i])
        region._bubble_group = groups[i] if isinstance(groups[i], int) else str(groups[i][1])
        region._bubble_original_lines = int(original_counts[i])
        region._bubble_boundary_kind = boundary_kind
        region._bubble_bounds = [left, top, right, bottom]
        region._bubble_layout_failed = None
        if avail_w < 1 or avail_h < 1:
            continue
        fg, bg = fg_bg_compare(*region.get_font_colors())
        if disable_font_border:
            bg = None
        text = text_render.compact_special_symbols(region.get_translation_for_rendering())
        if not text.strip():
            continue
        # Preserve genuinely small source lettering instead of imposing a 12px
        # floor that cannot fit its original footprint. Ordinary dialogue still
        # keeps at least 60% of its measured source ink; tiny labels keep their
        # own hierarchy, with a 4px floor only when the source is at least 4px.
        ratio_floor = min(18.0, max(heights[i], 1.0) * 0.60)
        readable_floor = max(
            min(4.0, heights[i]),
            min(ratio_floor, avail_h * 0.48, avail_w * 0.32),
        )
        font = max(6, int(np.ceil(max(targets[i], readable_floor))))
        best = None
        if region.horizontal:
            # Include narrow widths so SHORT translations can retain multiple
            # lines. calc_horizontal has a 2-em minimum; fewer breakable units
            # than source rows must fall back to fewer nonempty lines.
            upper = max(2 * font, avail_w)
            widths = sorted(set(int(round(w)) for w in np.linspace(2 * font, upper, 33)), reverse=True)
            # Always probe the compact one-line form. Wrapping is calculated at
            # the unscaled source font, but a short CJK translation can often fit
            # one readable line after a modest uniform scale (the old path never
            # discovered that candidate and instead forced two 9px rows).
            candidates = [(text.strip(),)]
            for width in widths:
                lines, _ = text_render.calc_horizontal(font, text, width, avail_h,
                                                       region.target_lang, hyphenate)
                candidates.append(tuple(line for line in lines if line.strip()))
            balanced = _bubble_balanced_lines(text, original_counts[i], font)
            if balanced:
                candidates.append(balanced)
        # Detector rows are a geometric hint, not a quota. If OCR metadata says
        # more rows than the source region's own geometry, a compact translation
        # must be allowed to collapse those surplus rows instead of becoming
        # microtext (e.g. one real line incorrectly hinted as two detector rows).
            seen = set()
            for lines in candidates:
                if not lines or lines in seen:
                    continue
                seen.add(lines)
                raster = text_render.put_text_horizontal(
                    font, text, avail_w, avail_h, region.alignment, region.direction == 'hl',
                    fg, bg, region.target_lang, hyphenate, line_spacing, prewrapped_lines=lines)
                if raster is None or not raster.size:
                    continue
                # Per-line visible height must come from the complete tight
                # raster divided by the known line count. Adjacent CJK rows often
                # touch, so alpha row-runs merge and `max(runs)` incorrectly sees
                # two 13px rows as one 26px glyph, halving the final text.
                visible = raster.shape[0] / max(len(lines), 1)
                cap = max(targets[i], readable_floor)
                scale = min(cap / max(visible, 1), avail_w / raster.shape[1], avail_h / raster.shape[0])
                displayed = visible * scale
                # Production readability floor: retain a meaningful fraction of
                # the source ink while never emitting microtext. A 1080px-wide
                # comic needs about 12px visible glyphs for ordinary dialogue;
                # larger source lettering gets a proportional floor as well.
                readable = displayed >= readable_floor
                # Readability is the hard gate. Prefer the source region's own
                # line count before detector-row hints; then keep as much of the
                # hinted structure as possible without sacrificing glyph size.
                source_count = _orig_line_count(region)
                score = (readable,
                         -abs(len(lines) - source_count) if readable else 0,
                         -abs(len(lines) - original_counts[i]) if readable else 0,
                         displayed, -len(lines))
                if best is None or score > best[0]:
                    best = (score, raster, scale, len(lines), visible)
        else:
            # Vertical text stays independent: no horizontal row normalization.
            raster = text_render.put_text_vertical(font, text, max(font, avail_h),
                                                   region.alignment, fg, bg, line_spacing)
            if raster is not None and raster.size:
                runs = _bubble_ink_runs(raster[:, :, 3].T)
                visible = max(runs, default=raster.shape[1])
                scale = min(heights[i] * 1.10 / max(visible, 1),
                            avail_w / raster.shape[1], avail_h / raster.shape[0])
                best = ((0, scale, 0), raster, scale, len(runs), visible)
        if best is not None:
            _, raster, scale, count, visible = best
            if region.horizontal and visible * scale + 0.01 < readable_floor:
                region._bubble_layout_failed = (
                    f'visible ink {visible * scale:.1f}px below {readable_floor:.1f}px floor')
            else:
                plans[i] = (raster, scale, count, visible, (left, top, right, bottom), font)

    # Resolve real raster collisions inside each physical balloon. Detector/VLM
    # splits can overlap slightly; keep source ordering and shift within the same
    # closed component instead of crossing into another bubble or panel.
    placed = {}
    points = []
    for i, region in enumerate(text_regions):
        region._bubble_raster = None
        if i not in plans:
            # Deliberate fail-closed behavior: do not let the generic renderer
            # silently emit unreadable 3px text after bubble fitting failed.
            if getattr(region, '_bubble_layout_failed', None):
                region._render_lines = 0
                region._bubble_retained_lines = False if region.horizontal else None
                region._bubble_visible_ink_height = 0.0
                cx, cy = (boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2
                points.append(np.array([[[cx, cy], [cx, cy], [cx, cy], [cx, cy]]], dtype=np.int64))
            else:
                points.append(np.array(region.min_rect, copy=True))
            continue
        raster, scale, count, visible, bounds, font = plans[i]
        width = max(1, int(np.floor(raster.shape[1] * scale)))
        height = max(1, int(np.floor(raster.shape[0] * scale)))
        raster = cv2.resize(raster, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        left, top, right, bottom = bounds
        x = left if region.alignment == 'left' else right - width if region.alignment == 'right' else left + (right - left - width) // 2
        y = top + (bottom - top - height) // 2
        group = physical_groups[i]
        source_x1, source_y1, source_x2, source_y2 = boxes[i]
        # Reserve strongly nested, separately-worded labels before placing a broad
        # parent block. UI command hints can sit inside the parent's detector AABB
        # without being OCR duplicates; placing the parent first would cover them.
        current_norm = re.sub(r'\s+', '', str(region.text or '')).casefold()
        current_area = max((source_x2 - source_x1) * (source_y2 - source_y1), 1)
        for j in range(i + 1, len(boxes)):
            fx1, fy1, fx2, fy2 = boxes[j]
            future_area = max((fx2 - fx1) * (fy2 - fy1), 1)
            future_norm = re.sub(r'\s+', '', str(text_regions[j].text or '')).casefold()
            nested = (fx1 >= source_x1 and fx2 <= source_x2
                      and fy1 >= source_y1 and fy2 <= source_y2
                      and future_area < current_area * 0.35)
            distinct = future_norm and future_norm not in current_norm
            if nested and distinct and x < fx2 and x + width > fx1 and y < fy2 and y + height > fy1:
                above, below = int(np.floor(fy1 - height - 2)), int(np.ceil(fy2 + 2))
                if above >= top:
                    y = above
                elif below + height <= bottom:
                    y = below
        # Source blocks that already overlap substantially are alternative OCR
        # interpretations/style fragments, not two independent labels requiring
        # collision separation. Preserve their source-band placement; only move
        # genuinely separate blocks that the expanded layout brought together.
        previous = placed.setdefault(group, [])
        source_x1, source_y1, source_x2, source_y2 = boxes[i]
        for j, px1, py1, px2, py2 in previous:
            sx1, sy1, sx2, sy2 = boxes[j]
            source_ox = min(source_x2, sx2) - max(source_x1, sx1)
            source_oy = min(source_y2, sy2) - max(source_y1, sy1)
            current_norm = re.sub(r'\s+', '', str(region.text or '')).casefold()
            previous_norm = re.sub(r'\s+', '', str(text_regions[j].text or '')).casefold()
            same_fragment_text = bool(
                current_norm and previous_norm
                and (current_norm in previous_norm or previous_norm in current_norm)
            )
            source_overlap = (
                same_fragment_text
                and source_ox > 0.85 * min(source_x2 - source_x1, sx2 - sx1)
                and source_oy > 0.30 * min(source_y2 - source_y1, sy2 - sy1)
            )
            previous_nested_duplicate = (
                previous_norm and previous_norm in current_norm
                and sx1 >= source_x1 and sx2 <= source_x2
                and sy1 >= source_y1 and sy2 <= source_y2
                and (sx2 - sx1) * (sy2 - sy1)
                    < (source_x2 - source_x1) * (source_y2 - source_y1) * 0.35
            )
            if previous_nested_duplicate:
                # The broad VLM block already owns this exact nested phrase. Keep
                # metadata for the small detector fragment but render it only once.
                marker_h = max(1, int(round(getattr(text_regions[j], '_bubble_visible_ink_height', 1))))
                text_regions[j]._bubble_raster = (
                    np.zeros((marker_h, 1, 4), dtype=np.uint8), int(round(sx1)), int(round(sy1)))
                points[j] = np.array([[[sx1, sy1], [sx1 + 1, sy1],
                                       [sx1 + 1, sy1 + marker_h], [sx1, sy1 + marker_h]]],
                                     dtype=np.int64)
            if (not source_overlap and x < px2 and x + width > px1
                    and y < py2 and y + height > py1):
                below, above = py2 + 2, py1 - height - 2
                if below + height <= bottom:
                    y = below
                elif above >= top:
                    y = above
                else:
                    region._bubble_layout_failed = 'no non-overlapping readable placement in bubble'
                    break
        if region._bubble_layout_failed:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            region._render_lines = 0
            region._bubble_retained_lines = False if region.horizontal else None
            region._bubble_visible_ink_height = 0.0
            points.append(np.array([[[cx, cy], [cx, cy], [cx, cy], [cx, cy]]], dtype=np.int64))
            continue
        # Source-component grouping is intentionally conservative and can split
        # adjacent labels that still share pixels after fitting. Enforce a final
        # global non-overlap pass without moving pre-existing source overlaps.
        source_x1, source_y1, source_x2, source_y2 = boxes[i]
        for other_group, other_boxes in placed.items():
            if other_group == group:
                continue
            for j, px1, py1, px2, py2 in other_boxes:
                sx1, sy1, sx2, sy2 = boxes[j]
                source_ox = min(source_x2, sx2) - max(source_x1, sx1)
                source_oy = min(source_y2, sy2) - max(source_y1, sy1)
                current_norm = re.sub(r'\s+', '', str(region.text or '')).casefold()
                previous_norm = re.sub(r'\s+', '', str(text_regions[j].text or '')).casefold()
                same_fragment_text = bool(
                    current_norm and previous_norm
                    and (current_norm in previous_norm or previous_norm in current_norm)
                )
                source_overlap = (
                    same_fragment_text
                    and source_ox > 0.85 * min(source_x2 - source_x1, sx2 - sx1)
                    and source_oy > 0.30 * min(source_y2 - source_y1, sy2 - sy1)
                )
                previous_nested_duplicate = (
                    previous_norm and previous_norm in current_norm
                    and sx1 >= source_x1 and sx2 <= source_x2
                    and sy1 >= source_y1 and sy2 <= source_y2
                    and (sx2 - sx1) * (sy2 - sy1)
                        < (source_x2 - source_x1) * (source_y2 - source_y1) * 0.35
                )
                if previous_nested_duplicate:
                    marker_h = max(1, int(round(getattr(
                        text_regions[j], '_bubble_visible_ink_height', 1))))
                    text_regions[j]._bubble_raster = (
                        np.zeros((marker_h, 1, 4), dtype=np.uint8),
                        int(round(sx1)), int(round(sy1)))
                    points[j] = np.array(
                        [[[sx1, sy1], [sx1 + 1, sy1],
                          [sx1 + 1, sy1 + marker_h], [sx1, sy1 + marker_h]]],
                        dtype=np.int64)
                    px1, py1, px2, py2 = sx1, sy1, sx1 + 1, sy1 + marker_h
                if (not source_overlap and x < px2 and x + width > px1
                        and y < py2 and y + height > py1):
                    below, above = py2 + 2, py1 - height - 2
                    if below + height <= bottom:
                        y = below
                    elif above >= top:
                        y = above
                    else:
                        region._bubble_layout_failed = 'no global non-overlapping readable placement'
                        break
            if region._bubble_layout_failed:
                break
        if region._bubble_layout_failed:
            cx, cy = (source_x1 + source_x2) / 2, (source_y1 + source_y2) / 2
            region._render_lines = 0
            region._bubble_retained_lines = False if region.horizontal else None
            region._bubble_visible_ink_height = 0.0
            points.append(np.array([[[cx, cy], [cx, cy], [cx, cy], [cx, cy]]], dtype=np.int64))
            continue
        previous.append((i, x, y, x + width, y + height))
        region.font_size = font
        region._render_lines = count
        region._bubble_retained_lines = count >= original_counts[i] if region.horizontal else None
        region._bubble_visible_ink_height = float(visible * scale)
        angle = float(getattr(region, 'angle', 0) or 0)
        if abs(angle) >= 3:
            # Keep the source lettering angle. Expand the temporary canvas, but
            # composite around the original footprint centre; do not clip corners.
            rgba = Image.fromarray(raster, mode='RGBA').rotate(
                angle, resample=Image.Resampling.BICUBIC, expand=True)
            raster = np.asarray(rgba)
            height, width = raster.shape[:2]
            # A rotated block's vertical alpha run includes width*sin(angle), so
            # it is not a glyph-height measurement. Keep the pre-rotation visible
            # per-line height and apply only the later containment scale.
            rotated_visible = visible * scale
            source_x1, source_y1, source_x2, source_y2 = map(float, region.xyxy)
            contain = min((source_x2-source_x1) / max(width, 1),
                          (source_y2-source_y1) / max(height, 1), 1.0)
            if contain < 1:
                width = max(1, int(np.floor(width * contain)))
                height = max(1, int(np.floor(height * contain)))
                raster = cv2.resize(raster, (width, height), interpolation=cv2.INTER_AREA)
                rotated_visible *= contain
            region._bubble_visible_ink_height = float(rotated_visible)
            cx, cy = (source_x1 + source_x2) / 2, (source_y1 + source_y2) / 2
            x, y = int(round(cx - width / 2)), int(round(cy - height / 2))
        region._bubble_raster = (raster, x, y)
        points.append(np.array([[[x, y], [x + width, y], [x + width, y + height], [x, y + height]]], dtype=np.int64))
    return points


async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = '',
    font_size_fixed: int = None,
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    hyphenate: bool = True,
    render_mask: np.ndarray = None,
    line_spacing: int = None,
    disable_font_border: bool = False,
    overflow_strategy: str = "cascade",
    max_font_shrink_ratio: float = 0.5,
    original_img: np.ndarray = None,
    ) -> np.ndarray:

    text_render.set_font(font_path)
    text_regions = list(filter(lambda region: region.translation, text_regions))

    # Clear private raster state when callers reuse TextBlocks for another mode.
    for region in text_regions:
        region._bubble_raster = None
    if overflow_strategy == "bubble":
        # Completely separate from product-page median/row-top/overlap policies.
        dst_points_list = _fit_regions_bubble(img, text_regions, original_img,
                                              hyphenate, line_spacing, disable_font_border)
    else:
        # Resize baseline (also sets vertical-text boxes) for non-comic modes.
        dst_points_list = resize_regions_to_font_size(img, text_regions, font_size_fixed, font_size_offset, font_size_minimum,
                                                     overflow_strategy=overflow_strategy, max_font_shrink_ratio=max_font_shrink_ratio)

    if overflow_strategy in ("cascade", "auto", "wrap", "expand_wrap"):
        # Neighbor-aware fit for horizontal text. All modes keep the
        # original per-line size and never overlap neighbours:
        #  - cascade/auto: expand (within neighbour bounds) -> wrap -> shrink+wrap
        #  - wrap: wrap-only at the ORIGINAL width, grow downward, NEVER shrink
        #  - expand_wrap: left-align, expand right to boundary, wrap, NEVER shrink
        dst_points_list = _fit_regions_cascade(img, text_regions, dst_points_list,
                                               font_size_minimum, max_font_shrink_ratio,
                                               original_img=original_img,
                                               strategy=overflow_strategy)
    elif overflow_strategy != "bubble":
        # Normalize uneven font sizes (product-image mode) so no line looks bold
        dst_points_list = _normalize_font_sizes(dst_points_list, text_regions)
        # Resolve overlapping text boxes to prevent text-on-text rendering
        dst_points_list = _resolve_overlaps(dst_points_list, text_regions)

    # Render text
    for region, dst_points in tqdm(zip(text_regions, dst_points_list), '[render]', total=len(text_regions)):
        # Store the FINAL render box (post expand/normalize/overlap-resolve) for diagnostics
        try:
            region._render_dst = np.array(dst_points).reshape(-1, 2).tolist()
        except Exception:
            pass
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)
        if overflow_strategy == "bubble":
            prepared = region._bubble_raster
            if prepared is not None:
                raster, x, y = prepared
                h, w = raster.shape[:2]
                alpha = raster[:, :, 3:4].astype(np.float32) / 255.0
                ix1, iy1 = max(0, x), max(0, y)
                ix2, iy2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
                if ix2 > ix1 and iy2 > iy1:
                    rx1, ry1 = ix1-x, iy1-y
                    rx2, ry2 = rx1+(ix2-ix1), ry1+(iy2-iy1)
                    crop = img[iy1:iy2, ix1:ix2]
                    local_alpha = alpha[ry1:ry2, rx1:rx2]
                    local_rgb = raster[ry1:ry2, rx1:rx2, :3]
                    crop[:] = np.clip(crop.astype(np.float32) * (1 - local_alpha) +
                                      local_rgb.astype(np.float32) * local_alpha, 0, 255).astype(np.uint8)
            # Raster has already been fitted using visible ink. No second wrap
            # or homography stretch back to a nominal n-line detection box.
            region._bubble_raster = None
        else:
            img = render(img, region, dst_points, hyphenate, line_spacing, disable_font_border)
    return img

def render(
    img,
    region: TextBlock,
    dst_points,
    hyphenate,
    line_spacing,
    disable_font_border
):
    fg, bg = region.get_font_colors()
    fg, bg = fg_bg_compare(fg, bg)

    if disable_font_border :
        bg = None

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    r_orig = np.mean(norm_h / norm_v)

    # If configuration is set to non-automatic mode, use configuration to determine direction directly
    forced_direction = region._direction if hasattr(region, "_direction") else region.direction
    if forced_direction != "auto":
        if forced_direction in ["horizontal", "h"]:
            render_horizontally = True
        elif forced_direction in ["vertical", "v"]:
            render_horizontally = False
        else:
            render_horizontally = region.horizontal
    else:
        render_horizontally = region.horizontal

    #print(f"Region text: {region.text}, forced_direction: {forced_direction}, render_horizontally: {render_horizontally}")

    if render_horizontally:
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_h[0]),
            round(norm_v[0]),
            region.alignment,
            region.direction == 'hl',
            fg,
            bg,
            region.target_lang,
            hyphenate,
            line_spacing,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_v[0]),
            region.alignment,
            fg,
            bg,
            line_spacing,
        )
    h, w, _ = temp_box.shape
    if h < 1 or w < 1:
        return img
    r_temp = float(w) / float(h)

    # Extend temporary box so that it has same ratio as original
    box = None  
    #print("\n" + "="*50)  
    #print(f"Processing text: \"{region.get_translation_for_rendering()}\"")  
    #print(f"Text direction: {'Horizontal' if region.horizontal else 'Vertical'}")  
    #print(f"Font size: {region.font_size}, Alignment: {region.alignment}")  
    #print(f"Target language: {region.target_lang}")      
    #print(f"Region horizontal: {region.horizontal}")  
    #print(f"Starting image adjustment: r_temp={r_temp}, r_orig={r_orig}, h={h}, w={w}")  
    if region.horizontal:  
        #print("Processing HORIZONTAL region")  
        
        if r_temp > r_orig:   
            #print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            h_ext = int((w / r_orig - h) // 2) if r_orig > 0 else 0  
            #print(f"Calculated h_ext = {h_ext}")  
            
            if h_ext >= 0:  
                #print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [h_ext:h_ext+h, :w] = [{h_ext}:{h_ext+h}, 0:{w}]")  
                # Columns fully filled, rows centered
                box[h_ext:h_ext+h, 0:w] = temp_box  
            else:  
                #print("h_ext < 0, using original temp_box")  
                box = temp_box.copy()  
        else:   
            #print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            w_ext = int((h * r_orig - w) // 2)  
            #print(f"Calculated w_ext = {w_ext}")  
            
            if w_ext >= 0:  
                #print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [:, :w] = [0:{h}, 0:{w}]")  
         
                # Position text based on alignment setting:
                # - center: center text in padded box (product images)
                # - right: right-align text
                # - left/auto: left-align (original behavior for comics)
                align = region.alignment
                if align in ('center', 'auto'):
                    # Anchor positioning: center text in box = preserve original center position
                    box[0:h, w_ext:w_ext+w] = temp_box
                elif align == 'right':
                    box[0:h, w_ext*2:w_ext*2+w] = temp_box
                else:
                    # 'left': left-align (original comic behavior)
                    box[0:h, 0:w] = temp_box
            else:  
                #print("w_ext < 0, using original temp_box")  
                box = temp_box.copy()  
    else:  
        #print("Processing VERTICAL region")  
        
        if r_temp > r_orig:   
            #print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            h_ext = int(w / (2 * r_orig) - h / 2) if r_orig > 0 else 0   
            #print(f"Calculated h_ext = {h_ext}")  
            
            if h_ext >= 0:   
                #print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [0:h, 0:w] = [0:{h}, 0:{w}]")  
                # The rows are full, and there should be no empty lines above the text; otherwise, when multiple text boxes have their top edges aligned, the text cannot be aligned. Common scenario: borderless comics, CG. 
                # When there are bubbles on the current page, it can be changed to center: box[h_ext:h_ext+h, 0:w] = temp_box, requiring more accurate bubble detection.
                box[0:h, 0:w] = temp_box  
            else:   
                #print("h_ext < 0, using original temp_box")  
                box = temp_box.copy()   
        else:   
            #print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            w_ext = int((h * r_orig - w) / 2)  
            #print(f"Calculated w_ext = {w_ext}")  
            
            if w_ext >= 0:  
                #print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [0:h, w_ext:w_ext+w] = [0:{h}, {w_ext}:{w_ext+w}]") 
                # Rows are fully filled, columns are centered
                box[0:h, w_ext:w_ext+w] = temp_box  
            else:   
                #print("w_ext < 0, using original temp_box")  
                box = temp_box.copy()   
    #print(f"Final box dimensions: {box.shape if box is not None else 'None'}")  

    src_points = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
    #src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
    #src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    rgba_region = cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x, y, w, h = cv2.boundingRect(dst_points.astype(np.int32))
    canvas_region = rgba_region[y:y+h, x:x+w, :3]
    mask_region = rgba_region[y:y+h, x:x+w, 3:4].astype(np.float32) / 255.0
    img[y:y+h, x:x+w] = np.clip((img[y:y+h, x:x+w].astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
    return img

async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/comic shanns 2.ttf')
    text_render.set_font(font_path)

    return render_textblock_list_eng(img_canvas, text_regions, line_spacing=line_spacing, size_tol=1.2, original_img=original_img, downscale_constraint=0.8,disable_font_border=disable_font_border)

async def dispatch_eng_render_pillow(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/NotoSansMonoCJK-VF.ttf.ttc')
    text_render.set_font(font_path)

    return render_textblock_list_eng_pillow(font_path, img_canvas, text_regions, original_img=original_img, downscale_constraint=0.95)
