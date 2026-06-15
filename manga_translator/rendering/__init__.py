import os
import cv2
import numpy as np
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
        #           "wrap" = no expansion, let text wrap at original font size
        #           "shrink" = no expansion, shrink font to fit in original lines
        img_h, img_w = img.shape[:2]
        
        if overflow_strategy in ("cascade", "auto") and dst_points is not None:
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
        
        # "expand" strategy: keep expanded dst_points as-is (allow overflow)

        # Store results and update font size
        dst_points_list.append(dst_points)  
        region.font_size = int(target_font_size)

    return dst_points_list

def _resolve_overlaps(dst_points_list, text_regions):
    """Detect and resolve overlapping dst_points boxes.
    
    When expanded text regions overlap, shrink them back toward their
    original min_rect to eliminate collisions.
    """
    if len(dst_points_list) <= 1:
        return dst_points_list
    
    def _get_xyxy(pts):
        """Get axis-aligned bounding box (x1, y1, x2, y2) from polygon points."""
        flat = pts.reshape(-1, 2)
        return (float(flat[:, 0].min()), float(flat[:, 1].min()),
                float(flat[:, 0].max()), float(flat[:, 1].max()))
    
    def _boxes_overlap(b1, b2, margin=2):
        """Check if two AABB boxes overlap (with small margin)."""
        return not (b1[2] <= b2[0] + margin or b1[0] >= b2[2] - margin or
                    b1[3] <= b2[1] + margin or b1[1] >= b2[3] - margin)
    
    def _overlap_area(b1, b2):
        """Calculate overlap area between two AABB boxes."""
        dx = min(b1[2], b2[2]) - max(b1[0], b2[0])
        dy = min(b1[3], b2[3]) - max(b1[1], b2[1])
        if dx > 0 and dy > 0:
            return dx * dy
        return 0
    
    # Get bounding boxes
    boxes = [_get_xyxy(pts) for pts in dst_points_list]
    
    # Detect and resolve overlaps
    max_iterations = 3
    for iteration in range(max_iterations):
        found_overlap = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not _boxes_overlap(boxes[i], boxes[j]):
                    continue
                found_overlap = True
                
                oa = _overlap_area(boxes[i], boxes[j])
                area_i = max((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]), 1)
                area_j = max((boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]), 1)
                
                # Shrink the region with larger overlap ratio back to min_rect
                ratio_i = oa / area_i
                ratio_j = oa / area_j
                
                # Shrink both overlapping regions back to their original min_rect
                if ratio_i > 0.1 or ratio_j > 0.1:
                    for idx in (i, j):
                        orig_rect = text_regions[idx].min_rect
                        orig_box = _get_xyxy(orig_rect)
                        # Blend: move 70% toward original min_rect
                        current = dst_points_list[idx].astype(np.float64)
                        original = orig_rect.astype(np.float64)
                        if current.shape == original.shape:
                            dst_points_list[idx] = (current * 0.3 + original * 0.7).astype(np.int64)
                            boxes[idx] = _get_xyxy(dst_points_list[idx])
        
        if not found_overlap:
            break
    
    return dst_points_list

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
    ) -> np.ndarray:

    text_render.set_font(font_path)
    text_regions = list(filter(lambda region: region.translation, text_regions))

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(img, text_regions, font_size_fixed, font_size_offset, font_size_minimum,
                                                     overflow_strategy=overflow_strategy, max_font_shrink_ratio=max_font_shrink_ratio)

    # Resolve overlapping text boxes to prevent text-on-text rendering
    dst_points_list = _resolve_overlaps(dst_points_list, text_regions)

    # Render text
    for region, dst_points in tqdm(zip(text_regions, dst_points_list), '[render]', total=len(text_regions)):
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)
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
