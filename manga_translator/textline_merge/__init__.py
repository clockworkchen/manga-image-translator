import itertools
import re
import numpy as np
from typing import List, Set
from collections import Counter
import networkx as nx
from shapely.geometry import Polygon

from ..utils import TextBlock, Quadrilateral, quadrilateral_can_merge_region

def split_text_region(
        bboxes: List[Quadrilateral],
        connected_region_indices: Set[int],
        width,
        height,
        gamma = 0.5,
        sigma = 2
    ) -> List[Set[int]]:

    connected_region_indices = list(connected_region_indices)

    # case 1
    if len(connected_region_indices) == 1:
        return [set(connected_region_indices)]

    # case 2
    if len(connected_region_indices) == 2:
        fs1 = bboxes[connected_region_indices[0]].font_size
        fs2 = bboxes[connected_region_indices[1]].font_size
        fs = max(fs1, fs2)

        # print(bboxes[connected_region_indices[0]].pts, bboxes[connected_region_indices[1]].pts)
        # print(fs, bboxes[connected_region_indices[0]].distance(bboxes[connected_region_indices[1]]), (1 + gamma) * fs)
        # print(bboxes[connected_region_indices[0]].angle, bboxes[connected_region_indices[1]].angle, 4 * np.pi / 180)

        if bboxes[connected_region_indices[0]].distance(bboxes[connected_region_indices[1]]) < (1 + gamma) * fs \
                and abs(bboxes[connected_region_indices[0]].angle - bboxes[connected_region_indices[1]].angle) < 0.2 * np.pi:
            return [set(connected_region_indices)]
        else:
            return [set([connected_region_indices[0]]), set([connected_region_indices[1]])]

    # case 3
    G = nx.Graph()
    for idx in connected_region_indices:
        G.add_node(idx)
    for (u, v) in itertools.combinations(connected_region_indices, 2):
        G.add_edge(u, v, weight=bboxes[u].distance(bboxes[v]))
    # Get distances from neighbouring bboxes
    edges = nx.algorithms.tree.minimum_spanning_edges(G, algorithm='kruskal', data=True)
    edges = sorted(edges, key=lambda a: a[2]['weight'], reverse=True)
    distances_sorted = [a[2]['weight'] for a in edges]
    fontsize = np.mean([bboxes[idx].font_size for idx in connected_region_indices])
    distances_std = np.std(distances_sorted)
    distances_mean = np.mean(distances_sorted)
    std_threshold = max(0.3 * fontsize + 5, 5)

    b1, b2 = bboxes[edges[0][0]], bboxes[edges[0][1]]
    max_poly_distance = Polygon(b1.pts).distance(Polygon(b2.pts))
    max_centroid_alignment = min(abs(b1.centroid[0] - b2.centroid[0]), abs(b1.centroid[1] - b2.centroid[1]))

    # print(edges)
    # print(f'std: {distances_std} < thrshold: {std_threshold}, mean: {distances_mean}')
    # print(f'{distances_sorted[0]} <= {distances_mean + distances_std * sigma}' \
    #         f' or {distances_sorted[0]} <= {fontsize * (1 + gamma)}' \
    #         f' or {distances_sorted[0] - distances_sorted[1]} < {distances_std * sigma}')

    if (distances_sorted[0] <= distances_mean + distances_std * sigma \
            or distances_sorted[0] <= fontsize * (1 + gamma)) \
            and (distances_std < std_threshold \
            or max_poly_distance == 0 and max_centroid_alignment < 5):
        return [set(connected_region_indices)]
    else:
        # (split_u, split_v, _) = edges[0]
        # print(f'split between "{bboxes[split_u].pts}", "{bboxes[split_v].pts}"')
        G = nx.Graph()
        for idx in connected_region_indices:
            G.add_node(idx)
        # Split out the most deviating bbox
        for edge in edges[1:]:
            G.add_edge(edge[0], edge[1])
        ans = []
        for node_set in nx.algorithms.components.connected_components(G):
            ans.extend(split_text_region(bboxes, node_set, width, height))
        return ans

# def get_mini_boxes(contour):
#     bounding_box = cv2.minAreaRect(contour)
#     points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

#     index_1, index_2, index_3, index_4 = 0, 1, 2, 3
#     if points[1][1] > points[0][1]:
#         index_1 = 0
#         index_4 = 1
#     else:
#         index_1 = 1
#         index_4 = 0
#     if points[3][1] > points[2][1]:
#         index_2 = 2
#         index_3 = 3
#     else:
#         index_2 = 3
#         index_3 = 2

#     box = [points[index_1], points[index_2], points[index_3], points[index_4]]
#     box = np.array(box)
#     startidx = box.sum(axis=1).argmin()
#     box = np.roll(box, 4 - startidx, 0)
#     box = np.array(box)
#     return box

def _style_heights(item):
    """Explicit VLM glyph measurements only; detector box size is not style."""
    values = getattr(item, 'ocr_style_heights', None)
    if values is None:
        values = [getattr(item, 'ocr_style_height', None)]
    return [float(value) for value in values
            if value is not None and np.isfinite(value) and value > 0]


def _compatible_styles(items):
    sizes = [size for item in items for size in _style_heights(item)]
    # Deliberately independent of caller-supplied stacked slack: geometric
    # uncertainty may be relaxed, but measured shouting/body styles may not.
    return not sizes or max(sizes) <= min(sizes) * 1.3


def merge_bboxes_text_region(bboxes: List[Quadrilateral], width, height,
                             font_size_ratio_tol: float = 1.3,
                             aspect_ratio_tol: float = 1.3,
                             char_gap_tolerance: float = 1,
                             char_gap_tolerance2: float = 3,
                             stacked_font_ratio_slack: float = 1.35):
    # step 0: merge quadrilaterals that belong to the same textline
    # u = 0
    # removed_counter = 0
    # while u < len(bboxes) - 1 - removed_counter:
    #     v = u
    #     while v < len(bboxes) - removed_counter:
    #         if quadrilateral_can_merge_region(bboxes[u], bboxes[v], aspect_ratio_tol=1.1, font_size_ratio_tol=1,
    #                                         char_gap_tolerance=1, char_gap_tolerance2=3, discard_connection_gap=0) \
    #            and abs(bboxes[u].centroid[0] - bboxes[v].centroid[0]) < 5 or abs(bboxes[u].centroid[1] - bboxes[v].centroid[1]) < 5:
    #                 bboxes[u] = merge_quadrilaterals(bboxes[u], bboxes[v])
    #                 removed_counter += 1
    #                 bboxes.pop(v)
    #         else:
    #             v += 1
    #     u += 1

    # step 1: divide into multiple text region candidates
    G = nx.Graph()
    for i, box in enumerate(bboxes):
        G.add_node(i, box=box)

    components = nx.utils.UnionFind(range(len(bboxes)))
    members = {i: [box] for i, box in enumerate(bboxes)}
    for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2):
        ur, vr = components[u], components[v]
        combined = members[ur] if ur == vr else members[ur] + members[vr]
        # Check the entire proposed component, not just this edge. An uncertain
        # intermediate line must not bridge a large and a small explicit style.
        if not _compatible_styles(combined):
            continue
        # Compatible measured styles may still need the original geometric
        # slack for ascenders/descenders. Conflicting styles were rejected
        # above, so slack can never override their explicit boundary.
        if quadrilateral_can_merge_region(ubox, vbox, aspect_ratio_tol=aspect_ratio_tol,
                                          font_size_ratio_tol=font_size_ratio_tol,
                                          char_gap_tolerance=char_gap_tolerance,
                                          char_gap_tolerance2=char_gap_tolerance2,
                                          stacked_font_ratio_slack=stacked_font_ratio_slack):
            G.add_edge(u, v)
            if ur != vr:
                components.union(ur, vr)
                members.pop(ur)
                members.pop(vr)
                members[components[u]] = combined

    # step 2: postprocess - further split each region
    region_indices: List[Set[int]] = []
    for node_set in nx.algorithms.components.connected_components(G):
         region_indices.extend(split_text_region(bboxes, node_set, width, height))

    # step 3: return regions
    for node_set in region_indices:
    # for node_set in nx.algorithms.components.connected_components(G):
        nodes = list(node_set)
        txtlns: List[Quadrilateral] = np.array(bboxes)[nodes]

        # calculate average fg and bg color
        fg_r = round(np.mean([box.fg_r for box in txtlns]))
        fg_g = round(np.mean([box.fg_g for box in txtlns]))
        fg_b = round(np.mean([box.fg_b for box in txtlns]))
        bg_r = round(np.mean([box.bg_r for box in txtlns]))
        bg_g = round(np.mean([box.bg_g for box in txtlns]))
        bg_b = round(np.mean([box.bg_b for box in txtlns]))

        # majority vote for direction
        dirs = [box.direction for box in txtlns]
        majority_dir_top_2 = Counter(dirs).most_common(2)
        if len(majority_dir_top_2) == 1 :
            majority_dir = majority_dir_top_2[0][0]
        elif majority_dir_top_2[0][1] == majority_dir_top_2[1][1] : # if top 2 have the same counts
            max_aspect_ratio = -100
            for box in txtlns :
                if box.aspect_ratio > max_aspect_ratio :
                    max_aspect_ratio = box.aspect_ratio
                    majority_dir = box.direction
                if 1.0 / box.aspect_ratio > max_aspect_ratio :
                    max_aspect_ratio = 1.0 / box.aspect_ratio
                    majority_dir = box.direction
        else :
            majority_dir = majority_dir_top_2[0][0]

        # sort textlines
        if majority_dir == 'h':
            nodes = sorted(nodes, key=lambda x: bboxes[x].centroid[1])
        elif majority_dir == 'v':
            nodes = sorted(nodes, key=lambda x: -bboxes[x].centroid[0])
        txtlns = np.array(bboxes)[nodes]

        # yield overall bbox and sorted indices
        yield txtlns, (fg_r, fg_g, fg_b), (bg_r, bg_g, bg_b)

def _merge_region_members(regions, members, boxes):
    """Construct one region after the caller has proved ownership/style."""
    members = sorted(members, key=lambda i: boxes[i][1])
    ordered = [regions[i] for i in members]
    entries = [(line, text) for region in ordered
               for line, text in zip(region.lines, region.texts)]
    entries.sort(key=lambda item: float(np.asarray(item[0])[:, 1].mean()))
    sizes = [float(region.font_size) for region in ordered]
    merged = TextBlock(
        [item[0] for item in entries], [item[1] for item in entries],
        font_size=float(np.median(sizes)), angle=0,
        prob=min(region.prob for region in ordered),
        fg_color=np.median([region.get_font_colors()[0] for region in ordered], axis=0),
        bg_color=np.median([region.get_font_colors()[1] for region in ordered], axis=0))
    merged.text_raw = merged.text
    merged.ocr_style_heights = [size for region in ordered for size in _style_heights(region)]
    merged.ocr_ink_heights = [size for region in ordered
                              for size in getattr(region, 'ocr_ink_heights', [])]
    merged.ocr_block_ids = set().union(
        *(getattr(region, 'ocr_block_ids', set()) for region in ordered))
    merged.ocr_detector_rows = sum(
        int(getattr(region, 'ocr_detector_rows', len(region.lines))) for region in ordered)
    merged.ocr_source_rows = max(
        (int(getattr(region, 'ocr_source_rows', 0)) for region in ordered), default=0)
    merged.ocr_complete_block = all(
        bool(getattr(region, 'ocr_complete_block', True)) for region in ordered)
    rects = {getattr(region, 'ocr_block_rect', None) for region in ordered}
    rects.discard(None)
    merged.ocr_block_rect = next(iter(rects)) if len(rects) == 1 else None
    return merged


def merge_closed_bubble_regions(regions, image):
    """Join near-equal stacked text blocks only inside one closed comic balloon.

    Called before translation, exclusively in bubble mode. Preserve distinct
    shouting/body sizes, columns, rotations and uncertain/open backgrounds.
    """
    from ..rendering import _bubble_groups, _aabb_of

    boxes = [_aabb_of(region.min_rect) for region in regions]
    groups = _bubble_groups(image, boxes)
    # VLM crop membership is stronger ownership evidence than flood-filled
    # balloon components, which often split at glyph holes or hand-drawn gaps.
    # It does not relax any distance/style rule below; it only supplies the
    # candidate set, avoiding the failed global-threshold experiments.
    for i, region in enumerate(regions):
        block_ids = getattr(region, 'ocr_block_ids', set())
        if len(block_ids) == 1:
            groups[i] = ('vlm', next(iter(block_ids)))
    replacements, removed = {}, set()
    for group in set(groups):
        members = [i for i, value in enumerate(groups) if value == group]
        if len(members) < 2:
            continue
        members.sort(key=lambda i: boxes[i][1])
        ordered = [regions[i] for i in members]
        if not _compatible_styles(ordered):
            continue
        sizes = [float(region.font_size) for region in ordered]
        if min(sizes) <= 0 or max(sizes) / min(sizes) > 1.15:
            continue
        if any(abs(region.angle) > 3 or len(region.lines) != len(region.texts)
               for region in ordered):
            continue
        # A complete VLM crop already supplied all detector rows for the block.
        # If the normal merger still split those rows, block ownership is stronger
        # evidence than the post-OCR box-height estimate. This is deliberately
        # local to one crop and does not widen global merge thresholds.
        same_complete_vlm = all(
            bool(getattr(region, 'ocr_complete_block', False))
            and len(getattr(region, 'ocr_block_ids', set())) == 1
            for region in ordered)
        if not same_complete_vlm and any(len(region.lines) > 1 for region in ordered):
            continue
        connected = True
        for a, b in zip(members, members[1:]):
            ax1, ay1, ax2, ay2 = boxes[a]
            bx1, by1, bx2, by2 = boxes[b]
            overlap = min(ax2, bx2) - max(ax1, bx1)
            gap = by1 - ay2
            if (overlap < 0.5 * min(ax2 - ax1, bx2 - bx1)
                    or gap < -1 or gap > 1.5 * max(sizes)):
                connected = False
                break
        if not connected:
            continue
        merged = _merge_region_members(regions, members, boxes)
        first = min(members)
        replacements[first] = merged
        removed.update(i for i in members if i != first)
    return [replacements.get(i, region) for i, region in enumerate(regions) if i not in removed]


def merge_stacked_open_regions(regions):
    """Conservative fallback for open/gradient balloons and mixed-script rows.

    No background ownership is assumed. Merge only adjacent, centred, same-style
    horizontal rows whose union cannot plausibly be two columns.
    """
    # Bracket-prefixed game commands are independent menu rows even when they
    # share type, colour and leading. Merging them produces one tall paragraph,
    # erases multiple UI rows at once and makes a partial OCR look like recall.
    bracketed = sum(bool(re.match(
        r'^\s*[【\[（(「『〈《].{1,30}[】\]）)」』〉》]', str(region.text or '')))
        for region in regions)
    if len(regions) >= 4 and bracketed >= 3 and bracketed / len(regions) >= 0.5:
        for region in regions:
            region._layout_profile = 'game_ui'
        return regions
    from ..rendering import _aabb_of
    boxes = [_aabb_of(region.min_rect) for region in regions]
    used, out = set(), []
    order = sorted(range(len(regions)), key=lambda i: (boxes[i][1], boxes[i][0]))
    for index in order:
        if index in used:
            continue
        chain = [index]
        while True:
            current = chain[-1]
            ax1, ay1, ax2, ay2 = boxes[current]
            candidates = []
            for other in order:
                if other in used or other in chain or boxes[other][1] < ay2 - 1:
                    continue
                bx1, by1, bx2, by2 = boxes[other]
                overlap = min(ax2, bx2) - max(ax1, bx1)
                gap = by1 - ay2
                center_delta = abs((ax1 + ax2) - (bx1 + bx2)) / 2
                scale = max(regions[current].font_size, regions[other].font_size, 1)
                same_colour = np.linalg.norm(
                    np.asarray(regions[current].get_font_colors()[0], float)
                    - np.asarray(regions[other].get_font_colors()[0], float)) <= 45
                min_width = min(ax2-ax1, bx2-bx1)
                # Mixed CJK/Latin rows can differ greatly in width while sharing
                # the same centre. Require meaningful overlap, but allow 35% when
                # centring is especially strong; cross-column rows still fail.
                overlap_requirement = 0.35 if center_delta <= scale * 1.0 else 0.55
                if (0 <= gap <= scale * 0.9
                        and overlap >= overlap_requirement * min_width
                        and center_delta <= scale * 2.5
                        and abs(regions[current].angle) <= 3 and abs(regions[other].angle) <= 3
                        and _compatible_styles([regions[current], regions[other]])
                        and 0.84 <= regions[current].font_size / max(regions[other].font_size, 1) <= 1.18
                        and same_colour):
                    candidates.append((gap, center_delta, other))
            if not candidates:
                break
            chain.append(min(candidates)[2])
        used.update(chain)
        out.append(_merge_region_members(regions, chain, boxes) if len(chain) > 1 else regions[index])
    return out


async def dispatch(textlines: List[Quadrilateral], width: int, height: int, verbose: bool = False,
                   merge_opts: dict = None) -> List[TextBlock]:
    # print(width, height)
    # import re
    # for l in textlines:
    #     s = str(l.pts)
    #     s = re.sub(r'([\d\]]) ', r'\1, ', s.replace('\n ', ', ')).replace(']]', ']],')
    #     print(s)

    text_regions: List[TextBlock] = []
    # Game/menu screens often consist of repeated bracketed commands. Their
    # vertical rhythm resembles a paragraph, but every row is a separate action.
    # Keep those detector rows independent before the generic graph merger can
    # collapse the whole menu into one region.
    bracketed = sum(bool(re.match(
        r'^\s*[【\[（(「『〈《].{1,30}[】\]）)」』〉》]', str(line.text or '')))
        for line in textlines)
    game_ui = (len(textlines) >= 4 and bracketed >= 3
               and bracketed / len(textlines) >= 0.5)
    if game_ui:
        merged_regions = [([line], line.fg_colors, line.bg_colors) for line in textlines]
    else:
        merged_regions = merge_bboxes_text_region(textlines, width, height,
                                                   **(merge_opts or {}))
    for (txtlns, fg_color, bg_color) in merged_regions:
        total_logprobs = 0
        for txtln in txtlns:
            total_logprobs += np.log(txtln.prob) * txtln.area
        total_logprobs /= sum([txtln.area for txtln in textlines])

        font_size = int(min([txtln.font_size for txtln in txtlns]))
        angle = np.rad2deg(np.mean([txtln.angle for txtln in txtlns])) - 90
        if abs(angle) < 3:
            angle = 0
        lines = [txtln.pts for txtln in txtlns]
        texts = [txtln.text for txtln in txtlns]
        region = TextBlock(lines, texts, font_size=font_size, angle=angle, prob=np.exp(total_logprobs),
                           fg_color=fg_color, bg_color=bg_color)
        # Carry source measurements through later bubble regrouping; these are
        # pre-render evidence, not a claim about final displayed font size.
        region.ocr_style_heights = [size for line in txtlns for size in _style_heights(line)]
        region.ocr_ink_heights = [float(line.ocr_ink_height) for line in txtlns
                                 if hasattr(line, 'ocr_ink_height')]
        region.ocr_detector_rows = sum(
            int(getattr(line, 'ocr_detector_rows', 1)) for line in txtlns)
        region.ocr_source_rows = max(
            (int(getattr(line, 'ocr_source_rows', 0)) for line in txtlns), default=0)
        region.ocr_complete_block = all(
            bool(getattr(line, 'ocr_complete_block', True)) for line in txtlns)
        block_rects = {getattr(line, 'ocr_block_rect', None) for line in txtlns}
        block_rects.discard(None)
        region.ocr_block_rect = next(iter(block_rects)) if len(block_rects) == 1 else None
        block_ids = {getattr(line, 'ocr_block_id', None) for line in txtlns}
        block_ids.discard(None)
        region.ocr_block_ids = block_ids
        if game_ui:
            region._layout_profile = 'game_ui'
        text_regions.append(region)
    return text_regions
