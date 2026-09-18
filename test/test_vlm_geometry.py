import numpy as np

from manga_translator.ocr.model_vlm import ModelVlmOCR
from manga_translator.utils import Quadrilateral


def _quad(x1, y1, x2, y2):
    return Quadrilateral(np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ], dtype=np.float32), "", 1.0)


def test_group_members_preserve_exact_detector_ownership():
    lines = [
        _quad(503, 499, 677, 521),
        _quad(503, 522, 659, 534),
        _quad(503, 535, 677, 562),
    ]

    groups = ModelVlmOCR._group_box_members(lines)

    assert len(groups) == 1
    rect, members = groups[0]
    assert rect == [503.0, 499.0, 677.0, 562.0]
    assert members == [0, 1, 2]


def test_vlm_lines_use_detector_geometry_one_to_one():
    image = np.full((100, 240, 3), 255, dtype=np.uint8)
    detected = [
        _quad(20, 10, 200, 30),
        _quad(40, 34, 160, 48),
        _quad(20, 52, 200, 75),
    ]

    result = ModelVlmOCR()._split_into_lines(
        image, [20, 10, 200, 75], ["FIRST", "MIDDLE", "LAST"],
        detected_lines=detected,
    )

    assert [line.text for line in result] == ["FIRST", "MIDDLE", "LAST"]
    assert [line.pts.tolist() for line in result] == [line.pts.tolist() for line in detected]
    assert [line.ocr_ink_height for line in result] == [20.0, 14.0, 23.0]


def test_short_vlm_answer_uses_detector_row_groups_not_equal_slices():
    image = np.full((120, 260, 3), 255, dtype=np.uint8)
    detected = [
        _quad(20, 10, 220, 30),
        _quad(25, 29, 210, 50),
        _quad(30, 49, 205, 70),
        _quad(50, 69, 190, 90),
    ]

    bands, groups = ModelVlmOCR._align_lines_to_detector(
        image, [20, 10, 220, 90], ["LONG FIRST SENTENCE", "LAST"], detected)

    assert groups == [3, 1]
    assert bands[0][:2] == (20.0, 10.0)
    assert bands[0][2] == 220.0
    assert bands[0][3] == bands[1][1]
    assert bands[1][2:] == (190.0, 90.0)


def test_vlm_crop_supports_horizontal_and_vertical_padding():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = ModelVlmOCR._crop(image, [50, 40, 100, 60], pad=(20, 5))
    assert crop.shape[:2] == (30, 90)


def test_overlapping_detector_rows_get_disjoint_vertical_ownership():
    result = ModelVlmOCR._regularize_vertical_bands([
        (501.0, 497.0, 675.0, 524.0),
        (502.0, 511.0, 662.0, 560.0),
    ])
    assert result[0][3] == result[1][1]
    assert result[0][1] < result[0][3] < result[1][3]
