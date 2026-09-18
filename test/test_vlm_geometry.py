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
