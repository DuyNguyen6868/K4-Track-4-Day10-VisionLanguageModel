import json

import pytest

from labsrc.reward import iou_reward

GT = [{"label": "pothole", "bbox": (100, 100, 300, 300)}]


def ans(*boxes):
    return json.dumps([{"label": l, "bbox_2d": list(b)} for l, b in boxes])


def test_perfect():
    assert iou_reward(ans(("pothole", (100, 100, 300, 300))), GT) == pytest.approx(1.0)


def test_broken_json():
    assert iou_reward("không phải json", GT) == 0.0


def test_wrong_class_only_gets_validity_bonus():
    assert iou_reward(ans(("crack", (100, 100, 300, 300))), GT) == pytest.approx(0.2)


def test_extra_box_is_penalised():
    r = iou_reward(ans(("pothole", (100, 100, 300, 300)), ("pothole", (600, 600, 700, 700))), GT)
    assert r == pytest.approx(0.2 + 0.8 * 1.0 / 2)


def test_empty_vs_empty():
    assert iou_reward("[]", []) == 1.0


def test_empty_answer_when_gt_exists():
    assert iou_reward("[]", GT) == pytest.approx(0.2)


def test_low_iou_not_matched():
    assert iou_reward(ans(("pothole", (250, 250, 500, 500))), GT) == pytest.approx(0.2)
