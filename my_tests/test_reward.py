import json

from labsrc.reward import reward


def test_reward_accepts_valid_json():
    gt = [{"label": "pothole", "bbox": (400.0, 400.0, 600.0, 600.0)}]
    completion = json.dumps([{"label": "pothole", "bbox_2d": [400, 400, 600, 600]}])
    assert reward(completion, gt) > 0.8


def test_reward_rejects_malformed_json():
    assert reward("not json", []) == 0.0


def test_reward_penalizes_wrong_class():
    gt = [{"label": "pothole", "bbox": (400.0, 400.0, 600.0, 600.0)}]
    completion = json.dumps([{"label": "crack", "bbox_2d": [400, 400, 600, 600]}])
    assert reward(completion, gt) < 0.4


def test_reward_handles_empty_predictions():
    gt = [{"label": "pothole", "bbox": (400.0, 400.0, 600.0, 600.0)}]
    assert reward("[]", gt) < 0.5
