import numpy as np
import torch

from src.train.metrics import (
    score_boundary,
    score_brier,
    score_dice,
    score_ece,
    score_iou_logits,
)


def test_region_metrics_define_empty_and_perfect_cases():
    target = torch.zeros(1, 1, 8, 8)
    negative = torch.full_like(target, -20.0)
    assert score_iou_logits(negative, target).item() == 1.0
    assert score_dice(negative, target).item() == 1.0

    target[:, :, 2:6, 2:6] = 1
    perfect = torch.where(target > 0, torch.tensor(20.0), torch.tensor(-20.0))
    assert score_iou_logits(perfect, target).item() == 1.0
    assert score_dice(perfect, target).item() == 1.0


def test_boundary_f1_matching_shift_and_empty_contract():
    empty = np.zeros((16, 16), dtype=np.uint8)
    assert score_boundary(empty, empty, tolerance_px=1) == 1.0

    target = empty.copy()
    target[4:12, 4:12] = 1
    assert score_boundary(target, target, tolerance_px=1) == 1.0

    shifted = empty.copy()
    shifted[4:12, 6:14] = 1
    assert 0.0 < score_boundary(shifted, target, tolerance_px=1) < 1.0


def test_probability_calibration_metrics_are_masked():
    target = torch.tensor([[[[0.0, 1.0, 1.0]]]])
    logits = torch.tensor([[[[-20.0, 20.0, -20.0]]]])
    valid = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    assert score_brier(logits, target, valid).item() < 1e-6
    assert score_ece(logits, target, valid).item() < 1e-6
