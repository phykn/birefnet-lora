import numpy as np
import torch

from src.train.metrics import (
    boundary,
    brier,
    dice,
    ece,
    iou_at_thresholds,
    iou_logits,
)


def test_region_metrics_define_empty_and_perfect_cases():
    target = torch.zeros(1, 1, 8, 8)
    negative = torch.full_like(target, -20.0)
    assert iou_logits(negative, target).item() == 1.0
    assert dice(negative, target).item() == 1.0

    target[:, :, 2:6, 2:6] = 1
    perfect = torch.where(target > 0, torch.tensor(20.0), torch.tensor(-20.0))
    assert iou_logits(perfect, target).item() == 1.0
    assert dice(perfect, target).item() == 1.0


def test_boundary_f1_matching_shift_and_empty_contract():
    empty = np.zeros((16, 16), dtype=np.uint8)
    assert boundary(empty, empty, tolerance_px=1) == 1.0

    target = empty.copy()
    target[4:12, 4:12] = 1
    assert boundary(target, target, tolerance_px=1) == 1.0

    shifted = empty.copy()
    shifted[4:12, 6:14] = 1
    assert 0.0 < boundary(shifted, target, tolerance_px=1) < 1.0


def test_probability_calibration_metrics_are_masked():
    target = torch.tensor([[[[0.0, 1.0, 1.0]]]])
    logits = torch.tensor([[[[-20.0, 20.0, -20.0]]]])
    valid = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    assert brier(logits, target, valid).item() < 1e-6
    assert ece(logits, target, valid).item() < 1e-6


def test_iou_at_thresholds_matches_scalar_sweep_including_equal_values():
    probability = np.array(
        [[0.2, 0.3, 0.5], [0.7, 0.9, np.nan]],
        dtype=np.float32,
    )
    target = np.array(
        [[0, 1, 1], [0, 1, 0]],
        dtype=bool,
    )
    thresholds = [0.3, 0.5, 0.7]
    expected = []
    for threshold in thresholds:
        pred = probability >= threshold
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        expected.append(1.0 if union == 0 else intersection / union)

    np.testing.assert_allclose(
        iou_at_thresholds(probability, target, thresholds),
        expected,
        rtol=0,
        atol=0,
    )


def test_iou_at_thresholds_accumulates_chunks(monkeypatch):
    monkeypatch.setattr("src.train.metrics.CALIBRATION_CHUNK_PIXELS", 3)
    probability = np.array(
        [0.1, 0.3, 0.5, 0.7, 0.9, np.nan, 0.5],
        dtype=np.float32,
    )
    target = np.array([0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
    thresholds = [0.3, 0.5, 0.7]
    expected = []
    for threshold in thresholds:
        pred = probability >= threshold
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        expected.append(1.0 if union == 0 else intersection / union)

    np.testing.assert_allclose(
        iou_at_thresholds(probability, target, thresholds),
        expected,
        rtol=0,
        atol=0,
    )
