import numpy as np
import pytest
import torch
import torch.nn as nn

from src.adapt.wrap import Output
from src.predict.run import predict, predict_logits
from src.predict.tile import Tile, plan, weigh


class _ConstantModel(nn.Module):
    def __init__(self, logit: float = 0.0):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(logit))

    def forward(self, x):
        output = self.logit.expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return Output(logits=[output])


class _MeanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        value = x.mean(dim=(1, 2, 3), keepdim=True) + self.anchor
        output = value.expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return Output(logits=[output])


def test_tile_planner_covers_image_and_aligns_last_tile():
    boxes = plan(45, 61, grid=4, overlap=1 / 3)
    assert max(box.bottom for box in boxes) == 45
    assert max(box.right for box in boxes) == 61
    coverage = np.zeros((45, 61), dtype=np.int32)
    for box in boxes:
        coverage[box.top : box.bottom, box.left : box.right] += 1
    assert np.all(coverage > 0)


def test_tile_planner_uses_requested_grid():
    for grid in (1, 2, 3, 4):
        assert len(plan(1500, 1500, grid=grid)) == grid**2


def test_tile_planner_keeps_overlap_at_least_one_third():
    for height, width, grid in [
        (1500, 1500, 2),
        (2000, 2000, 3),
        (3000, 4000, 4),
    ]:
        boxes = plan(height, width, grid=grid, overlap=1 / 3)
        for box in boxes:
            if box.overlap_top:
                assert box.overlap_top / box.height >= 1 / 3
            if box.overlap_bottom:
                assert box.overlap_bottom / box.height >= 1 / 3
            if box.overlap_left:
                assert box.overlap_left / box.width >= 1 / 3
            if box.overlap_right:
                assert box.overlap_right / box.width >= 1 / 3

    with pytest.raises(ValueError, match="overlap"):
        plan(100, 100, grid=2, overlap=0.2)


def test_blend_window_is_symmetric_and_keeps_outer_edges_nonzero():
    box = Tile(
        0,
        0,
        12,
        16,
        overlap_top=4,
        overlap_left=5,
        overlap_bottom=4,
        overlap_right=5,
    )
    window = weigh(box)
    np.testing.assert_allclose(window, window[::-1, :])
    np.testing.assert_allclose(window, window[:, ::-1])

    outer = weigh(Tile(0, 0, 12, 16, overlap_bottom=4, overlap_right=5))
    assert np.all(outer[0] > 0)
    assert np.all(outer[:, 0] > 0)


def test_neighboring_cosine_windows_are_complementary():
    left, right = plan(1500, 1500, grid=2)[:2]
    overlap = left.overlap_right
    assert overlap == right.overlap_left
    left_window = weigh(left)
    right_window = weigh(right)
    np.testing.assert_allclose(
        left_window[0, -overlap:] + right_window[0, :overlap],
        1.0,
        atol=1e-6,
    )


def test_one_by_one_skips_tile_planner(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("tile planner should not run")

    monkeypatch.setattr("src.predict.run.plan", fail)
    model = _ConstantModel(logit=1.25).eval()
    image = np.zeros((45, 61, 3), dtype=np.uint8)
    logits = predict_logits(model, image, size=32)

    assert logits.shape == image.shape[:2]
    np.testing.assert_allclose(logits, 1.25, atol=1e-6)


def test_tile_batch_does_not_change_logits():
    model = _ConstantModel(logit=1.25).eval()
    image = np.zeros((45, 61, 3), dtype=np.uint8)
    first = predict_logits(
        model,
        image,
        size=32,
        tiles=[3],
        overlap=1 / 3,
        tile_batch=1,
    )
    second = predict_logits(
        model,
        image,
        size=32,
        tiles=[3],
        overlap=1 / 3,
        tile_batch=3,
    )
    assert first.shape == image.shape[:2]
    np.testing.assert_allclose(first, 1.25, atol=1e-6)
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_multiple_grids_are_averaged_at_logit_level():
    model = _MeanModel().eval()
    image = np.zeros((45, 61, 3), dtype=np.uint8)
    image[:, 31:] = 255
    single = predict_logits(model, image, size=32, tiles=[1])
    tiled = predict_logits(model, image, size=32, tiles=[2])
    combined = predict_logits(model, image, size=32, tiles=[1, 2])
    np.testing.assert_allclose(combined, (single + tiled) / 2, atol=1e-6)


def test_tiles_accept_any_positive_grid():
    model = _ConstantModel(logit=1.25).eval()
    image = np.zeros((17, 23, 3), dtype=np.uint8)
    logits = predict_logits(model, image, size=32, tiles=[4], tile_batch=5)
    np.testing.assert_allclose(logits, 1.25, atol=1e-6)

    with pytest.raises(ValueError, match="positive integers"):
        predict_logits(model, image, size=32, tiles=[0])


def test_output_modes_are_explicit_and_restore_original_shape():
    model = _ConstantModel(logit=0.0).eval()
    image = np.zeros((17, 43, 3), dtype=np.uint8)
    binary = predict(
        model,
        image,
        output_mode="binary",
        threshold=0.5,
        size=32,
        overlap=1 / 3,
    )
    probability = predict(
        model,
        image,
        output_mode="probability",
        size=32,
        overlap=1 / 3,
    )
    assert binary.shape == probability.shape == image.shape[:2]
    assert np.all(binary == 255)
    assert np.all(probability == 128)
