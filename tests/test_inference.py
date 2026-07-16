import numpy as np
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


def test_tile_planner_covers_image_and_aligns_last_tile():
    boxes = plan(45, 61, size=24, overlap_ratio=1 / 3)
    assert max(box.bottom for box in boxes) == 45
    assert max(box.right for box in boxes) == 61
    coverage = np.zeros((45, 61), dtype=np.int32)
    for box in boxes:
        coverage[box.top : box.bottom, box.left : box.right] += 1
    assert np.all(coverage > 0)


def test_tile_planner_uses_two_or_three_by_two_or_three_grid():
    assert len(plan(1500, 1500, size=1024)) == 4
    assert len(plan(2000, 2000, size=1024)) == 9


def test_tile_planner_keeps_overlap_at_least_one_third():
    for height, width in [(1500, 1500), (2000, 2000), (3000, 4000)]:
        boxes = plan(height, width, size=1024, overlap_ratio=1 / 3)
        for box in boxes:
            if box.overlap_top:
                assert box.overlap_top / box.height >= 1 / 3
            if box.overlap_bottom:
                assert box.overlap_bottom / box.height >= 1 / 3
            if box.overlap_left:
                assert box.overlap_left / box.width >= 1 / 3
            if box.overlap_right:
                assert box.overlap_right / box.width >= 1 / 3


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
    left, right = plan(1500, 1500, size=1024)[:2]
    overlap = left.overlap_right
    assert overlap == right.overlap_left
    left_window = weigh(left)
    right_window = weigh(right)
    np.testing.assert_allclose(
        left_window[0, -overlap:] + right_window[0, :overlap],
        1.0,
        atol=1e-6,
    )


def test_tiling_is_opt_in(monkeypatch):
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
        tile=True,
        overlap_ratio=1 / 3,
        tile_batch=1,
    )
    second = predict_logits(
        model,
        image,
        size=32,
        tile=True,
        overlap_ratio=1 / 3,
        tile_batch=3,
    )
    assert first.shape == image.shape[:2]
    np.testing.assert_allclose(first, 1.25, atol=1e-6)
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_output_modes_are_explicit_and_restore_original_shape():
    model = _ConstantModel(logit=0.0).eval()
    image = np.zeros((17, 43, 3), dtype=np.uint8)
    binary = predict(
        model,
        image,
        output_mode="binary",
        threshold=0.5,
        size=32,
        overlap_ratio=1 / 3,
    )
    probability = predict(
        model,
        image,
        output_mode="probability",
        size=32,
        overlap_ratio=1 / 3,
    )
    assert binary.shape == probability.shape == image.shape[:2]
    assert np.all(binary == 255)
    assert np.all(probability == 128)
