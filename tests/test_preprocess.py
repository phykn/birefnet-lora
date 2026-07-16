import numpy as np

from src.prepare.convert import MEAN, STD, convert, normalize
from src.prepare.fit import fit_image, fit_mask, plan, restore


def test_normalize_known_rgb_pixel():
    image = np.array([[[255, 0, 127]]], dtype=np.uint8)
    expected = (image.astype(np.float32) / 255.0 - MEAN) / STD
    np.testing.assert_allclose(normalize(image), expected)


def test_modes_preserve_declared_channel_semantics():
    image = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(convert(image, "rgb"), image)
    gray = convert(image, "gray_repeat")
    assert gray.shape == image.shape
    np.testing.assert_array_equal(gray[..., 0], gray[..., 1])
    np.testing.assert_array_equal(gray[..., 1], gray[..., 2])
    assert convert(image, "gray_features").shape == image.shape


def test_fixed_canvas_and_valid_mask_for_wide_image():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    tensor, valid, fit = fit_image(image, size=32)
    assert tensor.shape == (3, 32, 32)
    assert valid.shape == (1, 32, 32)
    assert fit.dst_h == 16
    assert fit.dst_w == 32
    assert valid.sum() == 16 * 32
    assert np.all(valid[:, fit.top : fit.top + 16] == 1)


def test_mask_resize_stays_binary_and_matches_valid_geometry():
    mask = np.zeros((15, 30), dtype=np.float32)
    mask[:, 10:] = 1
    fit = plan(15, 30, size=32)
    prepared = fit_mask(mask, fit)
    assert set(np.unique(prepared)) <= {0.0, 1.0}
    assert prepared.shape == (1, 32, 32)


def test_restore_logit_returns_original_shape_and_removes_padding():
    image = np.zeros((17, 43, 3), dtype=np.uint8)
    _, _, fit = fit_image(image, size=32)
    canvas = np.zeros((32, 32), dtype=np.float32)
    y0, x0 = fit.top, fit.left
    canvas[y0 : y0 + fit.dst_h, x0 : x0 + fit.dst_w] = 3.0
    restored = restore(canvas, fit)
    assert restored.shape == (17, 43)
    np.testing.assert_allclose(restored, 3.0)
