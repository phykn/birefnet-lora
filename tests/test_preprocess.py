import numpy as np

from src.data.input import MEAN, STD, convert, normalize
from src.data.letterbox import plan, prepare_image, prepare_mask, restore_logit


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
    tensor, valid, geometry = prepare_image(image, size=32)
    assert tensor.shape == (3, 32, 32)
    assert valid.shape == (1, 32, 32)
    assert geometry.resized_h == 16
    assert geometry.resized_w == 32
    assert valid.sum() == 16 * 32
    assert np.all(valid[:, geometry.pad_top : geometry.pad_top + 16] == 1)


def test_mask_resize_stays_binary_and_matches_valid_geometry():
    mask = np.zeros((15, 30), dtype=np.float32)
    mask[:, 10:] = 1
    geometry = plan(15, 30, size=32)
    prepared = prepare_mask(mask, geometry)
    assert set(np.unique(prepared)) <= {0.0, 1.0}
    assert prepared.shape == (1, 32, 32)


def test_restore_logit_returns_original_shape_and_removes_padding():
    image = np.zeros((17, 43, 3), dtype=np.uint8)
    _, _, geometry = prepare_image(image, size=32)
    canvas = np.zeros((32, 32), dtype=np.float32)
    y0, x0 = geometry.pad_top, geometry.pad_left
    canvas[y0 : y0 + geometry.resized_h, x0 : x0 + geometry.resized_w] = 3.0
    restored = restore_logit(canvas, geometry)
    assert restored.shape == (17, 43)
    np.testing.assert_allclose(restored, 3.0)
