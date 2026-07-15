import numpy as np
import pytest

from src.data.augment import flip


def _pair():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    mask = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8, 1)
    return image, mask


def test_random_flip_preserves_shape_and_pair():
    image, mask = _pair()
    for _ in range(32):
        img2, msk2 = flip(image, mask)
        assert img2.shape == image.shape
        assert msk2.shape == mask.shape


def test_random_flip_is_contiguous():
    image, mask = _pair()
    for _ in range(32):
        img2, msk2 = flip(image, mask)
        assert img2.flags["C_CONTIGUOUS"]
        assert msk2.flags["C_CONTIGUOUS"]


def test_random_flip_covers_all_d4_elements(monkeypatch):
    image, mask = _pair()
    seen: set[bytes] = set()
    for rotation in range(4):
        for horizontal in range(2):
            values = iter((rotation, horizontal))
            monkeypatch.setattr(np.random, "randint", lambda *_: next(values))
            out, _ = flip(image, mask)
            seen.add(out.tobytes())
    assert len(seen) == 8, f"Expected 8 distinct D4 outputs, got {len(seen)}"


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("horizontal", range(2))
def test_random_flip_applies_same_transform_to_image_and_mask(
    monkeypatch, rotation, horizontal
):
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    mask = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8, 1)
    values = iter((rotation, horizontal))
    monkeypatch.setattr(np.random, "randint", lambda *_: next(values))
    img2, msk2 = flip(image, mask)
    expected_img = np.rot90(image, k=rotation, axes=(0, 1))
    expected_mask = np.rot90(mask, k=rotation, axes=(0, 1))
    if horizontal:
        expected_img = np.flip(expected_img, axis=1)
        expected_mask = np.flip(expected_mask, axis=1)
    assert np.array_equal(img2, np.ascontiguousarray(expected_img))
    assert np.array_equal(msk2, np.ascontiguousarray(expected_mask))
