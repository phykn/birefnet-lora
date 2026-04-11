import numpy as np

from src.ml.data.augment.flip import random_flip


def _pair():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    mask = rng.integers(0, 255, size=(8, 8, 1), dtype=np.uint8)
    return image, mask


def test_random_flip_preserves_shape_and_pair():
    image, mask = _pair()
    for _ in range(32):
        img2, msk2 = random_flip(image, mask)
        assert img2.shape == image.shape
        assert msk2.shape == mask.shape


def test_random_flip_is_contiguous():
    image, mask = _pair()
    for _ in range(32):
        img2, msk2 = random_flip(image, mask)
        assert img2.flags["C_CONTIGUOUS"]
        assert msk2.flags["C_CONTIGUOUS"]


def test_random_flip_covers_all_d4_elements():
    image, mask = _pair()
    seen: set[bytes] = set()
    for seed in range(200):
        np.random.seed(seed)
        out, _ = random_flip(image, mask)
        seen.add(out.tobytes())
    assert len(seen) == 8, f"Expected 8 distinct D4 outputs, got {len(seen)}"


def test_random_flip_applies_same_transform_to_image_and_mask():
    # Using an image where channels are a numbered gradient so any transform
    # is detectable. Mask is independent content — only the transform must
    # be consistent across both arguments.
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    mask = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8, 1)
    for seed in range(16):
        np.random.seed(seed)
        img2, msk2 = random_flip(image, mask)
        # Apply the same rng pulls to a manual implementation and compare.
        np.random.seed(seed)
        rot_k = int(np.random.randint(4))
        do_hflip = bool(np.random.randint(2))
        expected_img = np.rot90(image, k=rot_k, axes=(0, 1))
        if do_hflip:
            expected_img = np.flip(expected_img, axis=1)
        expected_mask = np.rot90(mask, k=rot_k, axes=(0, 1))
        if do_hflip:
            expected_mask = np.flip(expected_mask, axis=1)
        assert np.array_equal(img2, np.ascontiguousarray(expected_img))
        assert np.array_equal(msk2, np.ascontiguousarray(expected_mask))
