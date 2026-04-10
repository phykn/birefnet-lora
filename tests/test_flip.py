import numpy as np

from src.ai.data.augment.flip import (
    Compose,
    HorizontalFlip,
    Identity,
    TransposeFlip,
    VerticalFlip,
    random_flip,
)


def _pair():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    mask = rng.integers(0, 255, size=(8, 8, 1), dtype=np.uint8)
    return image, mask


def test_identity_returns_inputs_unchanged():
    image, mask = _pair()
    img2, msk2 = Identity()(image, mask)
    assert np.array_equal(image, img2)
    assert np.array_equal(mask, msk2)


def test_horizontal_flip_inverts_columns():
    image, mask = _pair()
    img2, msk2 = HorizontalFlip()(image, mask)
    assert np.array_equal(img2, image[:, ::-1])
    assert np.array_equal(msk2, mask[:, ::-1])


def test_vertical_flip_inverts_rows():
    image, mask = _pair()
    img2, msk2 = VerticalFlip()(image, mask)
    assert np.array_equal(img2, image[::-1])


def test_compose_applies_in_order():
    image, mask = _pair()
    composed = Compose(VerticalFlip(), HorizontalFlip())
    img2, _ = composed(image, mask)
    assert np.array_equal(img2, image[::-1, ::-1])


def test_transpose_flip_supports_2d_mask():
    rng = np.random.default_rng(0)
    mask_2d = rng.integers(0, 255, size=(8, 8), dtype=np.uint8)
    out = TransposeFlip().flip(mask_2d)
    assert out.shape == (8, 8)
    assert np.array_equal(out, mask_2d.T)


def test_random_flip_preserves_shape_and_pair():
    image, mask = _pair()
    for _ in range(8):
        img2, msk2 = random_flip(image, mask)
        assert img2.shape == image.shape
        assert msk2.shape == mask.shape
