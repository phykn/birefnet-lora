import numpy as np


def test_normalize_shape_preserved():
    image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    from src.ml.preprocess import normalize
    result = normalize(image)
    assert result.shape == (32, 32, 3)
    assert result.dtype == np.float32


def test_normalize_values():
    """Known-value check: a pixel of [124, 116, 104] ≈ [0.485, 0.456, 0.408] / 255
    should produce values near zero after ImageNet normalization."""
    from src.ml.preprocess import normalize
    image = np.full((1, 1, 3), [124, 116, 104], dtype=np.uint8)
    result = normalize(image)
    assert np.allclose(result, 0.0, atol=0.1)


def test_normalize_matches_legacy():
    """Ensure normalize produces the same result as the old inline logic."""
    from src.ml.preprocess import normalize

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    legacy = (image.astype(np.float32) / 255.0 - MEAN) / STD
    result = normalize(image)
    np.testing.assert_array_equal(result, legacy)
