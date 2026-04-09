import pytest

from src.build import build_pairs


def test_build_pairs_returns_sorted_pairs() -> None:
    image_paths = ["/tmp/data/image/b.png", "/tmp/data/image/a.png"]
    mask_paths = ["/tmp/data/mask/a.png", "/tmp/data/mask/b.png"]

    pairs = build_pairs(image_paths, mask_paths)

    assert pairs == [
        ("/tmp/data/image/a.png", "/tmp/data/mask/a.png"),
        ("/tmp/data/image/b.png", "/tmp/data/mask/b.png"),
    ]


def test_build_pairs_raises_on_filename_mismatch() -> None:
    image_paths = ["/tmp/data/image/a.png", "/tmp/data/image/b.png"]
    mask_paths = ["/tmp/data/mask/a.png"]

    with pytest.raises(ValueError, match="Image/mask filename mismatch"):
        build_pairs(image_paths, mask_paths)
