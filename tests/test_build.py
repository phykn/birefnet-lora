import pytest
from omegaconf import OmegaConf
from PIL import Image

from src.build.data import build_loaders, index


def test_index_by_stem_maps_stem_to_path():
    paths = ["/data/image/a.png", "/data/image/b.jpg"]
    result = index(paths)
    assert result == {"a": "/data/image/a.png", "b": "/data/image/b.jpg"}


def test_index_by_stem_raises_on_duplicate():
    paths = ["/x/a.png", "/y/a.jpg"]
    with pytest.raises(ValueError, match="Duplicate stem"):
        index(paths)


def test_index_by_stem_empty():
    assert index([]) == {}


def test_build_loaders_rejects_unpaired_stems(tmp_path):
    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "mask"
    image_dir.mkdir()
    mask_dir.mkdir()
    Image.new("RGB", (4, 4)).save(image_dir / "sample.png")
    cfg = OmegaConf.create(
        {
            "data": {
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
                "size": 8,
                "valid_ratio": 0.2,
                "calib_ratio": 0.2,
            },
            "augment": {
                "weak": {"brightness": 0.2, "contrast": 0.2},
                "strong": {"brightness": 0.4, "contrast": 0.4},
            },
            "loader": {"batch": 1, "num_workers": 0, "pin_memory": False},
        }
    )
    with pytest.raises(ValueError, match="stems do not match"):
        build_loaders(cfg)
