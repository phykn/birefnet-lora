import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import TrainDataset, ValidDataset


@pytest.fixture()
def sample_data(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    for name in ["a.png", "b.png"]:
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(img_dir / name)
        mask = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
        mask.save(mask_dir / name)

    img_paths = sorted(str(p) for p in img_dir.glob("*.png"))
    mask_paths = sorted(str(p) for p in mask_dir.glob("*.png"))
    return img_paths, mask_paths


def test_train_dataset_returns_dual_views(sample_data) -> None:
    img_paths, mask_paths = sample_data
    ds = TrainDataset(img_paths=img_paths, mask_paths=mask_paths, size=32)
    item = ds[0]

    assert "image_v1" in item
    assert "image_v2" in item
    assert "mask" in item
    assert item["image_v1"].shape == (3, 32, 32)
    assert item["image_v2"].shape == (3, 32, 32)
    assert item["mask"].shape == (1, 32, 32)


def test_train_dataset_dual_views_share_geometry(sample_data) -> None:
    img_paths, mask_paths = sample_data
    ds = TrainDataset(img_paths=img_paths, mask_paths=mask_paths, size=32)
    item = ds[0]
    assert item["mask"].shape == (1, 32, 32)


def test_train_dataset_dual_views_differ_in_color(sample_data) -> None:
    img_paths, mask_paths = sample_data
    ds = TrainDataset(img_paths=img_paths, mask_paths=mask_paths, size=32)

    any_different = False
    for i in range(len(img_paths)):
        item = ds[i]
        if not torch.equal(item["image_v1"], item["image_v2"]):
            any_different = True
            break

    assert any_different, "Dual views should sometimes differ in color"


def test_valid_dataset_returns_single_view(sample_data) -> None:
    img_paths, mask_paths = sample_data
    ds = ValidDataset(img_paths=img_paths, mask_paths=mask_paths, size=32)
    item = ds[0]

    assert "image" in item
    assert "mask" in item
    assert item["image"].shape == (3, 32, 32)
    assert item["mask"].shape == (1, 32, 32)
