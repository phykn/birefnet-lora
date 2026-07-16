import numpy as np
from PIL import Image

from src.prepare.load import MaskDataset
from src.prepare.read import read_image


def _write_pair(tmp_path):
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]
    mask = np.zeros((12, 20), dtype=np.uint8)
    mask[3:9, 5:15] = 255
    image_path = tmp_path / "sample.png"
    mask_path = tmp_path / "sample_mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    return image_path, mask_path


def test_loader_uses_rgb_channel_order(tmp_path):
    image_path, _ = _write_pair(tmp_path)
    assert read_image(str(image_path))[0, 0].tolist() == [255, 0, 0]


def test_validation_sample_has_fixed_canvas_binary_mask_and_valid_mask(tmp_path):
    image_path, mask_path = _write_pair(tmp_path)
    dataset = MaskDataset([(str(image_path), str(mask_path))], size=32, train=False)
    sample = dataset[0]
    assert sample["weak"].shape == (3, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)
    assert sample["valid"].shape == (1, 32, 32)
    assert set(np.unique(sample["mask"])) <= {0.0, 1.0}
    assert np.all(sample["mask"] <= sample["valid"])


def test_training_views_share_geometry(tmp_path):
    image_path, mask_path = _write_pair(tmp_path)
    dataset = MaskDataset(
        [(str(image_path), str(mask_path))],
        size=32,
        train=True,
        global_prob=1.0,
    )
    sample = dataset[0]
    assert sample["weak"].shape == sample["strong"].shape
    assert sample["mask"].shape == sample["valid"].shape
