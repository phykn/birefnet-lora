import os
import random
from glob import glob
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ..data.dataset import MaskDataset
from .options import get


def index(paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        stem = Path(path).stem
        if stem in result:
            raise ValueError(f"Duplicate stem {stem!r}: {result[stem]} vs {path}")
        result[stem] = path
    return result


def build_loaders(
    cfg: Any,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[str]]]:
    image_paths = glob(os.path.join(cfg.data.image_dir, "*"))
    mask_paths = glob(os.path.join(cfg.data.mask_dir, "*"))
    images = index(image_paths)
    masks = index(mask_paths)

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            "Image/mask stems do not match: "
            f"missing_masks={missing_masks[:5]}, missing_images={missing_images[:5]}"
        )
    data = [(images[key], masks[key]) for key in sorted(images)]

    random.shuffle(data)
    valid_ratio = float(cfg.data.valid_ratio)
    calib_ratio = float(cfg.data.calib_ratio)
    if valid_ratio <= 0 or calib_ratio <= 0:
        raise ValueError("validation/calibration ratios must be positive")
    if valid_ratio + calib_ratio >= 1:
        raise ValueError("validation/calibration ratios must sum to less than 1")
    if len(data) < 3:
        raise ValueError(
            "At least three image/mask pairs are required for train/valid/calibration"
        )

    n_valid = min(
        len(data) - 2,
        max(1, int(round(len(data) * valid_ratio))),
    )
    n_calib = min(
        len(data) - n_valid - 1,
        max(1, int(round(len(data) * calib_ratio))),
    )
    valid_data = data[:n_valid]
    calib_data = data[n_valid : n_valid + n_calib]
    train_data = data[n_valid + n_calib :]

    train_dataset = MaskDataset(
        data=train_data,
        size=cfg.data.size,
        train=True,
        mode=get(cfg.data, "mode", "rgb"),
        global_prob=float(get(cfg.data, "global_prob", 0.3)),
        boundary_prob=float(get(cfg.data, "boundary_prob", 0.5)),
        weak=(cfg.augment.weak.brightness, cfg.augment.weak.contrast),
        strong=(cfg.augment.strong.brightness, cfg.augment.strong.contrast),
    )
    valid_dataset = MaskDataset(
        data=valid_data,
        size=cfg.data.size,
        train=False,
        mode=get(cfg.data, "mode", "rgb"),
    )
    calib_set = MaskDataset(
        data=calib_data,
        size=cfg.data.size,
        train=False,
        mode=get(cfg.data, "mode", "rgb"),
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.loader.batch,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=False,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.loader.batch,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    )
    calib_loader = DataLoader(
        dataset=calib_set,
        batch_size=cfg.loader.batch,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    )
    splits = {
        "train_image": [Path(image).name for image, _ in train_data],
        "train_mask": [Path(mask).name for _, mask in train_data],
        "valid_image": [Path(image).name for image, _ in valid_data],
        "valid_mask": [Path(mask).name for _, mask in valid_data],
        "calib_image": [Path(image).name for image, _ in calib_data],
        "calib_mask": [Path(mask).name for _, mask in calib_data],
    }
    return train_loader, valid_loader, calib_loader, splits
