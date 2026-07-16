import os
from glob import glob
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ..prepare.load import MaskDataset
from .split import Splits, make, pack, restore


def index(paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        stem = Path(path).stem
        if stem in result:
            raise ValueError(f"Duplicate stem {stem!r}: {result[stem]} vs {path}")
        result[stem] = path
    return result


def build(
    cfg: Any,
    splits: Splits | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, Splits]:
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

    if len(data) < 3:
        raise ValueError(
            "At least three image/mask pairs are required for train/valid/calibration"
        )

    groups = (
        make(data, float(cfg.data.valid_ratio), float(cfg.data.calib_ratio))
        if splits is None
        else restore(data, splits)
    )
    train_data = groups["train"]
    valid_data = groups["valid"]
    calib_data = groups["calib"]

    train_dataset = MaskDataset(
        data=train_data,
        size=cfg.data.size,
        train=True,
        mode=cfg.data.get("mode", "rgb"),
        global_prob=float(cfg.data.get("global_prob", 0.3)),
        boundary_prob=float(cfg.data.get("boundary_prob", 0.5)),
        weak=(cfg.augment.weak.brightness, cfg.augment.weak.contrast),
        strong=(cfg.augment.strong.brightness, cfg.augment.strong.contrast),
    )
    valid_dataset = MaskDataset(
        data=valid_data,
        size=cfg.data.size,
        train=False,
        mode=cfg.data.get("mode", "rgb"),
    )
    calib_set = MaskDataset(
        data=calib_data,
        size=cfg.data.size,
        train=False,
        mode=cfg.data.get("mode", "rgb"),
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
    return train_loader, valid_loader, calib_loader, pack(groups)
