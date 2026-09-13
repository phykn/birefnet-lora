import os
from glob import glob
from typing import Any

from torch.utils.data import DataLoader

from ..data.dataset import MaskDataset
from ..data.pairs import pair_files
from ..data.split import Splits, make, pack, restore


def _loader(cfg: Any, dataset: MaskDataset, shuffle: bool) -> DataLoader:
    workers = int(cfg.loader.num_workers)
    if workers < 0:
        raise ValueError("loader.num_workers must be non-negative")

    persistent = bool(cfg.loader.get("persistent_workers", workers > 0))
    if workers == 0 and persistent:
        raise ValueError("persistent_workers requires num_workers > 0")

    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(cfg.loader.batch),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(cfg.loader.pin_memory),
    }
    if workers > 0:
        prefetch = int(cfg.loader.get("prefetch_factor", 2))
        if prefetch < 1:
            raise ValueError("loader.prefetch_factor must be positive")
        options.update(
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
    return DataLoader(**options)


def build(
    cfg: Any,
    splits: Splits | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, Splits]:
    image_paths = glob(os.path.join(cfg.data.image_dir, "*"))
    mask_paths = glob(os.path.join(cfg.data.mask_dir, "*"))
    data = pair_files(image_paths, mask_paths)

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

    train_loader = _loader(cfg, train_dataset, shuffle=True)
    valid_loader = _loader(cfg, valid_dataset, shuffle=False)
    calib_loader = _loader(cfg, calib_set, shuffle=False)
    return train_loader, valid_loader, calib_loader, pack(groups)
