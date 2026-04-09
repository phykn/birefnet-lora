import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data.dataset import FineTuneDataset, TrainDataset, ValidDataset
from .finetune.loss import SegmentationLoss
from .finetune.model import LoRABiRefNet
from .finetune.trainer import Trainer
from .models.birefnet import BiRefNet


def build_pairs(
    image_paths: list[str],
    mask_paths: list[str],
) -> list[tuple[str, str]]:
    image_map: dict[str, str] = {}
    for p in image_paths:
        stem = Path(p).stem
        if stem in image_map:
            raise ValueError(f"Duplicate image stem detected: {stem}")
        image_map[stem] = p

    mask_map: dict[str, str] = {}
    for p in mask_paths:
        stem = Path(p).stem
        if stem in mask_map:
            raise ValueError(f"Duplicate mask stem detected: {stem}")
        mask_map[stem] = p

    if set(image_map) != set(mask_map):
        raise ValueError(
            "Image/mask filename mismatch: "
            f"image_only={sorted(set(image_map) - set(mask_map))[:5]}, "
            f"mask_only={sorted(set(mask_map) - set(image_map))[:5]}"
        )

    common_stems = sorted(image_map)
    return [(image_map[s], mask_map[s]) for s in common_stems]


def build_dataloaders(cfg: Any) -> tuple[DataLoader, DataLoader, dict[str, list[str]]]:
    image_dir = Path(cfg.data.img_dir)
    mask_dir = Path(cfg.data.mask_dir)

    normalized_exts = {ext.lower() for ext in FineTuneDataset.EXTS}
    image_files = [
        str(file_path)
        for file_path in sorted(image_dir.glob("*"))
        if file_path.is_file() and file_path.suffix.lower() in normalized_exts
    ]
    mask_files = [
        str(file_path)
        for file_path in sorted(mask_dir.glob("*"))
        if file_path.is_file() and file_path.suffix.lower() in normalized_exts
    ]

    paired_paths = build_pairs(image_files, mask_files)

    rng = random.Random(42)
    rng.shuffle(paired_paths)

    split_ratio = float(cfg.data.split_ratio)
    num_valid = int(len(paired_paths) * split_ratio)
    valid_pairs = paired_paths[:num_valid]
    train_pairs = paired_paths[num_valid:]

    train_images, train_masks = zip(*train_pairs) if train_pairs else ([], [])
    valid_images, valid_masks = zip(*valid_pairs) if valid_pairs else ([], [])

    train_dataset = TrainDataset(
        img_paths=list(train_images),
        mask_paths=list(train_masks),
        size=cfg.data.size,
    )
    valid_dataset = ValidDataset(
        img_paths=list(valid_images),
        mask_paths=list(valid_masks),
        size=cfg.data.size,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.train.batch,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )

    split_filenames = {
        "train": [Path(path).name for path in train_images],
        "valid": [Path(path).name for path in valid_images],
    }

    return train_loader, valid_loader, split_filenames


def build_birefnet(cfg: Any) -> BiRefNet:
    model = BiRefNet(
        lateral_channels_in_collection=cfg.birefnet.lateral_channels_in_collection,
        mul_scl_ipt=cfg.birefnet.mul_scl_ipt,
        dec_ipt=cfg.birefnet.dec_ipt,
        dec_ipt_split=cfg.birefnet.dec_ipt_split,
        ms_supervision=cfg.birefnet.ms_supervision,
        out_ref=cfg.birefnet.out_ref,
    )

    if cfg.birefnet.weight:
        checkpoint_path = cfg.birefnet.weight
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"[Checkpoint] loaded BiRefNet weights from: {checkpoint_path}")

    return model


def build_lora_birefnet(
    cfg: Any,
    model: torch.nn.Module,
    ckpt_path: str | None = None,
) -> LoRABiRefNet:
    lora_model = LoRABiRefNet(model=model, rank=cfg.lora.rank, alpha=cfg.lora.alpha)

    if ckpt_path:
        lora_model.load_adapters(ckpt_path)
        print(f"[Checkpoint] loaded LoRA adapters from: {ckpt_path}")

    return lora_model


def build_trainer(
    cfg: Any,
    model: torch.nn.Module,
    train_dl: DataLoader,
    valid_dl: DataLoader,
) -> Trainer:
    from .finetune.loss import ConsistencyLoss

    device = next(model.parameters()).device
    criterion = SegmentationLoss()
    cons_criterion = ConsistencyLoss()
    optimizer = torch.optim.AdamW(params=model.get_adapter_params(), lr=cfg.train.lr)

    return Trainer(
        model=model,
        train_loader=train_dl,
        valid_loader=valid_dl,
        criterion=criterion,
        cons_criterion=cons_criterion,
        optimizer=optimizer,
        device=device,
        lambda_cons=cfg.train.lambda_cons,
        max_grad_norm=cfg.train.max_grad_norm,
        scheduler_name=cfg.train.scheduler,
        warmup_steps=cfg.train.warmup_steps,
        total_steps=cfg.train.steps,
    )
