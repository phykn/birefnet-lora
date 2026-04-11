import os
import random
from pathlib import Path
from typing import Any

import torch
from glob import glob
from torch.utils.data import DataLoader

from .data.dataset import Dataset
from .model.birefnet.birefnet import BiRefNet
from .model.lora.wrapper import LoRABiRefNet
from .training.loss import CustomLoss
from .training.scheduler import CosineAnnealingWarmupRestarts
from .training.trainer import Trainer


def index_by_stem(paths: list[str]) -> dict[str, str]:
    stem_to_path: dict[str, str] = {}
    for path in paths:
        stem = Path(path).stem
        if stem in stem_to_path:
            raise ValueError(f"Duplicate stem {stem!r}: {stem_to_path[stem]} vs {path}")
        stem_to_path[stem] = path
    return stem_to_path


def build_dl(cfg: Any) -> tuple[DataLoader, DataLoader, dict[str, list[str]]]:
    image_paths = glob(os.path.join(cfg.data.image_dir, "*"))
    mask_paths = glob(os.path.join(cfg.data.mask_dir, "*"))

    i_dict = index_by_stem(image_paths)
    m_dict = index_by_stem(mask_paths)

    keys = set(i_dict.keys()) & set(m_dict.keys())
    data = [(i_dict[key], m_dict[key]) for key in sorted(keys)]

    rng = random.Random(42)
    rng.shuffle(data)

    split_ratio = float(cfg.data.split_ratio)
    num_valid = int(len(data) * split_ratio)
    train_data = data[num_valid:]
    valid_data = data[:num_valid]

    train_dataset = Dataset(
        data=train_data,
        size=cfg.data.size,
        scales=cfg.data.scales,
        train=True,
        bc_weak=(cfg.augment.bc_weak.brightness, cfg.augment.bc_weak.contrast),
        bc_strong=(cfg.augment.bc_strong.brightness, cfg.augment.bc_strong.contrast),
    )
    valid_dataset = Dataset(
        data=valid_data,
        size=cfg.data.size,
        train=False,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.dl.batch,
        shuffle=True,
        num_workers=cfg.dl.num_workers,
        pin_memory=cfg.dl.pin_memory,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.dl.batch,
        shuffle=False,
        num_workers=cfg.dl.num_workers,
        pin_memory=cfg.dl.pin_memory,
    )

    split_filenames = {
        "train_image": [Path(image).name for image, _ in train_data],
        "train_mask": [Path(mask).name for _, mask in train_data],
        "valid_image": [Path(image).name for image, _ in valid_data],
        "valid_mask": [Path(mask).name for _, mask in valid_data],
    }

    return train_loader, valid_loader, split_filenames


def build_birefnet(cfg: Any) -> BiRefNet:
    model = BiRefNet(
        lateral_channels_in_collection=cfg.birefnet.lateral_channels_in_collection,
        dec_ipt=cfg.birefnet.dec_ipt,
        dec_ipt_split=cfg.birefnet.dec_ipt_split,
        ms_supervision=cfg.birefnet.ms_supervision,
        out_ref=cfg.birefnet.out_ref,
        gradient_checkpointing=cfg.birefnet.gradient_checkpointing,
    )

    if cfg.birefnet.weight:
        checkpoint_path = cfg.birefnet.weight
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"[LOAD] {checkpoint_path}")

    return model


def build_lora_birefnet_for_training(
    cfg: Any,
    model: torch.nn.Module,
) -> LoRABiRefNet:
    """Wrap `model` with fresh (random-initialized) LoRA adapters for training."""
    device = next(model.parameters()).device
    lora_model = LoRABiRefNet(model=model, rank=cfg.lora.rank, alpha=cfg.lora.alpha)
    return lora_model.to(device)


def build_lora_birefnet_for_inference(
    cfg: Any,
    model: torch.nn.Module,
    ckpt_path: str,
) -> LoRABiRefNet:
    """Wrap `model` with LoRA adapters and load trained weights from `ckpt_path`."""
    device = next(model.parameters()).device
    lora_model = LoRABiRefNet(model=model, rank=cfg.lora.rank, alpha=cfg.lora.alpha)
    lora_model.load_adapters(ckpt_path)
    print(f"[LOAD] {ckpt_path}")
    return lora_model.to(device)


def build_trainer(
    cfg: Any,
    model: LoRABiRefNet,
    train_dl: DataLoader,
    valid_dl: DataLoader,
) -> Trainer:
    criterion = CustomLoss(
        lambda_bce=cfg.loss.lambda_bce,
        lambda_iou=cfg.loss.lambda_iou,
        lambda_kl=cfg.loss.lambda_kl,
        lambda_aux=cfg.loss.lambda_aux,
    )

    optimizer = torch.optim.AdamW(
        params=model.get_adapter_params(),
        lr=cfg.train.max_lr,
    )

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer=optimizer,
        first_cycle_steps=cfg.train.steps,
        max_lr=cfg.train.max_lr,
        min_lr=cfg.train.min_lr,
        warmup_steps=cfg.train.warmup_steps,
    )

    return Trainer(
        model=model,
        train_loader=train_dl,
        valid_loader=valid_dl,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=cfg.train.max_grad_norm,
        accum_steps=cfg.train.accum_steps,
    )
