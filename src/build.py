import os
import random
import torch

from torch.utils.data import DataLoader
from typing import Any

from .data.dataset import collect_paths, FineTuneDataset, TrainDataset, ValidDataset
from .finetune.loss import SegmentationLoss
from .finetune.model import LoRABiRefNet
from .finetune.trainer import Trainer
from .models.birefnet import BiRefNet


def build_dl(cfg: Any) -> tuple[DataLoader, DataLoader, dict[str, list[str]]]:
    img_paths = collect_paths(path=cfg.data.img_dir, exts=FineTuneDataset.EXTS)
    mask_paths = collect_paths(path=cfg.data.mask_dir, exts=FineTuneDataset.EXTS)

    combined = list(zip(img_paths, mask_paths))
    rng = random.Random(42)
    rng.shuffle(combined)

    n_valid = int(len(combined) * cfg.data.split_ratio)
    valid_data = combined[:n_valid]
    train_data = combined[n_valid:]

    train_img, train_mask = zip(*train_data) if train_data else ([], [])
    valid_img, valid_mask = zip(*valid_data) if valid_data else ([], [])

    train_ds = TrainDataset(
        img_paths = list(train_img),
        mask_paths = list(train_mask),
        size = cfg.data.size
    )
    valid_ds = ValidDataset(
        img_paths = list(valid_img),
        mask_paths = list(valid_mask),
        size = cfg.data.size
    )

    train_dl = DataLoader(
        dataset = train_ds,
        batch_size = cfg.train.batch,
        shuffle = True,
        num_workers = 4,
        pin_memory = True,
        drop_last = True,
    )
    valid_dl = DataLoader(
        dataset = valid_ds,
        batch_size = cfg.train.batch,
        shuffle = False,
        num_workers = 4,
        pin_memory = True,
    )

    data = {
        "train": [os.path.basename(p) for p in train_img],
        "valid": [os.path.basename(p) for p in valid_img]
    }

    return train_dl, valid_dl, data


def build_birefnet(cfg: Any) -> BiRefNet:
    return BiRefNet(
        lateral_channels_in_collection = cfg.birefnet.lateral_channels_in_collection,
        mul_scl_ipt = cfg.birefnet.mul_scl_ipt,
        dec_ipt = cfg.birefnet.dec_ipt,
        dec_ipt_split = cfg.birefnet.dec_ipt_split,
        ms_supervision = cfg.birefnet.ms_supervision,
        out_ref = cfg.birefnet.out_ref,
    )


def build_lora_birefnet(cfg: Any) -> LoRABiRefNet:
    model = build_birefnet(cfg=cfg)

    if cfg.birefnet.weight:
        state = torch.load(cfg.birefnet.weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    return LoRABiRefNet(model=model, rank=cfg.lora.rank, alpha=cfg.lora.alpha)


def build_trainer(
    cfg: Any,
    model: torch.nn.Module,
    train_dl: DataLoader,
    valid_dl: DataLoader
) -> Trainer:
    device = next(model.parameters()).device
    criterion = SegmentationLoss()
    optimizer = torch.optim.AdamW(params=model.get_adapter_params(), lr=cfg.train.lr)

    return Trainer(
        model = model,
        train_loader = train_dl,
        valid_loader = valid_dl,
        criterion = criterion,
        optimizer = optimizer,
        device = device,
    )