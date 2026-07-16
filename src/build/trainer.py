import os
from datetime import datetime
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..adapt.wrap import LoRABiRefNet
from ..train.loss import TrainLoss
from ..train.schedule import CosineSchedule
from ..train.teacher import Teacher
from ..train.run import Trainer


def build(
    cfg: Any,
    model: LoRABiRefNet,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    calib_loader: DataLoader,
) -> Trainer:
    criterion = TrainLoss(
        gce_q=cfg.loss.gce_q,
        lambda_cls=cfg.loss.lambda_cls,
        lambda_region=cfg.loss.lambda_region,
        lambda_boundary=cfg.loss.lambda_boundary,
        region_loss=cfg.loss.region_loss,
        boundary_radius=cfg.loss.boundary_radius,
        lambda_aux=cfg.loss.lambda_aux,
        teacher_confidence=cfg.teacher.confidence,
        min_gt_weight=cfg.teacher.min_gt_weight,
        lambda_teacher=cfg.teacher.loss_weight,
    )

    lora_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".down." in name or ".up." in name:
            lora_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if lora_params:
        param_groups.append(
            {
                "name": "lora",
                "params": lora_params,
                "lr": float(cfg.train.max_lr),
                "max_lr": float(cfg.train.max_lr),
                "min_lr": float(cfg.train.min_lr),
                "weight_decay": float(cfg.train.get("weight_decay", 0.01)),
            }
        )
    if head_params:
        scale = float(cfg.train.get("head_lr_scale", 0.5))
        param_groups.append(
            {
                "name": "heads",
                "params": head_params,
                "lr": float(cfg.train.max_lr) * scale,
                "max_lr": float(cfg.train.max_lr) * scale,
                "min_lr": float(cfg.train.min_lr) * scale,
                "weight_decay": float(cfg.train.get("head_weight_decay", 0.01)),
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters were selected")

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = CosineSchedule(
        optimizer=optimizer,
        first_cycle_steps=cfg.train.steps,
        max_lr=cfg.train.max_lr,
        min_lr=cfg.train.min_lr,
        warmup_steps=cfg.train.warmup_steps,
    )
    save_dir = os.path.join("run", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    teacher = Teacher(
        model,
        decay=cfg.teacher.decay,
        start=cfg.teacher.start,
        ramp=cfg.teacher.ramp,
    )
    return Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        calib_loader=calib_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        teacher=teacher,
        save_dir=save_dir,
        max_grad_norm=cfg.train.max_grad_norm,
        accum_steps=cfg.train.accum_steps,
    )
