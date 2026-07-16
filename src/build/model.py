from typing import Any

import torch

from ..adapt.wrap import LoRABiRefNet
from ..model.birefnet import BiRefNet


def build(cfg: Any) -> BiRefNet:
    path = str(cfg.birefnet.weight) if cfg.birefnet.weight else None
    model = BiRefNet(
        channels=cfg.birefnet.channels,
        grad_checkpoint=cfg.birefnet.grad_checkpoint,
    )
    if path:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(value, torch.Tensor) for value in state.values()
        ):
            raise RuntimeError("Base checkpoint must be a flat tensor state_dict")
        model.load_state_dict(state, strict=True)
        print(f"[LOAD] {path}")

    return model


def adapt(cfg: Any, model: torch.nn.Module) -> LoRABiRefNet:
    device = next(model.parameters()).device
    wrapped = LoRABiRefNet(
        model=model,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        trainable_heads=list(cfg.lora.get("trainable_heads", [])),
    )
    return wrapped.to(device)


def load(
    cfg: Any,
    model: torch.nn.Module,
    path: str,
) -> LoRABiRefNet:
    device = next(model.parameters()).device
    wrapped = LoRABiRefNet(
        model=model,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        trainable_heads=list(cfg.lora.get("trainable_heads", [])),
    )
    wrapped.load_overlay(path)
    print(f"[LOAD] {path}")
    return wrapped.to(device)
