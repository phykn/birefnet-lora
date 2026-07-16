import torch
import torch.nn as nn

from .layer import LoRAConv2d, LoRALinear


def _copy(base: nn.Linear | nn.Conv2d, delta: torch.Tensor) -> None:
    dtype = base.weight.dtype
    weight = base.weight.detach().float().clone()
    weight.add_(delta)
    base.weight.copy_(weight.to(dtype=dtype))


def _linear(layer: LoRALinear) -> nn.Linear:
    up = layer.up.weight.detach().float()
    down = layer.down.weight.detach().float()
    _copy(layer.linear, (up @ down) * layer.scale)
    return layer.linear


def _conv(layer: LoRAConv2d) -> nn.Conv2d:
    up = layer.up.weight.detach().float()[:, :, 0, 0]
    down = layer.down.weight.detach().float()
    delta = (up @ down.flatten(1)).reshape_as(layer.conv.weight)
    _copy(layer.conv, delta * layer.scale)
    return layer.conv


@torch.no_grad()
def fuse(model: nn.Module) -> nn.Module:
    if model.training:
        raise RuntimeError("LoRA fusion requires an eval model")

    count = _replace(model)
    if count == 0:
        raise RuntimeError("Model has no LoRA adapters to fuse")
    return model


def _replace(module: nn.Module) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, _linear(child))
            count += 1
        elif isinstance(child, LoRAConv2d):
            setattr(module, name, _conv(child))
            count += 1
        else:
            count += _replace(child)
    return count
