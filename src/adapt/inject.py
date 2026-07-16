import torch.nn as nn

from .layer import LoRAConv2d, LoRALinear


def inject_linear(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> None:
    for _, module in list(model.named_modules()):
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            continue
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, nn.Linear):
                setattr(
                    module,
                    child_name,
                    LoRALinear(linear=child_module, rank=rank, alpha=alpha),
                )


def inject_conv(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    skip_names: list[str] | None = None,
    skip_paths: list[str] | None = None,
) -> None:
    names = set(skip_names or [])
    paths = set(skip_paths or [])
    for module_name, module in list(model.named_modules()):
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            continue
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, nn.Conv2d):
                child_path = (
                    f"{module_name}.{child_name}" if module_name else child_name
                )
                if child_name in names or child_path in paths:
                    continue
                setattr(
                    module,
                    child_name,
                    LoRAConv2d(conv=child_module, rank=rank, alpha=alpha),
                )
