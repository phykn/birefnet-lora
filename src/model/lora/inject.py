import torch.nn as nn

from .layers import LoRAConv2d, LoRALinear


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
    exclude_names: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> None:
    excludes = set(exclude_names or [])
    excluded_paths = set(exclude_paths or [])
    for module_name, module in list(model.named_modules()):
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            continue
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, nn.Conv2d):
                child_path = (
                    f"{module_name}.{child_name}" if module_name else child_name
                )
                if child_name in excludes or child_path in excluded_paths:
                    continue
                setattr(
                    module,
                    child_name,
                    LoRAConv2d(conv=child_module, rank=rank, alpha=alpha),
                )
