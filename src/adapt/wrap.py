from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .inject import inject_conv, inject_linear
from .overlay import OverlayMixin


@dataclass
class Output:
    logits: list[torch.Tensor]
    gdt: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        if not hasattr(module, part):
            raise ValueError(f"Unknown trainable module path: {path!r}")
        module = getattr(module, part)
        if not isinstance(module, nn.Module):
            raise ValueError(f"Path does not resolve to a module: {path!r}")
    return module


class LoRABiRefNet(OverlayMixin, nn.Module):
    def __init__(
        self,
        model: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        trainable_heads: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.trainable_heads = tuple(sorted(trainable_heads or ()))
        self.loaded_meta: dict[str, Any] | None = None

        for param in self.model.parameters():
            param.requires_grad = False

        inject_linear(model=self.model.bb, rank=self.rank, alpha=self.alpha)

        geometry_convs = ["regular_conv", "offset_conv", "modulator_conv"]
        inject_conv(
            model=self.model.squeeze_module,
            rank=self.rank,
            alpha=self.alpha,
            skip_names=geometry_convs,
        )

        head_paths = []
        for path in self.trainable_heads:
            prefix = "decoder."
            if not path.startswith(prefix):
                raise ValueError(
                    f"Full-trainable heads must be decoder paths, got {path!r}."
                )
            head_paths.append(path[len(prefix) :])

        inject_conv(
            model=self.model.decoder,
            rank=self.rank,
            alpha=self.alpha,
            skip_names=geometry_convs,
            skip_paths=head_paths,
        )

        for path in self.trainable_heads:
            module = _resolve_module(self.model, path)
            for param in module.parameters():
                param.requires_grad = True

        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        self._refresh_stats()

    def _refresh_stats(self) -> None:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        self.stats = {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    @staticmethod
    def _unpack(
        raw: Any,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise RuntimeError(
                "Raw BiRefNet training output must be [scaled_preds, class_preds_lst]."
            )
        scaled_preds = raw[0]
        if not isinstance(scaled_preds, (list, tuple)) or len(scaled_preds) != 2:
            raise RuntimeError(
                "BiRefNet scaled_preds must contain GDT outputs and mask outputs."
            )
        gdt_pair, preds = scaled_preds
        if not isinstance(gdt_pair, (list, tuple)) or len(gdt_pair) != 2:
            raise RuntimeError(
                "BiRefNet GDT output must contain predictions and labels."
            )
        gdt_preds, gdt_labels = gdt_pair
        return list(gdt_preds), list(gdt_labels), list(preds)

    def _train_step(self, x: torch.Tensor) -> Output:
        gdt_preds, gdt_labels, preds = self._unpack(self.model(x))
        return Output(logits=preds, gdt=(gdt_preds, gdt_labels))

    def _eval_step(self, x: torch.Tensor) -> Output:
        preds = self.model(x)
        if not isinstance(preds, (list, tuple)):
            raise RuntimeError("Raw BiRefNet eval output must be a prediction list.")
        return Output(logits=list(preds), gdt=None)

    def forward(self, x: torch.Tensor) -> Output:
        if self.training:
            return self._train_step(x)
        return self._eval_step(x)

    def train(self, mode: bool = True) -> "LoRABiRefNet":
        super().train(mode)
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def list_trainable(self) -> list[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]
