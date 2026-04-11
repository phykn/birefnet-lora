from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import LoRAConv2d, LoRALinear, apply_conv2d, apply_linear


@dataclass
class ModelOutput:
    preds: list[torch.Tensor]
    aux: torch.Tensor | None = None


class LoRABiRefNet(nn.Module):
    def __init__(self, model: nn.Module, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        apply_linear(model=self.model.bb, rank=rank, alpha=alpha)
        apply_conv2d(
            model=self.model.decoder,
            rank=rank,
            alpha=alpha,
            exclude_names=["regular_conv", "offset_conv", "modulator_conv"],
        )

        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )

        self.stats = {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
        }

    def _train_step(self, x: torch.Tensor) -> ModelOutput:
        (gdt_predictions, gdt_labels), predictions = self.model(x)

        auxiliary_loss = torch.tensor(0.0, device=x.device)
        for gdt_prediction, gdt_label in zip(gdt_predictions, gdt_labels):
            gdt_prediction = F.interpolate(
                gdt_prediction,
                size=gdt_label.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
            gdt_label = gdt_label.sigmoid()
            auxiliary_loss = auxiliary_loss + F.binary_cross_entropy_with_logits(
                gdt_prediction,
                gdt_label,
            )
        num_levels = max(len(gdt_predictions), 1)
        auxiliary_loss = auxiliary_loss / num_levels

        return ModelOutput(preds=predictions, aux=auxiliary_loss)

    def _eval_step(self, x: torch.Tensor) -> ModelOutput:
        predictions = self.model(x)
        return ModelOutput(preds=predictions, aux=None)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if self.training:
            return self._train_step(x)
        return self._eval_step(x)

    def train(self, mode: bool = True) -> "LoRABiRefNet":
        super().train(mode)
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def get_adapter_params(self) -> list[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

    def _adapter_state_keys(self) -> set[str]:
        keys: set[str] = set()
        for module_name, module in self.named_modules():
            if not isinstance(module, (LoRALinear, LoRAConv2d)):
                continue
            prefix = f"{module_name}." if module_name else ""
            keys.add(f"{prefix}down.weight")
            keys.add(f"{prefix}up.weight")
        return keys

    def save_adapters(self, path: str) -> None:
        adapter_keys = self._adapter_state_keys()
        full_state = self.state_dict()
        missing_adapter_keys = sorted(
            key for key in adapter_keys if key not in full_state
        )
        if missing_adapter_keys:
            missing_preview = ", ".join(missing_adapter_keys[:5])
            raise RuntimeError(
                "Failed to collect adapter state keys from model state dict. "
                f"Missing keys: {missing_preview}"
            )

        state = {key: full_state[key] for key in sorted(adapter_keys)}
        torch.save(state, path)

    def load_adapters(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        expected_keys = self._adapter_state_keys()
        loaded_keys = set(state.keys())

        missing_keys = sorted(expected_keys - loaded_keys)
        unexpected_keys = sorted(loaded_keys - expected_keys)
        if missing_keys or unexpected_keys:
            parts = []
            if missing_keys:
                parts.append(f"missing={missing_keys[:5]}")
            if unexpected_keys:
                parts.append(f"unexpected={unexpected_keys[:5]}")
            details = ", ".join(parts)
            raise RuntimeError(
                "Adapter checkpoint keys do not match current LoRA structure: "
                f"{details}"
            )

        self.load_state_dict(state, strict=False)
