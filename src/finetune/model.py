import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import apply_conv2d, apply_linear

ADAPTER_STATE_KEY_TOKENS = ("down", "up")


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
            exclude_names=["regular_conv"],
        )

        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)

        self.stats = {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
        }

    def _train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scaled_preds, _ = self.model(x)
        (gdt_predictions, gdt_labels), predictions = scaled_preds

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

        return predictions[-1], auxiliary_loss

    def _eval_step(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.model(x)
        return predictions[-1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.training:
            return self._train_step(x)
        return self._eval_step(x)

    def get_adapter_params(self) -> list[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

    def save_adapters(self, path: str) -> None:
        state = {
            key: value
            for key, value in self.state_dict().items()
            if any(token in key for token in ADAPTER_STATE_KEY_TOKENS)
        }
        torch.save(state, path)

    def load_adapters(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=False)
