from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.func import functional_call

from ..adapt.wrap import Output


class Teacher:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.99,
        start: int = 100,
        ramp: int = 100,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("teacher decay must be in [0, 1)")
        if start < 0 or ramp < 1:
            raise ValueError("teacher start must be >= 0 and ramp must be >= 1")
        self.decay = float(decay)
        self.start = int(start)
        self.ramp = int(ramp)
        self.params = {
            name: param.detach().float().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        if not self.params:
            raise RuntimeError("Teacher requires trainable model parameters")

    def scale(self, step: int) -> float:
        return max(0.0, min(1.0, (step - self.start) / self.ramp))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = dict(model.named_parameters())
        for name, avg in self.params.items():
            param = current[name].detach().to(device=avg.device, dtype=avg.dtype)
            avg.lerp_(param, 1.0 - self.decay)

    @torch.no_grad()
    def predict(self, model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        training = model.training
        try:
            model.eval()
            out: Output = functional_call(
                model,
                self.params,
                (image,),
                strict=False,
            )
            return out.logits[-1].detach()
        finally:
            model.train(training)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.params.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = set(self.params)
        loaded = set(state)
        if expected != loaded:
            missing = sorted(expected - loaded)
            unexpected = sorted(loaded - expected)
            raise RuntimeError(
                "Teacher state keys do not match the model: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        for name, avg in self.params.items():
            value = state[name]
            if tuple(value.shape) != tuple(avg.shape):
                raise RuntimeError(f"Teacher state shape does not match for {name}")
            avg.copy_(value.to(device=avg.device, dtype=avg.dtype))
