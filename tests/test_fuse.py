import pytest
import torch
import torch.nn as nn

from src.adapt.fuse import fuse
from src.adapt.layer import LoRAConv2d, LoRALinear
from src.adapt.wrap import LoRABiRefNet


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bb = _Backbone()
        self.squeeze_module = _Decoder()
        self.decoder = _Decoder()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        pooled = x.mean(dim=(-2, -1))
        bias = self.bb.fc2(torch.relu(self.bb.fc1(pooled)))[:, :, None, None]
        return [self.decoder.conv(x) + bias]


def _build() -> LoRABiRefNet:
    model = LoRABiRefNet(_Model(), rank=2, alpha=4.0)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                module.up.weight.normal_()
                module.down.weight.normal_()
    return model


def test_fuse_preserves_logits_and_removes_adapters():
    model = _build().eval()
    x = torch.randn(2, 3, 8, 8)
    expected = model(x).logits[-1]

    actual = fuse(model)(x).logits[-1]

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    assert not any(isinstance(module, LoRAConv2d) for module in model.modules())


def test_fuse_rejects_training_model_without_changing_it():
    model = _build().train()

    with pytest.raises(RuntimeError, match="eval model"):
        fuse(model)

    assert any(
        isinstance(module, (LoRALinear, LoRAConv2d)) for module in model.modules()
    )


def test_fuse_rejects_repeat_call():
    model = _build().eval()
    fuse(model)

    with pytest.raises(RuntimeError, match="no LoRA adapters"):
        fuse(model)


def test_api_loader_fuses_model_and_keeps_overlay_meta(monkeypatch):
    import run_api

    def load_overlay(cfg, base, path):
        model = LoRABiRefNet(base, rank=2, alpha=4.0)
        model.loaded_meta = {"selection": {"threshold": 0.42}}
        return model

    monkeypatch.setattr(run_api, "build_model", lambda cfg: _Model())
    monkeypatch.setattr(run_api, "load_model_overlay", load_overlay)

    model = run_api.load_model("overlay.pth", torch.device("cpu"))

    assert model.training is False
    assert not any(
        isinstance(module, (LoRALinear, LoRAConv2d)) for module in model.modules()
    )
    assert run_api.read_threshold(model) == 0.42
