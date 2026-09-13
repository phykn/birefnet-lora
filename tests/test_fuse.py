import pytest
import torch
import torch.nn as nn

from src.adapt.fuse import fuse
from src.adapt.layer import LoRAConv2d, LoRALinear
from src.adapt.wrap import LoRABiRefNet
from src.prepare.spec import PreprocessSpec


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
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                module.up.weight.normal_(std=0.1, generator=generator)
                module.down.weight.normal_(std=0.1, generator=generator)
    return model


def test_fuse_preserves_logits_and_removes_adapters():
    model = _build().eval()
    x = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(1))
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
    import src.build.model as builder
    from src.serve.app import read_preprocess, read_threshold

    def load_overlay(cfg, base, path):
        model = LoRABiRefNet(base, rank=2, alpha=4.0)
        model.loaded_meta = {
            "selection": {"threshold": 0.42},
            "preprocess": {"size": 640, "mode": "gray_features"},
        }
        return model

    monkeypatch.setattr(builder, "build", lambda cfg: _Model())
    monkeypatch.setattr(builder, "load", load_overlay)

    model = builder.build_predictor(None, "overlay.pth", torch.device("cpu"))

    assert model.training is False
    assert not any(
        isinstance(module, (LoRALinear, LoRAConv2d)) for module in model.modules()
    )
    assert read_threshold(model) == 0.42
    assert read_preprocess(model) == PreprocessSpec(
        size=640,
        mode="gray_features",
    )


def test_fuse_does_not_keep_removed_adapter_parameters_alive():
    model = _build().eval()
    adapters = [
        parameter
        for module in model.modules()
        if isinstance(module, (LoRALinear, LoRAConv2d))
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    fuse(model)

    live = {id(parameter) for parameter in model.parameters()}
    assert all(id(parameter) not in live for parameter in adapters)
    assert model.list_trainable() == []
