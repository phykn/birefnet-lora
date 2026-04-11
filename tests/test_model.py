import torch
import torch.nn as nn

from src.ml.model.birefnet.backbones.swin_v1 import BasicLayer
from src.ml.model.lora.adapters import LoRAConv2d, LoRALinear, apply_linear
from src.ml.model.lora.wrapper import LoRABiRefNet


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.offset_conv = nn.Conv2d(4, 4, 3, padding=1)
        self.modulator_conv = nn.Conv2d(4, 4, 3, padding=1)
        self.regular_conv = nn.Conv2d(4, 4, 3, padding=1)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
        self.bn = nn.BatchNorm2d(4)


class _FakeBiRefNet(nn.Module):
    """Minimal stand-in exposing the attributes LoRABiRefNet touches."""

    def __init__(self):
        super().__init__()
        self.bb = _Backbone()
        self.decoder = _Decoder()

    def forward(self, x):
        return x


def test_lora_birefnet_freezes_base_and_marks_adapters_trainable():
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    base_trainable = [
        p for n, p in lora.named_parameters()
        if p.requires_grad and ".down." not in n and ".up." not in n
    ]
    assert base_trainable == []
    assert lora.stats["trainable"] > 0
    assert lora.stats["frozen"] > 0


def test_lora_birefnet_excludes_geometry_convs_in_decoder():
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    dec = lora.model.decoder
    assert isinstance(dec.conv, LoRAConv2d)
    assert not isinstance(dec.offset_conv, LoRAConv2d)
    assert not isinstance(dec.modulator_conv, LoRAConv2d)
    assert not isinstance(dec.regular_conv, LoRAConv2d)


def test_lora_birefnet_applies_lora_to_backbone_linears():
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    assert isinstance(lora.model.bb.fc1, LoRALinear)
    assert isinstance(lora.model.bb.fc2, LoRALinear)


def test_lora_birefnet_keeps_batchnorm_in_eval_when_training():
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0).train()
    assert lora.training is True
    assert lora.model.bb.bn.training is False


def test_get_adapter_params_only_returns_trainable():
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    params = lora.get_adapter_params()
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_save_and_load_adapters_round_trip(tmp_path):
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    for p in lora.get_adapter_params():
        with torch.no_grad():
            p.add_(torch.randn_like(p))

    path = tmp_path / "adapters.pth"
    lora.save_adapters(str(path))
    assert path.exists()

    saved_state = {n: p.detach().clone() for n, p in lora.named_parameters() if p.requires_grad}

    fresh = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    fresh.load_adapters(str(path))
    loaded_state = {n: p for n, p in fresh.named_parameters() if p.requires_grad}

    assert set(saved_state.keys()) == set(loaded_state.keys())
    for k in saved_state:
        assert torch.allclose(saved_state[k], loaded_state[k])


def test_checkpointed_basic_layer_propagates_grad_to_lora_with_frozen_input():
    """Regression: gradient checkpointing must use use_reentrant=False so that
    LoRA adapters inside frozen swin blocks still receive gradients when the
    input tensor has requires_grad=False (which is the case under LoRA training,
    since the patch_embed conv is frozen)."""
    torch.manual_seed(0)
    dim = 12
    layer = BasicLayer(
        dim=dim,
        depth=2,
        num_heads=2,
        window_size=4,
        mlp_ratio=2.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=True,
        sdpa_enabled=False,
    )
    for p in layer.parameters():
        p.requires_grad = False
    apply_linear(layer, rank=2, alpha=4.0)

    adapter_params = [p for p in layer.parameters() if p.requires_grad]
    assert len(adapter_params) > 0

    h = w = 8
    x = torch.randn(1, h * w, dim, requires_grad=False)
    out, *_ = layer(x, h, w)
    out.sum().backward()

    grads = [p.grad for p in adapter_params]
    assert all(g is not None for g in grads), (
        "LoRA adapters inside checkpointed swin blocks did not receive gradients; "
        "gradient checkpointing likely fell back to use_reentrant=True"
    )
    assert any(g.abs().sum() > 0 for g in grads)


def test_load_adapters_rejects_mismatched_keys(tmp_path):
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    path = tmp_path / "bad.pth"
    torch.save({"unrelated.key": torch.zeros(1)}, str(path))
    try:
        lora.load_adapters(str(path))
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError on key mismatch")
