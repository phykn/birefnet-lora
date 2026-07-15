import torch
import torch.nn as nn

from src.model.birefnet.swin import BasicLayer
from src.model.lora.inject import inject_linear
from src.model.lora.layers import LoRAConv2d, LoRALinear
from src.model.lora.model import LoRABiRefNet


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
        self.squeeze_module = _Decoder()
        self.decoder = _Decoder()

    def forward(self, x):
        return x


def test_lora_birefnet_freezes_base_and_marks_adapters_trainable():
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    base_trainable = [
        p
        for n, p in lora.named_parameters()
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


def test_lora_birefnet_applies_lora_to_squeeze_module():
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    squeeze = lora.model.squeeze_module
    assert isinstance(squeeze.conv, LoRAConv2d)
    assert not isinstance(squeeze.offset_conv, LoRAConv2d)
    assert not isinstance(squeeze.modulator_conv, LoRAConv2d)
    assert not isinstance(squeeze.regular_conv, LoRAConv2d)


def test_lora_birefnet_applies_lora_to_backbone_linears():
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    assert isinstance(lora.model.bb.fc1, LoRALinear)
    assert isinstance(lora.model.bb.fc2, LoRALinear)


def test_zero_initialized_lora_preserves_base_eval_output():
    class _EvalStub(_FakeBiRefNet):
        def forward(self, x):
            return [self.decoder.conv(x)]

    base = _EvalStub().eval()
    x = torch.randn(1, 3, 8, 8)
    expected = base(x)[0].detach().clone()
    wrapped = LoRABiRefNet(base, rank=2, alpha=4.0).eval()
    actual = wrapped(x).preds[0]
    assert torch.allclose(actual, expected)


def test_lora_birefnet_keeps_batchnorm_in_eval_when_training():
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0).train()
    assert lora.training is True
    assert lora.model.bb.bn.training is False


def test_list_trainable_only_returns_trainable():
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    params = lora.list_trainable()
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_overlay_round_trip_includes_lora_and_full_head(tmp_path):
    lora = LoRABiRefNet(
        _FakeBiRefNet(),
        rank=2,
        alpha=4.0,
        trainable_heads=["decoder.conv"],
    )
    assert isinstance(lora.model.decoder.conv, nn.Conv2d)
    assert not isinstance(lora.model.decoder.conv, LoRAConv2d)
    for p in lora.list_trainable():
        with torch.no_grad():
            p.add_(torch.randn_like(p))

    path = tmp_path / "overlay.pth"
    lora.save_overlay(str(path))
    assert path.exists()

    saved_state = {
        n: p.detach().clone() for n, p in lora.named_parameters() if p.requires_grad
    }

    fresh = LoRABiRefNet(
        _FakeBiRefNet(),
        rank=2,
        alpha=4.0,
        trainable_heads=["decoder.conv"],
    )
    fresh.load_overlay(str(path))
    loaded_state = {n: p for n, p in fresh.named_parameters() if p.requires_grad}

    assert set(saved_state.keys()) == set(loaded_state.keys())
    for k in saved_state:
        assert torch.allclose(saved_state[k], loaded_state[k])


def test_checkpointed_basic_layer_propagates_grad_to_lora_with_frozen_input():
    """Regression: gradient checkpointing must use use_reentrant=False so that
    LoRA adapters inside frozen swin blocks still receive gradients when the
    input tensor has requires_grad=False (which is the case under LoRA training,
    since the patch_embed conv is frozen)."""
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
    )
    for p in layer.parameters():
        p.requires_grad = False
    inject_linear(layer, rank=2, alpha=4.0)

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


def test_overlay_rejects_invalid_format(tmp_path):
    lora = LoRABiRefNet(_FakeBiRefNet(), rank=2, alpha=4.0)
    path = tmp_path / "bad.pth"
    torch.save({"unrelated.key": torch.zeros(1)}, str(path))
    try:
        lora.load_overlay(str(path))
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError on key mismatch")


def test_lora_birefnet_forward_returns_modeloutput_in_both_modes():
    from src.model.lora.model import ModelOutput

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = _Backbone()
            self.squeeze_module = _Decoder()
            self.decoder = _Decoder()

        def forward(self, x):
            pred = torch.zeros(x.shape[0], 1, 4, 4)
            if self.training:
                gdt_preds = [torch.zeros(x.shape[0], 1, 4, 4)]
                gdt_labels = [torch.zeros(x.shape[0], 1, 4, 4)]
                scaled_preds = [[gdt_preds, gdt_labels], [pred, pred]]
                return [scaled_preds, [None]]
            return [pred, pred]

    lora = LoRABiRefNet(_Stub(), rank=2, alpha=4.0)

    lora.train()
    out_train = lora(torch.randn(1, 3, 8, 8))
    assert isinstance(out_train, ModelOutput)
    assert isinstance(out_train.preds, list)
    assert out_train.gdt is not None

    lora.eval()
    out_eval = lora(torch.randn(1, 3, 8, 8))
    assert isinstance(out_eval, ModelOutput)
    assert out_eval.gdt is None
