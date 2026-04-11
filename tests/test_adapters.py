import torch
import torch.nn as nn

from src.ml.model.lora.adapters import (
    LoRAConv2d,
    LoRALinear,
    apply_conv2d,
    apply_linear,
)


def test_lora_linear_initially_identity():
    linear = nn.Linear(8, 4)
    lora = LoRALinear(linear, rank=2, alpha=4.0)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), linear(x))


def test_lora_linear_freezes_base_and_only_adapters_trainable():
    lora = LoRALinear(nn.Linear(8, 4), rank=2, alpha=4.0)
    assert not lora.linear.weight.requires_grad
    assert lora.down.weight.requires_grad
    assert lora.up.weight.requires_grad


def test_lora_linear_output_shape():
    lora = LoRALinear(nn.Linear(8, 4), rank=2, alpha=4.0)
    out = lora(torch.randn(3, 8))
    assert out.shape == (3, 4)


def test_lora_conv2d_initially_identity_and_shape():
    conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
    lora = LoRAConv2d(conv, rank=2, alpha=4.0)
    x = torch.randn(2, 3, 8, 8)
    out = lora(x)
    assert out.shape == (2, 5, 8, 8)
    assert torch.allclose(out, conv(x))


def test_lora_conv2d_rejects_grouped_conv():
    grouped = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=2)
    try:
        LoRAConv2d(grouped, rank=2, alpha=4.0)
    except ValueError:
        return
    raise AssertionError("LoRAConv2d should reject grouped conv")


def test_apply_linear_replaces_all_linears():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    net = Net()
    apply_linear(net, rank=2, alpha=4.0)
    assert isinstance(net.fc1, LoRALinear)
    assert isinstance(net.fc2, LoRALinear)
    out = net(torch.randn(1, 4))
    assert out.shape == (1, 2)


def test_apply_conv2d_respects_excludes():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.offset_conv = nn.Conv2d(4, 4, 3, padding=1)

    net = Net()
    apply_conv2d(net, rank=2, alpha=4.0, exclude_names=["offset_conv"])
    assert isinstance(net.conv, LoRAConv2d)
    assert isinstance(net.offset_conv, nn.Conv2d)
    assert not isinstance(net.offset_conv, LoRAConv2d)
