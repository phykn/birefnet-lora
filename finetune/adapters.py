"""
어댑터 모듈 정의.

원본 파라미터를 하나도 변경하지 않고,
새로운 학습 가능한 경량 레이어만 추가하여 특징을 변환한다.

1. LoRALinear: Attention qkv / proj Linear에 병렬로 저랭크 경로를 추가
2. AdapterConv: Conv 출력에 잔차 보틀넥 어댑터를 추가
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# 1. LoRA for Linear layers (Backbone Attention)
# ─────────────────────────────────────────────
class LoRALinear(nn.Module):
    """
    원본 Linear를 감싸고, 병렬 저랭크 경로를 추가한다.
    output = original(x) + lora_up(lora_down(x)) * scale

    원본 가중치는 완전히 freeze 된다.
    """

    def __init__(self, original: nn.Linear, rank: int = 4, scale: float = 1.0):
        super().__init__()
        self.original = original
        self.scale = scale

        in_f = original.in_features
        out_f = original.out_features

        self.lora_down = nn.Linear(in_f, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_f, bias=False)

        # 시작 시 LoRA 출력 = 0 (원본 동작 그대로)
        nn.init.kaiming_normal_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

        # 원본 freeze
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_up(self.lora_down(x)) * self.scale


# ─────────────────────────────────────────────
# 2. Residual Adapter for Conv layers (Decoder)
# ─────────────────────────────────────────────
class AdapterConv(nn.Module):
    """
    원본 nn.Module을 감싸고, 출력에 잔차 보틀넥 어댑터를 추가한다.
    output = original(x) + adapter(original(x))

    adapter = Conv1x1(down) → ReLU → Conv1x1(up)
    초기 상태에서 adapter 출력 = 0 이므로 원본 동작 그대로.
    """

    def __init__(self, original: nn.Module, channels: int, reduction: int = 4):
        super().__init__()
        self.original = original
        mid = max(channels // reduction, 16)

        self.adapter = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

        # 시작 시 adapter 출력 = 0
        nn.init.zeros_(self.adapter[-1].weight)

        # 원본 freeze
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        return out + self.adapter(out)


# ─────────────────────────────────────────────
# 3. 모델에 어댑터를 적용하는 유틸리티 함수들
# ─────────────────────────────────────────────
def apply_lora_to_backbone(backbone: nn.Module, rank: int = 4, scale: float = 1.0) -> int:
    """
    Swin Transformer backbone 내의 모든 WindowAttention.qkv, .proj 에
    LoRA를 적용한다.

    Returns:
        적용된 LoRA 어댑터 수
    """
    count = 0
    for name, module in backbone.named_modules():
        # WindowAttention 찾기
        if module.__class__.__name__ == 'WindowAttention':
            # qkv
            if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
                module.qkv = LoRALinear(module.qkv, rank=rank, scale=scale)
                count += 1
            # proj
            if hasattr(module, 'proj') and isinstance(module.proj, nn.Linear):
                module.proj = LoRALinear(module.proj, rank=rank, scale=scale)
                count += 1
    return count


def apply_adapters_to_decoder(decoder: nn.Module, reduction: int = 4) -> int:
    """
    Decoder의 핵심 출력 지점에 AdapterConv를 삽입한다.

    적용 대상:
      - decoder_block4/3/2/1: 각 디코더 블록 출력에 어댑터
      - conv_out1: 최종 출력 conv 직전에 어댑터

    Returns:
        적용된 어댑터 수
    """
    count = 0

    # 디코더 블록 출력 어댑터
    for block_name in ['decoder_block4', 'decoder_block3', 'decoder_block2', 'decoder_block1']:
        if hasattr(decoder, block_name):
            original_block = getattr(decoder, block_name)
            # 출력 채널 수 추출: BasicDecBlk의 conv_out.out_channels
            if hasattr(original_block, 'conv_out'):
                out_ch = original_block.conv_out.out_channels
            else:
                # fallback
                out_ch = 64
            setattr(decoder, block_name, AdapterConv(original_block, out_ch, reduction))
            count += 1

    # 최종 출력 conv 어댑터 (1채널 출력이라 직접 어댑터 대신, 직전 특징에 어댑터를 넣는 것이 더 효과적)
    # conv_out1은 nn.Sequential(Conv2d(in, 1, 1)) 이므로, 입력 채널에 어댑터를 넣음
    if hasattr(decoder, 'conv_out1'):
        conv_out = decoder.conv_out1
        # conv_out1 내부의 Conv2d 입력 채널 가져오기
        for m in conv_out.modules():
            if isinstance(m, nn.Conv2d):
                in_ch = m.in_channels
                break
        # conv_out1 직전에 동작하는 어댑터를 삽입
        decoder.conv_out1 = AdapterConv(conv_out, in_ch, reduction)
        count += 1

    return count
