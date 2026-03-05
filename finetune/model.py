"""
BiRefNet Fine-tuning 모델 래퍼 (어댑터 버전).

원본 BiRefNet의 모든 파라미터를 완전히 freeze 하고,
새로운 학습 가능한 어댑터만 추가하여 학습한다.

    forward(image) → mask        (단일 텐서)
    self.aux_loss  → gradient loss (학습 시 자동 계산)

어댑터 적용 위치:
  1. Backbone (Swin) WindowAttention.qkv, .proj  →  LoRA (저랭크 병렬 경로)
  2. Decoder decoder_block4/3/2/1                →  AdapterConv (잔차 보틀넥)
  3. Decoder conv_out1 (최종 출력)               →  AdapterConv (잔차 보틀넥)

모든 어댑터는 초기 출력이 0으로 설정되어, 적용 직후 원본과 동일하게 동작한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.birefnet import BiRefNet
from adapters import apply_lora_to_backbone, apply_adapters_to_decoder


class FineTuneBiRefNet(nn.Module):
    """
    Args:
        pretrained:      backbone 사전학습 가중치 사용 여부
        lora_rank:       LoRA 랭크 (backbone attention)
        lora_scale:      LoRA 스케일
        adapter_reduction: Decoder 어댑터 보틀넥 축소 비율
    """

    def __init__(
        self,
        pretrained: bool = True,
        lora_rank: int = 4,
        lora_scale: float = 1.0,
        adapter_reduction: int = 4,
    ):
        super().__init__()
        self.model = BiRefNet(bb_pretrained=pretrained)

        # Bi-directional Reference 핵심 설정 확인
        assert self.model.config.ms_supervision, "ms_supervision must be True"
        assert self.model.config.out_ref, "out_ref must be True"
        assert self.model.config.dec_ipt, "dec_ipt must be True"

        # 보조 분류 헤드는 파인튜닝에 불필요
        self.model.config.auxiliary_classification = False

        # ── 1. 원본 전체 freeze ──
        for p in self.model.parameters():
            p.requires_grad = False

        # ── 2. 어댑터 삽입 (새 파라미터만 학습 가능) ──
        n_lora = apply_lora_to_backbone(
            self.model.bb, rank=lora_rank, scale=lora_scale
        )
        n_adapter = apply_adapters_to_decoder(
            self.model.decoder, reduction=adapter_reduction
        )

        # 정보 출력
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"[FineTuneBiRefNet] LoRA 어댑터 {n_lora}개, Decoder 어댑터 {n_adapter}개 적용")
        print(f"[FineTuneBiRefNet] 전체: {total:,} | 학습: {trainable:,} | 프리즈: {frozen:,}")

        # Gradient Loss 저장용
        self.aux_loss = torch.tensor(0.0)
        self._criterion_gdt = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] 입력 이미지
        Returns:
            [B, 1, H, W] 예측 마스크 (logit, sigmoid 전)
        """
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_eval(x)

    def _forward_train(self, x):
        """
        학습 모드.

        BiRefNet.forward() 반환 (out_ref=True, ms_supervision=True):
          [scaled_preds, class_preds_lst]
            scaled_preds = ([gdt_pred_list, gdt_label_list], [m4, m3, m2, p1_out])
            class_preds_lst = [None]
        """
        scaled_preds, _ = self.model(x)
        (gdt_preds, gdt_labels), preds_list = scaled_preds

        # Gradient Reference Loss (Bi-directional 핵심)
        loss_gdt = torch.tensor(0.0, device=x.device)
        for gdt_pred, gdt_label in zip(gdt_preds, gdt_labels):
            gdt_pred = F.interpolate(
                gdt_pred, size=gdt_label.shape[2:],
                mode='bilinear', align_corners=True
            ).sigmoid()
            gdt_label = gdt_label.sigmoid()
            loss_gdt = loss_gdt + self._criterion_gdt(gdt_pred, gdt_label)

        self.aux_loss = loss_gdt

        return preds_list[-1]

    def _forward_eval(self, x):
        """추론 모드."""
        preds = self.model(x)
        return preds[-1]

    def get_adapter_params(self):
        """학습 가능한 어댑터 파라미터만 반환 (옵티마이저용)."""
        return [p for p in self.parameters() if p.requires_grad]

    def save_adapters(self, path: str):
        """어댑터 가중치만 저장 (원본 가중치 제외, 용량 절약)."""
        adapter_state = {
            k: v for k, v in self.state_dict().items()
            if any(kw in k for kw in ['lora_', 'adapter'])
        }
        torch.save(adapter_state, path)
        print(f"어댑터 저장: {path} ({len(adapter_state)} tensors)")

    def load_adapters(self, path: str):
        """저장된 어댑터 가중치를 로드."""
        adapter_state = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(adapter_state, strict=False)
        print(f"어댑터 로드: {path} ({len(adapter_state)} tensors)")
