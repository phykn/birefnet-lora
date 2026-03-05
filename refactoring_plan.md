# BiRefNet Fine-tuning Refactoring Plan (Revised v2)

BiRefNet의 복잡한 학습 구조를 파인튜닝에 맞게 정리하되, 모델의 핵심인 **Bi-directional Reference** 로직은 완전히 보존한다.
외부(트레이너) 관점에서는 **Input: 이미지 → Output: 마스크** 라는 단순한 인터페이스를 유지하며, 내부적으로 필요한 중간 연산(Gradient/경계 참조)은 모두 모델 안에서 처리한다.

---

## 0. 사전 분석: 원본 코드의 핵심 흐름 정리

### 0-1. BiRefNet의 forward 흐름 (학습 모드, `out_ref=True` 기준)
```
BiRefNet.forward(x)
  ├─ forward_enc(x)         ← Backbone으로 multi-scale 특징 추출 (x1, x2, x3, x4)
  │   └─ mul_scl_ipt='cat'  ← 원본 이미지를 축소해서 backbone에 한번 더 넣고 cat
  │   └─ cxt                ← x1,x2,x3를 x4 해상도로 축소해서 concat (context)
  ├─ squeeze_module(x4)     ← ASPP/DecBlk로 x4 압축
  ├─ features = [x, x1, x2, x3, x4, laplacian(x)]
  │                           ↑ laplacian: 원본 이미지의 경계선 맵 (학습 시에만)
  │                           ↑ 이것이 "Bi-directional Reference"의 핵심
  └─ Decoder.forward(features)
      ├─ Pyramid Neck (ViT/DINO 백본일 때)
      ├─ dec_ipt: 원본 이미지 x를 패치로 쪼개서 각 디코더 단계에 주입 ← "Reference"
      ├─ 각 단계에서 gdt_convs로 Gradient 예측/어텐션 계산
      │   └─ gdt_gt(라플라시안) * 중간예측 = Gradient 정답 라벨
      │   └─ gdt_convs_pred = Gradient 예측값
      │   └─ gdt_convs_attn = Gradient 어텐션 (출력에 곱함)
      └─ 반환: ([outs_gdt_pred, outs_gdt_label], [m4, m3, m2, p1_out])
```

### 0-2. 학습 시 Loss 구성 (원본 train.py)
```
1. PixLoss(scaled_preds, gt)     ← BCE + IoU + SSIM (가중합, 모든 scale 출력에 대해)
2. ClsLoss(class_preds, labels)  ← 보조 분류 Loss (파인튜닝에서는 불필요)
3. loss_gdt (out_ref=True일 때)  ← Gradient 예측 vs 정답 BCE Loss
```

### 0-3. Config의 중요 설정값 (기본값 기준)
| 설정 | 기본값 | 의미 | 파인튜닝 시 |
|------|--------|------|-------------|
| `ms_supervision` | `True` | 다중 스케일 중간 출력에 대한 supervision | **True 유지 필수** (`out_ref`가 이에 의존) |
| `out_ref` | `ms_supervision and True` | Gradient 참조 (Bi-directional의 핵심) | **True 유지 필수** |
| `dec_ipt` | `True` | 디코더 각 단계에 원본 이미지 패치 주입 | **True 유지 필수** (Reference의 핵심) |
| `mul_scl_ipt` | `'cat'` | 원본 이미지 축소 후 backbone에 재투입, cat | **유지** |
| `squeeze_block` | `'BasicDecBlk_x1'` | x4 특징 압축 | **유지** |
| `auxiliary_classification` | `False` | 보조 분류 헤드 | False 유지 (불필요) |
| `bb` | `'swin_v1_l'` | 백본 종류 | **유지** |

> ⚠️ **주의**: `ms_supervision=False`로 설정하면 Decoder 초기화 시 `conv_ms_spvn_*`, `gdt_convs_*` 모듈이 **아예 생성되지 않는다**.
> 따라서 `ms_supervision`과 `out_ref`는 반드시 `True`로 유지해야 Bi-directional Reference가 작동한다.

> ⚠️ **주의**: `Config.__init__()`에서 `train.sh` 파일을 파싱하여 `save_last`, `save_step`을 설정한다.
> BiRefNet 디렉토리 밖에서 실행하면 FileNotFoundError가 발생할 수 있으므로, 작업 디렉토리(cwd)를 `BiRefNet/`으로 맞추거나 Config를 패치해야 한다.

---

## 1. Model (`simple_model.py`)

원본 `BiRefNet`을 감싸는 래퍼 클래스. 핵심 원칙:
- **내부**: `out_ref=True`, `ms_supervision=True`, `dec_ipt=True` 등 Bi-directional 로직 **전부 유지**
- **외부**: `forward(x)` → 마스크 텐서 1개 반환 (학습/추론 무관)
- **학습 시 부산물**: Gradient Loss를 `self.aux_loss`에 저장하여 트레이너에서 꺼내 쓸 수 있게 함

### 구현 방향
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.birefnet import BiRefNet


class FineTuneBiRefNet(nn.Module):
    """
    BiRefNet 래퍼.
    - 내부적으로 Bi-directional Reference (Gradient 참조, 원본 이미지 패치 주입) 모두 유지
    - 외부 인터페이스: forward(image) → mask (단일 텐서)
    - 학습 시 중간 Loss는 self.aux_loss 에 저장
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = BiRefNet(bb_pretrained=pretrained)
        
        # ★ Bi-directional Reference 핵심 설정 유지 확인
        assert self.model.config.ms_supervision == True
        assert self.model.config.out_ref == True
        assert self.model.config.dec_ipt == True
        
        # 보조 분류는 파인튜닝에 불필요
        self.model.config.auxiliary_classification = False
        
        # 학습 시 부산물 Loss 저장용
        self.aux_loss = torch.tensor(0.0)
        
    def forward(self, x):
        """
        Args:
            x: 입력 이미지 텐서 [B, 3, H, W]
        Returns:
            최종 예측 마스크 [B, 1, H, W] (logit, sigmoid 전)
        """
        if self.training:
            # ── 학습 모드 ──
            # BiRefNet.forward() 반환: [scaled_preds, class_preds_lst]
            #   scaled_preds (out_ref=True): ([outs_gdt_pred, outs_gdt_label], [m4, m3, m2, p1_out])
            #   class_preds_lst: [None] (auxiliary_classification=False이므로)
            scaled_preds, class_preds_lst = self.model(x)
            
            # out_ref=True이므로 scaled_preds는 튜플
            (outs_gdt_pred, outs_gdt_label), preds_list = scaled_preds
            
            # Gradient Loss 내부 계산 (Bi-directional Reference Loss)
            loss_gdt = torch.tensor(0.0, device=x.device)
            criterion_gdt = nn.BCELoss()
            for gdt_pred, gdt_label in zip(outs_gdt_pred, outs_gdt_label):
                gdt_pred = F.interpolate(
                    gdt_pred, size=gdt_label.shape[2:],
                    mode='bilinear', align_corners=True
                ).sigmoid()
                gdt_label = gdt_label.sigmoid()
                loss_gdt = loss_gdt + criterion_gdt(gdt_pred, gdt_label)
            
            self.aux_loss = loss_gdt
            
            # 최종 마스크만 반환 (리스트의 마지막 = 가장 고해상도 출력 = p1_out)
            return preds_list[-1]
        else:
            # ── 추론 모드 ──
            # BiRefNet.forward() 반환: scaled_preds (리스트)
            #   out_ref=True이지만 추론 시에는 gdt 관련 출력이 없음
            #   → [p1_out] (ms_supervision의 중간출력 m4,m3,m2은 학습 시에만 추가됨)
            preds = self.model(x)
            return preds[-1]
```

> **왜 `ms_supervision`을 끄지 않는가?**
> `ms_supervision=True`여야 Decoder 초기화 시 `conv_ms_spvn_*` 및 `gdt_convs_*` 레이어가 생성된다.
> 이 레이어들이 있어야 Gradient Attention이 작동하므로 Bi-directional Reference의 핵심이 유지된다.
> 다만 중간 출력(m4, m3, m2)에 대한 supervision loss는 `PixLoss`에서 처리하므로, Loss 쪽에서 이를 다루면 된다.

---

## 2. Dataset (`simple_dataset.py`)

원본의 복잡한 데이터 분배, 클래스 라벨, dynamic_size, 다중 데이터셋 결합 등을 모두 제거.
파인튜닝 대상 단일 폴더의 `(이미지, 마스크)` 쌍만 제공한다.

### 구현 방향
```python
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class FineTuneDataset(Dataset):
    """
    단순한 이미지-마스크 쌍 데이터셋.
    image_dir/  와  mask_dir/  아래 파일명이 정렬 기준으로 1:1 대응되어야 한다.
    """
    def __init__(self, image_dir, mask_dir, size=(1024, 1024)):
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])
        assert len(self.image_paths) == len(self.mask_paths), \
            f"이미지({len(self.image_paths)})와 마스크({len(self.mask_paths)}) 개수가 다릅니다"

        # 원본 BiRefNet과 동일한 Normalize 값 사용
        self.img_transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_transform = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),  # 0~255 → 0.0~1.0
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        return self.img_transform(image), self.mask_transform(mask)
```

---

## 3. Loss (`simple_loss.py`)

원본의 `PixLoss`는 모든 scale 출력(m4, m3, m2, p1_out)에 대해 BCE+IoU+SSIM 가중합을 계산한다.
래퍼에서 최종 출력(p1_out) 하나만 반환하므로, 이 출력에 대해서만 Loss를 계산한다.

### 주의사항
- 원본 `PixLoss`는 `pred.sigmoid()` 를 먼저 적용한 뒤 `nn.BCELoss()`를 사용한다 (BCEWithLogitsLoss가 **아님**).
- 파인튜닝에서는 우선 **BCE + IoU** 조합으로 시작하고, 필요 시 SSIM을 추가한다.

### 구현 방향
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    """원본 loss.py의 IoULoss와 동일"""
    def forward(self, pred, target):
        b = pred.shape[0]
        iou_loss = 0.0
        for i in range(b):
            iand = torch.sum(target[i] * pred[i])
            ior = torch.sum(target[i]) + torch.sum(pred[i]) - iand
            iou_loss += (1 - iand / ior)
        return iou_loss


class SegmentationLoss(nn.Module):
    """
    예측 마스크와 정답 마스크 간 Loss.
    원본 PixLoss의 핵심 구성을 유지하되 단일 출력에 대해서만 계산한다.
    """
    def __init__(self, lambda_bce=30.0, lambda_iou=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.iou = IoULoss()
        self.lambda_bce = lambda_bce
        self.lambda_iou = lambda_iou

    def forward(self, pred, target):
        """
        Args:
            pred: 모델 출력 logit [B, 1, H, W]
            target: 정답 마스크 [B, 1, H, W], 값 범위 0~1
        """
        # 해상도 맞추기
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=True)

        # ★ 원본과 동일하게 sigmoid 먼저 적용 후 BCELoss 사용
        pred_sigmoid = pred.sigmoid()
        target = torch.clamp(target, 0, 1)

        loss_bce = self.bce(pred_sigmoid, target) * self.lambda_bce
        loss_iou = self.iou(pred_sigmoid, target) * self.lambda_iou

        return loss_bce + loss_iou
```

---

## 4. Trainer (`simple_train.py`)

위의 3가지 컴포넌트를 조합하는 최소한의 학습 루프.

### 주의사항
- **cwd를 `BiRefNet/` 디렉토리로 설정**해야 `from config import Config` 등 내부 import가 정상 동작한다.
- `Config()`가 `train.sh`를 파싱하려 하므로, `train.sh`가 없으면 `save_last`/`save_step` 관련 에러가 난다. → 별도 처리 필요.
- `model.aux_loss` (Gradient Reference Loss)는 `total_loss`에 합산해야 Bi-directional Reference가 학습된다.

### 구현 방향
```python
import os
import sys

# ★ BiRefNet 내부 import가 정상 동작하도록 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BiRefNet'))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from simple_model import FineTuneBiRefNet
from simple_dataset import FineTuneDataset
from simple_loss import SegmentationLoss


def train(
    image_dir: str,
    mask_dir: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    img_size: tuple = (1024, 1024),
    save_dir: str = "ckpts",
    freeze_backbone: bool = False,
    lambda_gdt: float = 1.0,       # Gradient Reference Loss 가중치
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Dataset ──
    dataset = FineTuneDataset(image_dir, mask_dir, size=img_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=min(4, batch_size), pin_memory=True, drop_last=True,
    )

    # ── 2. Model ──
    model = FineTuneBiRefNet(pretrained=True).to(device)

    if freeze_backbone:
        for param in model.model.bb.parameters():
            param.requires_grad = False
        print("Backbone frozen.")

    # ── 3. Loss & Optimizer ──
    criterion = SegmentationLoss(lambda_bce=30.0, lambda_iou=0.5)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-2,
    )

    # ── 4. Train Loop ──
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward: 이미지 → 마스크 (내부에서 Bi-directional Reference 처리)
            preds = model(images)

            # Loss 1: 예측 마스크 vs 정답 마스크
            loss_seg = criterion(preds, masks)

            # Loss 2: Bi-directional Reference (Gradient) Loss (모델 내부에서 계산됨)
            loss_gdt = model.aux_loss * lambda_gdt

            # Total
            total_loss = loss_seg + loss_gdt

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix({
                "seg": f"{loss_seg.item():.4f}",
                "gdt": f"{loss_gdt.item():.4f}",
                "total": f"{total_loss.item():.4f}",
            })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} — Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pth"))


if __name__ == "__main__":
    train(
        image_dir="data/images",
        mask_dir="data/masks",
        epochs=10,
        batch_size=4,
        lr=1e-4,
        img_size=(1024, 1024),
    )
```

---

## 5. 파일 구조 (최종)
```
260305_birefnet/
├── BiRefNet/                    ← 원본 레포 (수정하지 않음)
│   ├── config.py
│   ├── dataset.py
│   ├── loss.py
│   ├── models/
│   │   ├── birefnet.py
│   │   ├── backbones/
│   │   └── modules/
│   ├── train.py
│   └── ...
│
├── simple_model.py              ← 1. 모델 래퍼 (FineTuneBiRefNet)
├── simple_dataset.py            ← 2. 데이터셋 (FineTuneDataset)
├── simple_loss.py               ← 3. 손실 함수 (SegmentationLoss)
├── simple_train.py              ← 4. 트레이너 (train loop)
│
├── data/                        ← 파인튜닝 데이터 (사용자 준비)
│   ├── images/
│   └── masks/
├── ckpts/                       ← 체크포인트 저장
└── refactoring_plan.md          ← 이 문서
```

---

## 6. 실행 시 주의사항 체크리스트

- [ ] **cwd 또는 sys.path**: `BiRefNet/` 디렉토리가 Python 경로에 포함되어야 내부 import(`from config import Config` 등)가 동작한다.
- [ ] **`train.sh` 파일 의존성**: `Config.__init__()`이 `train.sh`를 파싱하므로, `BiRefNet/` 디렉토리에 해당 파일이 존재해야 한다. 없으면 `save_last`/`save_step` 관련 에러 발생 → Config 패치 또는 `train.sh` 존재 보장이 필요하다.
- [ ] **pretrained 가중치**: `BiRefNet(bb_pretrained=True)` 시, `Config.weights`에 지정된 경로(`/workspace/weights/cv/...`)에 사전학습 가중치가 존재해야 한다. HuggingFace에서 다운로드하거나 경로를 수정해야 할 수 있다.
- [ ] **GPU 메모리**: swin_v1_l + 1024×1024 + batch_size=4 기준 약 40~60GB VRAM이 필요하다. 메모리가 부족하면 `img_size`를 줄이거나 `batch_size`를 낮춘다.
- [ ] **`ms_supervision` / `out_ref` / `dec_ipt` 설정은 절대 False로 바꾸지 않는다** — Bi-directional Reference의 핵심이다.
