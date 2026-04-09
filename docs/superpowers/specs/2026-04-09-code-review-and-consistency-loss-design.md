# Code Review & Consistency Loss Design

## 1. Code Review Findings

### 1.1 Code Quality / Structure

#### Config 이중 경로 정리

`train.py:normalize_config()`이 `dl`/`trainer` 섹션을 `train`으로 병합하지만, `build.py`는 여전히 `cfg.dl.batch`, `cfg.trainer.lr`을 직접 접근한다. 두 경로가 공존하면 어떤 값이 실제로 사용되는지 추적하기 어렵다.

**변경:** `build.py`의 모든 config 접근을 `cfg.train.*`으로 통일. `normalize_config()`를 `train.py`에서 `build.py`나 config 유틸로 이동하여 config 로딩 시점에 정규화.

#### 불필요한 wrapper 제거

`build_dataloaders()`(`build.py:94-95`)는 `build_dl()`을 감싸기만 한다. 하나로 합친다.

#### 깨진 테스트 수정

`test_build.py`가 `match_image_and_mask_paths`를 import하지만 이 함수는 `build.py`에 존재하지 않는다. 테스트가 현재 코드와 맞지 않으므로 수정하거나, 해당 함수를 `build.py`에 추출한다.

#### conftest.py sys.path 조작 제거

`pip install -e .`로 설치하면 불필요. 제거한다.

### 1.2 Training Pipeline

#### LR Scheduler 추가

`CosineAnnealingLR` 또는 linear warmup + cosine decay. `tune.yaml`에 scheduler 설정 추가.

```yaml
trainer:
  lr: 0.0001
  scheduler: cosine  # "cosine" | "none"
  warmup_steps: 50
  steps: 1000
```

#### Gradient Clipping

LoRA + AMP 조합에서 gradient explosion 방지. `max_grad_norm` 설정 추가.

```yaml
trainer:
  max_grad_norm: 1.0
```

`scaler.unscale_(optimizer)` 후 `torch.nn.utils.clip_grad_norm_()` 적용.

#### Best Model 저장

Validation loss 기준 best adapter를 `weights/best.pth`로 별도 저장. Trainer에 `best_val_loss` 상태 추가.

#### Resume 지원

Checkpoint에 optimizer state, scaler state, current step을 함께 저장. `tune.yaml`에 `resume_from` 경로를 지정하면 해당 시점부터 재개.

### 1.3 Inference

`predict.py` 추가: LoRA adapter 로드 → 이미지 입력 → 마스크 출력. 단일 이미지 또는 디렉토리 batch 처리.

---

## 2. Consistency Loss

### 2.1 배경

배터리 공극률 인식에서 조명/대비 변화에 따라 예측이 달라지는 문제를 해결한다. 동일 이미지에 서로 다른 augmentation을 적용한 두 뷰의 예측이 일관되도록 regularization을 추가한다.

### 2.2 설계

#### Augmentation 전략

하나의 이미지에 대해 두 개의 뷰를 생성한다:
- **기하학적 변환**: 두 뷰에 **동일하게** 적용 (D4: flip + rotation)
- **색상 변환**: 두 뷰에 **독립적으로** 적용 (RandomBrightnessContrast)

이렇게 하면 두 뷰의 spatial layout이 동일하므로 prediction을 직접 비교할 수 있다.

#### Dataset 변경

`TrainDataset.__getitem__`이 기존 `{"image": tensor, "mask": tensor}` 대신 `{"image_v1": tensor, "image_v2": tensor, "mask": tensor}`를 반환한다.

구현:
1. 이미지와 마스크를 로드
2. 기하학적 augmentation을 적용 (D4) → 두 뷰에 공유
3. 색상 augmentation을 독립적으로 두 번 적용 → view1, view2
4. 각각 ImageNet normalize 적용

albumentations 파이프라인을 2단계로 분리한다:
- `geo_transform`: `A.Compose([A.Resize, A.D4])` — image+mask 동시 적용
- `color_transform`: `A.Compose([A.RandomBrightnessContrast(p=1.0)])` — image에만 적용

#### Loss 계산

```python
# Forward
pred1, aux1 = model(images_v1)   # logits
pred2, aux2 = model(images_v2)   # logits

# Supervised loss (양쪽 뷰 모두)
seg_loss1 = criterion(pred1, masks)
seg_loss2 = criterion(pred2, masks)

# Consistency loss (logits 공간에서 MSE)
cons_loss = F.mse_loss(pred1, pred2)

# Total
total = seg_loss1 + seg_loss2 + aux1 + aux2 + lambda_cons * cons_loss
```

Logits MSE를 사용하는 이유:
- Sigmoid 후에는 0/1 근처에서 gradient 포화 발생
- Logits 공간에서는 모든 영역에서 균등하게 일관성을 강제
- 경계 영역(공극/비공극 경계)에서의 일관성이 특히 중요한데, logits MSE가 이를 더 잘 포착

#### Config 추가

```yaml
trainer:
  lambda_cons: 0.1
```

`lambda_cons: 0.0`이면 consistency loss 비활성화 (기존 동작과 동일).

#### Trainer 변경

`Trainer._step()`에서:
- batch에서 `image_v1`, `image_v2`, `mask` 추출
- 두 번 forward → seg_loss 2개 + consistency_loss 1개
- Validation은 단일 뷰로 기존과 동일 (consistency는 학습 시에만)

#### LoRABiRefNet 변경

`_train_step()`이 현재 `(prediction, aux_loss)` 튜플을 반환한다. Consistency loss 계산을 위해 **sigmoid 이전의 logits**를 반환해야 하므로, 반환 형태는 동일하되 `predictions[-1]`이 이미 logits임을 확인 — 현재 코드에서 `predictions[-1]`은 `conv_out1`의 출력이므로 이미 logits. 변경 불필요.

### 2.3 Validation

Validation에서는 consistency loss를 계산하지 않는다. 단일 뷰(색상 augmentation 없음)로 seg_loss만 측정하여 기존 validation과 비교 가능하게 유지.

### 2.4 TensorBoard Logging

추가 metric:
- `Train/Cons`: consistency loss
- `Train/Seg1`, `Train/Seg2`: 각 뷰의 segmentation loss (optional, 디버깅용)

---

## 3. 변경 대상 파일 요약

| 파일 | 변경 내용 |
|------|-----------|
| `src/config/tune.yaml` | scheduler, warmup_steps, max_grad_norm, lambda_cons 추가 |
| `src/data/dataset.py` | `TrainDataset` augmentation 2단계 분리, 두 뷰 반환 |
| `src/finetune/loss.py` | `ConsistencyLoss` 클래스 추가 |
| `src/finetune/trainer.py` | dual-view 학습 루프, scheduler, grad clipping, best model 저장 |
| `src/build.py` | config 접근 통일, scheduler 빌드, wrapper 제거 |
| `train.py` | normalize_config 정리, resume 지원 |
| `predict.py` | 신규: inference 파이프라인 |
| `tests/test_build.py` | 깨진 테스트 수정 |
| `tests/conftest.py` | sys.path 조작 제거 |
