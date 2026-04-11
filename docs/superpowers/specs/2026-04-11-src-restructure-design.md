# src/ 디렉토리 MECE 재구성 설계

**Date:** 2026-04-11
**Status:** Design approved, pending implementation plan

## 배경

현재 `src/` 구조는 여러 축이 섞여 있어 MECE하지 않다.

- `ai/` 래퍼는 내용 대부분이 AI 코드라 정보량이 없음 (단, 향후 `api/` 추가 예정이라 구분 자체는 유지).
- `ai/modeling/finetune/`에 모델 래퍼(`model.py`), 어댑터(`adapters.py`), 학습 루프(`trainer.py`), 손실(`loss.py`), 스케줄러(`scheduler.py`)가 모두 들어 있어 카테고리 이름과 내용이 불일치.
- `ai/modeling/models/` 이중 중첩.
- `build.py`, `utils/`, `config/`는 `src/` 최상단인데 `inference/`는 `ai/` 아래라 일관성 없음.
- BiRefNet은 외부에서 가져온 코드(수정은 이미 이루어졌지만 격리하고 싶음)이고, LoRA는 자체 구현 — 둘을 같은 레벨에 두는 게 맞음.
- `ai/`와 `api/`는 글자 하나 차이라 시각적으로 혼동되기 쉬움 → `ai/` 대신 `ml/` 사용.

## 목표

1. 책임(data / model / training / inference)을 축으로 MECE 분리.
2. 외부 기원 코드(BiRefNet)와 자체 코드(LoRA, training loop)의 경계 명확화.
3. 향후 `api/` 추가를 염두에 두고 ML 파이프라인 전체를 `ml/` 아래에 모음.
4. 의존 방향 단방향 유지: `training → model/lora → model/birefnet`, `training → data`.

## 최종 구조

```
src/
├── ml/
│   ├── build.py                      # factory: birefnet, lora, dataloader, trainer
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   └── augment/
│   │       ├── crop.py
│   │       └── flip.py
│   ├── model/
│   │   ├── birefnet/                 # 벤더링된 BiRefNet (수정 허용, 격리 유지)
│   │   │   ├── __init__.py           # public API 노출
│   │   │   ├── birefnet.py
│   │   │   ├── backbones/
│   │   │   │   ├── build.py          # ← build_backbone.py
│   │   │   │   └── swin_v1.py
│   │   │   └── blocks/               # ← modules/
│   │   │       ├── aspp.py
│   │   │       ├── decoder.py        # ← decoder_blocks.py
│   │   │       ├── deform_conv.py
│   │   │       └── lateral.py        # ← lateral_blocks.py
│   │   └── lora/
│   │       ├── adapters.py           # LoRALinear, LoRAConv2d
│   │       └── wrapper.py            # ← finetune/model.py (LoRABiRefNet)
│   ├── training/
│   │   ├── trainer.py
│   │   ├── loss.py
│   │   └── scheduler.py
│   └── inference/
│       └── predict.py
├── config/
│   ├── model.yaml
│   └── tune.yaml
└── utils/
    └── io.py
```

## 파일 이동 매핑

| 기존 | 신규 |
|---|---|
| `src/build.py` | `src/ml/build.py` |
| `src/ai/data/**` | `src/ml/data/**` (그대로 이동) |
| `src/ai/inference/predict.py` | `src/ml/inference/predict.py` |
| `src/ai/modeling/models/__init__.py` | `src/ml/model/birefnet/__init__.py` |
| `src/ai/modeling/models/birefnet.py` | `src/ml/model/birefnet/birefnet.py` |
| `src/ai/modeling/models/backbones/build_backbone.py` | `src/ml/model/birefnet/backbones/build.py` |
| `src/ai/modeling/models/backbones/swin_v1.py` | `src/ml/model/birefnet/backbones/swin_v1.py` |
| `src/ai/modeling/models/modules/aspp.py` | `src/ml/model/birefnet/blocks/aspp.py` |
| `src/ai/modeling/models/modules/decoder_blocks.py` | `src/ml/model/birefnet/blocks/decoder.py` |
| `src/ai/modeling/models/modules/deform_conv.py` | `src/ml/model/birefnet/blocks/deform_conv.py` |
| `src/ai/modeling/models/modules/lateral_blocks.py` | `src/ml/model/birefnet/blocks/lateral.py` |
| `src/ai/modeling/finetune/adapters.py` | `src/ml/model/lora/adapters.py` |
| `src/ai/modeling/finetune/model.py` | `src/ml/model/lora/wrapper.py` |
| `src/ai/modeling/finetune/trainer.py` | `src/ml/training/trainer.py` |
| `src/ai/modeling/finetune/loss.py` | `src/ml/training/loss.py` |
| `src/ai/modeling/finetune/scheduler.py` | `src/ml/training/scheduler.py` |
| `src/config/**` | 변경 없음 |
| `src/utils/**` | 변경 없음 |

삭제되는 빈 디렉토리: `src/ai/`, `src/ai/modeling/`, `src/ai/modeling/models/`, `src/ai/modeling/models/modules/`, `src/ai/modeling/finetune/`.

## 의존 방향

단방향 (역방향 금지):

```
ml/training  ──▶  ml/model/lora  ──▶  ml/model/birefnet
     │                                       ▲
     └──────────▶  ml/data                    │
                                              │
ml/inference ─────────────────────────────────┘
ml/build     ─▶ 위 전부
```

- `model/birefnet/`는 `model/lora/`, `training/`, `data/`를 import하지 않음.
- `model/lora/wrapper.py`만 `birefnet`을 import해서 LoRA를 주입.
- `training/`은 `model/*`와 `data/*`를 import하지만 역방향은 없음.
- `build.py`가 최상위 팩토리로 모든 것을 조립.

## 영향 범위 (수정 필요 파일)

- `run_train.py` — import 경로 갱신 (`from src.build import …` 등)
- `tests/test_adapters.py`, `tests/test_loss.py`, `tests/test_model.py`, `tests/test_scheduler.py`, `tests/test_trainer.py` — 전부 import 경로 갱신
- `test_notebooks/02_loss.ipynb` — import 경로 갱신
- `src/ml/build.py` 내부 import 경로 전면 갱신
- `src/ml/model/birefnet/` 내부 상호 import 갱신 (`backbones`, `blocks` 상대 경로)
- `src/ml/model/lora/wrapper.py` 내부 birefnet import 갱신
- `src/ml/training/trainer.py` 내부 model/data import 갱신
- `src/ml/inference/predict.py` 내부 import 갱신
- `CLAUDE.md` 내 경로 레퍼런스 갱신

## 검증 방법

1. `python -m compileall src run_train.py` — 문법/import 에러 없음
2. `pytest tests/ -v` — 전부 통과
3. `python run_train.py`를 몇 스텝만 실행해 실제 학습 루프가 동작하는지 smoke test

## 비목표

- 코드 로직 변경 없음 (순수 이동 + 이름 변경 + import 갱신).
- 새 기능 추가 없음.
- 테스트 자체 수정은 import 경로 변경에 국한.
- `CLAUDE.md`의 "Trainer stale" 상태나 기타 기능적 이슈는 이 작업 범위 밖.
