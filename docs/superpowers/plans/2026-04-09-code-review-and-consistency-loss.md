# Code Review & Consistency Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up code quality issues, strengthen the training pipeline (scheduler, grad clipping, best model, resume), add consistency regularization loss for robustness to brightness/contrast variations, and add an inference script.

**Architecture:** The training loop gains dual-view forward passes with a logits-space MSE consistency loss. Dataset returns two color-augmented views sharing the same geometric transform. Training infra gets cosine LR scheduler, gradient clipping, best-model checkpointing, and resume support. A standalone predict.py handles inference.

**Tech Stack:** Python 3.12, PyTorch, albumentations, torchvision, tqdm, tensorboard

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `tests/conftest.py` | Test configuration | Modify: remove sys.path hack |
| `tests/test_build.py` | Build function tests | Modify: fix broken import, test `build_pairs` |
| `src/build.py` | Factory functions | Modify: extract `build_pairs`, remove wrapper, unify config access, add scheduler build |
| `src/config/tune.yaml` | Runtime config | Modify: add scheduler, grad clip, consistency loss settings |
| `train.py` | Entry point | Modify: move `normalize_config` to build, add resume support |
| `src/data/dataset.py` | Dataset classes | Modify: split TrainDataset augmentation into geo+color, return dual views |
| `tests/test_dataset.py` | Dataset tests | Create: test dual-view output |
| `src/finetune/loss.py` | Loss functions | Modify: add `ConsistencyLoss` |
| `tests/test_loss.py` | Loss tests | Create: test consistency loss |
| `src/finetune/trainer.py` | Training loop | Modify: dual-view step, scheduler, grad clip, best model, resume |
| `tests/test_trainer.py` | Trainer tests | Create: test step dict keys, best model tracking |
| `predict.py` | Inference entry point | Create: load adapter, predict, save masks |

---

### Task 1: Fix conftest.py and broken test_build.py

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_build.py`
- Modify: `src/build.py` (extract `build_pairs` from `build_dl`)

- [ ] **Step 1: Clear conftest.py**

Replace `tests/conftest.py` contents with an empty file (the sys.path hack is unnecessary with `pip install -e .`):

```python
```

(Empty file — pytest still needs it to exist for package discovery.)

- [ ] **Step 2: Extract `build_pairs` in `src/build.py`**

Extract the image/mask pairing logic from `build_dl` into a standalone function. Add it right before `build_dl`:

```python
def build_pairs(
    image_paths: list[str],
    mask_paths: list[str],
) -> list[tuple[str, str]]:
    image_map: dict[str, str] = {}
    for p in image_paths:
        stem = Path(p).stem
        if stem in image_map:
            raise ValueError(f"Duplicate image stem detected: {stem}")
        image_map[stem] = p

    mask_map: dict[str, str] = {}
    for p in mask_paths:
        stem = Path(p).stem
        if stem in mask_map:
            raise ValueError(f"Duplicate mask stem detected: {stem}")
        mask_map[stem] = p

    if set(image_map) != set(mask_map):
        raise ValueError(
            "Image/mask filename mismatch: "
            f"image_only={sorted(set(image_map) - set(mask_map))[:5]}, "
            f"mask_only={sorted(set(mask_map) - set(image_map))[:5]}"
        )

    common_stems = sorted(image_map)
    return [(image_map[s], mask_map[s]) for s in common_stems]
```

Then update `build_dl` to call `build_pairs` instead of duplicating the logic:

```python
def build_dl(cfg: Any) -> tuple[DataLoader, DataLoader, dict[str, list[str]]]:
    image_dir = Path(cfg.data.img_dir)
    mask_dir = Path(cfg.data.mask_dir)

    normalized_exts = {ext.lower() for ext in FineTuneDataset.EXTS}
    image_files = sorted(
        str(p) for p in image_dir.glob("*")
        if p.is_file() and p.suffix.lower() in normalized_exts
    )
    mask_files = sorted(
        str(p) for p in mask_dir.glob("*")
        if p.is_file() and p.suffix.lower() in normalized_exts
    )

    paired_paths = build_pairs(image_files, mask_files)

    rng = random.Random(42)
    rng.shuffle(paired_paths)

    split_ratio = float(cfg.data.split_ratio)
    num_valid = int(len(paired_paths) * split_ratio)
    valid_pairs = paired_paths[:num_valid]
    train_pairs = paired_paths[num_valid:]

    train_images, train_masks = zip(*train_pairs) if train_pairs else ([], [])
    valid_images, valid_masks = zip(*valid_pairs) if valid_pairs else ([], [])

    train_dataset = TrainDataset(
        img_paths=list(train_images),
        mask_paths=list(train_masks),
        size=cfg.data.size,
    )
    valid_dataset = ValidDataset(
        img_paths=list(valid_images),
        mask_paths=list(valid_masks),
        size=cfg.data.size,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.train.batch,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )

    split_filenames = {
        "train": [Path(path).name for path in train_images],
        "valid": [Path(path).name for path in valid_images],
    }

    return train_loader, valid_loader, split_filenames
```

Also remove the `build_dataloaders` wrapper. Rename `build_dl` to `build_dataloaders`:

```python
def build_dataloaders(cfg: Any) -> tuple[DataLoader, DataLoader, dict[str, list[str]]]:
    # ... (the full body from build_dl above)
```

- [ ] **Step 3: Update test_build.py**

```python
import pytest

from src.build import build_pairs


def test_build_pairs_returns_sorted_pairs() -> None:
    image_paths = ["/tmp/data/image/b.png", "/tmp/data/image/a.png"]
    mask_paths = ["/tmp/data/mask/a.png", "/tmp/data/mask/b.png"]

    pairs = build_pairs(image_paths, mask_paths)

    assert pairs == [
        ("/tmp/data/image/a.png", "/tmp/data/mask/a.png"),
        ("/tmp/data/image/b.png", "/tmp/data/mask/b.png"),
    ]


def test_build_pairs_raises_on_filename_mismatch() -> None:
    image_paths = ["/tmp/data/image/a.png", "/tmp/data/image/b.png"]
    mask_paths = ["/tmp/data/mask/a.png"]

    with pytest.raises(ValueError, match="Image/mask filename mismatch"):
        build_pairs(image_paths, mask_paths)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_build.py tests/test_misc.py tests/test_io.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_build.py src/build.py
git commit -m "refactor: extract build_pairs, fix broken test, clean conftest"
```

---

### Task 2: Unify config access and move normalize_config

**Files:**
- Modify: `src/config/tune.yaml`
- Modify: `train.py`
- Modify: `src/build.py`

- [ ] **Step 1: Update tune.yaml**

Replace the entire file with the new flat `train` section plus new fields:

```yaml
data:
  img_dir: "data/image"
  mask_dir: "data/mask"
  size: [1024, 1024]
  split_ratio: 0.1

birefnet:
  weight: "weight/BiRefNet-general-epoch_244.pth"
  lateral_channels_in_collection: [1536, 768, 384, 192]
  mul_scl_ipt: "cat"
  dec_ipt: True
  dec_ipt_split: True
  ms_supervision: True
  out_ref: True

lora:
  rank: 8
  alpha: 16.0

train:
  batch: 4
  num_workers: 0
  pin_memory: false
  lr: 0.0001
  scheduler: "cosine"
  warmup_steps: 50
  max_grad_norm: 1.0
  steps: 1000
  val_freq: 50
  save_freq: 100
  lambda_cons: 0.1
```

- [ ] **Step 2: Remove normalize_config and dl/trainer references from train.py**

Replace `train.py` with:

```python
import csv
import json
import os

import torch

from src.build import build_birefnet, build_dataloaders, build_lora_birefnet, build_trainer
from src.utils.io import load_yaml, save_yaml
from src.utils.misc import ConfigDict


def save_split_csv(split_filenames: dict[str, list[str]], output_dir: str) -> None:
    for split_name in ["train", "valid"]:
        csv_path = f"{output_dir}/{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(["filename"])
            for filename in split_filenames.get(split_name, []):
                writer.writerow([filename])


def print_run_info(
    cfg: ConfigDict,
    split_filenames: dict[str, list[str]],
    model: torch.nn.Module,
) -> None:
    train_count = len(split_filenames.get("train", []))
    valid_count = len(split_filenames.get("valid", []))

    print("[Config]")
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    print(f"[Dataset] train={train_count}, valid={valid_count}")
    if hasattr(model, "stats"):
        total_params = model.stats.get("total", 0)
        trainable_params = model.stats.get("trainable", 0)
        ratio = (trainable_params / total_params) if total_params else 0.0
        print(
            "[LoRABiRefNet] "
            f"total={total_params:,}  "
            f"trainable={trainable_params:,}  "
            f"ratio={ratio:.2%}"
        )


def train() -> None:
    config_data = load_yaml(path="src/config/tune.yaml")
    cfg = ConfigDict(config_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = build_birefnet(cfg=cfg)
    lora_ckpt_path = cfg.get("lora", {}).get("ckpt", None)
    model = build_lora_birefnet(cfg=cfg, model=base_model, ckpt_path=lora_ckpt_path).to(device)
    train_loader, valid_loader, split_filenames = build_dataloaders(cfg=cfg)

    trainer = build_trainer(
        cfg=cfg,
        model=model,
        train_dl=train_loader,
        valid_dl=valid_loader,
    )

    print_run_info(cfg=cfg, split_filenames=split_filenames, model=model)

    save_yaml(data=cfg.to_dict(), path=f"{trainer.save_dir}/config.yaml")
    save_split_csv(split_filenames=split_filenames, output_dir=trainer.save_dir)

    trainer.train(
        steps=cfg.train.steps,
        val_freq=cfg.train.val_freq,
        save_freq=cfg.train.save_freq,
    )

    if hasattr(model, "save_adapters"):
        weights_dir = f"{trainer.save_dir}/weights"
        os.makedirs(weights_dir, exist_ok=True)
        model.save_adapters(f"{weights_dir}/model.pth")


if __name__ == "__main__":
    train()
```

- [ ] **Step 3: Update build.py config access**

In `build_dataloaders` (the renamed `build_dl`), DataLoader construction already uses `cfg.train.*` from Step 2 of Task 1. Now update `build_trainer`:

```python
def build_trainer(
    cfg: Any,
    model: torch.nn.Module,
    train_dl: DataLoader,
    valid_dl: DataLoader,
) -> Trainer:
    device = next(model.parameters()).device
    criterion = SegmentationLoss()
    optimizer = torch.optim.AdamW(params=model.get_adapter_params(), lr=cfg.train.lr)

    return Trainer(
        model=model,
        train_loader=train_dl,
        valid_loader=valid_dl,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )
```

- [ ] **Step 4: Verify syntax**

Run: `.venv/bin/python -m compileall src train.py`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add src/config/tune.yaml train.py src/build.py
git commit -m "refactor: unify config access under train section, remove normalize_config"
```

---

### Task 3: Dual-view TrainDataset

**Files:**
- Modify: `src/data/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_dataset.py`:

```python
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import TrainDataset, ValidDataset


@pytest.fixture()
def sample_data(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    for name in ["a.png", "b.png"]:
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(img_dir / name)
        mask = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
        mask.save(mask_dir / name)

    img_paths = sorted(str(p) for p in img_dir.glob("*.png"))
    mask_paths = sorted(str(p) for p in mask_dir.glob("*.png"))
    return img_paths, mask_paths


def test_train_dataset_returns_dual_views(sample_data) -> None:
    img_paths, mask_paths = sample_data
    ds = TrainDataset(img_paths=img_paths, mask_paths=mask_paths, size=32)
    item = ds[0]

    assert "image_v1" in item
    assert "image_v2" in item
    assert "mask" in item
    assert item["image_v1"].shape == (3, 32, 32)
    assert item["image_v2"].shape == (3, 32, 32)
    assert item["mask"].shape == (1, 32, 32)


def test_train_dataset_dual_views_share_geometry(sample_data) -> None:
    img_paths, mask_paths = sample_data
    ds = TrainDataset(img_paths=img_paths, mask_paths=mask_paths, size=32)
    item = ds[0]

    # Mask is identical for both views (same geometric transform)
    # Views differ only in color — mask is unaffected by color aug
    assert item["mask"].shape == (1, 32, 32)


def test_train_dataset_dual_views_differ_in_color(sample_data) -> None:
    img_paths, mask_paths = sample_data
    ds = TrainDataset(img_paths=img_paths, mask_paths=mask_paths, size=32)

    # Run multiple times — at least one should produce different views
    any_different = False
    for i in range(len(img_paths)):
        item = ds[i]
        if not torch.equal(item["image_v1"], item["image_v2"]):
            any_different = True
            break

    assert any_different, "Dual views should sometimes differ in color"


def test_valid_dataset_returns_single_view(sample_data) -> None:
    img_paths, mask_paths = sample_data
    ds = ValidDataset(img_paths=img_paths, mask_paths=mask_paths, size=32)
    item = ds[0]

    assert "image" in item
    assert "mask" in item
    assert item["image"].shape == (3, 32, 32)
    assert item["mask"].shape == (1, 32, 32)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `image_v1` key not in item

- [ ] **Step 3: Implement dual-view TrainDataset**

Replace `src/data/dataset.py`:

```python
import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FineTuneDataset(Dataset):
    EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    def __init__(
        self,
        img_paths: list[str],
        mask_paths: list[str],
        size: int | tuple[int, int],
    ) -> None:
        if len(img_paths) != len(mask_paths):
            raise ValueError(
                "Image and mask path counts must match. "
                f"Got {len(img_paths)} images and {len(mask_paths)} masks."
            )

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        return image, mask

    def _to_tensors(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = TF.normalize(TF.to_tensor(image), mean=MEAN, std=STD)
        mask_tensor = TF.to_tensor(mask)
        return image_tensor, mask_tensor


class ValidDataset(FineTuneDataset):
    def __init__(
        self,
        img_paths: list[str],
        mask_paths: list[str],
        size: int | tuple[int, int],
    ) -> None:
        super().__init__(img_paths=img_paths, mask_paths=mask_paths, size=size)
        height, width = self.size
        self.transform = A.Compose([A.Resize(height=height, width=width)])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, mask = self._load(idx)
        transformed = self.transform(image=image, mask=mask)
        image_tensor, mask_tensor = self._to_tensors(transformed["image"], transformed["mask"])
        return {"image": image_tensor, "mask": mask_tensor}


class TrainDataset(FineTuneDataset):
    def __init__(
        self,
        img_paths: list[str],
        mask_paths: list[str],
        size: int | tuple[int, int],
    ) -> None:
        super().__init__(img_paths=img_paths, mask_paths=mask_paths, size=size)
        height, width = self.size
        self.geo_transform = A.Compose([
            A.Resize(height=height, width=width),
            A.D4(),
        ])
        self.color_transform = A.Compose([
            A.RandomBrightnessContrast(p=1.0),
        ])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, mask = self._load(idx)

        geo = self.geo_transform(image=image, mask=mask)
        geo_image = geo["image"]
        geo_mask = geo["mask"]

        view1 = self.color_transform(image=geo_image)["image"]
        view2 = self.color_transform(image=geo_image)["image"]

        v1_tensor, mask_tensor = self._to_tensors(view1, geo_mask)
        v2_tensor, _ = self._to_tensors(view2, geo_mask)

        return {"image_v1": v1_tensor, "image_v2": v2_tensor, "mask": mask_tensor}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py tests/test_dataset.py
git commit -m "feat: dual-view TrainDataset with shared geometry, independent color aug"
```

---

### Task 4: ConsistencyLoss

**Files:**
- Modify: `src/finetune/loss.py`
- Create: `tests/test_loss.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_loss.py`:

```python
import torch

from src.finetune.loss import ConsistencyLoss, SegmentationLoss


def test_consistency_loss_zero_for_identical_logits() -> None:
    loss_fn = ConsistencyLoss()
    logits = torch.randn(2, 1, 32, 32)
    loss = loss_fn(logits, logits)
    assert loss.item() == 0.0


def test_consistency_loss_positive_for_different_logits() -> None:
    loss_fn = ConsistencyLoss()
    logits1 = torch.randn(2, 1, 32, 32)
    logits2 = torch.randn(2, 1, 32, 32)
    loss = loss_fn(logits1, logits2)
    assert loss.item() > 0.0


def test_consistency_loss_interpolates_mismatched_sizes() -> None:
    loss_fn = ConsistencyLoss()
    logits1 = torch.randn(2, 1, 64, 64)
    logits2 = torch.randn(2, 1, 32, 32)
    loss = loss_fn(logits1, logits2)
    assert loss.item() > 0.0


def test_segmentation_loss_returns_scalar() -> None:
    loss_fn = SegmentationLoss()
    pred = torch.randn(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    loss = loss_fn(pred, target)
    assert loss.dim() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_loss.py -v`
Expected: FAIL — `ConsistencyLoss` not found

- [ ] **Step 3: Implement ConsistencyLoss**

Add to the end of `src/finetune/loss.py`:

```python
class ConsistencyLoss(nn.Module):
    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        if logits1.shape[2:] != logits2.shape[2:]:
            logits2 = F.interpolate(
                logits2,
                size=logits1.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        return F.mse_loss(logits1, logits2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loss.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/finetune/loss.py tests/test_loss.py
git commit -m "feat: add ConsistencyLoss (logits-space MSE)"
```

---

### Task 5: Trainer — dual-view step, scheduler, grad clipping, best model

**Files:**
- Modify: `src/finetune/trainer.py`
- Modify: `src/build.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_trainer.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.finetune.loss import ConsistencyLoss, SegmentationLoss
from src.finetune.trainer import Trainer


def _make_dummy_model():
    """Simple model mimicking LoRABiRefNet interface."""
    model = nn.Conv2d(3, 1, 1)

    def train_forward(x):
        pred = model(x)
        aux = torch.tensor(0.0, device=x.device)
        return pred, aux

    def eval_forward(x):
        return model(x)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = model

        def forward(self, x):
            if self.training:
                return train_forward(x)
            return eval_forward(x)

        def get_adapter_params(self):
            return list(self.parameters())

        def save_adapters(self, path):
            torch.save(self.state_dict(), path)

    return DummyModel()


def _make_dual_view_loader(batch_size=2, num_samples=4):
    v1 = torch.randn(num_samples, 3, 32, 32)
    v2 = torch.randn(num_samples, 3, 32, 32)
    masks = torch.rand(num_samples, 1, 32, 32)
    ds = TensorDataset(v1, v2, masks)

    class DualViewDataset:
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            v1, v2, m = self.ds[idx]
            return {"image_v1": v1, "image_v2": v2, "mask": m}

    return DataLoader(DualViewDataset(ds), batch_size=batch_size, drop_last=True)


def _make_single_view_loader(batch_size=2, num_samples=4):
    images = torch.randn(num_samples, 3, 32, 32)
    masks = torch.rand(num_samples, 1, 32, 32)
    ds = TensorDataset(images, masks)

    class SingleViewDataset:
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, m = self.ds[idx]
            return {"image": img, "mask": m}

    return DataLoader(SingleViewDataset(ds), batch_size=batch_size)


def test_trainer_step_returns_expected_keys():
    model = _make_dummy_model()
    train_loader = _make_dual_view_loader()
    valid_loader = _make_single_view_loader()
    criterion = SegmentationLoss()
    cons_criterion = ConsistencyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        cons_criterion=cons_criterion,
        optimizer=optimizer,
        device=device,
        lambda_cons=0.1,
        use_tensorboard=False,
    )
    losses = trainer._step()

    assert "loss" in losses
    assert "seg" in losses
    assert "aux" in losses
    assert "cons" in losses


def test_trainer_tracks_best_val_loss():
    model = _make_dummy_model()
    train_loader = _make_dual_view_loader()
    valid_loader = _make_single_view_loader()
    criterion = SegmentationLoss()
    cons_criterion = ConsistencyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        cons_criterion=cons_criterion,
        optimizer=optimizer,
        device=device,
        lambda_cons=0.1,
        use_tensorboard=False,
    )

    assert trainer.best_val_loss == float("inf")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_trainer.py -v`
Expected: FAIL — Trainer doesn't accept `cons_criterion`/`lambda_cons`

- [ ] **Step 3: Implement updated Trainer**

Replace `src/finetune/trainer.py`:

```python
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module,
        cons_criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lambda_cons: float = 0.1,
        max_grad_norm: float = 1.0,
        scheduler_name: str = "cosine",
        warmup_steps: int = 50,
        total_steps: int = 1000,
        use_tensorboard: bool = True,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.cons_criterion = cons_criterion
        self.optimizer = optimizer
        self.device = device
        self.lambda_cons = lambda_cons
        self.max_grad_norm = max_grad_norm

        self.amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=self.use_amp)

        self.scheduler = self._build_scheduler(scheduler_name, warmup_steps, total_steps)

        self.best_val_loss = float("inf")

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join("run", run_timestamp)
        os.makedirs(self.save_dir, exist_ok=True)

        log_dir = os.path.join(self.save_dir, "logs")
        self.writer = SummaryWriter(log_dir=log_dir) if use_tensorboard else None
        self.train_iter = iter(self.train_loader)

    def _build_scheduler(
        self,
        name: str,
        warmup_steps: int,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if name == "none":
            return None

        warmup = LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(warmup_steps, 1)))
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(total_steps - warmup_steps, 1))
        return SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    def train(self, steps: int, val_freq: int = 500, save_freq: int = 1000) -> None:
        progress_bar = tqdm(range(1, steps + 1), desc="Training")

        for step in progress_bar:
            losses = self._step()
            progress_bar.set_postfix(
                loss=f"{losses['loss']:.4f}",
                seg=f"{losses['seg']:.4f}",
                cons=f"{losses['cons']:.4f}",
            )

            if self.writer:
                self.writer.add_scalar("Train/Loss", losses["loss"], step)
                self.writer.add_scalar("Train/Seg", losses["seg"], step)
                self.writer.add_scalar("Train/Aux", losses["aux"], step)
                self.writer.add_scalar("Train/Cons", losses["cons"], step)
                if self.scheduler:
                    self.writer.add_scalar("Train/LR", self.scheduler.get_last_lr()[0], step)

            if step % val_freq == 0:
                valid_loss = self._validate()
                if self.writer:
                    self.writer.add_scalar("Val/Loss", valid_loss, step)
                if valid_loss < self.best_val_loss:
                    self.best_val_loss = valid_loss
                    self._save(filename="best.pth")

            if step % save_freq == 0:
                self._save(filename="last.pth")

        if self.writer:
            self.writer.flush()
            self.writer.close()

    def _step(self) -> dict[str, float]:
        self.model.train()
        batch = self._get_batch()
        images_v1 = batch["image_v1"].to(self.device)
        images_v2 = batch["image_v2"].to(self.device)
        masks = batch["mask"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.amp_device_type, enabled=self.use_amp):
            pred1, aux1 = self.model(images_v1)
            pred2, aux2 = self.model(images_v2)

            seg_loss1 = self.criterion(pred1, masks)
            seg_loss2 = self.criterion(pred2, masks)
            seg_loss = seg_loss1 + seg_loss2

            cons_loss = self.cons_criterion(pred1, pred2) * self.lambda_cons
            aux_loss = aux1 + aux2

            total_loss = seg_loss + aux_loss + cons_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler:
            self.scheduler.step()

        return {
            "loss": total_loss.item(),
            "seg": seg_loss.item(),
            "aux": aux_loss.item(),
            "cons": cons_loss.item(),
        }

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in self.valid_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            with torch.amp.autocast(self.amp_device_type, enabled=self.use_amp):
                prediction = self.model(images)
                loss = self.criterion(prediction, masks)

            total_loss += loss.item()

        num_batches = len(self.valid_loader)
        return total_loss / num_batches if num_batches > 0 else 0.0

    def _save(self, filename: str = "last.pth") -> None:
        if hasattr(self.model, "save_adapters"):
            weights_dir = os.path.join(self.save_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            path = os.path.join(weights_dir, filename)
            self.model.save_adapters(path)

    def _get_batch(self) -> dict[str, torch.Tensor]:
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)
```

- [ ] **Step 4: Update build_trainer in src/build.py**

Replace the `build_trainer` function:

```python
def build_trainer(
    cfg: Any,
    model: torch.nn.Module,
    train_dl: DataLoader,
    valid_dl: DataLoader,
) -> Trainer:
    from .finetune.loss import ConsistencyLoss

    device = next(model.parameters()).device
    criterion = SegmentationLoss()
    cons_criterion = ConsistencyLoss()
    optimizer = torch.optim.AdamW(params=model.get_adapter_params(), lr=cfg.train.lr)

    return Trainer(
        model=model,
        train_loader=train_dl,
        valid_loader=valid_dl,
        criterion=criterion,
        cons_criterion=cons_criterion,
        optimizer=optimizer,
        device=device,
        lambda_cons=cfg.train.lambda_cons,
        max_grad_norm=cfg.train.max_grad_norm,
        scheduler_name=cfg.train.scheduler,
        warmup_steps=cfg.train.warmup_steps,
        total_steps=cfg.train.steps,
    )
```

Also update the import at the top of `src/build.py` — no new imports needed since `ConsistencyLoss` is imported inside `build_trainer`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_trainer.py -v`
Expected: All 2 tests PASS

- [ ] **Step 6: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS (test_build, test_misc, test_io, test_dataset, test_loss, test_trainer)

- [ ] **Step 7: Commit**

```bash
git add src/finetune/trainer.py src/build.py tests/test_trainer.py
git commit -m "feat: dual-view training, LR scheduler, grad clipping, best model save"
```

---

### Task 6: Resume support

**Files:**
- Modify: `src/finetune/trainer.py`
- Modify: `train.py`

- [ ] **Step 1: Add save_checkpoint and load_checkpoint to Trainer**

Add these methods to the `Trainer` class in `src/finetune/trainer.py`:

```python
    def save_checkpoint(self, path: str, step: int) -> None:
        state = {
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler:
            state["scheduler"] = self.scheduler.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        if self.scheduler and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        return state["step"]
```

- [ ] **Step 2: Update _save to also save checkpoint**

Replace the `_save` method:

```python
    def _save(self, filename: str = "last.pth", step: int = 0) -> None:
        weights_dir = os.path.join(self.save_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        if hasattr(self.model, "save_adapters"):
            self.model.save_adapters(os.path.join(weights_dir, filename))
        self.save_checkpoint(os.path.join(weights_dir, "checkpoint.pth"), step=step)
```

- [ ] **Step 3: Update train loop to pass step to _save and support start_step**

Replace the `train` method:

```python
    def train(self, steps: int, val_freq: int = 500, save_freq: int = 1000, start_step: int = 0) -> None:
        progress_bar = tqdm(range(start_step + 1, steps + 1), desc="Training")

        for step in progress_bar:
            losses = self._step()
            progress_bar.set_postfix(
                loss=f"{losses['loss']:.4f}",
                seg=f"{losses['seg']:.4f}",
                cons=f"{losses['cons']:.4f}",
            )

            if self.writer:
                self.writer.add_scalar("Train/Loss", losses["loss"], step)
                self.writer.add_scalar("Train/Seg", losses["seg"], step)
                self.writer.add_scalar("Train/Aux", losses["aux"], step)
                self.writer.add_scalar("Train/Cons", losses["cons"], step)
                if self.scheduler:
                    self.writer.add_scalar("Train/LR", self.scheduler.get_last_lr()[0], step)

            if step % val_freq == 0:
                valid_loss = self._validate()
                if self.writer:
                    self.writer.add_scalar("Val/Loss", valid_loss, step)
                if valid_loss < self.best_val_loss:
                    self.best_val_loss = valid_loss
                    self._save(filename="best.pth", step=step)

            if step % save_freq == 0:
                self._save(filename="last.pth", step=step)

        if self.writer:
            self.writer.flush()
            self.writer.close()
```

- [ ] **Step 4: Update train.py for resume**

Add resume logic in the `train()` function, after `build_trainer` and before `trainer.train`:

```python
    resume_from = cfg.train.get("resume_from", None)
    start_step = 0
    if resume_from:
        import os as _os
        checkpoint_path = _os.path.join(resume_from, "weights", "checkpoint.pth")
        adapter_path = _os.path.join(resume_from, "weights", "last.pth")
        if hasattr(model, "load_adapters"):
            model.load_adapters(adapter_path)
        start_step = trainer.load_checkpoint(checkpoint_path)
        print(f"[Resume] from step {start_step}, dir={resume_from}")
```

And update the `trainer.train()` call:

```python
    trainer.train(
        steps=cfg.train.steps,
        val_freq=cfg.train.val_freq,
        save_freq=cfg.train.save_freq,
        start_step=start_step,
    )
```

- [ ] **Step 5: Verify syntax**

Run: `.venv/bin/python -m compileall src train.py`
Expected: No errors

- [ ] **Step 6: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/finetune/trainer.py train.py
git commit -m "feat: add checkpoint save/load for training resume"
```

---

### Task 7: Inference script

**Files:**
- Create: `predict.py`

- [ ] **Step 1: Create predict.py**

```python
import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.build import build_birefnet, build_lora_birefnet
from src.data.dataset import MEAN, STD
from src.utils.io import load_yaml
from src.utils.misc import ConfigDict

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image(path: str, size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=MEAN, std=STD)
    tensor = TF.resize(tensor, list(size), antialias=True)
    return tensor.unsqueeze(0)


def save_mask(tensor: torch.Tensor, path: str, original_size: tuple[int, int]) -> None:
    mask = tensor.squeeze(0).squeeze(0).sigmoid()
    mask = TF.resize(mask.unsqueeze(0), list(original_size), antialias=True).squeeze(0)
    mask = (mask * 255).byte().cpu().numpy()
    Image.fromarray(mask, mode="L").save(path)


def predict() -> None:
    parser = argparse.ArgumentParser(description="Run inference with LoRA-adapted BiRefNet")
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--config", "-c", default="src/config/tune.yaml", help="Config file")
    parser.add_argument("--adapter", "-a", required=True, help="Path to LoRA adapter weights")
    args = parser.parse_args()

    config_data = load_yaml(path=args.config)
    cfg = ConfigDict(config_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = tuple(cfg.data.size)

    base_model = build_birefnet(cfg=cfg)
    model = build_lora_birefnet(cfg=cfg, model=base_model, ckpt_path=args.adapter).to(device)
    model.eval()

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(
            p for p in input_path.glob("*")
            if p.is_file() and p.suffix.lower() in EXTS
        )

    os.makedirs(args.output, exist_ok=True)

    for image_path in image_paths:
        original_image = Image.open(image_path).convert("RGB")
        original_size = (original_image.height, original_image.width)

        image_tensor = load_image(str(image_path), size).to(device)

        with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            prediction = model(image_tensor)

        output_path = os.path.join(args.output, f"{image_path.stem}_mask.png")
        save_mask(prediction, output_path, original_size)
        print(f"{image_path.name} -> {output_path}")


if __name__ == "__main__":
    predict()
```

- [ ] **Step 2: Verify syntax**

Run: `.venv/bin/python -m compileall predict.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add predict.py
git commit -m "feat: add inference script for LoRA-adapted BiRefNet"
```

---

### Task 8: Final integration test and cleanup

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Syntax check all Python files**

Run: `.venv/bin/python -m compileall src train.py predict.py`
Expected: No errors

- [ ] **Step 3: Verify config loads correctly**

Run: `.venv/bin/python -c "from src.utils.io import load_yaml; from src.utils.misc import ConfigDict; cfg = ConfigDict(load_yaml('src/config/tune.yaml')); print(cfg.train.lambda_cons, cfg.train.scheduler, cfg.train.max_grad_norm)"`
Expected: `0.1 cosine 1.0`

- [ ] **Step 4: Commit any remaining fixes**

Only if needed based on test results.
