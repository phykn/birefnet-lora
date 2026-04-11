# Refactor Groups A+B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the 9 findings selected from the 2026-04-11 refactor diagnosis (Groups A silent-bug defenses + B dead-code cleanup), leaving the codebase with tighter contracts, no dead slots, and tests still green.

**Architecture:** Work bottom-up from trivial deletions to the structural fix for the forward-path contract. Each finding is one task (or a small cluster), each task ends with a commit, tests run after every change. The largest task (T8) introduces a `ModelOutput` dataclass so `LoRABiRefNet.forward` has a single return type, eliminating the mode-dependent unpacking that drives F-09 and allowing F-15 (aux normalization) to be solved in the same place.

**Tech Stack:** Python 3.12, PyTorch, OmegaConf, pytest.

**Source spec:** `docs/superpowers/specs/2026-04-11-refactor-ab-design.md`

---

## Pre-flight

- [ ] **Step 0.1: Verify clean working tree**

```bash
git status
```
Expected: working tree clean (or only the spec file committed).

- [ ] **Step 0.2: Establish baseline — all tests pass before any changes**

```bash
pytest tests/ -v
```
Expected: all tests pass. If any test is already failing on `main`, STOP and report — do not start the refactor on a broken baseline.

- [ ] **Step 0.3: Compile check**

```bash
python -m compileall src run_train.py
```
Expected: no errors.

- [ ] **Step 0.4: Coordinate with in-flight trainer rewrite**

Per project memory, `src/ml/training/trainer.py` and `src/ml/training/loss.py` are being rewritten. Before starting T8, re-read both files and the associated tests (`tests/test_trainer.py`, `tests/test_loss.py`) to confirm they still match the contract this plan assumes: `model(x)` returns `(list[Tensor], Tensor)` in train and `Tensor` in eval. If the trainer rewrite has already changed the contract, flag to user before proceeding with T8 — the remaining tasks (T1–T7) are independent and can still run.

---

## File Structure

Files touched by this plan, grouped by responsibility:

- `src/ml/model/lora/wrapper.py` — F-05 (delete unused attr), F-09 (return dataclass), F-10 (drop class_preds unpacking), F-15 (normalize aux)
- `src/ml/model/birefnet/birefnet.py` — F-10 (remove class_preds plumbing), F-11 (remove `mul_scl_ipt` + "add" branch)
- `src/ml/build.py` — F-03 (call `Dataset` directly), F-08 (require ckpt for inference), F-11 (drop `mul_scl_ipt` kwarg), F-14 (narrow type hint)
- `src/ml/data/dataset.py` — F-03 (delete subclasses)
- `src/ml/training/loss.py` — F-09 (consume `ModelOutput`), F-15 (aux moves here)
- `src/ml/inference/predict.py` — F-09 (consume `ModelOutput`)
- `src/config/model.yaml` — F-11 (remove `mul_scl_ipt` key)
- `run_train.py` — F-04 (inline `OmegaConf.load/save`)
- `tests/test_io.py` — F-04 (delete)
- `tests/test_loss.py` — F-09 (update stub models)
- `tests/test_trainer.py` — F-09 (update stub models)
- `src/utils/io.py` — F-04 (delete)
- `src/utils/` — F-04 (delete if empty)

---

## Task 1: Remove unused `LoRABiRefNet.size` (F-05)

**Files:**
- Modify: `src/ml/model/lora/wrapper.py:12`

- [ ] **Step 1.1: Confirm it is unused**

```bash
rg "\.size" src/ml/model/lora/ tests/ run_train.py src/ml/inference/
```
Expected: the only match inside `lora/wrapper.py` is the declaration on line 12. No other file reads `lora.size` or `lora_model.size`.

- [ ] **Step 1.2: Delete the line**

In `src/ml/model/lora/wrapper.py`, remove line 12:

```python
        self.size = (1024, 1024)
```

- [ ] **Step 1.3: Run tests**

```bash
pytest tests/test_model.py -v
```
Expected: all pass.

- [ ] **Step 1.4: Commit**

```bash
git add src/ml/model/lora/wrapper.py
git commit -m "refactor: drop unused LoRABiRefNet.size attribute (F-05)"
```

---

## Task 2: Narrow `build_trainer` type hint (F-14)

**Files:**
- Modify: `src/ml/build.py` (add import + hint)

- [ ] **Step 2.1: Confirm `LoRABiRefNet` is already imported**

```bash
rg "LoRABiRefNet" src/ml/build.py
```
Expected: already imported on line 12 (`from .model.lora.wrapper import LoRABiRefNet`).

- [ ] **Step 2.2: Narrow the `model` parameter hint**

In `src/ml/build.py`, change the `build_trainer` signature (around line 120-125) from:

```python
def build_trainer(
    cfg: Any,
    model: torch.nn.Module,
    train_dl: DataLoader,
    valid_dl: DataLoader,
) -> Trainer:
```

to:

```python
def build_trainer(
    cfg: Any,
    model: LoRABiRefNet,
    train_dl: DataLoader,
    valid_dl: DataLoader,
) -> Trainer:
```

- [ ] **Step 2.3: Run tests**

```bash
pytest tests/test_build.py -v && python -m compileall src/ml/build.py
```
Expected: tests pass, compile clean.

- [ ] **Step 2.4: Commit**

```bash
git add src/ml/build.py
git commit -m "refactor: narrow build_trainer hint to LoRABiRefNet (F-14)"
```

---

## Task 3: Remove trivial `TrainDataset` / `ValidDataset` subclasses (F-03)

**Files:**
- Modify: `src/ml/data/dataset.py` (delete lines 112-119)
- Modify: `src/ml/build.py` (update imports + call sites)

- [ ] **Step 3.1: Check for external references**

```bash
rg "TrainDataset|ValidDataset" src/ tests/ run_train.py
```
Record every hit. Expected hits: definitions in `src/ml/data/dataset.py`, imports + calls in `src/ml/build.py`. No test references (confirmed during planning).

- [ ] **Step 3.2: Delete the subclasses**

In `src/ml/data/dataset.py`, delete lines 112-119 (both class definitions, keeping the trailing newline at EOF):

```python
class TrainDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, train=True)


class ValidDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, train=False)
```

- [ ] **Step 3.3: Update `build.py` import**

In `src/ml/build.py`, change:

```python
from .data.dataset import TrainDataset, ValidDataset
```

to:

```python
from .data.dataset import Dataset
```

- [ ] **Step 3.4: Update `build_dl` call sites**

In `src/ml/build.py`, replace the `train_dataset` / `valid_dataset` constructions (lines 46-57) with:

```python
    train_dataset = Dataset(
        data=train_data,
        size=cfg.data.size,
        scales=cfg.data.scales,
        train=True,
        bc_weak=(cfg.augment.bc_weak.brightness, cfg.augment.bc_weak.contrast),
        bc_strong=(cfg.augment.bc_strong.brightness, cfg.augment.bc_strong.contrast),
    )
    valid_dataset = Dataset(
        data=valid_data,
        size=cfg.data.size,
        train=False,
    )
```

Note: the dead `scales=(1.0, 1.0)` kwarg is dropped from the valid construction (scales is only consulted under `self.train`).

- [ ] **Step 3.5: Run tests**

```bash
pytest tests/test_build.py -v && python -m compileall src/ml/build.py src/ml/data/dataset.py
```
Expected: pass.

- [ ] **Step 3.6: Final grep**

```bash
rg "TrainDataset|ValidDataset" src/ tests/ run_train.py
```
Expected: no matches.

- [ ] **Step 3.7: Commit**

```bash
git add src/ml/build.py src/ml/data/dataset.py
git commit -m "refactor: remove trivial Train/ValidDataset subclasses (F-03)"
```

---

## Task 4: Inline `io.py` and delete the module (F-04)

**Files:**
- Modify: `run_train.py` (inline `OmegaConf.load/save`)
- Delete: `src/utils/io.py`
- Delete: `tests/test_io.py`
- Possibly delete: `src/utils/` (if empty after removal)

- [ ] **Step 4.1: List all import sites**

```bash
rg "from src\.utils\.io|from \.io|src\.utils\.io" src/ tests/ run_train.py
```
Expected hits: `run_train.py`, `tests/test_io.py`. Record any additional hits and update them in subsequent steps.

- [ ] **Step 4.2: Update `run_train.py` imports and calls**

In `run_train.py`, replace:

```python
from src.ml.build import build_birefnet, build_dl, build_lora_birefnet, build_trainer
from src.utils.io import load_yaml, save_yaml
```

with:

```python
from src.ml.build import build_birefnet, build_dl, build_lora_birefnet, build_trainer
```

Then in `main()`, replace:

```python
    cfg = OmegaConf.merge(
        load_yaml("src/config/tune.yaml"),
        load_yaml("src/config/model.yaml"),
    )
```

with:

```python
    cfg = OmegaConf.merge(
        OmegaConf.load("src/config/tune.yaml"),
        OmegaConf.load("src/config/model.yaml"),
    )
```

And replace:

```python
    save_yaml(cfg=cfg, path=f"{trainer.save_dir}/config.yaml")
```

with:

```python
    OmegaConf.save(cfg, f"{trainer.save_dir}/config.yaml")
```

- [ ] **Step 4.3: Delete `src/utils/io.py`**

```bash
rm src/utils/io.py
```

- [ ] **Step 4.4: Delete `tests/test_io.py`**

```bash
rm tests/test_io.py
```

- [ ] **Step 4.5: Check if `src/utils/` is now empty**

```bash
ls src/utils/
```

If empty or only `__pycache__`: delete the directory.

```bash
rm -rf src/utils/
```

If other files exist, leave the directory.

- [ ] **Step 4.6: Check for dangling references**

```bash
rg "src\.utils\.io|utils\.io|load_yaml|save_yaml" src/ tests/ run_train.py docs/
```
Expected: no matches in source or tests. (Matches in `docs/` are acceptable historical references.)

- [ ] **Step 4.7: Update CLAUDE.md reference if present**

```bash
rg "src/utils/io|io\.py" CLAUDE.md README.md
```
If matches: remove the bullet that describes `src/utils/io.py` (CLAUDE.md section "Key modules" mentions it).

- [ ] **Step 4.8: Run full suite and compile**

```bash
pytest tests/ -v && python -m compileall src run_train.py
```
Expected: pass.

- [ ] **Step 4.9: Commit**

```bash
git add -A
git commit -m "refactor: inline OmegaConf load/save, delete utils/io.py (F-04)"
```

---

## Task 5: Remove `mul_scl_ipt` parameter and "add" branch (F-11)

**Files:**
- Modify: `src/ml/model/birefnet/birefnet.py` (delete param, "add" branch, self.mul_scl_ipt refs)
- Modify: `src/ml/build.py:85-94` (stop passing `mul_scl_ipt` kwarg)
- Modify: `src/config/model.yaml` (remove `mul_scl_ipt` key)

- [ ] **Step 5.1: Verify pretrained state_dict is not affected**

Confirm that `self.mul_scl_ipt` is stored as a plain attribute (not `nn.Parameter` / `register_buffer`). Read `src/ml/model/birefnet/birefnet.py:412-418`: it is a string stored as `self.mul_scl_ipt = mul_scl_ipt`. No state_dict key. Safe to remove.

- [ ] **Step 5.2: Edit `BiRefNet.__init__`**

In `src/ml/model/birefnet/birefnet.py`:

Delete the `mul_scl_ipt: str = "cat"` parameter from `__init__` (line 403).

Replace the conditional channel doubling (lines 412-415) with unconditional doubling:

```python
        lateral_channels_in_collection = [
            ch * 2 for ch in lateral_channels_in_collection
        ]
```

Delete the line `self.mul_scl_ipt = mul_scl_ipt` (line 418).

- [ ] **Step 5.3: Edit `forward_enc`**

In the same file, simplify `forward_enc` around lines 445-515. Delete the outer `if self.mul_scl_ipt:` guard and the entire `elif self.mul_scl_ipt == "add":` branch. The pyramid-cat logic becomes unconditional.

After: the block should read (approximately; preserve exact indentation and formatting of the surrounding code):

```python
        x1, x2, x3, x4 = self.bb(x)

        B, C, H, W = x.shape
        x_pyramid = F.interpolate(
            x, size=(H // 2, W // 2), mode="bilinear", align_corners=True
        )
        pyramid_x1, pyramid_x2, pyramid_x3, pyramid_x4 = self.bb(x_pyramid)
        x1 = torch.cat(
            [
                x1,
                F.interpolate(
                    pyramid_x1,
                    size=x1.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ],
            dim=1,
        )
        x2 = torch.cat(
            [
                x2,
                F.interpolate(
                    pyramid_x2,
                    size=x2.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ],
            dim=1,
        )
        x3 = torch.cat(
            [
                x3,
                F.interpolate(
                    pyramid_x3,
                    size=x3.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ],
            dim=1,
        )
        x4 = torch.cat(
            [
                x4,
                F.interpolate(
                    pyramid_x4,
                    size=x4.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ],
            dim=1,
        )

        class_preds = None
```

Note: `class_preds = None` is kept in this task and removed in Task 6.

- [ ] **Step 5.4: Remove `mul_scl_ipt` kwarg from `build_birefnet`**

In `src/ml/build.py`, delete line 88:

```python
        mul_scl_ipt=cfg.birefnet.mul_scl_ipt,
```

- [ ] **Step 5.5: Remove `mul_scl_ipt` from `model.yaml`**

In `src/config/model.yaml`, delete the line:

```yaml
  mul_scl_ipt: "cat"
```

- [ ] **Step 5.6: Verify pretrained weight still loads**

```bash
python -c "
from omegaconf import OmegaConf
from src.ml.build import build_birefnet
cfg = OmegaConf.merge(
    OmegaConf.load('src/config/tune.yaml'),
    OmegaConf.load('src/config/model.yaml'),
)
model = build_birefnet(cfg=cfg)
print('OK:', sum(p.numel() for p in model.parameters()), 'params')
"
```
Expected: prints `OK: <number> params` and `[LOAD] weight/BiRefNet-general-epoch_244.pth`. If the pretrained weight file is absent locally, the `load_state_dict` call will fail — that is acceptable as long as the error is about the file, not about unexpected/missing keys.

If the error shows unexpected keys or size mismatches, STOP: the pretrained checkpoint's structure was implicitly coupled to `mul_scl_ipt="cat"` doubling, and this task's assumption that the doubling is structurally identical may be wrong. Investigate before proceeding.

- [ ] **Step 5.7: Run tests and compile**

```bash
pytest tests/ -v && python -m compileall src run_train.py
```
Expected: pass.

- [ ] **Step 5.8: Commit**

```bash
git add -A
git commit -m "refactor: drop dead mul_scl_ipt='add' branch and param (F-11)"
```

---

## Task 6: Remove `class_preds` return slot (F-10)

**Files:**
- Modify: `src/ml/model/birefnet/birefnet.py`
- Modify: `src/ml/model/lora/wrapper.py`

- [ ] **Step 6.1: Grep for every reference**

```bash
rg "class_preds" src/ tests/
```
Record all hits.

- [ ] **Step 6.2: Edit `forward_enc`**

In `src/ml/model/birefnet/birefnet.py`:

Delete the line `class_preds = None` (inside `forward_enc`, currently around line 517).

Change the `forward_enc` return statement from:

```python
        return (x1, x2, x3, x4), class_preds
```

to:

```python
        return (x1, x2, x3, x4)
```

Update the `forward_enc` return type annotation (lines 439-442) from:

```python
    def forward_enc(
        self, x: torch.Tensor
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor | None,
    ]:
```

to:

```python
    def forward_enc(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
```

- [ ] **Step 6.3: Edit `forward_ori`**

Change the body (lines 543-554) to remove the `class_preds` unpacking and second return element:

```python
    def forward_ori(
        self, x: torch.Tensor
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        x1, x2, x3, x4 = self.forward_enc(x)

        x4 = self.squeeze_module(x4)

        features = [x, x1, x2, x3, x4]

        if self.training and self.out_ref:
            features.append(laplacian(torch.mean(x, dim=1).unsqueeze(1), kernel_size=5))

        scaled_preds = self.decoder(features)

        return scaled_preds
```

- [ ] **Step 6.4: Edit `forward`**

Change the body (lines 556-568) to return `scaled_preds` directly in both modes:

```python
    def forward(
        self, x: torch.Tensor
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.forward_ori(x)
```

Note: training-mode callers now receive `[[gdt_pred, gdt_label], outs]` from `Decoder.forward` directly, without the extra `[..., [class_preds]]` wrapping. This is the whole point of F-10.

- [ ] **Step 6.5: Edit `LoRABiRefNet._train_step`**

In `src/ml/model/lora/wrapper.py`, change the first line of `_train_step` (currently line 41):

```python
        scaled_preds, _ = self.model(x)
        (gdt_predictions, gdt_labels), predictions = scaled_preds
```

to:

```python
        (gdt_predictions, gdt_labels), predictions = self.model(x)
```

- [ ] **Step 6.6: Grep verification**

```bash
rg "class_preds" src/ tests/
```
Expected: no matches.

- [ ] **Step 6.7: Verify pretrained weight still loads**

```bash
python -c "
from omegaconf import OmegaConf
from src.ml.build import build_birefnet
cfg = OmegaConf.merge(
    OmegaConf.load('src/config/tune.yaml'),
    OmegaConf.load('src/config/model.yaml'),
)
model = build_birefnet(cfg=cfg)
print('OK')
"
```
Expected: loads without state_dict errors.

- [ ] **Step 6.8: Run tests and compile**

```bash
pytest tests/ -v && python -m compileall src run_train.py
```
Expected: pass.

- [ ] **Step 6.9: Commit**

```bash
git add src/ml/model/birefnet/birefnet.py src/ml/model/lora/wrapper.py
git commit -m "refactor: remove always-None class_preds return slot (F-10)"
```

---

## Task 7: Require `ckpt_path` at inference — split `build_lora_birefnet` (F-08)

**Files:**
- Modify: `src/ml/build.py`
- Modify: `run_train.py` (if call-site name changes)
- Modify: `tests/test_build.py` (if it references the function)

- [ ] **Step 7.1: Decide the API shape**

The spec's suggested direction offers two options. **Pick option A**: split into two explicit functions — `build_lora_birefnet_for_training(cfg, model)` (no ckpt) and `build_lora_birefnet_for_inference(cfg, model, ckpt_path)` (required positional). Rationale: a required positional arg at inference makes the silent-random-adapter failure mode impossible, and splitting the name documents intent at every call site.

- [ ] **Step 7.2: Grep current call sites**

```bash
rg "build_lora_birefnet" src/ tests/ run_train.py docs/
```
Record all hits.

- [ ] **Step 7.3: Replace the single function with two**

In `src/ml/build.py`, replace the existing `build_lora_birefnet` function (lines 105-117) with:

```python
def build_lora_birefnet_for_training(
    cfg: Any,
    model: torch.nn.Module,
) -> LoRABiRefNet:
    """Wrap `model` with fresh (random-initialized) LoRA adapters for training."""
    device = next(model.parameters()).device
    lora_model = LoRABiRefNet(model=model, rank=cfg.lora.rank, alpha=cfg.lora.alpha)
    return lora_model.to(device)


def build_lora_birefnet_for_inference(
    cfg: Any,
    model: torch.nn.Module,
    ckpt_path: str,
) -> LoRABiRefNet:
    """Wrap `model` with LoRA adapters and load trained weights from `ckpt_path`."""
    device = next(model.parameters()).device
    lora_model = LoRABiRefNet(model=model, rank=cfg.lora.rank, alpha=cfg.lora.alpha)
    lora_model.load_adapters(ckpt_path)
    print(f"[LOAD] {ckpt_path}")
    return lora_model.to(device)
```

- [ ] **Step 7.4: Update `run_train.py` call site**

Change the import:

```python
from src.ml.build import build_birefnet, build_dl, build_lora_birefnet, build_trainer
```

to:

```python
from src.ml.build import (
    build_birefnet,
    build_dl,
    build_lora_birefnet_for_training,
    build_trainer,
)
```

Change the call site (around line 115):

```python
    model = build_lora_birefnet(cfg=cfg, model=base_model)
```

to:

```python
    model = build_lora_birefnet_for_training(cfg=cfg, model=base_model)
```

- [ ] **Step 7.5: Check `tests/test_build.py` for references**

```bash
rg "build_lora_birefnet" tests/
```
If any test imports or calls the old name, update it. (At planning time: `tests/test_build.py` only tests `index_by_stem`, so no changes expected.)

- [ ] **Step 7.6: Verify no old name survives**

```bash
rg "build_lora_birefnet\b" src/ tests/ run_train.py
```
Expected: only the two new names appear; no bare `build_lora_birefnet(` calls remain.

- [ ] **Step 7.7: Run tests and compile**

```bash
pytest tests/ -v && python -m compileall src run_train.py
```
Expected: pass.

- [ ] **Step 7.8: Commit**

```bash
git add src/ml/build.py run_train.py
git commit -m "refactor: split build_lora_birefnet into training/inference (F-08)"
```

---

## Task 8: Unify forward return via `ModelOutput` dataclass + normalize aux (F-09, F-15)

This is the largest task. It introduces a single typed return shape for `LoRABiRefNet.forward` in both modes, moves the aux-loss accumulation into a helper on the model (where `gdt_predictions`/`gdt_labels` live), normalizes the aux by the number of GDT levels, and updates every consumer (`CustomLoss`, `predict.py`, test stubs).

**Files:**
- Modify: `src/ml/model/lora/wrapper.py` — add `ModelOutput`, rewrite `_train_step`/`_eval_step`/`forward`, normalize aux
- Modify: `src/ml/training/loss.py` — consume `ModelOutput`
- Modify: `src/ml/inference/predict.py` — consume `ModelOutput`
- Modify: `tests/test_loss.py` — update stub model
- Modify: `tests/test_trainer.py` — update stub model

### Sub-task 8a: Write the failing test first (TDD)

- [ ] **Step 8.1: Add a test asserting the new contract**

In `tests/test_model.py`, append this test at the end of the file:

```python
def test_lora_birefnet_forward_returns_modeloutput_in_both_modes():
    from src.ml.model.lora.wrapper import ModelOutput

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = _Backbone()
            self.decoder = _Decoder()

        def forward(self, x):
            pred = torch.zeros(x.shape[0], 1, 4, 4)
            if self.training:
                gdt_preds = [torch.zeros(x.shape[0], 1, 4, 4)]
                gdt_labels = [torch.zeros(x.shape[0], 1, 4, 4)]
                return [[gdt_preds, gdt_labels], [pred, pred]]
            return [pred, pred]

    lora = LoRABiRefNet(_Stub(), rank=2, alpha=4.0)

    lora.train()
    out_train = lora(torch.randn(1, 3, 8, 8))
    assert isinstance(out_train, ModelOutput)
    assert isinstance(out_train.preds, list)
    assert out_train.aux is not None
    assert out_train.aux.dim() == 0

    lora.eval()
    out_eval = lora(torch.randn(1, 3, 8, 8))
    assert isinstance(out_eval, ModelOutput)
    assert out_eval.aux is None
```

- [ ] **Step 8.2: Run the new test — expect failure**

```bash
pytest tests/test_model.py::test_lora_birefnet_forward_returns_modeloutput_in_both_modes -v
```
Expected: FAIL with `ImportError` (`ModelOutput` does not exist yet).

### Sub-task 8b: Introduce `ModelOutput` and new forward plumbing

- [ ] **Step 8.3: Add `ModelOutput` dataclass and rewrite `_train_step`/`_eval_step`/`forward`**

In `src/ml/model/lora/wrapper.py`:

At the top, add:

```python
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import LoRAConv2d, LoRALinear, apply_conv2d, apply_linear


@dataclass
class ModelOutput:
    preds: list[torch.Tensor]
    aux: torch.Tensor | None = None
```

(Merge the import block with the existing imports; do not leave duplicate `import torch` lines.)

Replace the existing `_train_step`, `_eval_step`, and `forward` methods with:

```python
    def _train_step(self, x: torch.Tensor) -> ModelOutput:
        (gdt_predictions, gdt_labels), predictions = self.model(x)

        auxiliary_loss = torch.tensor(0.0, device=x.device)
        for gdt_prediction, gdt_label in zip(gdt_predictions, gdt_labels):
            gdt_prediction = F.interpolate(
                gdt_prediction,
                size=gdt_label.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
            gdt_label = gdt_label.sigmoid()
            auxiliary_loss = auxiliary_loss + F.binary_cross_entropy_with_logits(
                gdt_prediction,
                gdt_label,
            )
        num_levels = max(len(gdt_predictions), 1)
        auxiliary_loss = auxiliary_loss / num_levels

        return ModelOutput(preds=predictions, aux=auxiliary_loss)

    def _eval_step(self, x: torch.Tensor) -> ModelOutput:
        predictions = self.model(x)
        return ModelOutput(preds=predictions, aux=None)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if self.training:
            return self._train_step(x)
        return self._eval_step(x)
```

The key changes:
- Both branches return `ModelOutput`. Callers never need `isinstance` / mode checks.
- `auxiliary_loss / num_levels` (F-15): normalizes by the number of GDT supervision levels so `lambda_aux` has a stable meaning.
- Train-mode unpacking is now a single `(gdt_pred, gdt_label), predictions = self.model(x)` — no class_preds wrapping (relies on T6).

- [ ] **Step 8.4: Run the new test — expect pass**

```bash
pytest tests/test_model.py::test_lora_birefnet_forward_returns_modeloutput_in_both_modes -v
```
Expected: PASS.

- [ ] **Step 8.5: Run the full test suite — expect test_loss and test_trainer failures**

```bash
pytest tests/ -v
```
Expected: `tests/test_loss.py` and `tests/test_trainer.py` now FAIL because the stub models and `CustomLoss` still use the old `(list, Tensor)` tuple contract. This is expected — we fix them next.

### Sub-task 8c: Update `CustomLoss` to consume `ModelOutput`

- [ ] **Step 8.6: Rewrite `CustomLoss.forward`**

In `src/ml/training/loss.py`, replace the `CustomLoss` class body (lines 56-99). Add the import at the top of the file:

```python
from src.ml.model.lora.wrapper import ModelOutput
```

Rewrite the class:

```python
class CustomLoss(nn.Module):
    def __init__(
        self,
        lambda_bce: float = 30.0,
        lambda_iou: float = 0.5,
        lambda_kl: float = 1.0,
        lambda_aux: float = 1.0,
    ):
        super().__init__()
        self.seg = SegmentationLoss(lambda_bce=lambda_bce, lambda_iou=lambda_iou)
        self.cons = SymmetricBinaryKLLoss(lambda_kl=lambda_kl)
        self.lambda_aux = lambda_aux

    def forward(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = next(model.parameters()).device
        image_1 = batch["image_1"].to(device)
        mask = batch["mask"].to(device)

        if model.training:
            image_2 = batch["image_2"].to(device)
            batch_size = image_1.shape[0]
            images = torch.cat([image_1, image_2], dim=0)
            masks = torch.cat([mask, mask], dim=0)
            out: ModelOutput = model(images)
            preds = out.preds

            seg_loss = sum(self.seg(p, masks) for p in preds) / len(preds)
            cons_loss = self.cons(preds[-1][:batch_size], preds[-1][batch_size:])
            aux_loss = self.lambda_aux * (out.aux if out.aux is not None else torch.tensor(0.0, device=device))
            loss = seg_loss + cons_loss + aux_loss

            return {
                "loss": loss,
                "seg": seg_loss,
                "cons": cons_loss,
                "aux": aux_loss,
            }, loss

        out = model(image_1)
        pred = out.preds[-1]
        seg_loss = self.seg(pred, mask)
        return {"loss": seg_loss, "seg": seg_loss}, seg_loss
```

Note the change at the last branch: eval mode uses `out.preds[-1]` (the final-resolution map) since `_eval_step` now returns the whole `preds` list instead of only the last entry.

### Sub-task 8d: Update `predict.py`

- [ ] **Step 8.7: Consume `ModelOutput` in `predict.py`**

In `src/ml/inference/predict.py`, change the inference line (currently line 30):

```python
    logits = model(x)
```

to:

```python
    logits = model(x).preds[-1]
```

### Sub-task 8e: Update test stubs

- [ ] **Step 8.8: Update `tests/test_loss.py` `_TrainModel` stub**

Replace the `_TrainModel` class (lines 61-72) with:

```python
class _TrainModel(nn.Module):
    """Mimics LoRABiRefNet output: ModelOutput in both modes."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        from src.ml.model.lora.wrapper import ModelOutput

        pred = self.conv(x)
        if self.training:
            return ModelOutput(preds=[pred, pred], aux=torch.tensor(0.5, device=x.device))
        return ModelOutput(preds=[pred], aux=None)
```

- [ ] **Step 8.9: Update `tests/test_trainer.py` `_DummyModel` and `_DummyCriterion` stubs**

Replace the `_DummyModel` class (lines 11-26) with:

```python
class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        from src.ml.model.lora.wrapper import ModelOutput

        pred = self.conv(x)
        if self.training:
            return ModelOutput(
                preds=[pred, pred],
                aux=torch.tensor(0.0, device=x.device),
            )
        return ModelOutput(preds=[pred], aux=None)

    def get_adapter_params(self):
        return list(self.parameters())

    def save_adapters(self, path):
        torch.save(self.state_dict(), path)
```

Replace the `_DummyCriterion` class (lines 29-40) with:

```python
class _DummyCriterion(nn.Module):
    def forward(self, model, batch):
        device = next(model.parameters()).device
        if model.training:
            out = model(batch["image_1"].to(device))
            preds = out.preds
            seg = sum(p.mean() ** 2 for p in preds) / len(preds)
            cons = (preds[0] - preds[-1]).pow(2).mean()
            aux = out.aux if out.aux is not None else torch.tensor(0.0, device=device)
            loss = seg + cons + aux
            return {"loss": loss, "seg": seg, "cons": cons, "aux": aux}, loss
        out = model(batch["image_1"].to(device))
        pred = out.preds[-1]
        seg = pred.mean() ** 2
        return {"loss": seg, "seg": seg}, seg
```

### Sub-task 8f: Verify and commit

- [ ] **Step 8.10: Run the full suite**

```bash
pytest tests/ -v
```
Expected: all pass.

- [ ] **Step 8.11: Compile check**

```bash
python -m compileall src run_train.py tests
```
Expected: no errors.

- [ ] **Step 8.12: End-to-end smoke test**

Temporarily set `train.steps: 10`, `val_freq: 5`, `save_freq: 10` in `src/config/tune.yaml`, then run:

```bash
python run_train.py
```
Expected: completes without error, creates a fresh `run/<YYYYMMDD_HHMMSS>/` with `config.yaml`, `train.csv`, `valid.csv`, `logs/`, and `weights/last.pth`. If the pretrained weight file or local dataset is missing, note it and skip — do NOT commit the reduced `tune.yaml` values.

**Revert `tune.yaml` to its original values before committing.**

```bash
git diff src/config/tune.yaml
```
Expected: no diff.

- [ ] **Step 8.13: Commit**

```bash
git add src/ml/model/lora/wrapper.py src/ml/training/loss.py src/ml/inference/predict.py tests/test_model.py tests/test_loss.py tests/test_trainer.py
git commit -m "refactor: unify LoRABiRefNet forward via ModelOutput + normalize aux (F-09, F-15)"
```

---

## Post-flight

- [ ] **Step 9.1: Final test sweep**

```bash
pytest tests/ -v && python -m compileall src run_train.py
```
Expected: pass.

- [ ] **Step 9.2: Verify every success criterion from the spec**

Run these greps; each must return no matches:

```bash
rg "TrainDataset|ValidDataset" src/ tests/ run_train.py
rg "from src\.utils\.io|src/utils/io" src/ tests/ run_train.py
test -f src/utils/io.py && echo "STILL EXISTS" || echo "deleted"
rg "self\.size" src/ml/model/lora/wrapper.py
rg "class_preds" src/
rg "mul_scl_ipt" src/ src/config/
rg "build_lora_birefnet\b" src/ run_train.py | grep -v "_for_training\|_for_inference" && echo "OLD NAME STILL USED" || echo "clean"
```

And these must return matches:

```bash
rg "model: LoRABiRefNet" src/ml/build.py        # F-14
rg "class ModelOutput" src/ml/model/lora/wrapper.py  # F-09
rg "/ num_levels" src/ml/model/lora/wrapper.py   # F-15
```

- [ ] **Step 9.3: Show commit log since spec**

```bash
git log --oneline e39d9a4..HEAD
```
Expected: 8 new commits, one per task (T1-T8).

- [ ] **Step 9.4: Report to user**

Report the task list with pass/fail per task, the final test output, and any smoke-test findings from Step 8.12.

---

## Notes for the executor

- **Every task ends with a commit.** Do not bundle tasks into one commit, even if they feel small.
- **If a test fails that was passing before the current task, STOP** and investigate before continuing. Do not comment out or delete failing tests.
- **If the pretrained weight file `weight/BiRefNet-general-epoch_244.pth` is absent**, Step 5.6 and Step 6.7 will fail on the `load_state_dict` call — but the failure will be about the file path, not about structural key mismatches. File-absent failures are acceptable; key-mismatch failures must be investigated.
- **Revert any temporary config edits** (e.g., Step 8.12 reducing `train.steps`) before committing.
- **Do not touch trainer.py beyond what F-15 requires** — the project memory notes trainer.py is being rewritten in parallel. If you find the trainer already uses a different contract than what this plan assumes, flag to user before continuing T8.
