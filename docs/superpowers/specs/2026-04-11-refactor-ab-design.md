# Refactor Spec — Groups A (Silent-Bug Defenses) + B (Dead-Code Cleanup)

## Scope

Targeted refactor of the `birefnet-lora` repository selected from the
Feynman-style diagnosis performed on 2026-04-11. Two groups of findings are in
scope:

- **Group A — "Silent bugs"**: behaviors that succeed without error but produce
  wrong or misleading results (F-08, F-09, F-15).
- **Group B — "Dead-code cleanup"**: unused attributes, dead branches, and
  vestigial indirection layers (F-03, F-04, F-05, F-10, F-11, F-14).

Out of scope: F-01/F-02/F-17 (preprocessing/normalization/multi-scale
supervision design decisions — deferred to a separate brainstorming session).

## Context Summary

Sources read during diagnosis:

- `CLAUDE.md`, `README.md`
- `src/config/tune.yaml`, `src/config/model.yaml`
- `src/ml/build.py`, `run_train.py`
- `src/ml/training/{trainer,loss,scheduler}.py`
- `src/ml/data/{dataset,preprocess}.py`, `src/ml/data/augment/*`
- `src/ml/model/birefnet/birefnet.py`, `backbones/swin_v1.py`, `blocks/*`
- `src/ml/model/lora/{wrapper,adapters}.py`
- `src/ml/inference/predict.py`, `src/utils/io.py`
- `tests/` (test_model, test_build, test_loss, test_trainer, test_io, etc.)
- Git log for churn / recent restructuring history
- Project memory notes on vendored BiRefNet constraints and in-flight trainer rewrite

---

## Group A — Silent-Bug Defenses

### F-08: `build_lora_birefnet(ckpt_path=None)` silent inference failure

- **Location**: `src/ml/build.py:105-117`
- **Category**: logic
- **Observation**: `build_lora_birefnet` accepts `ckpt_path: str | None = None`.
  When `None`, `load_adapters` is skipped and the returned model has **randomly
  initialized LoRA adapters**. Training flow (`run_train.py:115`) intentionally
  passes no `ckpt_path`. Inference callers must remember to pass one; if they
  forget, `predict()` returns garbage with no error.
- **Reconstruction attempt**: The factory should produce a LoRA-wrapped model
  whose purpose differs for training (fresh adapters) vs. inference (loaded
  adapters). The optional kwarg encodes both modes in one signature, defaulting
  to the training path.
- **Failure point**: Defaulting to "no checkpoint" makes the inference bug
  silent. A function whose default value is wrong for the most dangerous caller
  is mis-designed — the default should force the error-free case.
- **Suggested direction**: Split into two entry points, one for training
  (fresh adapters) and one for inference (requires `ckpt_path`), OR make
  `ckpt_path` required and pass an explicit sentinel for training. The plan
  should decide which.
- **Axes**: Impact: high, Confidence: high, Effort: S

### F-09: BiRefNet / LoRABiRefNet forward return type asymmetry

- **Location**: `src/ml/model/birefnet/birefnet.py:393-396,566-568` and
  `src/ml/model/lora/wrapper.py:40-62`
- **Category**: structure
- **Observation**: In training mode, `Decoder.forward` returns
  `([outs_gdt_pred, outs_gdt_label], outs)`; in eval mode it returns a flat
  `list[Tensor]`. `BiRefNet.forward` additionally wraps the training output
  with `class_preds` (always `None`). `LoRABiRefNet._train_step` has to perform
  `scaled_preds, _ = self.model(x); (gdt_predictions, gdt_labels), predictions
  = scaled_preds` — a two-stage unpacking whose correctness depends on
  `self.training`.
- **Reconstruction attempt**: Training needs auxiliary GDT targets; eval does
  not. A single `forward()` is expected to serve both modes, so the return type
  encodes the mode.
- **Failure point**: Mode-dependent return types cannot be statically typed
  and force every caller into fragile unpacking. Any caller that forgets to
  check `self.training` crashes. The contract is legible only to someone who
  reads `_train_step` and `_eval_step` side by side.
- **Suggested direction**: Unify the contract. Options: (a) always return a
  dataclass/TypedDict with `main` and optional `aux` fields; (b) keep two
  physical methods on `LoRABiRefNet` and make the external `forward()` a thin
  dispatch with a documented, stable shape; (c) move the GDT auxiliary loss
  computation out of `_train_step` entirely so the model returns only
  predictions and `CustomLoss` computes aux itself (this also fixes F-15).
  Option (c) is preferred — it collapses F-09 and F-15 into one structural fix.
- **Axes**: Impact: high, Confidence: high, Effort: M

### F-15: Asymmetric aux_loss normalization

- **Location**: `src/ml/model/lora/wrapper.py:44-57` vs
  `src/ml/training/loss.py:85`
- **Category**: logic
- **Observation**: `_train_step` accumulates `auxiliary_loss` by summing BCE
  over every `(gdt_prediction, gdt_label)` pair with no division by the number
  of pairs. `seg_loss` in `CustomLoss.forward` is explicitly averaged:
  `sum(self.seg(p, masks) for p in preds) / len(preds)`. `lambda_aux=1.0`
  (tune.yaml) is therefore implicitly tuned against the current GDT level
  count.
- **Reconstruction attempt**: Both seg and aux are multi-scale supervisory
  terms combined into a single loss. They should be on comparable magnitude
  scales so `lambda_aux` has a stable meaning.
- **Failure point**: The normalization asymmetry breaks that comparability —
  if the number of GDT levels changes (e.g., someone adds or removes a decoder
  stage), `aux_loss` magnitude shifts while `seg_loss` does not, and
  `lambda_aux` silently takes on a different effective weight.
- **Suggested direction**: Divide `auxiliary_loss` by the number of GDT levels
  in `_train_step`, matching the seg averaging convention. If F-09 is fixed
  via option (c), aux computation moves to `CustomLoss` and can be normalized
  there instead.
- **Axes**: Impact: med, Confidence: med, Effort: S

---

## Group B — Dead-Code Cleanup

### F-03: Trivial `TrainDataset` / `ValidDataset` subclasses

- **Location**: `src/ml/data/dataset.py:112-119`, `src/ml/build.py:46-57`
- **Category**: dead-code
- **Observation**: Both subclasses exist only to hardcode `train=True` /
  `train=False` in `super().__init__`. `build_dl` additionally passes
  `scales=(1.0, 1.0)` to `ValidDataset`, but `scales` is only consulted under
  `if self.train:` — the argument is dead for valid.
- **Reconstruction attempt**: Subclasses should exist when they add behavior.
  These add none — they are alias indirection.
- **Failure point**: The subclasses create the illusion that validation has
  its own logic (and that `scales=(1.0, 1.0)` is a meaningful config). Neither
  is true.
- **Suggested direction**: Delete both subclasses. Have `build_dl` instantiate
  `Dataset(..., train=True)` and `Dataset(..., train=False)` directly, dropping
  the `scales` argument from the valid call site.
- **Axes**: Impact: low, Confidence: high, Effort: S

### F-04: `src/utils/io.py` thin OmegaConf wrapper

- **Location**: `src/utils/io.py` (9 lines total)
- **Category**: dead-code
- **Observation**: `load_yaml` / `save_yaml` forward verbatim to
  `OmegaConf.load` / `OmegaConf.save`. No validation, defaults, or
  transformation. Callers already handle `DictConfig`/`ListConfig` directly
  (e.g., `tests/test_io.py:13`).
- **Reconstruction attempt**: A utility module should exist to centralize
  logic that would otherwise be duplicated or fragile. This module centralizes
  nothing — the underlying API is already one line.
- **Failure point**: Indirection without abstraction. Readers must follow an
  import chain to discover that the wrapper is a no-op.
- **Suggested direction**: Inline `OmegaConf.load` / `OmegaConf.save` at all
  call sites (`run_train.py`, `tests/test_io.py`, any others). Delete
  `src/utils/io.py`. If `src/utils/` becomes empty, delete the directory.
- **Axes**: Impact: low, Confidence: high, Effort: S

### F-05: `LoRABiRefNet.size` unused attribute

- **Location**: `src/ml/model/lora/wrapper.py:12`
- **Category**: dead-code
- **Observation**: `self.size = (1024, 1024)` is set in `__init__` and never
  read. A repo-wide grep for `\.size` confirms the only reads are on other
  objects (`Dataset.size`, `Tensor.size()`, etc.).
- **Reconstruction attempt**: A model-level `size` might exist to enforce
  input dimensions or to inform downstream consumers. Nothing consumes it.
- **Failure point**: The attribute lies — it implies the model cares about a
  fixed input size, but it does not.
- **Suggested direction**: Delete the line.
- **Axes**: Impact: low, Confidence: high, Effort: S

### F-10: `class_preds` always-None return slot

- **Location**: `src/ml/model/birefnet/birefnet.py:517,535,554,566`
- **Category**: dead-code
- **Observation**: `forward_enc` sets `class_preds = None` and always returns
  it as `None`. `BiRefNet.forward` includes it in the training-mode output
  (`[scaled_preds, [class_preds]]`). `LoRABiRefNet._train_step` discards the
  second return value entirely (`scaled_preds, _ = self.model(x)`).
- **Reconstruction attempt**: Upstream BiRefNet may have supported a
  classification head. In this codebase it is neither wired nor tested.
- **Failure point**: A return slot that is always `None` is indistinguishable
  from a bug placeholder. It also entrenches the fragile contract in F-09.
- **Suggested direction**: Remove `class_preds` from `forward_enc`,
  `BiRefNet.forward`, and anywhere else it is constructed. If the return shape
  simplification makes F-09 structurally cleaner, fold the changes together.
- **Axes**: Impact: low, Confidence: high, Effort: M

### F-11: `mul_scl_ipt == "add"` dead branch

- **Location**: `src/ml/model/birefnet/birefnet.py:502-515`
- **Category**: dead-code
- **Observation**: `forward_enc` branches on `mul_scl_ipt`. `model.yaml` fixes
  it to `"cat"`. The `"add"` branch has no config exposure and no test.
- **Reconstruction attempt**: The branch is vendored upstream code preserved
  for option parity. This repo never uses it.
- **Failure point**: Vendored dead branches increase surface area without
  payoff. If the `"cat"` path is the only tested one, the `"add"` path may
  already be broken and nobody would notice.
- **Suggested direction**: Remove the `"add"` branch and either delete
  `mul_scl_ipt` from `BiRefNet.__init__` entirely (hardcoding concat) or keep
  the parameter but validate that it equals `"cat"`. Prefer deletion — LoRA
  adapters are rank-tied to the "cat" channel widths anyway, so this flag is
  not tunable without retraining from scratch.
- **Axes**: Impact: low, Confidence: high, Effort: S

### F-14: `build_trainer` misleading type hint

- **Location**: `src/ml/build.py:120-125`
- **Category**: naming
- **Observation**: `build_trainer(cfg, model: torch.nn.Module, ...)` accepts a
  generic `nn.Module` but immediately calls `model.get_adapter_params()`, a
  method only defined on `LoRABiRefNet`.
- **Reconstruction attempt**: The type hint should document the actual
  contract the function requires.
- **Failure point**: A generic hint where the method call demands a specific
  subclass is a documentation bug. Static analysis will not catch it.
- **Suggested direction**: Narrow the type hint to
  `model: LoRABiRefNet`. Import `LoRABiRefNet` at the top of `build.py`.
- **Axes**: Impact: low, Confidence: high, Effort: S

---

## Refactoring Constraints

1. **Tests must still pass.** `pytest tests/ -v` is the gate. No test may be
   deleted or weakened solely to accommodate a refactor; if a test becomes
   misaligned, update the test to reflect the new correct contract and
   document the change.
2. **Pretrained BiRefNet weights must still load.** Any change that alters
   `BiRefNet.state_dict()` keys (e.g., F-10 removing `class_preds` plumbing,
   F-11 removing the `"add"` branch) must be verified to still accept
   `weight/BiRefNet-general-epoch_244.pth` via `load_state_dict`. Confirm with
   a smoke run of `build_birefnet(cfg)`.
3. **LoRA adapter checkpoints must still load.** If `LoRABiRefNet` module
   structure changes (F-05 deletion, F-09 restructuring), existing
   `last.pth` files produced before the refactor may break. This is
   acceptable — note it in the PR description — but verify that a fresh
   training → save → load cycle completes end-to-end.
4. **Public API stability is NOT required** — this is a solo research
   project with no external consumers. Callers can be updated in lockstep.
5. **Vendored BiRefNet internals beyond the listed findings are out of
   scope.** Resist the temptation to clean up adjacent code.
6. **No backwards-compatibility shims.** Per project convention, delete
   cleanly; do not add `# removed` markers, re-exports, or deprecation stubs.
7. **Group A and Group B can be interleaved.** F-09 and F-10 in particular
   should be implemented together because they touch the same return-path
   plumbing; F-09, F-10, and F-15 together cover the entire train-mode
   forward / loss contract.
8. **trainer.py rewrite is in flight** (per project memory). The plan must
   not conflict with ongoing trainer changes — coordinate via TodoWrite or
   hold F-15 if it materially overlaps.

## Success Criteria

After the refactor, re-running the Feynman reconstruction on the affected
units should **no longer get stuck at the failure points** identified above.
Specifically:

- **F-08**: Reading `build_lora_birefnet` should make it immediately obvious
  whether the returned model has loaded adapters or not. No caller can
  silently get random adapters at inference time.
- **F-09**: `LoRABiRefNet.forward` has a single, documented return shape (or
  two physically separate methods with unambiguous names). No caller needs a
  mode-conditional unpack.
- **F-15**: `aux_loss` and `seg_loss` are normalized on the same basis, so
  `lambda_aux` has a stable meaning independent of GDT level count.
- **F-03**: `grep "TrainDataset\|ValidDataset"` returns no matches.
- **F-04**: `grep "from src.utils.io"` returns no matches; `src/utils/io.py`
  does not exist.
- **F-05**: `grep "self.size" src/ml/model/lora/wrapper.py` returns no
  matches.
- **F-10**: `grep "class_preds" src/` returns no matches.
- **F-11**: `grep '"add"' src/ml/model/birefnet/birefnet.py` returns no
  matches related to `mul_scl_ipt`. The parameter is either deleted or
  validated to equal `"cat"`.
- **F-14**: `build_trainer` type hint reads `model: LoRABiRefNet`.
- **Gate**: `pytest tests/ -v` passes, `python -m compileall src run_train.py`
  passes, and `python run_train.py` with `train.steps=10` produces a fresh
  `run/<timestamp>/` directory without error.
