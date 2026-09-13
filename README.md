# BiRefNet LoRA

Fine-tune BiRefNet with LoRA while keeping the original BiRefNet checkpoint compatible.

- Noise-robust GCE, Dice, boundary, and EMA-teacher losses
- Aspect-ratio-preserving input with valid-region masking
- Caller-selected grids such as `tiles=[1]`, `[2]`, or `[1, 3]`
- Logit-level cosine blending
- LoRA fusion at API startup

## Structure

- `src/data`: dataset discovery, split persistence, image I/O, and augmentation
- `src/prepare`: shared in-memory pixel conversion and fit/restore geometry
- `src/model`: base network and the normalized model-output contract
- `src/adapt`: LoRA layers, injection, overlays, and fusion
- `src/predict`: inference orchestration and tiling
- `src/train`: losses, training objective, metrics, validation, checkpoints, and the trainer
- `src/build`: model, data-loader, and trainer assembly only
- `src/serve`: HTTP schemas, codecs, and routes
- `src/config.py`: shared YAML loading, overrides, and saved run configuration

Use `src.train.loss` for loss formulas, `src.train.objective` for the combined
training objective, `src.train.schedule`, `src.train.validate`, and
`src.predict.tile`. Former internal compatibility modules have been removed.
`src.build.trainer` injects the deployment prediction function into the trainer;
training modules do not import the inference loop. HTTP app construction lives
in `src.serve.app`.

## Setup

```bash
pip install -r requirements.txt
```

Place the base checkpoint at `weight/BiRefNet-general-epoch_244.pth`. Put paired images and binary masks under `local_data/image` and `local_data/mask` using matching filename stems.

Configuration is in `config/model.yaml` and `config/train.yaml` (formerly
`tune.yaml`). Run commands from the project root; relative data and weight paths
keep that existing interpretation. `--config` selects a YAML file whose values
override the shared model defaults. No personal config is loaded automatically.

## Train

```bash
python run_train.py
python run_train.py --config config/train.yaml
python run_train.py --resume run/<run-id>/weights/last.train.pth
```

Runs are saved under a collision-safe `run/<run-id>` directory. The final
training step is always evaluated, so a run can produce best checkpoints even
when `steps` is not divisible by `val_freq`. The default loader uses two
persistent workers and pinned memory; use `num_workers: 0`,
`persistent_workers: false`, and `pin_memory: false` for a low-memory or
CPU-only setup.

Resume loads the run's saved configuration and split membership; `--config`
cannot be combined with `--resume`. Checkpoint filenames and overlay metadata
are unchanged. Resume restores model, optimizer, scheduler, AMP scaler, EMA,
step, best scores, and threshold. RNG and data-loader position are not saved,
so the exact sequence of samples and augmentations is not restored.

Use `notebooks/01_predict.ipynb` to compare the base and LoRA model paths.

## Serve

```bash
python run_api.py --host 0.0.0.0 --port 8000 --weight run/<run-id>/weights/best_boundary.overlay.pth
```

`POST /predict` accepts base64-encoded image bytes and returns a PNG mask. Output mode can be `binary` or `probability`; positive integers in `tiles` select N×N grids, and `overlap` sets their overlap ratio. Binary output with any grid other than 1 requires an explicit `threshold`. Requests are processed one at a time. New overlays store the training
preprocess `size` and `mode`, and both deployment validation and the API use
that same contract. Legacy overlays without this metadata retain the previous
`1024`/`rgb` behavior.

## Test

```bash
python -m pytest -q
```

The base network targets the upstream Swin-L / multi-scale concatenation /
ASPPDeformable configuration. Upstream comparison is separate from the default
tests and uses an explicitly supplied checkout and local weights:

```bash
python scripts/check_upstream.py --upstream <BiRefNet-checkout> --weights weight/BiRefNet-general-epoch_244.pth --device cuda
```

Compared against [upstream revision ebcc0bc8](https://github.com/ZhengPeng7/BiRefNet/tree/ebcc0bc8ec7fe919cec829f2dea656b3078acddc):
both implementations strictly load the same base checkpoint, with identical
FP32 outputs for 64x96 and 128x128 evaluation inputs and a 2x64x64 training batch
(BatchNorm frozen, matching this project's LoRA training). This checks model
calculation and output containers, not task accuracy on a real dataset.
Aspect-ratio fitting, valid-region losses, logit blending, EMA, and LoRA-specific
non-reentrant gradient checkpointing remain intentional local improvements.
LoRA injection preserves the base model's train/eval mode and each layer's
device/dtype. Boolean or fractional LoRA ranks and gradient accumulation counts
are rejected instead of silently converted to integers.
