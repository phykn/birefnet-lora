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

Older internal import paths such as `src.train.run` and `src.prepare.load`
remain as compatibility exports; new code should use the owner modules above.

## Setup

```bash
pip install -r requirements.txt
```

Place the base checkpoint at `weight/BiRefNet-general-epoch_244.pth`. Put paired images and binary masks under `local_data/image` and `local_data/mask` using matching filename stems.

Configuration is in `config/model.yaml` and `config/tune.yaml`.

## Train

```bash
python run_train.py
python run_train.py --resume run/<run-id>/weights/last.train.pth
```

Runs are saved under a collision-safe `run/<run-id>` directory. The final
training step is always evaluated, so a run can produce best checkpoints even
when `steps` is not divisible by `val_freq`. The default loader uses two
persistent workers and pinned memory; use `num_workers: 0`,
`persistent_workers: false`, and `pin_memory: false` for a low-memory or
CPU-only setup.

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
