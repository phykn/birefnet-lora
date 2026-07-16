# BiRefNet LoRA

Fine-tune BiRefNet with LoRA while keeping the original BiRefNet checkpoint compatible.

- Noise-robust GCE, Dice, boundary, and EMA-teacher losses
- Aspect-ratio-preserving input with valid-region masking
- Optional 2×2 or 3×3 tiling through `predict(..., tile=True)`
- Logit-level cosine blending

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

Runs are saved under `run/<run-id>`. Use `notebooks/01_predict.ipynb` to compare the base and LoRA model paths.

## Serve

```bash
python run_api.py --host 0.0.0.0 --port 8000 --weight run/<run-id>/weights/best_boundary.overlay.pth
```

`POST /predict` accepts base64-encoded image bytes and returns a PNG mask. Output mode can be `binary` or `probability`. Requests are processed one at a time.

## Test

```bash
python -m pytest -q
```
