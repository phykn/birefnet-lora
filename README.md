# BiRefNet LoRA

Retrain BiRefNet with LoRA adapters and trainable output/GDT heads while keeping the original BiRefNet checkpoint compatible.

- RGB letterbox input with a valid-region mask
- BCEWithLogits, Dice, and boundary BCE losses
- 2×2 or 3×3 tiling with at least 33% overlap
- Logit-level cosine blending

## Setup

```bash
pip install -r requirements.txt
```

Place the base checkpoint and paired data as follows. Image and mask filenames must use the same stem.

```text
weight/BiRefNet-general-epoch_244.pth
local_data/image/sample.png
local_data/mask/sample.png
```

Training options are in `config/tune.yaml`. Model and LoRA options are in `config/model.yaml`.

## Training

```bash
python run_train.py
```

Checkpoints and run metadata are written to `run/<run-id>/`. The `weights` directory contains deployment overlays, resume state, and the best region/boundary checkpoints.

## API

```bash
python run_api.py --host 0.0.0.0 --port 8000 --weight run/<run-id>/weights/best_boundary.overlay.pth
```

Send an image to `POST /predict` as standard base64-encoded file bytes.

```json
{
  "id": "sample",
  "base64_str": "<base64-encoded image bytes>",
  "output_mode": "binary",
  "threshold": 0.5
}
```

`binary` returns a `0/255 uint8` PNG. `probability` returns a `0-255 uint8` probability PNG. Threshold priority is request, checkpoint, then `0.5`.

Requests are processed one at a time. Inputs are limited to 32 MiB and 36 million pixels. `GET /health` returns the service status and device.

## Tests

```bash
python -m pytest -q
```
