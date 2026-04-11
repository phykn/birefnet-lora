# BiRefNet LoRA Fine-Tuning

LoRA fine-tuning pipeline for [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).
The project keeps base weights frozen and trains lightweight adapters on the Swin backbone and decoder modules.

## What You Get
- LoRA adapters on all backbone `Linear` layers and decoder `Conv2d` layers (deformable geometry convs excluded); only adapter params are optimized.
- Paired image/mask dataset with seeded (42) train/valid split — `valid` takes the first `split_ratio` slice, `train` the rest. Multi-scale supervision via `data.scales`, ImageNet normalization, AMP on CUDA (bf16 on Ampere+, fp16 on Turing and older — auto-detected by hardware).
- Per-run artifacts in `run/<YYYYMMDD_HHMMSS>/`: config snapshot, train/valid CSVs, TensorBoard logs, and `last.pth` adapter weights (overwritten every `train.save_freq` steps; no resume-from-checkpoint).
- FastAPI inference server (`run_api.py`) with `GET /health` and `POST /predict` (imstr-encoded JSON I/O) — see the **Inference API** section below.

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Place data and pretrained weights:
- images: `local_data/image/`
- masks: `local_data/mask/`
- checkpoint: `weight/BiRefNet-general-epoch_244.pth` (or your own)
3. Review configuration in `src/config/`:
- `tune.yaml` — `data`, `augment`, `dl`, `train`, `loss` (paths, augmentation, batch, schedule, loss weights)
- `model.yaml` — `birefnet` (architecture + pretrained `weight`) and `lora` (`rank`, `alpha`)
4. Start training:
```bash
python run_train.py
```

## Monitoring
```bash
tensorboard --logdir run
```

## Inference (Python)
Wrap a base BiRefNet with trained adapter weights, then run `predict`:
```python
from src.ml.build import build_lora_birefnet_for_inference
from src.ml.inference.predict import predict

model = build_lora_birefnet_for_inference(cfg, checkpoint="run/<timestamp>/last.pth").cuda()
mask = predict(model, image)  # HxWx3 uint8 RGB → HxW uint8 mask
```

`predict()` auto-enables bf16/fp16 autocast on CUDA (bf16 on Ampere+, fp16 elsewhere) — typically a 30–40% latency saving at fp32 quality for segmentation masks.

## Inference API
Launch the FastAPI server against a trained adapter checkpoint:
```bash
python run_api.py --host 0.0.0.0 --port 8000 --weight run/<timestamp>/last.pth
# optional: --max-concurrency N   (default 2)
```

**`GET /health`** → `{"status": "ok", "device": "cpu" | "cuda"}`

**`POST /predict`** — JSON body:
```json
{
  "image": {
    "base64_str": "<imstr.encode(np.ndarray)>",
    "height": 720,
    "width": 1280,
    "channels": 3
  },
  "threshold": 0.5
}
```
- `image.base64_str` is required; the dimension fields are optional metadata.
- `threshold ∈ [0, 1]` returns a binary `0/255` mask.
- `threshold < 0` (e.g. `-1`) returns a probability map as `uint8` (`prob * 255`).

Response:
```json
{
  "mask": {
    "base64_str": "<imstr str>",
    "height": 720,
    "width": 1280,
    "channels": null
  }
}
```

GPU concurrency is capped by an `asyncio.Semaphore(max_concurrency)`; `torch.cuda.empty_cache()` is called automatically after each prediction on CUDA to prevent allocator fragmentation.

Minimal client example:
```python
import numpy as np, requests
from imstr import encode, decode

img = np.array(...)  # HxWx3 uint8 RGB
payload = {
    "image": {
        "base64_str": encode(img),
        "height": img.shape[0],
        "width": img.shape[1],
        "channels": img.shape[2],
    },
    "threshold": 0.5,
}
r = requests.post("http://localhost:8000/predict", json=payload)
mask = decode(r.json()["mask"]["base64_str"])
```

## Validation Checklist
```bash
python -m compileall src run_train.py
pytest tests/ -v
```

Optional smoke test:
1. Reduce `train.steps` in `src/config/tune.yaml` (for example, `10`).
2. Run `python run_train.py` and confirm a new `run/<timestamp>/` directory is created.

## Project Layout
- `run_train.py`: training entrypoint.
- `run_api.py`: inference API entrypoint and `build_app` factory.
- `src/ml/build.py`: factory functions for model, data, trainer.
- `src/ml/model/lora/`: LoRA adapters and `LoRABiRefNet` wrapper.
- `src/ml/inference/predict.py`: shared inference call with auto autocast.
- `src/api/`: FastAPI routes and Pydantic schemas (`routes.py`, `schema.py`).
- `src/config/`: `tune.yaml` (runtime) and `model.yaml` (architecture + LoRA).

## Reference
- [BiRefNet official repository](https://github.com/ZhengPeng7/BiRefNet)
