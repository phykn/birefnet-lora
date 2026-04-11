# BiRefNet LoRA Fine-Tuning

LoRA fine-tuning pipeline for [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).
The project keeps base weights frozen and trains lightweight adapters on the Swin backbone and decoder modules.

## What You Get
- LoRA adapters on all backbone `Linear` layers and decoder `Conv2d` layers (deformable geometry convs excluded); only adapter params are optimized.
- Paired image/mask dataset with seeded (42) train/valid split — `valid` takes the first `split_ratio` slice, `train` the rest. Multi-scale supervision via `data.scales`, ImageNet normalization, AMP on CUDA.
- Per-run artifacts in `run/<YYYYMMDD_HHMMSS>/`: config snapshot, train/valid CSVs, TensorBoard logs, and `last.pth` adapter weights (overwritten every `train.save_freq` steps; no resume-from-checkpoint).

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

## Inference
Wrap a base BiRefNet with trained adapter weights, then run `predict`:
```python
from src.ml.build import build_lora_birefnet_for_inference
from src.ml.inference.predict import predict

model = build_lora_birefnet_for_inference(cfg, checkpoint="run/<timestamp>/last.pth").cuda()
mask = predict(model, image)  # HxWx3 uint8 RGB → HxW uint8 mask
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
- `src/ml/build.py`: factory functions for model, data, trainer.
- `src/ml/model/lora/`: LoRA adapters and `LoRABiRefNet` wrapper.
- `src/config/`: `tune.yaml` (runtime) and `model.yaml` (architecture + LoRA).

## Reference
- [BiRefNet official repository](https://github.com/ZhengPeng7/BiRefNet)
