# BiRefNet LoRA Fine-Tuning

LoRA fine-tuning pipeline for [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).
The project keeps base weights frozen and trains lightweight adapters on the Swin backbone and decoder modules.

## What You Get
- LoRA modules for linear and convolution layers.
- Dataset pipeline with train/valid split export.
- Per-run artifacts in `run/<YYYYMMDD_HHMMSS>/` (config, CSV splits, logs, adapter weights).

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
- `src/ml/model/birefnet/`: BiRefNet architecture (backbones, blocks).
- `src/ml/model/lora/`: LoRA adapters and `LoRABiRefNet` wrapper.
- `src/ml/training/`: trainer, loss, and scheduler.
- `src/ml/data/`: datasets, augmentations, and preprocessing.
- `src/ml/inference/`: prediction entry point.
- `src/config/`: `tune.yaml` (runtime) and `model.yaml` (architecture + LoRA).

## Reference
- [BiRefNet official repository](https://github.com/ZhengPeng7/BiRefNet)
