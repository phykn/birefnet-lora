# BiRefNet LoRA Fine-Tuning

LoRA fine-tuning pipeline for [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).
The project keeps base weights frozen and trains lightweight adapters on the Swin backbone and decoder modules.

## What You Get
- LoRA modules for linear and convolution layers.
- Dataset pipeline with Albumentations and train/valid split export.
- Per-run artifacts in `run/<YYYYMMDD_HHMMSS>/` (config, CSV splits, logs, adapter weights).

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Place data and pretrained weights:
- images: `data/image/`
- masks: `data/mask/`
- checkpoint: `weight/BiRefNet-general-epoch_244.pth` (or your own)
3. Update `src/config/tune.yaml`:
```yaml
data:
  img_dir: "data/image"
  mask_dir: "data/mask"
  size: [1024, 1024]
birefnet:
  weight: "weight/BiRefNet-general-epoch_244.pth"
```
4. Start training:
```bash
python train.py
```

## Monitoring
```bash
tensorboard --logdir run
```

## Validation Checklist
```bash
python -m compileall src train.py
```

Optional smoke test:
1. Reduce `train.steps` in `src/config/tune.yaml` (for example, `10`).
2. Run `python train.py`.
3. Confirm a new `run/<timestamp>/` directory is created.

## Project Layout
- `train.py`: training entrypoint.
- `src/models/`: BiRefNet architecture and backbones.
- `src/finetune/`: LoRA adapters, loss, and trainer.
- `src/data/`: datasets and augmentations.
- `src/config/tune.yaml`: runtime configuration.

## Reference
- [BiRefNet official repository](https://github.com/ZhengPeng7/BiRefNet)
