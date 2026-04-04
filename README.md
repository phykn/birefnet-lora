# BiRefNet LoRA Fine-Tuning

A clean, modular, and efficient pipeline for fine-tuning [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) using **Low-Rank Adaptation (LoRA)**. By applying LoRA to the Swin Transformer backbone and the decoder, this project allows for memory-efficient adaptation with significantly fewer trainable parameters while keeping baseline weights frozen.

## Features
- **Efficient Fine-Tuning**: Integrates LoRA (`LoRALinear` & `LoRAConv2d`).
- **Data Pipeline**: Built-in augmentations via Albumentations, automatic train/valid splitting.
- **Tracking**: Automatically saves configs, dataset splits (`train.csv`, `valid.csv`), and TensorBoard logs per run.

## Setup & Training

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Configuration
Prepare your data (images and masks with matching filenames) and pre-trained weights. Update `src/config/tune.yaml` accordingly:
```yaml
data:
  img_dir: "data/image"
  mask_dir: "data/mask"
  size: [1024, 1024]
birefnet:
  weight: "weight/BiRefNet-general-epoch_244.pth"
```

### 3. Run
```bash
python train.py
```

Outputs containing checkpoints, logs, and configs will be automatically saved to `run/<timestamp>/`.

## Validation Checklist
Run the following commands before committing:

```bash
python -m compileall src train.py
pytest -q
ruff check .
```

## References
- [BiRefNet Official Repository](https://github.com/ZhengPeng7/BiRefNet)
