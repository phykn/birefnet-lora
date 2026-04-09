import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.build import build_birefnet, build_lora_birefnet
from src.data.dataset import MEAN, STD
from src.utils.io import load_yaml
from src.utils.misc import ConfigDict

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image(path: str, size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(image)
    tensor = TF.resize(tensor, list(size), antialias=True)
    tensor = TF.normalize(tensor, mean=MEAN, std=STD)
    return tensor.unsqueeze(0)


def save_mask(tensor: torch.Tensor, path: str, original_size: tuple[int, int]) -> None:
    mask = tensor.squeeze(0).squeeze(0).sigmoid()
    mask = TF.resize(mask.unsqueeze(0), list(original_size), antialias=True).squeeze(0)
    mask = (mask * 255).byte().cpu().numpy()
    Image.fromarray(mask, mode="L").save(path)


def predict() -> None:
    parser = argparse.ArgumentParser(description="Run inference with LoRA-adapted BiRefNet")
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--config", "-c", default="src/config/tune.yaml", help="Config file")
    parser.add_argument("--adapter", "-a", required=True, help="Path to LoRA adapter weights")
    args = parser.parse_args()

    config_data = load_yaml(path=args.config)
    cfg = ConfigDict(config_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = tuple(cfg.data.size)

    base_model = build_birefnet(cfg=cfg)
    model = build_lora_birefnet(cfg=cfg, model=base_model, ckpt_path=args.adapter).to(device)
    model.eval()

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(
            p for p in input_path.glob("*")
            if p.is_file() and p.suffix.lower() in EXTS
        )

    os.makedirs(args.output, exist_ok=True)

    for image_path in image_paths:
        original_image = Image.open(image_path).convert("RGB")
        original_size = (original_image.height, original_image.width)

        image_tensor = load_image(str(image_path), size).to(device)

        with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            prediction = model(image_tensor)

        output_path = os.path.join(args.output, f"{image_path.stem}_mask.png")
        save_mask(prediction, output_path, original_size)
        print(f"{image_path.name} -> {output_path}")


if __name__ == "__main__":
    predict()
