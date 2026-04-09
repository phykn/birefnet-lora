import csv
import json
import os

import torch

from src.build import build_birefnet, build_dataloaders, build_lora_birefnet, build_trainer
from src.utils.io import load_yaml, save_yaml
from src.utils.misc import ConfigDict


def save_split_csv(split_filenames: dict[str, list[str]], output_dir: str) -> None:
    for split_name in ["train", "valid"]:
        csv_path = f"{output_dir}/{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(["filename"])
            for filename in split_filenames.get(split_name, []):
                writer.writerow([filename])


def print_run_info(
    cfg: ConfigDict,
    split_filenames: dict[str, list[str]],
    model: torch.nn.Module,
) -> None:
    train_count = len(split_filenames.get("train", []))
    valid_count = len(split_filenames.get("valid", []))

    print("[Config]")
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    print(f"[Dataset] train={train_count}, valid={valid_count}")
    if hasattr(model, "stats"):
        total_params = model.stats.get("total", 0)
        trainable_params = model.stats.get("trainable", 0)
        ratio = (trainable_params / total_params) if total_params else 0.0
        print(
            "[LoRABiRefNet] "
            f"total={total_params:,}  "
            f"trainable={trainable_params:,}  "
            f"ratio={ratio:.2%}"
        )


def train() -> None:
    config_data = load_yaml(path="src/config/tune.yaml")
    cfg = ConfigDict(config_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = build_birefnet(cfg=cfg)
    lora_ckpt_path = cfg.get("lora", {}).get("ckpt", None)
    model = build_lora_birefnet(cfg=cfg, model=base_model, ckpt_path=lora_ckpt_path).to(device)
    train_loader, valid_loader, split_filenames = build_dataloaders(cfg=cfg)

    trainer = build_trainer(
        cfg=cfg,
        model=model,
        train_dl=train_loader,
        valid_dl=valid_loader,
    )

    print_run_info(cfg=cfg, split_filenames=split_filenames, model=model)

    save_yaml(data=cfg.to_dict(), path=f"{trainer.save_dir}/config.yaml")
    save_split_csv(split_filenames=split_filenames, output_dir=trainer.save_dir)

    trainer.train(
        steps=cfg.train.steps,
        val_freq=cfg.train.val_freq,
        save_freq=cfg.train.save_freq,
    )

    if hasattr(model, "save_adapters"):
        weights_dir = f"{trainer.save_dir}/weights"
        os.makedirs(weights_dir, exist_ok=True)
        model.save_adapters(f"{weights_dir}/model.pth")


if __name__ == "__main__":
    train()
