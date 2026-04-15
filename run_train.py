import csv
import shutil
from itertools import cycle
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.ml.build import (
    build_birefnet,
    build_dl,
    build_lora_birefnet_for_training,
    build_trainer,
)

RESET = "\033[0m"
BOLD = "\033[1m"
COLORS = ["\033[96m", "\033[92m", "\033[93m", "\033[95m", "\033[94m", "\033[91m"]


def print_cfg(cfg: DictConfig) -> None:
    def walk(node: Any, path: str = "") -> list[str]:
        if isinstance(node, dict):
            return [
                p
                for k, v in node.items()
                for p in walk(v, f"{path}.{k}" if path else k)
            ]
        return [f"{path}={node}"]

    container = OmegaConf.to_container(cfg, resolve=True)
    groups = [(k, walk(v, k)) for k, v in container.items()]
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    col_w = max(len(p) for _, pairs in groups for p in pairs) + 2
    ncols = max(1, width // col_w)

    for (name, pairs), color in zip(groups, cycle(COLORS)):
        print(f"{color}{BOLD}[{name}]{RESET}")
        for i in range(0, len(pairs), ncols):
            row = pairs[i : i + ncols]
            print(color + "".join(p.ljust(col_w) for p in row) + RESET)


def save_split_csv(split_filenames: dict[str, list[str]], output_dir: str) -> None:
    for split in ["train", "valid"]:
        with open(f"{output_dir}/{split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            writer.writerows(
                zip(split_filenames[f"{split}_image"], split_filenames[f"{split}_mask"])
            )


def main() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load("src/config/tune.yaml"),
        OmegaConf.load("src/config/model.yaml"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = build_birefnet(cfg=cfg).to(device)
    model = build_lora_birefnet_for_training(cfg=cfg, model=base_model)
    train_loader, valid_loader, split_filenames = build_dl(cfg=cfg)
    trainer = build_trainer(
        cfg=cfg, model=model, train_dl=train_loader, valid_dl=valid_loader
    )

    print_cfg(cfg)
    n_train = len(split_filenames["train_image"])
    n_valid = len(split_filenames["valid_image"])
    total, trainable = model.stats["total"], model.stats["trainable"]
    print(f"\n[Dataset] train={n_train}, valid={n_valid}")
    print(
        f"[LoRABiRefNet] total={total:,}  trainable={trainable:,}  "
        f"ratio={trainable / total:.2%}"
    )

    OmegaConf.save(cfg, f"{trainer.save_dir}/config.yaml")
    save_split_csv(split_filenames=split_filenames, output_dir=trainer.save_dir)

    trainer.train(
        steps=cfg.train.steps,
        val_freq=cfg.train.val_freq,
        save_freq=cfg.train.save_freq,
    )
    print(f"\n{BOLD}Training finished. Weights saved to: {trainer.save_dir}{RESET}")


if __name__ == "__main__":
    main()
