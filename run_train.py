import argparse
import csv

import torch
from omegaconf import OmegaConf

from src.build.data import build_loaders
from src.build.model import build_base, wrap_lora
from src.build.train import build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    return parser.parse_args()


def save_splits(splits: dict[str, list[str]], output_dir: str) -> None:
    for split in ["train", "valid", "calib"]:
        with open(f"{output_dir}/{split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            writer.writerows(
                zip(splits[f"{split}_image"], splits[f"{split}_mask"])
            )


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.load("config/tune.yaml"),
        OmegaConf.load("config/model.yaml"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = build_base(cfg).to(device)
    model = wrap_lora(cfg, base)
    train_loader, valid_loader, calib_loader, splits = build_loaders(cfg)
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        calib_loader=calib_loader,
    )
    if args.resume:
        trainer.load_resume(args.resume)

    n_train = len(splits["train_image"])
    n_valid = len(splits["valid_image"])
    n_calib = len(splits["calib_image"])
    total, trainable = model.stats["total"], model.stats["trainable"]
    print(f"\n[Dataset] train={n_train}, valid={n_valid}, calib={n_calib}")
    print(
        f"[LoRABiRefNet] total={total:,}  trainable={trainable:,}  "
        f"ratio={trainable / total:.2%}"
    )

    OmegaConf.save(cfg, f"{trainer.save_dir}/config.yaml")
    save_splits(splits=splits, output_dir=trainer.save_dir)

    trainer.train(
        steps=cfg.train.steps,
        val_freq=cfg.train.val_freq,
        save_freq=cfg.train.save_freq,
    )
    print(f"\nTraining finished. Weights saved to: {trainer.save_dir}")


if __name__ == "__main__":
    main()
