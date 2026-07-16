import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.build.data import build as build_data
from src.build.model import adapt
from src.build.model import build as build_model
from src.build.split import load as load_splits
from src.build.split import save as save_splits
from src.build.trainer import build as build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.merge(
        OmegaConf.load("config/tune.yaml"),
        OmegaConf.load("config/model.yaml"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = build_model(cfg).to(device)
    model = adapt(cfg, base)
    saved_splits = None
    if args.resume:
        run_dir = Path(args.resume).resolve().parent.parent
        saved_splits = load_splits(run_dir)
    train_loader, valid_loader, calib_loader, splits = build_data(
        cfg,
        saved_splits,
    )
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
    save_splits(splits, trainer.save_dir)

    trainer.train(
        steps=cfg.train.steps,
        val_freq=cfg.train.val_freq,
        save_freq=cfg.train.save_freq,
    )
    print(f"\nTraining finished. Weights saved to: {trainer.save_dir}")


if __name__ == "__main__":
    main()
