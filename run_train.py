import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.build.data import build as build_data
from src.build.model import adapt
from src.build.model import build as build_model
from src.build.split import Splits
from src.build.split import load as load_splits
from src.build.split import save as save_splits
from src.build.trainer import build as build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    return parser.parse_args()


def load_run(
    resume: str | None,
) -> tuple[DictConfig, Path | None, Path | None, Splits | None]:
    if resume is None:
        cfg = OmegaConf.merge(
            OmegaConf.load("config/tune.yaml"),
            OmegaConf.load("config/model.yaml"),
        )
        return cfg, None, None, None

    checkpoint = Path(resume).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    if checkpoint.parent.name != "weights":
        raise ValueError("Resume checkpoint must be inside a run weights directory")

    run_dir = checkpoint.parent.parent
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    return cfg, checkpoint, run_dir, load_splits(run_dir)


def main() -> None:
    args = parse_args()
    cfg, checkpoint, run_dir, saved_splits = load_run(args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = build_model(cfg).to(device)
    model = adapt(cfg, base)
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
        save_dir=run_dir,
    )
    if checkpoint is not None:
        trainer.load_resume(str(checkpoint))

    n_train = len(splits["train_image"])
    n_valid = len(splits["valid_image"])
    n_calib = len(splits["calib_image"])
    total, trainable = model.stats["total"], model.stats["trainable"]
    print(f"\n[Dataset] train={n_train}, valid={n_valid}, calib={n_calib}")
    print(
        f"[LoRABiRefNet] total={total:,}  trainable={trainable:,}  "
        f"ratio={trainable / total:.2%}"
    )

    if run_dir is None:
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
