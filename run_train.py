import csv

import torch
from omegaconf import DictConfig, OmegaConf

from src.ml.build import (
    build_birefnet,
    build_dl,
    build_lora_birefnet_for_training,
    build_trainer,
)


def save_split_csv(split_filenames: dict[str, list[str]], output_dir: str) -> None:
    for split in ["train", "valid"]:
        images = split_filenames[f"{split}_image"]
        masks = split_filenames[f"{split}_mask"]
        with open(f"{output_dir}/{split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            writer.writerows(zip(images, masks))


def print_run_info(
    cfg: DictConfig,
    split_filenames: dict[str, list[str]],
    model: torch.nn.Module,
) -> None:
    train_count = len(split_filenames["train_image"])
    valid_count = len(split_filenames["valid_image"])
    total = model.stats["total"]
    trainable = model.stats["trainable"]

    print("[Config]")
    print(OmegaConf.to_yaml(cfg))
    print(f"[Dataset] train={train_count}, valid={valid_count}")
    print(
        f"[LoRABiRefNet] total={total:,}  trainable={trainable:,}  "
        f"ratio={trainable / total:.2%}"
    )


def main() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load("src/config/tune.yaml"),
        OmegaConf.load("src/config/model.yaml"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    base_model = build_birefnet(cfg=cfg).to(device)
    model = build_lora_birefnet_for_training(cfg=cfg, model=base_model)
    train_loader, valid_loader, split_filenames = build_dl(cfg=cfg)
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        train_dl=train_loader,
        valid_dl=valid_loader,
    )

    print_run_info(cfg=cfg, split_filenames=split_filenames, model=model)
    OmegaConf.save(cfg, f"{trainer.save_dir}/config.yaml")
    save_split_csv(split_filenames=split_filenames, output_dir=trainer.save_dir)

    trainer.train(
        steps=cfg.train.steps,
        val_freq=cfg.train.val_freq,
        save_freq=cfg.train.save_freq,
    )
    print(f"[Done] saved to {trainer.save_dir}")


if __name__ == "__main__":
    main()
