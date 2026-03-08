import csv
import os
import torch

from src.build import build_dl, build_lora_birefnet, build_trainer
from src.utils.io import load_yaml, save_yaml
from src.utils.misc import ConfigDict


def save_data_csv(
    data: dict[str, list[str]],
    wdir: str
) -> None:
    for split in ["train", "valid"]:
        path = f"{wdir}/{split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename"])
            for name in data.get(split, []):
                writer.writerow([name])


def train() -> None:
    data = load_yaml(path="src/config/finetune.yaml")
    cfg = ConfigDict(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_lora_birefnet(cfg=cfg).to(device)
    train_dl, valid_dl, data = build_dl(cfg=cfg)

    trainer = build_trainer(
        cfg = cfg,
        model = model,
        train_dl = train_dl,
        valid_dl = valid_dl,
    )

    save_yaml(data=cfg.to_dict(), path=f"{trainer.save_dir}/config.yaml")
    save_data_csv(data=data, wdir=trainer.save_dir)

    trainer.train(
        steps = cfg.train.steps,
        val_freq = cfg.train.val_freq,
        save_freq = cfg.train.save_freq,
    )

    if hasattr(model, "save_adapters"):
        wdir = f"{trainer.save_dir}/weights"
        os.makedirs(wdir, exist_ok=True)
        model.save_adapters(f"{wdir}/model.pth")


if __name__ == "__main__":
    train()