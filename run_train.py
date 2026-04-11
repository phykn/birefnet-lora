import csv
import shutil

import torch
from omegaconf import DictConfig, OmegaConf

from src.ml.build import build_birefnet, build_dl, build_lora_birefnet, build_trainer


CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _color_value(value) -> str:
    if isinstance(value, bool) or value is None:
        return f"{MAGENTA}{value}{RESET}"
    if isinstance(value, (int, float)):
        return f"{YELLOW}{value}{RESET}"
    return f"{GREEN}{value}{RESET}"


def _flatten(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            items.extend(_flatten(v, key))
        else:
            items.append((key, v))
    return items


def format_config(cfg: DictConfig) -> str:
    container = OmegaConf.to_container(cfg, resolve=True)
    width = shutil.get_terminal_size((100, 20)).columns
    lines: list[str] = []

    for section, body in container.items():
        header = f"{CYAN}{section}{RESET}: "
        plain_indent = " " * (len(section) + 2)
        pairs = _flatten(body) if isinstance(body, dict) else [("", body)]
        tokens = [
            f"{CYAN}{k}{RESET}={_color_value(v)}" if k else _color_value(v)
            for k, v in pairs
        ]

        current = header
        current_len = len(section) + 2
        first = True
        for tok, (k, v) in zip(tokens, pairs):
            tok_plain_len = (len(k) + 1 if k else 0) + len(str(v))
            sep = "" if first else ", "
            sep_len = 0 if first else 2
            if not first and current_len + sep_len + tok_plain_len > width:
                lines.append(current)
                current = plain_indent + tok
                current_len = len(plain_indent) + tok_plain_len
            else:
                current += sep + tok
                current_len += sep_len + tok_plain_len
            first = False
        lines.append(current)

    return "\n".join(lines)


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

    print(f"{BOLD}[Config]{RESET}")
    print(format_config(cfg))
    print(
        f"{BOLD}[Dataset]{RESET} train={YELLOW}{train_count}{RESET}, "
        f"valid={YELLOW}{valid_count}{RESET}"
    )
    print(
        f"{BOLD}[LoRABiRefNet]{RESET} total={YELLOW}{total:,}{RESET}  "
        f"trainable={YELLOW}{trainable:,}{RESET}  "
        f"ratio={YELLOW}{trainable / total:.2%}{RESET}"
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
    model = build_lora_birefnet(cfg=cfg, model=base_model)
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
    print(f"{BOLD}[Done]{RESET} saved to {GREEN}{trainer.save_dir}{RESET}")


if __name__ == "__main__":
    main()
