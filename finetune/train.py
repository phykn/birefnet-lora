"""
BiRefNet 어댑터 파인튜닝 학습 스크립트.

원본 파라미터는 완전히 freeze 되어 있으며, 어댑터 파라미터만 학습한다.

사용법:
    python finetune/train.py \\
        --image_dir data/images \\
        --mask_dir  data/masks \\
        --epochs 10 \\
        --batch_size 4 \\
        --lr 1e-4 \\
        --lora_rank 4
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from model import FineTuneBiRefNet
from dataset import FineTuneDataset
from loss import SegmentationLoss


def parse_args():
    parser = argparse.ArgumentParser(description="BiRefNet Adapter Fine-tuning")
    # 데이터
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, nargs=2, default=[1024, 1024],
                        help="width height")
    # 학습
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_gdt", type=float, default=1.0,
                        help="Gradient Reference Loss 가중치")
    # 어댑터
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--adapter_reduction", type=int, default=4)
    # 저장
    parser.add_argument("--save_dir", type=str, default="ckpts")
    parser.add_argument("--resume", type=str, default=None,
                        help="어댑터 체크포인트 경로")
    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. Dataset ──
    dataset = FineTuneDataset(
        args.image_dir, args.mask_dir,
        size=tuple(args.img_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, args.batch_size),
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # ── 2. Model (어댑터 삽입 + 원본 freeze) ──
    model = FineTuneBiRefNet(
        pretrained=True,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        adapter_reduction=args.adapter_reduction,
    ).to(device)

    if args.resume:
        model.load_adapters(args.resume)

    # ── 3. Loss & Optimizer (어댑터 파라미터만) ──
    criterion = SegmentationLoss()
    optimizer = AdamW(
        model.get_adapter_params(),
        lr=args.lr,
        weight_decay=1e-2,
    )

    # ── 4. Train Loop ──
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward: 이미지 → 마스크
            preds = model(images)

            # Loss = 세그먼테이션 + Gradient Reference
            loss_seg = criterion(preds, masks)
            loss_gdt = model.aux_loss * args.lambda_gdt
            total_loss = loss_seg + loss_gdt

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix(
                seg=f"{loss_seg.item():.4f}",
                gdt=f"{loss_gdt.item():.4f}",
            )

        avg = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} — Avg Loss: {avg:.4f}")

        # 어댑터 가중치만 저장 (원본 가중치 제외, 용량 절약)
        model.save_adapters(os.path.join(args.save_dir, f"adapter_ep{epoch}.pth"))


if __name__ == "__main__":
    train(parse_args())
