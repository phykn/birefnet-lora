import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml.model.lora.wrapper import ModelOutput


class IoULoss(nn.Module):
    """Expects `pred` to be probabilities in [0, 1] (already sigmoid'd)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, pred.ndim))
        intersection = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
        return (1 - intersection / (union + 1e-6)).mean()


class SegmentationLoss(nn.Module):
    def __init__(self, lambda_bce: float = 30.0, lambda_iou: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.iou = IoULoss()
        self.lambda_bce = lambda_bce
        self.lambda_iou = lambda_iou

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp(0, 1)
        if pred.shape[2:] != target.shape[2:]:
            target = F.interpolate(target, size=pred.shape[2:], mode="nearest")
        bce = self.bce(pred, target) * self.lambda_bce
        iou = self.iou(pred.sigmoid(), target) * self.lambda_iou
        return bce + iou


class SymmetricBinaryKLLoss(nn.Module):
    def __init__(self, lambda_kl: float = 1.0):
        super().__init__()
        self.lambda_kl = lambda_kl

    @staticmethod
    def _binary_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(p_logits)
        log_p, log_1mp = F.logsigmoid(p_logits), F.logsigmoid(-p_logits)
        log_q, log_1mq = F.logsigmoid(q_logits), F.logsigmoid(-q_logits)
        return p * (log_p - log_q) + (1 - p) * (log_1mp - log_1mq)

    def forward(self, logits_1: torch.Tensor, logits_2: torch.Tensor) -> torch.Tensor:
        assert logits_1.shape == logits_2.shape, (
            f"SymmetricBinaryKLLoss expects matching shapes, got "
            f"{tuple(logits_1.shape)} vs {tuple(logits_2.shape)}"
        )
        kl = self._binary_kl(logits_1, logits_2) + self._binary_kl(logits_2, logits_1)
        return self.lambda_kl * 0.5 * kl.mean()


class MAELoss(nn.Module):
    def __init__(self, lambda_mae: float = 1.0):
        super().__init__()
        self.lambda_mae = lambda_mae

    def forward(self, logits_1: torch.Tensor, logits_2: torch.Tensor) -> torch.Tensor:
        m_1 = torch.sigmoid(logits_1).mean()
        m_2 = torch.sigmoid(logits_2).mean()
        return self.lambda_mae * (m_1 - m_2).abs()


class CustomLoss(nn.Module):
    def __init__(
        self,
        lambda_bce: float = 30.0,
        lambda_iou: float = 0.5,
        lambda_kl: float = 1.0,
        lambda_aux: float = 1.0,
        lambda_mae: float = 1.0,
    ):
        super().__init__()
        self.seg = SegmentationLoss(lambda_bce=lambda_bce, lambda_iou=lambda_iou)
        self.con = SymmetricBinaryKLLoss(lambda_kl=lambda_kl)
        self.mae = MAELoss(lambda_mae=lambda_mae)
        self.lambda_aux = lambda_aux

    def forward(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = next(model.parameters()).device
        image_1 = batch["image_1"].to(device)
        mask = batch["mask"].to(device)

        if model.training:
            image_2 = batch["image_2"].to(device)
            batch_size = image_1.shape[0]
            images = torch.cat([image_1, image_2], dim=0)
            masks = torch.cat([mask, mask], dim=0)
            out: ModelOutput = model(images)
            preds = out.preds

            logits_1 = preds[-1][:batch_size]
            logits_2 = preds[-1][batch_size:]

            seg_loss = sum(self.seg(p, masks) for p in preds) / len(preds)
            con_loss = self.con(logits_1, logits_2)
            mae_loss = self.mae(logits_1, logits_2)
            aux_loss = self.lambda_aux * (
                out.aux if out.aux is not None else torch.tensor(0.0, device=device)
            )

            loss = seg_loss + con_loss + mae_loss + aux_loss
            loss_dict = {
                "loss": loss,
                "seg": seg_loss,
                "con": con_loss,
                "mae": mae_loss,
                "aux": aux_loss,
            }
            return loss_dict, loss

        out = model(image_1)
        pred = out.preds[-1]
        seg_loss = self.seg(pred, mask)
        return {"seg": seg_loss}, seg_loss
