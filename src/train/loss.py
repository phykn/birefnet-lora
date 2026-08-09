"""Compatibility exports for loss functions and the training objective."""

from .losses import BoundaryBCELoss as BoundaryBCELoss
from .losses import DiceLoss as DiceLoss
from .losses import GCELoss as GCELoss
from .losses import IoULoss as IoULoss
from .losses import SegmentationLoss as SegmentationLoss
from .objective import TrainLoss as TrainLoss
from .losses import make_band as make_band

__all__ = [
    "BoundaryBCELoss",
    "DiceLoss",
    "GCELoss",
    "IoULoss",
    "SegmentationLoss",
    "TrainLoss",
    "make_band",
]
