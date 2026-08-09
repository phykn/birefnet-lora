"""Compatibility exports for prediction."""

from .inference import OutputMode as OutputMode
from .inference import predict as predict
from .inference import predict_logits as predict_logits

__all__ = ["OutputMode", "predict", "predict_logits"]
