"""Compatibility exports for data image I/O."""

from ..data.image import read_image as read_image
from ..data.image import read_mask as read_mask

__all__ = ["read_image", "read_mask"]
