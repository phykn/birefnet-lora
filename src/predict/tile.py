"""Compatibility exports for tiled prediction."""

from .tiling import Tile as Tile
from .tiling import plan as plan
from .tiling import weigh as weigh

__all__ = ["Tile", "plan", "weigh"]
