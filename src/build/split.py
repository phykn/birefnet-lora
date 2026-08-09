"""Compatibility exports for data split ownership."""

from ..data.split import Groups as Groups
from ..data.split import Pair as Pair
from ..data.split import Splits as Splits
from ..data.split import load as load
from ..data.split import make as make
from ..data.split import pack as pack
from ..data.split import restore as restore
from ..data.split import save as save

__all__ = [
    "Groups",
    "Pair",
    "Splits",
    "load",
    "make",
    "pack",
    "restore",
    "save",
]
