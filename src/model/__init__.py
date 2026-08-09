from typing import TYPE_CHECKING, Any

from .output import Output as Output

if TYPE_CHECKING:
    from .net import BiRefNet as BiRefNet

__all__ = ["BiRefNet", "Output"]


def __getattr__(name: str) -> Any:
    if name == "BiRefNet":
        from .net import BiRefNet

        globals()[name] = BiRefNet
        return BiRefNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
