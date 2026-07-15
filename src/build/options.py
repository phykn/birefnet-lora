from typing import Any


def get(node: Any, key: str, default: Any) -> Any:
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)
