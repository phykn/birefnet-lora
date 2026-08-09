import os
import uuid
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: dict[str, Any], path: str | os.PathLike[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
