import os
import uuid
from pathlib import Path
from typing import Any

import torch


def _save(payload: dict[str, Any], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


class OverlayMixin:
    def _list_keys(self) -> set[str]:
        return {name for name, param in self.named_parameters() if param.requires_grad}

    def make_meta(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "lora": {
                "rank": self.rank,
                "alpha": self.alpha,
            },
            "trainable_heads": list(self.trainable_heads),
        }
        if extra:
            meta.update(extra)
        return meta

    def make_overlay(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        expected_keys = self._list_keys()
        full_state = self.state_dict()
        missing = sorted(expected_keys - set(full_state))
        if missing:
            raise RuntimeError(f"Missing trainable state keys: {missing[:5]}")
        state = {key: full_state[key].detach().cpu() for key in sorted(expected_keys)}
        meta = self.make_meta(extra)
        return {
            "meta": meta,
            "state": state,
        }

    def save_overlay(self, path: str, extra: dict[str, Any] | None = None) -> None:
        _save(self.make_overlay(extra), path)

    def _check_meta(self, meta: dict[str, Any]) -> None:
        lora = meta.get("lora", {})
        if (
            int(lora.get("rank", -1)) != self.rank
            or float(lora.get("alpha", float("nan"))) != self.alpha
        ):
            raise RuntimeError("Overlay LoRA rank/alpha do not match the model.")

        heads = tuple(sorted(meta.get("trainable_heads", [])))
        if heads != self.trainable_heads:
            raise RuntimeError(
                "Overlay trainable-head allowlist does not match the model: "
                f"expected={self.trainable_heads}, loaded={heads}."
            )

    def load_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError("Unsupported overlay checkpoint format.")

        meta = payload.get("meta")
        state = payload.get("state")
        if not isinstance(meta, dict) or not isinstance(state, dict):
            raise RuntimeError("Overlay checkpoint is missing meta/state.")
        self._check_meta(meta)

        expected_keys = self._list_keys()
        loaded_keys = set(state)
        missing = sorted(expected_keys - loaded_keys)
        unexpected = sorted(loaded_keys - expected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "Overlay state keys do not match the model: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}."
            )

        current = self.state_dict()
        with torch.no_grad():
            for key in sorted(expected_keys):
                loaded = state[key]
                if not isinstance(loaded, torch.Tensor):
                    raise RuntimeError(f"Overlay state {key!r} is not a tensor.")
                if tuple(loaded.shape) != tuple(current[key].shape):
                    raise RuntimeError(
                        f"Overlay shape mismatch for {key}: "
                        f"expected={tuple(current[key].shape)}, "
                        f"loaded={tuple(loaded.shape)}."
                    )
                current[key].copy_(
                    loaded.to(device=current[key].device, dtype=current[key].dtype)
                )

        self.loaded_meta = meta
        return meta

    def load_overlay(self, path: str) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return self.load_payload(payload)
