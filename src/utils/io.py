from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        parsed = yaml.safe_load(file_obj)

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return parsed


def save_yaml(data: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(data=data, stream=file_obj, sort_keys=False)
