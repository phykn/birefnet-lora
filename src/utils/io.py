import yaml

from typing import Any


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(
    data: dict[str, Any],
    path: str
) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(
            data = data,
            stream = f,
            sort_keys = False
        )