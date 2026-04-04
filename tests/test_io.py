from pathlib import Path

import pytest

from src.utils.io import load_yaml, save_yaml


def test_save_and_load_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    expected = {"train": {"steps": 10}, "seed": 42}

    save_yaml(expected, str(path))

    loaded = load_yaml(str(path))
    assert loaded == expected


def test_load_yaml_raises_for_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML root must be a mapping"):
        load_yaml(str(path))
