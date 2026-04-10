from omegaconf import DictConfig, OmegaConf

from src.utils.io import load_yaml, save_yaml


def test_yaml_round_trip(tmp_path):
    cfg = OmegaConf.create({"a": 1, "b": {"c": "x", "d": [1, 2, 3]}})
    path = tmp_path / "cfg.yaml"
    save_yaml(cfg, str(path))
    assert path.exists()

    loaded = load_yaml(str(path))
    assert isinstance(loaded, DictConfig)
    assert loaded.a == 1
    assert loaded.b.c == "x"
    assert list(loaded.b.d) == [1, 2, 3]


def test_select_default_for_missing_key(tmp_path):
    cfg = OmegaConf.create({"present": 1})
    path = tmp_path / "cfg.yaml"
    save_yaml(cfg, str(path))
    loaded = load_yaml(str(path))
    assert OmegaConf.select(loaded, "missing.key", default="fallback") == "fallback"
