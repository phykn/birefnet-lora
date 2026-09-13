from pathlib import Path

from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def load_config(path: str | Path | None = None) -> DictConfig:
    cfg = _read(ROOT / "config/model.yaml")
    if path is not None:
        cfg = OmegaConf.merge(cfg, _read(path))
    return cfg


def load_run(
    resume: str | None = None,
    config: str | Path | None = None,
) -> tuple[DictConfig, Path | None, Path | None]:
    if resume is None:
        return load_config(config or ROOT / "config/train.yaml"), None, None
    if config is not None:
        raise ValueError("--config cannot override a resumed run's saved config")

    checkpoint = Path(resume).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    if checkpoint.parent.name != "weights":
        raise ValueError("Resume checkpoint must be inside a run weights directory")
    run_dir = checkpoint.parent.parent
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    return _read(config_path), checkpoint, run_dir
