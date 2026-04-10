from omegaconf import DictConfig, ListConfig, OmegaConf


def load_yaml(path: str) -> DictConfig | ListConfig:
    return OmegaConf.load(path)


def save_yaml(cfg: DictConfig | ListConfig, path: str) -> None:
    OmegaConf.save(config=cfg, f=path)
