from argparse import Namespace

import pytest
from omegaconf import OmegaConf

import run_train
from src.build.split import save as save_splits


def _make_run(tmp_path):
    run_dir = tmp_path / "run" / "sample"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    checkpoint = weights_dir / "last.train.pth"
    checkpoint.touch()
    cfg = OmegaConf.create({"marker": "saved"})
    OmegaConf.save(cfg, run_dir / "config.yaml")
    splits = {
        "train_image": ["train.png"],
        "train_mask": ["train.png"],
        "valid_image": ["valid.png"],
        "valid_mask": ["valid.png"],
        "calib_image": ["calib.png"],
        "calib_mask": ["calib.png"],
    }
    save_splits(splits, run_dir)
    return run_dir, checkpoint, splits


def test_load_run_uses_saved_config_and_splits(tmp_path):
    run_dir, checkpoint, expected_splits = _make_run(tmp_path)

    cfg, actual_checkpoint, actual_run_dir, splits = run_train.load_run(
        str(checkpoint)
    )

    assert cfg.marker == "saved"
    assert actual_checkpoint == checkpoint.resolve()
    assert actual_run_dir == run_dir.resolve()
    assert splits == expected_splits


def test_load_run_requires_saved_config(tmp_path):
    run_dir, checkpoint, _ = _make_run(tmp_path)
    (run_dir / "config.yaml").unlink()

    with pytest.raises(FileNotFoundError, match="Run config not found"):
        run_train.load_run(str(checkpoint))


def test_main_resumes_in_existing_run(monkeypatch, tmp_path):
    run_dir, checkpoint, splits = _make_run(tmp_path)
    cfg = OmegaConf.create(
        {
            "marker": "saved",
            "train": {"steps": 7, "val_freq": 2, "save_freq": 3},
        }
    )
    OmegaConf.save(cfg, run_dir / "config.yaml")
    calls = {}

    class Base:
        def to(self, device):
            return self

    class Model:
        stats = {"total": 10, "trainable": 2}

    class Trainer:
        save_dir = str(run_dir)

        def load_resume(self, path):
            calls["checkpoint"] = path

        def train(self, **kwargs):
            calls["train"] = kwargs

    monkeypatch.setattr(
        run_train,
        "parse_args",
        lambda: Namespace(resume=str(checkpoint)),
    )
    monkeypatch.setattr(run_train, "build_model", lambda actual: Base())
    monkeypatch.setattr(run_train, "adapt", lambda actual, base: Model())

    def build_data(actual, saved):
        calls["config"] = actual
        calls["splits"] = saved
        return "train", "valid", "calib", splits

    def build_trainer(**kwargs):
        calls["save_dir"] = kwargs["save_dir"]
        return Trainer()

    monkeypatch.setattr(run_train, "build_data", build_data)
    monkeypatch.setattr(run_train, "build_trainer", build_trainer)

    run_train.main()

    assert calls["config"].marker == "saved"
    assert calls["splits"] == splits
    assert calls["save_dir"] == run_dir.resolve()
    assert calls["checkpoint"] == str(checkpoint.resolve())
    assert calls["train"] == {"steps": 7, "val_freq": 2, "save_freq": 3}
