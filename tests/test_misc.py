import pytest

from src.utils.misc import ConfigDict


def test_config_dict_supports_nested_attribute_access() -> None:
    cfg = ConfigDict({"train": {"steps": 10}})

    assert cfg.train.steps == 10


def test_config_dict_to_dict_returns_plain_mapping() -> None:
    cfg = ConfigDict({"train": {"steps": 10}})

    plain = cfg.to_dict()

    assert plain == {"train": {"steps": 10}}
    assert isinstance(plain, dict)
    assert isinstance(plain["train"], dict)


def test_config_dict_raises_attribute_error_for_missing_key() -> None:
    cfg = ConfigDict({"train": {"steps": 10}})

    with pytest.raises(AttributeError):
        _ = cfg.missing
