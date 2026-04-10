import pytest

pytest.importorskip("imrw")
pytest.importorskip("sdimg")

from src.build import index_by_stem  # noqa: E402


def test_index_by_stem_maps_stem_to_path():
    paths = ["/data/image/a.png", "/data/image/b.jpg"]
    result = index_by_stem(paths)
    assert result == {"a": "/data/image/a.png", "b": "/data/image/b.jpg"}


def test_index_by_stem_raises_on_duplicate():
    paths = ["/x/a.png", "/y/a.jpg"]
    with pytest.raises(ValueError, match="Duplicate stem"):
        index_by_stem(paths)


def test_index_by_stem_empty():
    assert index_by_stem([]) == {}
