import numpy as np

from abc import ABC, abstractmethod


class FlipBase(ABC):
    @abstractmethod
    def flip(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.flip(image), self.flip(mask)


class Identity(FlipBase):
    def flip(self, x: np.ndarray) -> np.ndarray:
        return x


class HorizontalFlip(FlipBase):
    def flip(self, x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.flip(x, axis=1))


class VerticalFlip(FlipBase):
    def flip(self, x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.flip(x, axis=0))


class TransposeFlip(FlipBase):
    def flip(self, x: np.ndarray) -> np.ndarray:
        axes = (1, 0) + tuple(range(2, x.ndim))
        return np.ascontiguousarray(np.transpose(x, axes))


class Compose(FlipBase):
    def __init__(self, *flips: FlipBase) -> None:
        self._flips = flips

    def flip(self, x: np.ndarray) -> np.ndarray:
        for f in self._flips:
            x = f.flip(x)
        return x


_hflip = HorizontalFlip()
_vflip = VerticalFlip()
_tflip = TransposeFlip()

_transforms: tuple[FlipBase, ...] = (
    Identity(),
    _hflip,
    _vflip,
    _tflip,
    Compose(_vflip, _hflip),
    Compose(_tflip, _hflip),
    Compose(_tflip, _vflip),
    Compose(_tflip, _vflip, _hflip),
)


def random_flip(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = _transforms[np.random.randint(len(_transforms))]
    return transform(image, mask)
