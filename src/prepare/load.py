from pathlib import Path

import numpy as np

from .augment import crop, flip, jitter
from .convert import InputMode, convert
from .fit import fit_image, fit_mask
from .read import read_image, read_mask


class MaskDataset:
    def __init__(
        self,
        data: list[tuple[str, str]],
        size: int = 1024,
        train: bool = False,
        weak: tuple[float, float] = (0.2, 0.2),
        strong: tuple[float, float] = (0.4, 0.4),
        mode: InputMode = "rgb",
        global_prob: float = 0.3,
        boundary_prob: float = 0.5,
    ) -> None:
        if not 0.0 <= global_prob <= 1.0:
            raise ValueError("global_prob must be in [0, 1]")
        if not 0.0 <= boundary_prob <= 1.0:
            raise ValueError("boundary_prob must be in [0, 1]")
        self.data = data
        self.size = int(size)
        self.train = train
        self.weak = weak
        self.strong = strong
        self.mode = mode
        self.global_prob = float(global_prob)
        self.boundary_prob = float(boundary_prob)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        image, mask = self._load(index)
        image = convert(image, mode=self.mode)

        if self.train:
            image, mask = crop(
                image,
                mask,
                self.size,
                self.global_prob,
                self.boundary_prob,
            )
            image, mask = flip(image, mask)
            weak = jitter(image, self.weak)
            strong = jitter(image, self.strong)
        else:
            weak = image

        weak, valid, fit = fit_image(
            weak,
            size=self.size,
            mode="rgb",
        )
        binary_mask = (mask > 127).astype(np.float32)
        sample = {
            "weak": weak,
            "mask": fit_mask(binary_mask, fit),
            "valid": valid,
        }
        if self.train:
            strong, strong_valid, strong_fit = fit_image(
                strong,
                size=self.size,
                mode="rgb",
            )
            if strong_fit != fit or not np.array_equal(strong_valid, valid):
                raise RuntimeError("Two-view augmentation produced mismatched geometry.")
            sample["strong"] = strong
        return sample

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.data[index]
        image = read_image(image_path)
        mask = read_mask(mask_path)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                "Image and mask dimensions differ: "
                f"{Path(image_path).name}={image.shape[:2]}, "
                f"{Path(mask_path).name}={mask.shape[:2]}"
            )
        return image, mask
