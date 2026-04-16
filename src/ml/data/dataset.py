import os
import random

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import numpy as np

from imrw import imread
from sdimg.image import to_gray, adjust_brightness_contrast
from sdimg.spatial import resize

from .augment import random_crop, random_flip
from ..preprocess import normalize, preprocess


class Dataset:
    def __init__(
        self,
        data: list,
        size: int = 1024,
        scales: tuple[float, float] = (1.0, 1.0),
        train: bool = False,
        bc_weak: tuple[float, float] = (0.2, 0.2),
        bc_strong: tuple[float, float] = (0.4, 0.4),
    ):
        self.data = data
        self.size = size
        self.scales = scales
        self.train = train
        self.bc_weak_range = bc_weak
        self.bc_strong_range = bc_strong

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        image, mask = self.load(idx)

        if self.train:
            scale = random.uniform(*self.scales)
            if scale < 1.0:
                image, mask = random_crop(image, mask, scale=scale)
            image, mask = random_flip(image, mask)

            image_1 = self.prepare(self.bc_weak(image))
            image_2 = self.prepare(self.bc_strong(image))
        else:
            image_1 = self.prepare(image)

        mask = self.to_binary(mask)
        mask = resize(
            mask,
            height=self.size,
            width=self.size,
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        mask = np.expand_dims(mask, axis=0)

        sample = dict(image_1=image_1, mask=mask)
        if self.train:
            sample["image_2"] = image_2
        return sample

    def load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.data[idx]
        image = imread(image_path)
        mask = imread(mask_path)
        return image, mask

    def prepare(self, image: np.ndarray) -> np.ndarray:
        image = preprocess(image)
        image = resize(
            image, height=self.size, width=self.size, interpolation=cv2.INTER_CUBIC
        )
        image = normalize(image)
        return np.transpose(image, (2, 0, 1))

    def to_binary(self, mask: np.ndarray) -> np.ndarray:
        return (to_gray(mask) > 127).astype(np.float32)

    def bc_weak(self, image: np.ndarray) -> np.ndarray:
        return self._jitter(image, self.bc_weak_range)

    def bc_strong(self, image: np.ndarray) -> np.ndarray:
        return self._jitter(image, self.bc_strong_range)

    @staticmethod
    def _jitter(image: np.ndarray, bc_range: tuple[float, float]) -> np.ndarray:
        b, c = bc_range
        return adjust_brightness_contrast(
            image=image,
            brightness=random.uniform(-b, b),
            contrast=random.uniform(-c, c),
        )
