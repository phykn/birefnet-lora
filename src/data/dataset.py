import albumentations as A
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def collect_paths(
    path: str,
    exts: list[str],
) -> list[str]:
    """Return sorted file paths under `path` whose suffix is in `exts`."""
    return sorted(
        str(p) for p in Path(path).iterdir()
        if p.suffix.lower() in exts
    )


class FineTuneDataset(Dataset):
    EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    def __init__(
        self,
        img_paths: list[str],
        mask_paths: list[str],
        size: int | tuple[int, int],
    ) -> None:
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.transform = self._build_transform()

    def _build_transform(self) -> A.Compose:
        h, w = self.size
        return A.Compose([A.Resize(height=h, width=w)])

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        return img, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img, mask = self._load(idx)
        res = self.transform(image=img, mask=mask)
        img_t = TF.normalize(TF.to_tensor(res["image"]), mean=MEAN, std=STD)
        mask_t = TF.to_tensor(res["mask"])
        return {"image": img_t, "mask": mask_t}


class ValidDataset(FineTuneDataset):
    pass


class TrainDataset(FineTuneDataset):
    def _build_transform(self) -> A.Compose:
        h, w = self.size
        return A.Compose([
            A.Resize(height=h, width=w),
            A.D4(),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(num_holes_range=(1, 4), p=0.5),
        ])
