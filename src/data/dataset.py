import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FineTuneDataset(Dataset):
    EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    def __init__(
        self,
        img_paths: list[str],
        mask_paths: list[str],
        size: int | tuple[int, int],
    ) -> None:
        if len(img_paths) != len(mask_paths):
            raise ValueError(
                "Image and mask path counts must match. "
                f"Got {len(img_paths)} images and {len(mask_paths)} masks."
            )

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.transform = self._build_transform()

    def _build_transform(self) -> A.Compose:
        height, width = self.size
        return A.Compose([A.Resize(height=height, width=width)])

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        return image, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, mask = self._load(idx)
        transformed = self.transform(image=image, mask=mask)
        image_tensor = TF.normalize(TF.to_tensor(transformed["image"]), mean=MEAN, std=STD)
        mask_tensor = TF.to_tensor(transformed["mask"])
        return {"image": image_tensor, "mask": mask_tensor}


class ValidDataset(FineTuneDataset):
    pass


class TrainDataset(FineTuneDataset):
    def _build_transform(self) -> A.Compose:
        height, width = self.size
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                A.D4(),
                A.RandomBrightnessContrast(p=0.5),
            ]
        )
