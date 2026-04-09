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

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        return image, mask

    def _to_tensors(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = TF.normalize(TF.to_tensor(image), mean=MEAN, std=STD)
        mask_tensor = TF.to_tensor(mask)
        return image_tensor, mask_tensor


class ValidDataset(FineTuneDataset):
    def __init__(
        self,
        img_paths: list[str],
        mask_paths: list[str],
        size: int | tuple[int, int],
    ) -> None:
        super().__init__(img_paths=img_paths, mask_paths=mask_paths, size=size)
        height, width = self.size
        self.transform = A.Compose([A.Resize(height=height, width=width)])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, mask = self._load(idx)
        transformed = self.transform(image=image, mask=mask)
        image_tensor, mask_tensor = self._to_tensors(transformed["image"], transformed["mask"])
        return {"image": image_tensor, "mask": mask_tensor}


class TrainDataset(FineTuneDataset):
    def __init__(
        self,
        img_paths: list[str],
        mask_paths: list[str],
        size: int | tuple[int, int],
    ) -> None:
        super().__init__(img_paths=img_paths, mask_paths=mask_paths, size=size)
        height, width = self.size
        self.geo_transform = A.Compose([
            A.Resize(height=height, width=width),
            A.D4(),
        ])
        self.color_transform = A.Compose([
            A.RandomBrightnessContrast(p=1.0),
        ])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, mask = self._load(idx)

        geo = self.geo_transform(image=image, mask=mask)
        geo_image = geo["image"]
        geo_mask = geo["mask"]

        view1 = self.color_transform(image=geo_image)["image"]
        view2 = self.color_transform(image=geo_image)["image"]

        v1_tensor, mask_tensor = self._to_tensors(view1, geo_mask)
        v2_tensor, _ = self._to_tensors(view2, geo_mask)

        return {"image_v1": v1_tensor, "image_v2": v2_tensor, "mask": mask_tensor}
