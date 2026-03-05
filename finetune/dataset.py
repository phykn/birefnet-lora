"""
파인튜닝용 단순 데이터셋.

이미지 폴더와 마스크 폴더를 받아 (image, mask) 쌍을 반환한다.
원본의 복잡한 다중 데이터셋 결합, 클래스 라벨, dynamic_size 등은 모두 제거.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class FineTuneDataset(Dataset):
    """
    Args:
        image_dir: 이미지 폴더 경로
        mask_dir:  마스크 폴더 경로 (grayscale, 0~255)
        size:      리사이즈 크기 (width, height)

    파일명 정렬 기준으로 image_dir과 mask_dir이 1:1 대응되어야 한다.
    """

    VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, image_dir: str, mask_dir: str, size=(1024, 1024)):
        self.image_paths = self._collect(image_dir)
        self.mask_paths = self._collect(mask_dir)
        assert len(self.image_paths) == len(self.mask_paths), (
            f"이미지({len(self.image_paths)})와 마스크({len(self.mask_paths)}) 개수 불일치"
        )

        # 원본 BiRefNet과 동일한 정규화
        self.img_transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_transform = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def _collect(self, directory: str) -> list:
        return sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in self.VALID_EXT
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        return self.img_transform(image), self.mask_transform(mask)
