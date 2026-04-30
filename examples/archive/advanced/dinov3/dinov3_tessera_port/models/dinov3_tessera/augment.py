from typing import List, Tuple, Optional
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class MultiCropTransform:
    def __init__(self, global_size=224, local_size=96, n_local=8):
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        common = [
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(9, sigma=(0.1, 2.0)),
        ]
        self.global_tf = T.Compose([
            T.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            *common,
            normalize,
        ])
        self.local_tf = T.Compose([
            T.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
            T.RandomHorizontalFlip(),
            *common,
            normalize,
        ])
        self.n_local = n_local

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        views = [self.global_tf(img), self.global_tf(img)]
        views += [self.local_tf(img) for _ in range(self.n_local)]
        return views


class FakeMultiCropDataset(Dataset):
    """Synthetic data for smoke tests."""
    def __init__(self, length=128, img_size=224, transform=None):
        self.length = length
        self.img_size = img_size
        self.transform = transform or MultiCropTransform()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = Image.fromarray((torch.rand(3, self.img_size, self.img_size).numpy().transpose(1,2,0) * 255).astype('uint8'))
        views = self.transform(img)
        return views, 0  # label unused
