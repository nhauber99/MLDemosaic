import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class DemosaicDataset(Dataset):
    def __init__(self, root_dir, len_factor, crop_transform, augment_transform=None):
        self.root_dir = root_dir
        self.len_factor = len_factor
        self.image_paths = glob.glob(os.path.join(root_dir, '*'))
        self.crop_transform = crop_transform
        self.augment_transform = augment_transform

    def __len__(self):
        return len(self.image_paths) * self.len_factor

    def __getitem__(self, idx):
        img_path = self.image_paths[idx % len(self.image_paths)]
        image = Image.open(img_path).convert('RGB')
        target = v2.ToImage()(image)
        if self.crop_transform is not None:
            target = self.crop_transform(target)
        target = v2.ToDtype(torch.float32, scale=True)(target) ** 2.2
        if self.augment_transform is not None:
            target = self.augment_transform(target)
        bayer = torch.zeros((1, target.size(1), target.size(2)), dtype=target.dtype)
        bayer[:, 0::2, 0::2] = target[0, 0::2, 0::2]
        bayer[:, 0::2, 1::2] = target[1, 0::2, 1::2]
        bayer[:, 1::2, 0::2] = target[1, 1::2, 0::2]
        bayer[:, 1::2, 1::2] = target[2, 1::2, 1::2]

        return bayer, target
