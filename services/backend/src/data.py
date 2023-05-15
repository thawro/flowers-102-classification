"""Dataset from https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html"""

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Callable, Optional
import numpy as np
import torchvision
import json
from src.transforms import TransformWrapper
import urllib.request
import PIL

# labels mapping from: https://gist.github.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1
# saved to: https://raw.githubusercontent.com/thawro/flowers-species-classification/main/data/flowers-102/mapping.txt
"""
wget https://raw.githubusercontent.com/thawro/flowers-species-classification/main/data/flowers-102/mapping.txt -O data/flowers-102/mapping.txt
"""


class FlowersDataset(torchvision.datasets.Flowers102):
    """
    source: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
    benchmark: https://paperswithcode.com/sota/image-classification-on-flowers-102
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        transform = TransformWrapper(transform)
        super().__init__(
            root, split, transform=transform, target_transform=target_transform, download=download
        )
        with open(self._base_folder / "mapping.txt") as f:
            labels = f.readlines()
            labels = [label.strip() for label in labels]
            self.idx2label = {i: label for i, label in enumerate(labels)}

        self._named_labels = np.array([self.idx2label[target] for target in self._labels])
        self.classes = list(self.idx2label.values())

    def get_raw_img(self, idx: int):
        image_file = self._image_files[idx]
        return PIL.Image.open(image_file).convert("RGB")

    def download(self):
        super().download()
        mapping_url = "https://raw.githubusercontent.com/thawro/flowers-species-classification/main/data/flowers-102/mapping.txt"
        mapping_dst = self._base_folder / "mapping.txt"
        urllib.request.urlretrieve(mapping_url, mapping_dst)


class FlowersDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ds: FlowersDataset,
        val_ds: FlowersDataset,
        test_ds: FlowersDataset,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )  # shuffle for plotting purposes only
