import torchvision.transforms as T
import torch
from torch import nn
import numpy as np
from PIL import Image
from typing import Optional, Callable


class ImgToTensor:
    def __call__(self, x: np.ndarray | torch.Tensor | Image.Image) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif isinstance(x, Image.Image):
            x = T.functional.pil_to_tensor(x).permute(1, 2, 0)
        out = x.to(torch.uint8)
        return out


class Divide:
    def __init__(self, scalar: float | int):
        self.scalar = scalar

    def __call__(self, x: torch.Tensor):
        return x / self.scalar


class Permute:
    def __init__(self, dims: list[int]):
        self.dims = dims

    def __call__(self, sample: torch.Tensor):
        return torch.permute(sample, self.dims)


class CenterCrop:
    """Based on torchvision implementation"""

    def __init__(self, size: int | list[int] | tuple[int]):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, (tuple, list)) and len(size) == 1:
            size = (size[0], size[0])
        self.size = size

    def __call__(self, img: torch.Tensor):
        _, image_height, image_width = img.shape
        crop_height, crop_width = self.size

        # extra lines to ensure that scalars arent of Tensor type
        if not isinstance(image_height, int) and not isinstance(image_width, int):
            image_height, image_width = image_height.item(), image_width.item()

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            img = T.functional.pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
            _, image_height, image_width = img.shape
            if crop_width == image_width and crop_height == image_height:
                return img

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return T.functional.crop(img, crop_top, crop_left, crop_height, crop_width)


class TransformWrapper(nn.Module):
    def __init__(self, transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform is None:
            return x
        return self.transform(x)


MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]
new_mean = [-m / s for m, s in zip(MEAN_IMAGENET, STD_IMAGENET)]
new_std = [1 / s for s in STD_IMAGENET]


img_unnormalizer = T.Normalize(new_mean, new_std)
