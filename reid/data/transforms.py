import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class ToTensorNoScale:
    """PIL image -> float tensor kept in [0, 255]. The model normalizes to [-1, 1] itself."""

    def __call__(self, pic):
        return torch.from_numpy(np.array(pic, copy=True)).permute(2, 0, 1).contiguous().float()

    def __repr__(self):
        return self.__class__.__name__ + "()"


def build_transforms(img_size, is_train=True):
    """Standard TransReID / BoT augmentations."""
    height, width = img_size
    if not is_train:
        return T.Compose([T.Resize([height, width]), ToTensorNoScale()])

    return T.Compose(
        [
            T.Resize([height, width], interpolation=InterpolationMode.BICUBIC),
            T.Pad(10),
            T.RandomCrop([height, width]),
            T.RandomHorizontalFlip(p=0.5),
            ToTensorNoScale(),
            T.RandomErasing(p=0.5, value=[123.675, 116.28, 103.53]),  # ImageNet mean, in [0, 255]
        ]
    )
