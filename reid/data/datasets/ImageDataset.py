from pathlib import Path

from PIL import Image
from torch.utils.data.dataset import Dataset


def load_image(path):
    """Load an image as RGB. RGBA images (segmented MOO renders) are composited on black."""
    with Image.open(path) as img:
        img = img.copy()
    if img.mode == "RGBA":
        black = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(black, img)
    return img.convert("RGB")


class ImageDataset(Dataset):
    """data is a list of (path, domain, pid, camid), with domain = 0 for real images and 1 for synthetic views."""

    def __init__(self, data, transform, name):
        self.data = data
        self.transform = transform
        self.name = name
        self.pids = [d[2] for d in data]
        self.num_classes = len(set(self.pids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, domain, pid, camid = self.data[idx]
        return {
            "x": self.transform(load_image(path)),
            "pid": pid,
            "camid": camid,
            "domain_label": domain,
            "path": str(path),
            "filename": Path(path).stem,
        }
