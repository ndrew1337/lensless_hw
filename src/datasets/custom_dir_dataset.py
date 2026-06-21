from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from lensless_helpers.psf import simulate_psf_from_mask
from src.datasets.lensless_common import make_item


class CustomDirDataset(Dataset):
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.ids = sorted(p.stem for p in (self.dir / "lensless").glob("*.png"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image_id = self.ids[i]
        lensless = Image.open(self.dir / "lensless" / f"{image_id}.png").convert("RGB")
        mask = np.load(self.dir / "masks" / f"{image_id}.npy")
        lensed_path = self.dir / "lensed" / f"{image_id}.png"
        lensed = Image.open(lensed_path).convert("RGB") if lensed_path.exists() else None
        return make_item(lensed, lensless, simulate_psf_from_mask(mask), image_id)
