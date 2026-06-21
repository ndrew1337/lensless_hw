import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from lensless_helpers.psf import simulate_psf_from_mask
from src.datasets.lensless_common import make_item

REPO = "bezzam/DigiCam-Mirflickr-MultiMask-10K"


class DigiCamDataset(Dataset):
    def __init__(self, split, limit=None):
        ds = load_dataset(REPO, split=split)
        if limit is not None:
            ds = ds.select(range(limit))
        self.ds = ds
        self.psfs = {}

    def __len__(self):
        return len(self.ds)

    def _psf(self, label):
        if label not in self.psfs:
            path = hf_hub_download(REPO, f"masks/mask_{label}.npy", repo_type="dataset")
            self.psfs[label] = simulate_psf_from_mask(np.load(path))
        return self.psfs[label]

    def __getitem__(self, i):
        row = self.ds[i]
        return make_item(row["lensed"], row["lensless"], self._psf(row["mask_label"]), str(i))
