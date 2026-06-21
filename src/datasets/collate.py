import torch


def collate_fn(items):
    batch = {
        "lensless": torch.stack([x["lensless"] for x in items]),
        "psf": torch.stack([x["psf"] for x in items]),
        "image_id": [x["image_id"] for x in items],
    }
    if all("lensed" in x for x in items):
        batch["lensed"] = torch.stack([x["lensed"] for x in items])
    return batch
