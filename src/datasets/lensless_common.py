import numpy as np
import torch

from lensless_helpers.preprocessor import (
    convert_image_to_float,
    force_rgb,
    get_cropped_lensed,
)


def _chw(t):
    return t.permute(2, 0, 1).contiguous()


def make_item(lensed_img, lensless_img, psf, image_id):
    lensless = convert_image_to_float(force_rgb(np.array(lensless_img)))
    lensless = torch.rot90(torch.from_numpy(lensless), dims=(-3, -2), k=2)
    item = {"lensless": _chw(lensless), "psf": _chw(psf[0]), "image_id": image_id}
    if lensed_img is not None:
        lensed = convert_image_to_float(force_rgb(np.array(lensed_img)))
        item["lensed"] = _chw(torch.from_numpy(get_cropped_lensed(lensed, lensless)))
    return item
