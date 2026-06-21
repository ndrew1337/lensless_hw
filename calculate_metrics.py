import argparse
from pathlib import Path

import numpy as np
import torch
import torchmetrics as tm
from PIL import Image

from lensless_helpers.preprocessor import get_cropped_lensed
from src.utils.roi import align_to_gt, crop_roi


def load_rgb01(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="directory with ground-truth images")
    parser.add_argument("--pred", required=True, help="directory with reconstructions")
    args = parser.parse_args()

    gt_dir, pred_dir = Path(args.gt), Path(args.pred)
    metrics = {
        "PSNR": tm.image.PeakSignalNoiseRatio(data_range=1.0),
        "SSIM": tm.image.StructuralSimilarityIndexMeasure(data_range=1.0),
        "LPIPS": tm.image.LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True),
        "MSE": tm.MeanSquaredError(),
    }

    matched = 0
    sums = {name: 0.0 for name in metrics}
    for pred_path in sorted(pred_dir.glob("*.png")):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue
        rec = load_rgb01(pred_path)
        gt = crop_roi(torch.from_numpy(get_cropped_lensed(load_rgb01(gt_path), rec)).permute(2, 0, 1)[None])
        rec = align_to_gt(crop_roi(torch.from_numpy(rec).permute(2, 0, 1)[None]), gt)
        for name, metric in metrics.items():
            sums[name] += metric(rec, gt).item()
            metric.reset()
        matched += 1

    print(f"matched images: {matched}")
    for name in metrics:
        print(f"{name:6s}: {sums[name] / matched:.4f}")


if __name__ == "__main__":
    main()
