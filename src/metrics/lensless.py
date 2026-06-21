import torch

from src.metrics.base_metric import BaseMetric
from src.utils.roi import align_to_gt, crop_roi


class LenslessMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, reconstructed, lensed, **kwargs):
        gt = crop_roi(lensed)
        rec = align_to_gt(crop_roi(reconstructed), gt)
        self.metric.reset()
        return torch.stack([self.metric(rec[i : i + 1], gt[i : i + 1]) for i in range(rec.shape[0])]).mean()
