import lpips
import torch.nn as nn

from src.utils.roi import align_to_gt, crop_roi


class LenslessLoss(nn.Module):
    def __init__(self, lpips_weight):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lpips = lpips.LPIPS(net="vgg")
        self.lpips.requires_grad_(False)
        self.lpips_weight = lpips_weight

    def forward(self, reconstructed, lensed, **batch):
        gt = crop_roi(lensed)
        rec = align_to_gt(crop_roi(reconstructed), gt)
        mse = self.mse(rec, gt)
        perceptual = self.lpips(rec * 2 - 1, gt * 2 - 1).mean()
        loss = mse + self.lpips_weight * perceptual
        return {"loss": loss, "mse": mse.detach(), "lpips": perceptual.detach()}
