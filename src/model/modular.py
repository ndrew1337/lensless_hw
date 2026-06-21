import torch
import torch.nn as nn


class ModularReconstruction(nn.Module):
    """pre-processor (optional) -> camera inversion -> post-processor (optional)"""

    def __init__(self, inversion, pre=None, post=None):
        super().__init__()
        self.pre = pre
        self.inversion = inversion
        self.post = post
        self.frozen_inversion = not any(p.requires_grad for p in inversion.parameters())

    def forward(self, lensless, psf, **batch):
        x = lensless if self.pre is None else self.pre(lensless)
        if self.frozen_inversion and not x.requires_grad:
            with torch.no_grad():
                x = self.inversion(x, psf)
        else:
            x = self.inversion(x, psf)
        if self.post is not None:
            x = self.post(x)
        return {"reconstructed": x}

    def __str__(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super().__str__() + f"\nAll parameters: {total}\nTrainable parameters: {trainable}"
