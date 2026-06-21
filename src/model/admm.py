import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from src.physics.convolution import LenslessOperator
from src.physics.finite_diff import tv_grad, tv_div, tv_laplacian_freq


def soft_threshold(z, t):
    return z.sign() * (z.abs() - t).clamp(min=0)


class LeADMM(nn.Module):
    def __init__(self, n_iters, learnable, mu, tau):
        super().__init__()
        self.n_iters = n_iters
        raw = torch.log(torch.expm1(torch.tensor([mu, mu, mu, tau]))).repeat(n_iters, 1)
        self.raw = nn.Parameter(raw, requires_grad=learnable)

    def forward(self, b, psf):
        b, psf = b.float(), psf.float()
        B, C = b.shape[:2]
        op = LenslessOperator(psf)

        Ctb = op.pad(b)
        CtC = op.pad(torch.ones_like(b))
        otf2 = op.otf.abs() ** 2
        lap = tv_laplacian_freq(op.padded_shape).to(otf2)
        mus = F.softplus(self.raw)

        img = (B, C, *op.padded_shape)
        x = b.new_zeros(img)
        a1 = b.new_zeros(img)
        a2 = b.new_zeros(*img, 2)
        a3 = b.new_zeros(img)

        for  k in range(self.n_iters):
            mu1, mu2, mu3, tau = mus[k]
            u = soft_threshold(tv_grad(x) + a2 / mu2, tau / mu2)
            v = (mu1 * op.H(x) + a1 + Ctb) / (CtC + mu1)
            w = torch.relu(x + a3 / mu3)
            r = (mu3 * w - a3) + tv_div(mu2 * u - a2) + op.HT(mu1 * v - a1)
            x = fft.irfft2(fft.rfft2(r) / (mu1 * otf2 + mu2 * lap + mu3), s=op.padded_shape)
            a1 = a1 + mu1 * (op.H(x) - v)
            a2 = a2 + mu2 * (tv_grad(x) - u)
            a3 = a3 + mu3 * (x - w)

        return op.crop(x)
