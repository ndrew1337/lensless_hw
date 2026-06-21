import torch
import torch.nn as nn

from src.physics.convolution import LenslessOperator
from src.physics.finite_diff import tv_grad, tv_div


class FISTA(nn.Module):
    def __init__(self, n_iters, tau, tv_eps):
        super().__init__()
        self.n_iters = n_iters
        self.tau = tau
        self.tv_eps = tv_eps

    def forward(self, b, psf):
        b, psf = b.float(), psf.float() 
        op = LenslessOperator(psf)
        step = 1.0 / (op.otf.abs() ** 2).amax().clamp(min=1e-6)

        x = op.pad(torch.zeros_like(b))
        y = x
        t = 1.0
        for _ in range(self.n_iters):
            grad = op.HT(op.pad(op.crop(op.H(y)) - b))
            g = tv_grad(y)
            grad = grad + self.tau * tv_div(g / (g * g + self.tv_eps**2).sqrt())
            x_next = torch.relu(y - step * grad)
            t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            y = x_next + (t - 1.0) / t_next * (x_next - x)
            x, t = x_next, t_next

        return op.crop(x)
