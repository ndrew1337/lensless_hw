import torch
import torch.fft as fft


def tv_grad(x):
    dx = torch.roll(x, -1, dims=-1) - x
    dy = torch.roll(x, -1, dims=-2) - x
    return torch.stack([dx, dy], dim=-1)


def tv_div(g):
    gx, gy = g[..., 0], g[..., 1]
    return torch.roll(gx, 1, dims=-1) - gx + torch.roll(gy, 1, dims=-2) - gy


def tv_laplacian_freq(shape):
    delta = torch.zeros(shape)
    delta[0, 0] = 1.0
    g = tv_grad(delta)
    return fft.rfft2(g[..., 0]).abs() ** 2 + fft.rfft2(g[..., 1]).abs() ** 2
