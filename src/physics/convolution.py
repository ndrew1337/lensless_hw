import torch.nn.functional as F
import torch.fft as fft


def pad_center(x, shape):
    th, tw = shape
    h, w = x.shape[-2:]
    dh = (th - h) // 2
    dw = (tw - w) // 2
    return F.pad(x, (dw, tw - w - dw, dh, th - h - dh))


def crop_center(x, shape):
    th, tw = shape
    h, w = x.shape[-2:]
    dh = (h - th) // 2
    dw = (w - tw) // 2
    return x[..., dh:dh+th, dw:dw+tw]


class LenslessOperator:
    def __init__(self, psf):
        H, W = psf.shape[-2:]
        self.sensor_shape = (H, W)
        self.padded_shape = (2 * H, 2 * W)
        psf = fft.ifftshift(pad_center(psf, self.padded_shape), dim=(-2, -1))
        self.otf = fft.rfft2(psf)

    def H(self, x):
        return fft.irfft2(fft.rfft2(x) * self.otf, s=self.padded_shape)

    def HT(self, x):
        return fft.irfft2(fft.rfft2(x) * self.otf.conj(), s=self.padded_shape)

    def crop(self, x):
        return crop_center(x, self.sensor_shape)

    def pad(self, b):
        return pad_center(b, self.padded_shape)
