import torch

from src.physics.convolution import LenslessOperator


def test_centered_impulse_is_identity():
    psf = torch.zeros(3, 16, 24)
    psf[:, 8, 12] = 1.0
    op = LenslessOperator(psf)
    x = torch.randn(2, 3, *op.padded_shape)
    assert torch.allclose(op.H(x), x, atol=1e-5)


def test_H_adjoint():
    op = LenslessOperator(torch.rand(3, 16, 24))
    x = torch.randn(2, 3, *op.padded_shape)
    y = torch.randn(2, 3, *op.padded_shape)
    assert torch.allclose((op.H(x) * y).sum(), (x * op.HT(y)).sum(), rtol=1e-4)


def test_crop_pad_roundtrip():
    op = LenslessOperator(torch.rand(3, 16, 24))
    b = torch.randn(2, 3, 16, 24)
    assert torch.allclose(op.crop(op.pad(b)), b)
