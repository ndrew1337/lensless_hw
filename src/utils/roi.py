TOP, LEFT, HEIGHT = 80, 100, 200
WIDTH = int(HEIGHT * 1200 / 900)


def crop_roi(x):
    return x[..., TOP : TOP + HEIGHT, LEFT : LEFT + WIDTH].contiguous()


def normalize01(x):
    x = x.clamp(min=0)
    return x / x.amax(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-8)


def align_to_gt(rec, gt):
    rec = rec.clamp(min=0)
    num = (rec * gt).sum(dim=(-3, -2, -1), keepdim=True)
    den = (rec * rec).sum(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-8)
    return (num / den * rec).clamp(0, 1)
