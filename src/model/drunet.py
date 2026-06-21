import torch.nn as nn
import torch.nn.functional as F


def conv3(cin, cout):
    return nn.Conv2d(cin, cout, 3, padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(conv3(c, c), nn.ReLU(inplace=True), conv3(c, c))

    def forward(self, x):
        return x + self.body(x)


class DRUNet(nn.Module):
    def __init__(self, channels, in_ch=3, out_ch=3, n_blocks=4):
        super().__init__()
        c0, c1, c2, c3 = channels

        def blocks(c):
            return nn.Sequential(*[ResBlock(c) for _ in range(n_blocks)])

        def down(cin, cout):
            return nn.Conv2d(cin, cout, 2, stride=2, bias=False)

        def up(cin, cout):
            return nn.ConvTranspose2d(cin, cout, 2, stride=2, bias=False)

        self.input = conv3(in_ch, c0)
        self.down1 = nn.Sequential(blocks(c0), down(c0, c1))
        self.down2 = nn.Sequential(blocks(c1), down(c1, c2))
        self.down3 = nn.Sequential(blocks(c2), down(c2, c3))
        self.inside = blocks(c3)
        self.up3 = nn.Sequential(up(c3, c2), blocks(c2))
        self.up2 = nn.Sequential(up(c2, c1), blocks(c1))
        self.up1 = nn.Sequential(up(c1, c0), blocks(c0))
        self.output = conv3(c0, out_ch)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 16, 0, (-h) % 16), mode="replicate")
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.inside(x4)
        x = self.up3(x + x4)
        x = self.up2(x + x3)
        x = self.up1(x + x2)
        x = self.output(x + x1)
        return x[..., :h, :w]
