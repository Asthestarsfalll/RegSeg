import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, in_ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        atten = self.gap(x)
        atten = self.stem(atten)
        return x * atten


ATTENTIONS = dict(se=SEBlock)
