import torch
import torch.nn as nn
import torch.nn.functional as F
from config import RegSegConfig
from blocks import DilationBlock, ConvBnAct
from backbone import RegSegBody
from decoder import Decoder26


class RegSeg(nn.Module):
    def __init__(self, cfg: RegSegConfig):
        super(RegSeg, self).__init__()
        self.stem = ConvBnAct(3, cfg.stem_out, 3, 2, 1)
        # build
        print("use backbone {}".format(cfg.backbone))
        self.backbone = eval(cfg.backbone)(cfg.backbone_cfg)
        channels = self.backbone.get_channels()

        print("use decoder {}".format(cfg.decoder))
        self.decoder = eval(cfg.decoder)(cfg.decoder_cfg, channels)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.stem(x)
        x = self.backbone(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_shape,
                          mode='bilinear', align_corners=False)
        return x


if __name__ == "__main__":
    cfg = RegSegConfig()
    model = RegSeg(cfg)
    inp = torch.randn((2, 3, 512, 512))
    out = model(inp)
    print(out.shape)
