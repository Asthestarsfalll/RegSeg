import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BACKBONES
from .blocks import ConvBnAct, DilationBlock
from .config import RegSegConfig
from .decoder import DECODERS


class RegSeg(nn.Module):
    def __init__(self, cfg: RegSegConfig):
        super(RegSeg, self).__init__()
        self.stem = ConvBnAct(3, cfg.stem_out, 3, 2, 1)
        # build
        print("use backbone {}".format(cfg.backbone))
        self.backbone = BACKBONES[cfg.backbone](cfg.backbone_cfg)
        channels = self.backbone.get_channels()

        print("use decoder {}".format(cfg.decoder))
        self.decoder = DECODERS[cfg.decoder](cfg.decoder_cfg, channels)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.stem(x)
        x = self.backbone(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x
