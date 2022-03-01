import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBnAct


# TODO: build decoder due to input cfg
class Decoder26(nn.Module):
    def __init__(self, cfg, channels):
        super(Decoder26, self).__init__()
        num_classes = cfg['num_classes']
        header_out = cfg['header_out']
        channels4, channels8, channels16 = channels[0], channels[1], channels[2]
        self.head16 = ConvBnAct(channels16, header_out[0], 1)
        self.head8 = ConvBnAct(channels8, header_out[1], 1)
        self.head4 = ConvBnAct(channels4, header_out[2], 1)
        self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv4 = ConvBnAct(64 + 8, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x4, x8, x16 = x["4"], x["8"], x["16"]
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = F.interpolate(
            x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(
            x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4
