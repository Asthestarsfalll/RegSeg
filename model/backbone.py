import torch.nn as nn
from blocks import DilationBlock


# TODO: build backbone due to input cfg
class RegSegBody(nn.Module):
    def __init__(self, cfg):
        super(RegSegBody, self).__init__()
        channels, group_width, attention, dilations = cfg['channels'], cfg[
            'group_width'], cfg['attention'], cfg['dilations']
        self.out_channels = [
            channels[1], channels[2], channels[-1]]
        # 4 stands for 4 times downsample
        self.stage4 = DilationBlock(
            channels[0], channels[1], 2, [1], group_width, attention)

        self.stage8 = nn.Sequential(
            DilationBlock(channels[1], channels[2], 2,
                          [1], group_width, attention),
            DilationBlock(channels[2], channels[2], 1,
                          [1], group_width, attention),
            DilationBlock(channels[2], channels[2], 1,
                          [1], group_width, attention)
        )
        self.stage16 = nn.Sequential(
            DilationBlock(channels[2], channels[3], 2,
                          [1], group_width, attention),
            *self._make_layers(dilations[:-1], lambda d:  DilationBlock(channels[3],
                               channels[3], 1, d, group_width, attention)),
            DilationBlock(channels[3], channels[4], 1,
                          dilations[-1], group_width, attention)
        )

    @staticmethod
    def _make_layers(dilations, func):
        layers = []
        for dilation in dilations:
            layers.append(func(dilation))
        return layers

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        return {"4": x4, "8": x8, "16": x16}

    def get_channels(self):
        return self.out_channels
