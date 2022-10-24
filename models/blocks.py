from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .attention import ATTENTIONS

_ACTIVATION = dict(
    relu=partial(nn.ReLU, inplace=True),
)


class BnAct(nn.Module):
    def __init__(self, out_ch: int, act_type: Optional[str] = None):
        super(BnAct, self).__init__()
        self.bn = nn.BatchNorm2d(out_ch)
        if act_type is None:
            self.act = nn.Identity()
        else:
            self.act = _ACTIVATION[act_type]()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvBnAct(BnAct):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        act_type: Optional[str] = None,
    ):
        super(ConvBnAct, self).__init__(out_ch=out_ch, act_type=act_type)
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias
        )


class DilationConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        stride: int,
        dilations: Sequence[int],
        group_width: int,
        bias: bool = False,
    ):
        super(DilationConv, self).__init__()
        self.num_splits = len(dilations)
        assert in_ch % self.num_splits == 0
        each_conv_chan = in_ch // self.num_splits
        assert each_conv_chan % group_width == 0
        num_groups = each_conv_chan // group_width
        self.branches = self._make_layers(
            dilations, each_conv_chan, stride, bias, num_groups
        )

    def _make_layers(self, dilations, each_conv_chan, stride, bias, num_groups):
        convs = []
        for dilation in dilations:
            convs.append(
                nn.Conv2d(
                    each_conv_chan,
                    each_conv_chan,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    stride=stride,
                    bias=bias,
                    groups=num_groups,
                )
            )
        return nn.ModuleList(convs)

    def forward(self, x):
        x = torch.tensor_split(x, self.num_splits, dim=1)
        res = []
        for i in range(self.num_splits):
            res.append(self.branches[i](x[i]))
        return torch.cat(res, dim=1)


class DilationConvBnAct(BnAct):
    def __init__(
        self,
        in_ch: int,
        stride: int,
        dilations: Sequence[int],
        group_width: int,
        bias: bool = False,
        act_type: Optional[str] = None,
    ):
        super(DilationConvBnAct, self).__init__(out_ch=in_ch, act_type=act_type)
        self.conv = DilationConv(in_ch, stride, dilations, group_width, bias)


class ShortCut(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, stride: int, avg_downsample: bool = False
    ):
        super(ShortCut, self).__init__()
        if avg_downsample and stride != 1:
            self.avgpool = nn.AvgPool2d(2, 2, ceil_mode=True)
            self.conv_bn = ConvBnAct(in_ch, out_ch, 1, bias=False)
        else:
            self.avgpool = nn.Identity()
            self.conv_bn = ConvBnAct(in_ch, out_ch, 1, stride, bias=False)

    def forward(self, x):
        return self.conv_bn(self.avgpool(x))


class DilationBlock(nn.Module):
    """
    DBlock
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        dilations: Sequence[int],
        group_width: int,
        attention: Optional[str] = None,
    ):
        super(DilationBlock, self).__init__()
        groups = out_ch // group_width

        self.cba1 = ConvBnAct(in_ch, out_ch, act_type="relu")

        # multi-dilations-branch
        if len(dilations) == 1:
            dilation = dilations[0]
            self.cba2 = ConvBnAct(
                out_ch,
                out_ch,
                kernel_size=3,
                stride=stride,
                groups=groups,
                padding=dilation,
                dilation=dilation,
                bias=False,
                act_type="relu",
            )
        else:
            self.cba2 = DilationConvBnAct(
                out_ch, stride, dilations, group_width, bias=False, act_type="relu"
            )

        self.cba3 = ConvBnAct(out_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        # attention
        if attention is None:
            self.atten = nn.Identity()
        elif attention == "se":
            self.atten = ATTENTIONS["se"](out_ch, in_ch // 4)

        # shortcut
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ShortCut(in_ch, out_ch, stride, avg_downsample=True)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.cba1(x)
        x = self.cba2(x)
        x = self.atten(x)
        x = self.cba3(x)
        x = self.act(x + identity)
        return x
