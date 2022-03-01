import torch
import torch.nn as nn


class BnAct(nn.Module):
    def __init__(self, out_ch, apply_act='none'):
        super(BnAct, self).__init__()
        self.bn = nn.BatchNorm2d(out_ch)
        if apply_act == 'none':
            self.act = nn.Identity()
        elif apply_act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError()  # custom yourself

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvBnAct(BnAct):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        apply_act='none'
    ):
        super(ConvBnAct, self).__init__(out_ch=out_ch, apply_act=apply_act)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride, padding, dilation, groups, bias)


class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super(SEBlock, self).__init__()
        assert in_ch // reduction > 0
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//reduction, in_ch, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        atten = self.gap(x)
        atten = self.stem(atten)
        return x * atten


class DilationConv(nn.Module):
    def __init__(
        self,
        in_ch,
        stride,
        dilations,
        group_width,
        bias=False
    ):
        super(DilationConv, self).__init__()
        self.num_splits = len(dilations)
        assert in_ch % self.num_splits == 0
        each_conv_chan = in_ch // self.num_splits
        assert each_conv_chan % group_width == 0
        num_groups = each_conv_chan // group_width
        self.branches = self._make_layers(
            dilations, each_conv_chan, stride, bias, num_groups)

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
                    groups=num_groups
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
        in_ch,
        stride,
        dilations,
        group_width,
        bias=False,
        apply_act='none'
    ):
        super(DilationConvBnAct, self).__init__(
            out_ch=in_ch, apply_act=apply_act)
        self.conv = DilationConv(in_ch, stride, dilations, group_width)


class ShortCut(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        stride,
        avg_downsample: False
    ):
        super(ShortCut, self).__init__()
        if avg_downsample and stride != 1:
            self.avgpool = nn.AvgPool2d(2, 2, ceil_mode=True)
            self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(self.avgpool(x)))


class DilationBlock(nn.Module):
    """
        DBlock
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        stride,
        dilations,
        group_width,
        attention='se'
    ):
        super(DilationBlock, self).__init__()
        groups = out_ch // group_width

        self.cba1 = ConvBnAct(in_ch, out_ch, apply_act='relu')

        # multi-dilations-branch
        if len(dilations) == 1:
            dilation = dilations[0]
            self.cba2 = ConvBnAct(out_ch, out_ch, kernel_size=3, stride=stride,
                                  groups=groups, padding=dilation, dilation=dilation, bias=False, apply_act='relu')
        else:
            self.cba2 = DilationConvBnAct(
                out_ch, stride, dilations, group_width, bias=False, apply_act='relu')

        self.cba3 = ConvBnAct(out_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        # attention
        if attention == 'none':
            self.atten = nn.Identity()
        elif attention == 'se':
            self.atten = SEBlock(out_ch, reduction=4)
        else:
            raise NotImplementedError()

        # shortcut
        if stride != 1 or in_ch != out_ch:
            # self.shortcut =
            self.shortcut = ShortCut(
                in_ch, out_ch, stride, avg_downsample=True)
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


if __name__ == "__main__":
    se = SEBlock(32, 16)
    conv = ConvBnAct(in_ch=32, out_ch=16)
    dc = DilationConv(16, 1, [2, 4], group_width=2)
    inp = torch.randn((2, 32, 224, 224))
    out = se(inp)
    print(out.shape)
    out = conv(out)
    print(out.shape)
    out = dc(out)
    print(out.shape)
