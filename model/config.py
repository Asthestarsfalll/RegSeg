

class RegSegConfig(object):
    def __init__(self):
        # base cfg
        self.name = 'base'
        # model cfg
        self.stem_out = 32
        self.backbone = 'RegSegBody'
        self.backbone_cfg = dict(
            group_width=16,
            channels=[self.stem_out, 48, 128, 256, 320],
            attention='se',
            dilations=[[1], [1, 2]]+4*[[1, 4]]+7*[[1, 14]],
            pretrained=False
        )
        self.decoder = 'Decoder26'
        self.decoder_cfg = dict(
            header_out=[128, 128, 8],
            conv_out=[64, 64],
            num_classes=19
        )
        # train cfg?

        # test cfg?


def log_cfg(cfg):
    pass
