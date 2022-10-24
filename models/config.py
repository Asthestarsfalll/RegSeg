import copy
import json


class BaseConfig:
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class RegSegConfig(BaseConfig):
    def __init__(self, num_classes: int):
        # base cfg
        self.name = "base"
        # model cfg
        self.stem_out = 32
        self.backbone = "RegSegBody"
        self.backbone_cfg = dict(
            group_width=16,
            channels=[self.stem_out, 48, 128, 256, 320],
            attention="se",
            dilations=[[1], [1, 2]] + 4 * [[1, 4]] + 7 * [[1, 14]],
            pretrained=False,
        )
        self.decoder = "Decoder26"
        self.decoder_cfg = dict(
            header_out=[128, 128, 8], conv_out=[64, 64], num_classes=num_classes
        )


MODEL_CONFIGS = {"base": RegSegConfig}
