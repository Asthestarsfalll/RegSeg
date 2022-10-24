from models.config import BaseConfig


class Config(BaseConfig):
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)
