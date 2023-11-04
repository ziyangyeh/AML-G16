from omegaconf import OmegaConf

from utils import Registry

LOSSES = Registry("losses")

def build_loss_from_omegaconf(cfg: OmegaConf, **kwargs):
    return {f"{name}": LOSSES.build(dict(NAME=name) if name != "Dice" else dict(NAME=name, to_onehot_y=True, softmax=True), **kwargs) for name in cfg.losses}
    # return {f"{name}": LOSSES.build(dict(NAME=name)) for name in cfg.losses}
