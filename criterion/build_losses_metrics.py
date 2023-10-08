from omegaconf import OmegaConf
from utils import Registry

LOSSES = Registry("losses")
METRICS = Registry("metrics")


def build_loss_from_omegaconf(cfg: OmegaConf, **kwargs):
    if len(cfg.losses) == 1:
        return {f"{cfg.losses[0]}": LOSSES.build(dict(NAME=cfg.losses[0]), **kwargs)}
    else:
        return {
            f"{name}": LOSSES.build(dict(NAME=name), **kwargs) for name in cfg.losses
        }


def build_metrics_from_omegaconf(cfg: OmegaConf, **kwargs):
    if len(cfg.metrics) == 1:
        return {f"{cfg.metrics[0]}": METRICS.build(dict(NAME=cfg.kwargs[0]), **kwargs)}
    else:
        return {
            f"{name}": METRICS.build(dict(NAME=name), **kwargs) for name in cfg.metrics
        }
