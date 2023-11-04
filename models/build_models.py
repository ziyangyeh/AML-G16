from omegaconf import OmegaConf

from utils import Registry, count_channels

MODELS = Registry("models")

# def _fix(cfg: OmegaConf):
#     cfg.model.xyz = True if ["xyz"] == cfg.dataset.mesh_feature_select else False
#     cfg.model.channel = count_channels(cfg.dataset.mesh_feature_select)
#     cfg.model.npoints = cfg.dataset.npoints if "npoints" in cfg.dataset.keys() else None  # PointNet2 use
#     return cfg

def build_model_from_omegaconf(cfg: OmegaConf, **kwargs):
    """
    Build a model, defined by `NAME`.
    Args:
        cfg (eDICT):
    Returns:
        Model: a constructed model specified by NAME.
    """
    # res = _fix(cfg)
    return MODELS.build(dict(cfg.model), **kwargs)
