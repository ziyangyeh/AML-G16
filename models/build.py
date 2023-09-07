from omegaconf import OmegaConf
from external import Registry, EXT_MODELS
from utils import count_channels

MODELS = Registry("models", parent=EXT_MODELS)

def _fix(cfg: OmegaConf):
    cfg.model.xyz = True if ["xyz"] == cfg.dataset.mesh_feature_select else False
    cfg.model.channel = count_channels(cfg.dataset.mesh_feature_select)
    return cfg

def build_model_from_omegaconf(cfg: OmegaConf, **kwargs):
    """
    Build a model, defined by `NAME`.
    Args:
        cfg (eDICT):
    Returns:
        Model: a constructed model specified by NAME.
    """
    res = _fix(cfg)
    return MODELS.build(dict(res.model), **kwargs)
