import torch.nn as nn

from .build_models import MODELS

@MODELS.register_module()
class TFuseNet(nn.Module):
    def __init__(self):
        super(TFuseNet, self).__init__()
