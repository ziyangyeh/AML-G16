import torch.nn as nn
from kornia.losses import DiceLoss
from monai.losses import GeneralizedDiceLoss

# from monai.losses import *
from torch.nn import CrossEntropyLoss, NLLLoss

from .build_losses import LOSSES
from .lovasz import LovaszLoss

LOSSES.register_module(name="NLL", module=NLLLoss)
LOSSES.register_module(name="Dice", module=GeneralizedDiceLoss)
LOSSES.register_module(name="CE", module=CrossEntropyLoss)
LOSSES.register_module(name="Lovasz", module=LovaszLoss)
