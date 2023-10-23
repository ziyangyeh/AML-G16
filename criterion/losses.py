from monai.losses import *
from torch.nn import *
import torch.nn as nn

from .build_losses import LOSSES
from .lovasz_losses import lovasz_softmax_flat

@LOSSES.register_module()
class Lovasz(nn.Module):
    def __init__(self) -> None:
        super(Lovasz, self).__init__()
    def forward(self, probas, labels):
        return lovasz_softmax_flat(probas, labels)

LOSSES.register_module(name="NLL", module=NLLLoss)
LOSSES.register_module(name="Dice", module=DiceLoss)
LOSSES.register_module(name="CE", module=CrossEntropyLoss)
