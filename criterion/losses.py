from monai.losses import *
from torch.nn import *
import torch
import torch.nn as nn

from .build_losses_metrics import LOSSES
from .lovasz_losses import lovasz_softmax

LOSSES.register_module(name="NLL", module=NLLLoss)
LOSSES.register_module(name="Dice", module=DiceLoss)
# LOSSES.register_module(name="CE", module=CrossEntropyLoss)

LOSSES.register_module()
class Lovasz(nn.Module):
    def __init__(self) -> None:
        super(Lovasz, self).__init__()
    def forward(self, probas,labels):
        _labels=torch.argmax(labels, dim=2)
        return lovasz_softmax(probas,_labels)
