import inspect
from typing import Callable, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchmetrics.classification import *

from criterion.build_losses import LOSSES

from .build_models import build_model_from_omegaconf

class LitModule(pl.LightningModule):
    def __init__(self, cfg: OmegaConf, train_test: str) -> None:
        super(LitModule, self).__init__()
        self.cfg = cfg
        self.lr = self.cfg.train.optimizer.lr
        self.model = build_model_from_omegaconf(cfg)
        self.forward_paramlist = [name for name, _ in inspect.signature(self.model.forward).parameters.items()]
        self.loss_lst = []
        for name in cfg[train_test].losses:
            setattr(
                self,
                f"{name.lower()}_fn",
                LOSSES.build(dict(NAME=name) if name != "Dice" else dict(NAME=name, to_onehot_y=True, softmax=True)),
            )
            self.loss_lst.append(name.lower())
        self.losses_weights = [1 / len(cfg[train_test].losses)] * len(cfg[train_test].losses) if "losses_weights" not in cfg.train.keys() else cfg.train.losses_weights

        self.dice_metric = Dice(multiclass=True, num_classes=cfg.model.num_classes, average="macro")
        self.precision_metric = Precision(task="multiclass", num_classes=cfg.model.num_classes, average="macro")
        self.recall_metric = Recall(task="multiclass", num_classes=cfg.model.num_classes, average="macro")
        self.specificity_metric = Specificity(task="multiclass", num_classes=cfg.model.num_classes, average="macro")
        self.metric_lst = ["dice", "precision", "recall", "specificity"]
        self.save_hyperparameters(ignore=["cfg"])
    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        _input = {k: v for k, v in X.items() if k in self.forward_paramlist}
        outputs = self.model(**_input)
        return outputs
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.cfg.train.optimizer.weight_decay,
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cfg.train.scheduler.T_0,
            T_mult=self.cfg.train.scheduler.T_mult,
            eta_min=self.cfg.train.scheduler.eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self(batch).transpose(2, 1).softmax(dim=-1)
    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        outputs = self(batch)
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        outputs_m = outputs.clone().detach()
        loss_lst = []
        for name, w in zip(self.loss_lst, self.losses_weights):
            loss_fn = getattr(self, f"{name}_fn")
            if name == "dice":
                _loss = loss_fn(rearrange(outputs, "b n c -> b c n"), batch["labels"].unsqueeze(1))
                self.log(f"{step}_{name}_loss", _loss, sync_dist=True)
                _loss *= w
            else:
                _loss = loss_fn(
                    rearrange(outputs, "b n c -> (b n) c"),
                    rearrange(batch["labels"], "b n ->(b n)"),
                )
                self.log(f"{step}_{name}_loss", _loss, sync_dist=True)
                _loss *= w
            loss_lst.append(_loss)

        loss = torch.stack(loss_lst).sum()
        self.log(f"{step}_loss", loss, sync_dist=True, prog_bar=True)

        with torch.no_grad():
            for name in self.metric_lst:
                metric_fn = getattr(self, f"{name}_metric")
                _mertric = metric_fn(
                    rearrange(outputs_m, "b n c -> (b n) c"),
                    rearrange(batch["labels"], "b n ->(b n)"),
                )
                self.log(f"{step}_{name}_metric", _mertric, sync_dist=True)
        return loss
