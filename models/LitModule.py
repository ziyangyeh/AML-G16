import importlib
import inspect
from typing import Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from einops import rearrange
from omegaconf import OmegaConf
from torch.optim import *
from torch.optim.lr_scheduler import *

from .build_models import build_model_from_omegaconf
from criterion.build_losses_metrics import (build_loss_from_omegaconf,
                                            build_metrics_from_omegaconf)

class LitModule(pl.LightningModule):
    def __init__(self, cfg: OmegaConf, train_test: str) -> None:
        super(LitModule, self).__init__()
        self.cfg = cfg
        self.model = build_model_from_omegaconf(cfg)
        self.forward_paramlist = [name for name, _ in inspect.signature(self.model.forward).parameters.items()]
        self.losses = build_loss_from_omegaconf(cfg[train_test])
        self.losses_weights = [1 / len(self.losses)] * len(self.losses) if "losses_weights" not in cfg.train.keys() else cfg.train.losses_weights
        if cfg[train_test].metrics:
            self.metrics = build_metrics_from_omegaconf(cfg[train_test])
            self.acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
            self.metrics.update({"Accuracy": self.acc_fn})
        else:
            self.metrics=None
        self.save_hyperparameters(ignore=["cfg"])
    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        _input = {k: v for k, v in X.items() if k in self.forward_paramlist}
        _input["x"] = rearrange(_input["x"], "b n c -> b c n")
        outputs = self.model(**_input)
        return outputs
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.train.optimizer.lr,
            weight_decay=self.cfg.train.optimizer.weight_decay,
        )

        scheduler = StepLR(
            optimizer,
            step_size=self.cfg.train.scheduler.step_size,
            gamma=self.cfg.train.scheduler.gamma,
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
        outputs_m = outputs.clone()
        loss_lst = []
        for w, (name, loss_fn) in zip(self.losses_weights, self.losses.items()):
            if name=="NLL":
                outputs = rearrange(outputs, "b n c -> (b n c)")
                labels = rearrange(batch["labels"], "b n c -> (b n c)")
                _loss = w * loss_fn(outputs, labels)
            else:
                _loss = w * loss_fn(outputs, batch["labels"])
            loss_lst.append(_loss)
            if len(loss_lst) > 1:
                self.log(f"{step}_{name}_loss", _loss, sync_dist=True)
        loss = loss_lst[0] if len(loss_lst) == 1 else torch.stack([loss_lst]).sum()
        self.log(f"{step}_loss", loss, sync_dist=True, prog_bar=True)
        
        if self.metrics:
            with torch.no_grad():
                for name, metric_fn in self.metrics.items():
                    _metric = metric_fn(outputs_m, batch["labels"])
                    self.log(f"{step}_{name}", _metric.mean(), sync_dist=True)
        return loss
