import inspect
from typing import Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import Dice, Recall, Specificity, F1Score
from einops import rearrange
from omegaconf import OmegaConf
from torch.optim import *
from torch.optim.lr_scheduler import *

from .build_models import build_model_from_omegaconf
from criterion.build_losses import LOSSES

class LitModule(pl.LightningModule):
    def __init__(self, cfg: OmegaConf, train_test: str) -> None:
        super(LitModule, self).__init__()
        self.cfg = cfg
        self.model = build_model_from_omegaconf(cfg)
        self.forward_paramlist = [name for name, _ in inspect.signature(self.model.forward).parameters.items()]
        self.loss_lst = []
        for name in cfg[train_test].losses:
            setattr(
                self,
                f"{name.lower()}_fn",
                LOSSES.build(dict(NAME=name) if name != "Dice" else dict(NAME=name, to_onehot_y=True)),
            )
            self.loss_lst.append(name.lower())
        self.losses_weights = (
            [1 / len(cfg[train_test].losses)] * len(cfg[train_test].losses) if "losses_weights" not in cfg.train.keys() else cfg.train.losses_weights
        )

        self.dice_metric = Dice(multiclass=True, num_classes=cfg.model.num_classes)
        self.recall_metric = Recall(task="multiclass", num_classes=cfg.model.num_classes)
        self.specificity_metric = Specificity(task="multiclass", num_classes=cfg.model.num_classes)
        self.f1_metric = F1Score(task="multiclass", num_classes=cfg.model.num_classes)
        self.metric_lst = ["dice", "recall", "specificity", "f1"]
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
        for name, w in zip(self.loss_lst, self.losses_weights):
            loss_fn = getattr(self, f"{name}_fn")
            if name == "Dice":
                _loss = w * loss_fn(rearrange(outputs, "b n c -> b c n"), batch["labels"].unsqueeze(1))
            else:
                _loss = w * loss_fn(
                    rearrange(outputs, "b n c -> (b n) c"),
                    rearrange(batch["labels"], "b n ->(b n)"),
                )
            loss_lst.append(_loss)
            self.log(f"{step}_{name}_loss", _loss, sync_dist=True)
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
