import argparse
import gc
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold, train_test_split

from dataset import LitDataModule
from models import LitModule

torch.set_float32_matmul_precision("high")

def train(
    cfg: OmegaConf,
    train_dataframe: Optional[pd.DataFrame] = None,
    val_test_dataframe: Optional[pd.DataFrame] = None,
    train_test: str = "train",
):
    pl.seed_everything(cfg.seed)  # ENSURE REPRODUCIBILITY

    datamodule = LitDataModule(cfg=cfg, train_dataframe=train_dataframe, val_test_dataframe=val_test_dataframe)

    datamodule.setup()

    module = LitModule(cfg, train_test=train_test)

    loss_model_checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_loss",
        mode="min",
        filename=f"{cfg.model.NAME}_{cfg.model.num_classes}_classes_{cfg.train.precision}_f_best_loss",
        verbose="True",
    )

    dsc_model_checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_dice_metric",
        mode="max",
        filename=f"{cfg.model.NAME}_{cfg.model.num_classes}_classes_{cfg.train.precision}_f_best_dice",
        verbose="True",
    )

    trainer = pl.Trainer(
        # fast_dev_run=True,
        callbacks=[loss_model_checkpoint, dsc_model_checkpoint],
        # benchmark=False,  # ENSURE REPRODUCIBILITY
        # deterministic=False,  # ENSURE REPRODUCIBILITY
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy="ddp" if cfg.train.ddp else "auto",
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=WandbLogger(project="TFuseNet", name=f"{cfg.model.NAME}-{cfg.train.precision}"),
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
    )

    trainer.fit(module, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "-cfg",
        "--config_file",
        type=str,
        metavar="",
        help="configuration file",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)
    dataframe = pd.read_csv(cfg.dataset.csv_path)
    train_df, val_df = train_test_split(dataframe, train_size=cfg.dataset.train_val_ratio, random_state=cfg.seed, shuffle=True)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train(cfg, train_df, val_df)
