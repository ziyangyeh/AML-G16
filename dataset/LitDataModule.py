from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .teeth3ds import Teeth3DS


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: OmegaConf,
        train_dataframe: Optional[pd.DataFrame] = None,
        val_test_dataframe: Optional[pd.DataFrame] = None,
    ):
        super(LitDataModule, self).__init__()
        self.train_dataframe = train_dataframe
        self.val_test_dataframe = val_test_dataframe
        self.num_classes = (cfg.num_classes,)
        self.path_size = cfg.patch_size
        self.batch_size = cfg.batch_size
        self.rearrange = cfg.rearrange
        self.num_workers = cfg.num_workers
        self.transform = None

        self.save_hyperparameters(
            ignore=["cfg", "train_dataframe", "val_test_dataframe"]
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = Teeth3DS(
                dataframe=self.train_dataframe,
                jaw=self.jaw,
                image=self.image_flag,
                depth=self.depth_flag,
                decimated=self.decimated_flag,
                mesh_feature_select=self.feat_select,
                num_classes=self.num_classes,
                transform=self.transform,
            )
            self.val_dataset = Teeth3DS(
                dataframe=self.val_test_dataframe,
                jaw=self.jaw,
                image=self.image_flag,
                depth=self.depth_flag,
                decimated=self.decimated_flag,
                mesh_feature_select=self.feat_select,
                num_classes=self.num_classes,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = Teeth3DS(
                dataframe=self.val_test_dataframe,
                jaw=self.jaw,
                image=self.image_flag,
                depth=self.depth_flag,
                decimated=self.decimated_flag,
                mesh_feature_select=self.feat_select,
                num_classes=self.num_classes,
            )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True, val=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, train=False, val=True)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(
        self, dataset: Teeth3DS, train: bool = False, val: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if train and val else False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if train and val else False,
        )