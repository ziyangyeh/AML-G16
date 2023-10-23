from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import vedo
from utils import GetVTKTransformationMatrix
from .teeth3ds import Teeth3DS

def augment(mesh: vedo.Mesh):
    vtk_matrix = GetVTKTransformationMatrix(
        rotate_X=[-180, 180],
        rotate_Y=[-180, 180],
        rotate_Z=[-180, 180],
        translate_X=[-10, 10],
        translate_Y=[-10, 10],
        translate_Z=[-10, 10],
        scale_X=[0.8, 1.2],
        scale_Y=[0.8, 1.2],
        scale_Z=[0.8, 1.2],
    )
    mesh.apply_transform(vtk_matrix)
    return mesh

class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: OmegaConf,
        train_dataframe: Optional[pd.DataFrame] = None,
        val_test_dataframe: Optional[pd.DataFrame] = None,
    ):
        super(LitDataModule, self).__init__()
        self.cfg = cfg
        self.train_dataframe = train_dataframe
        self.val_test_dataframe = val_test_dataframe
        self.num_classes = cfg.model.num_classes
        # self.path_size = cfg.patch_size
        self.batch_size = cfg.dataloader.batch_size
        self.num_workers = cfg.dataloader.num_workers
        self.transform = False
        if cfg.dataset.transform:
            self.transform = augment

        self.save_hyperparameters(ignore=["cfg", "train_dataframe", "val_test_dataframe"])
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = Teeth3DS(
                dataframe=self.train_dataframe,
                num_classes=self.num_classes,
                mesh_aug=self.transform,
                **self.cfg.dataset,
            )
            self.val_dataset = Teeth3DS(
                dataframe=self.val_test_dataframe,
                num_classes=self.num_classes,
                **self.cfg.dataset,
            )
        if stage == "test" or stage is None:
            self.test_dataset = Teeth3DS(
                dataframe=self.val_test_dataframe,
                num_classes=self.num_classes,
                **self.cfg.dataset,
            )
    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True, val=True)
    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, train=False, val=True)
    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)
    def _dataloader(self, dataset: Teeth3DS, train: bool = False, val: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if train and val else False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if train and val else False,
        )
