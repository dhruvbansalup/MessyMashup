import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path

from src.data.dataset import MessyMashDataset
from src.config import AudioConfig

class MessyMashDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir,
            batch_size=32,
            num_workers=4,
            val_split=0.1,            
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
    def setup(self, stage=None):
        full_dataset=MessyMashDataset(
            self.data_dir,
            split="train",
            sample_rate=AudioConfig.SAMPLE_RATE,
            duration=AudioConfig.DURATION,
        )

        val_size=int(len(full_dataset) * self.val_split)
        train_size=len(full_dataset) - val_size

        self.train_dataset, self.val_dataset=torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        self.test_dataset=MessyMashDataset(
            self.data_dir,
            split="test"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )