import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path

from src.data.dataset import ProcessedDataset, TestDataset
from src.config import AudioConfig, TrainConfig


class MashupDataModule(pl.LightningDataModule):
    '''
    Data module for handling data loading and splitting for training, validation, and testing.
    '''
    def __init__(
            self,
            processed_data_dir,
            test_wav_dir,
            test_csv,
            batch_size=TrainConfig.BATCH_SIZE,
            num_workers=TrainConfig.NUM_WORKERS,
    ):
        super().__init__()

        self.processed_data_dir = Path(processed_data_dir)
        self.test_wav_dir = Path(test_wav_dir)
        self.test_csv = Path(test_csv)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage=None):
        '''
        Build Dataset objects
        '''
        self.train_dataset = ProcessedDataset(self.processed_data_dir, split="train")
        self.val_dataset = ProcessedDataset(self.processed_data_dir, split="val")
        self.test_dataset = TestDataset(self.test_wav_dir, self.test_csv)


    def _make_dataloader(self, dataset, shuffle=False):
    #Helper function to create dataloader with common settings
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=shuffle, # drop last batch during training for consistent batch sizes
        )

    def train_dataloader(self):
        # shuffle for different data order each epoch
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataset, shuffle=False)