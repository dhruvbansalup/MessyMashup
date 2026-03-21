import torch
from torch.utils.data import Dataset
import torchaudio
import random
from pathlib import Path
import pandas as pd

from src.config import AudioConfig
from src.utils import genre_to_idx


class ProcessedDataset(Dataset):
    '''
    Dataset for pre-mixed spectograms .pt files.

    Directory structure expected:
        processed_dir/
            blues/
                blues_mix_0000.pt
                blues_mix_0001.pt
                ...
            classical/
                classical_mix_0000.pt
                ...

    train — labelled processed .pt files, with augmentation
    val   — labelled processed .pt files, no augmentation

    '''
    def __init__(
            self,
            processed_dir,
            split="train", # train, val
    ):
        self.processed_dir = Path(processed_dir)
        self.split = split

        self.files = []
        self.labels = []
        
        for genre in AudioConfig.GENRES:
            genre_dir = self.processed_dir / split / genre
            
            pt_files = list(genre_dir.glob("*.pt"))

            self.files.extend(pt_files)
            self.labels.extend([genre_to_idx(genre)] * len(pt_files))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load the pre-mixed spectrogram from the .pt file
        spec = torch.load(self.files[idx], weights_only=True)

        label = self.labels[idx]
        return spec, label

class TestDataset(Dataset):
    '''
    Dataset for raw .wav files in the test set.
    '''
    def __init__(
            self,
            test_dir,
            test_csv, # For getting ID
    ):
        self.test_dir = Path(test_dir)

        # Using test.csv to get the list of test IDs, and then loading corresponding .wav files
        df=pd.read_csv(test_csv)
        self.ids=df['id'].tolist()
        self.files=[self.test_dir / f for f in df['filename']]
        
        print(f"Test Dataset: {len(self.files)} samples.")

        # Storing - Resempler to ensure all test audio is at the same sample rate as training data
        self.resampler = {}
        
    
    def _get_resampler(self, original_SR):
        if original_SR not in self.resampler:
            self.resampler[original_SR] = torchaudio.transforms.Resample(orig_freq=original_SR, new_freq=AudioConfig.SAMPLE_RATE)
        return self.resampler[original_SR]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load the raw waveform from the .wav file
        waveform, sr = torchaudio.load(self.files[idx])

        # Fixed Length
        target_len = AudioConfig.SAMPLE_RATE * 30
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_len - waveform.shape[1])
            )

        # Resample if needed
        if sr != AudioConfig.SAMPLE_RATE:
            resampler = self._get_resampler(sr)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        #[1,22050]

        # Convert to spectogram
        mel = torchaudio.transforms.MelSpectrogram(
                        sample_rate=AudioConfig.SAMPLE_RATE,
                        n_mels=AudioConfig.N_MELS
                    )(waveform)

        #[1, n_mels, time_frames] -> [1, 128, 128]

        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
        
        # Interpolate to 128x128
        mel_db = torch.nn.functional.interpolate(
            mel_db.unsqueeze(0),  
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return mel_db, self.ids[idx]