import torch
from torch.utils.data import Dataset
import torchaudio
import random

from ..config import GENRES, STEM_FILES
from ..utils import genre_to_idx

class MessyMashDataset(Dataset):
    def __init__(
            self,
            data_dir,
            split="train", # train, val, test
            sample_rate=22050,
            duration=10,
    ):
        self.data_dir = data_dir
        self.split = split
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples=self.sample_rate * self.duration

        if split in ["train", "val"]:
            self.data=self._load_train_data()
        else:
            self.data=self._load_test_data()
        
    def _load_train_data(self):
        data=[]
        genres_path=self.data_dir / "genres_stems"

        for genre in GENRES:
            genre_dir=genres_path / genre
            for song_dir in genre_dir.iterdir():
                if song_dir.is_dir():
                    data.append((song_dir, genre_to_idx(genre)))
        return data


    def _load_test_data(self):
        mashup_dir=self.data_dir / "mashups"
        files=list(mashup_dir.glob("*.wav"))
        return files
    
    def _mix_stems(self, genre):
        stems=[]
        genre_dir=self.data_dir / "genres_stems" / genre
        all_songs=list(genre_dir.iterdir())

        for stem_name in STEM_FILES:

            #Randomly select a song from the genre
            random_song=random.choice(all_songs)

            stem_file=random_song / stem_name

            #Loading the stem audio file
            waveform, sr=torchaudio.load(stem_file) 

            #Resample the audio to the desired sample rate
            waveform=torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )(waveform)

            #Convert to mono by averaging the channels if audio is stereo
            waveform=waveform.mean(dim=0, keepdim=True)
            
            # random instrument gain (simulating balance shifts in the mix)
            gain=random.uniform(0.5, 1.0)
            waveform=waveform * gain

            stems.append(waveform)
        
        #Mix the stems together by summing them
        mixed_waveform=sum(stems)

        return mixed_waveform

    def _add_noise(self, audio):
        # Randomly selecting the noise
        noise_files=list((self.data_dir / "noise").glob("*.wav"))
        noise_file=random.choice(noise_files)

        noise_waveform, sr=torchaudio.load(noise_file)
        noise_waveform=torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=self.sample_rate
        )(noise_waveform)

        # Convert to mono
        noise_waveform=noise_waveform.mean(dim=0, keepdim=True)

        # Pad the noise to match the length of the audio i.e adding silence
        if len(noise_waveform) < len(audio):
            noise_waveform=torch.nn.functional.pad(
                noise_waveform, (0, len(audio) - len(noise_waveform))
            )
        else:
            noise_waveform=noise_waveform[:len(audio)]
        
        # Randomly selecting the noise level
        noise_level=random.uniform(0.01, 0.1)
        noisy_audio=audio + noise_level * noise_waveform
        return noisy_audio

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if self.split in ["train", "val"]:
            
            song_dir, genre_idx=self.data[idx]
            mixed_audio=self._mix_stems(GENRES[genre_idx])
            noisy_audio=self._add_noise(mixed_audio)

            # Ensure the audio is the correct length by trimming or padding
            if len(noisy_audio) > self.samples:
                noisy_audio=noisy_audio[:self.samples]
            else:
                noisy_audio=torch.nn.functional.pad(
                    noisy_audio, (0, self.samples - len(noisy_audio))
                )
                           
            return noisy_audio, genre_idx
    
        else:
            mashup_file=self.data[idx]
            audio, sr=torchaudio.load(mashup_file)

            audio=torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )(audio)

            # Convert to mono
            audio=audio.mean(dim=0, keepdim=True)

            # Ensure the audio is the correct length by trimming or padding
            if len(audio) > self.samples:
                audio=audio[:self.samples]
            else:
                audio=torch.nn.functional.pad(
                    audio, (0, self.samples - len(audio))
                )
            
            return audio