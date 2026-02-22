import torch
from torch.utils.data import Dataset
import torchaudio
import random
from pathlib import Path

from src.config import AudioConfig
from src.utils import genre_to_idx

class MessyMashDataset(Dataset):
    def __init__(
            self,
            data_dir,
            split="train", # train, val, test
            sample_rate=AudioConfig.SAMPLE_RATE,
            duration=AudioConfig.DURATION,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples=self.sample_rate * self.duration

        # Preloading noise files
        self.noise_dir = self.data_dir / "ESC-50-master" / "audio"
        self.noise_files = list(self.noise_dir.glob("*.wav"))

        if split in ["train", "val"]:
            self.data=self._load_train_data()
        else:
            self.data=self._load_test_data()
        
    def _load_train_data(self):
        data=[]
        genres_path=self.data_dir / "genres_stems"

        for genre in AudioConfig.GENRES:
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

        for stem_name in AudioConfig.STEM_FILES:

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
        
        min_length = min(stem.shape[1] for stem in stems)
        # Trim all stems
        stems = [stem[:, :min_length] for stem in stems]

        # Mix the stems together by summing them
        mixed_waveform = sum(stems)

        return mixed_waveform

    def _add_noise(self, audio):
       
        if not self.noise_files:
            print(f"No noise files found in {self.noise_dir}. Returning original audio.")
            return audio

        # Randomly selecting the noise
        noise_file=random.choice(self.noise_files)

        noise_waveform, sr=torchaudio.load(noise_file)

        noise_waveform=torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=self.sample_rate
        )(noise_waveform)

        # Convert to mono
        noise_waveform=noise_waveform.mean(dim=0, keepdim=True)

        audio_length=audio.shape[1]
        noise_length=noise_waveform.shape[1]

        # Trim noise if longer
        if noise_length > audio_length:
            noise_waveform = noise_waveform[:, :audio_length]
            noise_length = audio_length
        
        #Random position insertion
        start=random.randint(0, audio_length - noise_length)

        padded_noise=torch.nn.functional.pad(
            noise_waveform, (start, audio_length - noise_length - start)
        )

        # Randomly selecting the noise level
        noise_level=random.uniform(0.01, 0.1)

        noisy_audio=audio + noise_level * padded_noise
        return noisy_audio
    
    def _random_crop(self, audio):
        audio_len = audio.shape[1]

        if audio_len <= self.samples:
            # pad if too short
            pad_size = self.samples - audio_len
            audio = torch.nn.functional.pad(audio, (0, pad_size))
            return audio

        start = random.randint(0, audio_len - self.samples)
        return audio[:, start:start + self.samples]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if self.split in ["train", "val"]:
            
            song_dir, genre_idx=self.data[idx]
            mixed_audio=self._mix_stems(AudioConfig.GENRES[genre_idx])
            noisy_audio=self._add_noise(mixed_audio)

            noisy_audio=self._random_crop(noisy_audio)
                
            return noisy_audio, genre_idx
    
        else:
            mashup_file=self.data[idx]
            audio, sr=torchaudio.load(mashup_file)

            audio=torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )(audio)

            # Convert to mono
            audio=audio.mean(dim=0, keepdim=True)

            audio = self._random_crop(audio)
            
            return audio
