# Script to make the training dataset from the provided stems and noise files. 
#This is a CPU intensive process and should be done before training to avoid bottlenecks.

import os
import random
import torch
from pathlib import Path
import torchaudio
import librosa
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
import shutil # for copying files

from src.config import EnvConfig
from src.data.dataset import AudioConfig

TEMPO_BINS=[(0, 80), (80, 100), (100, 120), (120, 300)]

def get_tempo_bins(stems_dir):
    '''
    Get the tempo bins for the given stems directory.
    '''
     #Loading only 10 sec to get tempo to save time, as tempo is usally same throughout
    y, sr= librosa.load(stems_dir, sr=AudioConfig.SAMPLE_RATE, duration=10)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # return index of bin it belongs
    for i, (lo, hi) in enumerate(TEMPO_BINS):
        if lo <= float(tempo) < hi:
            return i
    return len(TEMPO_BINS) - 1 #index of last bin if tempo is above 300 BPM

def load_and_resample(path, target_sr=AudioConfig.SAMPLE_RATE):
    '''
    Load the audio file and resample it to the target sample rate.
    '''
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform.mean(dim=0, keepdim=True) # Convert to mono by averaging channels

def mix_stems(songs_in_bin, noise_waveforms):
    '''
    Mix the stems of the given songs in the bin and add noise.
    '''
    stems = []

    for stem_name in AudioConfig.STEM_FILES:
        #Randomly selecting song from bin to get stem
        song = random.choice(songs_in_bin) 
        waveform = load_and_resample(song / stem_name)

        # Adding random gain to stem to different loudness the mix
        gain = random.uniform(0.5, 1.0)
        stems.append(waveform * gain)

    min_len = min(s.shape[1] for s in stems)

    mixed = sum(s[:, :min_len] for s in stems)

    # Add noise - Randomly
    noise = random.choice(noise_waveforms).clone()

    audio_len = mixed.shape[1] #duration
    noise_len = noise.shape[1]

    #trim & pad noise to fit the audio length
    if noise_len > audio_len:
        noise = noise[:, :audio_len]
        noise_len = audio_len
    start = random.randint(0, audio_len - noise_len) #randomly placing noise in the audio
    padded = torch.nn.functional.pad(noise, (start, audio_len - noise_len - start)) 
    mixed = mixed + random.uniform(0.01, 0.1) * padded

    # Crop to exactly 30s
    target = AudioConfig.SAMPLE_RATE * 30
    if mixed.shape[1] >= target:
        start = random.randint(0, mixed.shape[1] - target)
        mixed = mixed[:, start:start + target]
    else:
        mixed = torch.nn.functional.pad(mixed, (0, target - mixed.shape[1]))

    return mixed

def build_mashup_dataset(mixes_per_song=15, val_songs_per_genre=10):
    '''
    Build the dataset by mixing the stems and adding noise.
    Also splits into train and val
    '''
    # Paths
    noise_dir = Path(EnvConfig.DATA_DIR) / "ESC-50-master" / "audio"
    output_dir = Path(EnvConfig.PROCESSED_DATA_DIR)
    stems_dir = Path(EnvConfig.DATA_DIR) / "genres_stems"
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    # Create the output directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True) 

    # Preload noise
    noise_waveforms = [load_and_resample(f) for f in tqdm(noise_dir.glob("*.wav"), desc="Loading noise")]

    random.seed(42)

    for genre in AudioConfig.GENRES:
        genre_dir = Path(stems_dir) / genre

        songs = list(genre_dir.iterdir())
        
        # Bucketing songs by tempo using drums stem
        print(f"Computing tempos for {genre}...")
        
        bins = defaultdict(list)
        for song in tqdm(songs):
            drums = song / "drums.wav"
            if drums.exists():
                tempo_bin = get_tempo_bins(drums)
                bins[tempo_bin].append(song)

        # If a tempo bin is empty, use all songs in the genre
        for b in range(len(TEMPO_BINS)):
            if not bins[b]:
                bins[b] = songs
        
        train_out_genre_dir = train_dir / genre
        val_out_genre_dir = val_dir / genre
        train_out_genre_dir.mkdir(parents=True, exist_ok=True)
        val_out_genre_dir.mkdir(parents=True, exist_ok=True)

        # Randomly select which songs go to val — computed ONCE per genre
        index_of_val_songs = set(random.sample(range(len(songs)), min(val_songs_per_genre, len(songs))))

        # Cache tempo bins so get_tempo_bins isn't called twice per song
        song_tempo_bins = {}
        for song in songs:
            drums = song / "drums.wav"
            if drums.exists():
                song_tempo_bins[song] = get_tempo_bins(drums)

        mix_idx = 0
        for i, song in tqdm(enumerate(songs), desc=f"Processing {genre}"):
            tempo_bin = song_tempo_bins.get(song, 0)

            for _ in range(mixes_per_song):
                # Mix the stems of songs in the same tempo bin and add noise
                mixed = mix_stems(bins[tempo_bin], noise_waveforms)

                # Save the mixed audio to val / train folder based on the song index and val_songs_per_genre
                if i in index_of_val_songs:
                    out_path = val_out_genre_dir / f"{song.stem}_mix_{mix_idx}.pt"
                else:
                    out_path = train_out_genre_dir / f"{song.stem}_mix_{mix_idx}.pt"
                torch.save(mixed, out_path) # Saving as .pt for faster loading
                mix_idx += 1

        print(f"{genre}: saved {mix_idx} mixes")

if __name__ == "__main__":
    build_mashup_dataset()