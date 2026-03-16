# Deterministic transformations for data augmentation and preprocessing

import torch
import torchaudio.transforms as T

from src.config import AudioConfig

class BaseTransform:
    """
    Tranformation pipeline
    """
    def __init__(self):
        self.fix_length = WaveformToFixedLength()
        self.log_mel = LogMelSpectogramRransform()
    def __call__(self, waveform):
        fixed_length_waveform = self.fix_length(waveform)
        log_mel = self.log_mel(fixed_length_waveform)
        return log_mel

class LogMelSpectogramRransform:
    """
    Convert a raw waveform into a normalized log mel spectrogram.
    """
    def __init__(self):

        self.mel_transform = T.MelSpectrogram(
            sample_rate=AudioConfig.SAMPLE_RATE,
            n_fft=AudioConfig.N_FFT,
            hop_length=AudioConfig.HOP_LENGTH,
            n_mels=AudioConfig.N_MELS
        )

        # Used to convert to log scale (dB)
        self.db_transform = T.AmplitudeToDB()


    def __call__(self, waveform):
        # Compute the mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale (dB)
        log_mel_spec = self.db_transform(mel_spec)

        # Normalize to zero mean and unit variance
        mean = log_mel_spec.mean()
        std = log_mel_spec.std()
        normalized_log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)

        return normalized_log_mel_spec

class WaveformToFixedLength:
    """
    Pad or trim the waveform to a fixed target_samples (length).
    """
    def __init__(self, target_samples=AudioConfig.SAMPLES_PER_TRACK):
        self.target_length = target_samples

    def __call__(self, waveform):
        current_length = waveform.shape[1]

        if current_length < self.target_length:
            # Pad with zeros
            padding = self.target_length - current_length
            padded_waveform = torch.nn.functional.pad(waveform, (0, padding))
            return padded_waveform
        else:
            # Trim to target length
            return waveform[:, :self.target_length]
