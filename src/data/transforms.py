# Deterministic transformations for data augmentation and preprocessing

import torch
import torchaudio.transforms as T

from src.config import AudioConfig

class BaseTransform:
    """
    Tranformation pipeline
    """
    def __init__(self):
        pass
    def __call__(self, spec):
        return spec