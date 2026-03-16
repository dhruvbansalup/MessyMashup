# Training Augmentations

# Applied only while training

import torch
import random
import torchaudio.transforms as T

from src.config import AudioConfig

class TrainAugmentation:
    '''
    Pipeline of augmentations applied to the training data.
    '''

    def __init__(
        self
    ):
        pass

    def __call__(self, spec):

        return spec

class ValTransform:
    '''
    Pipeline of augmentations applied to the validation data.
    Using for making the tensor of the same shape as the training data, without any augmentation.
    '''

    def __init__(
        self,
    ):
        pass

    def __call__(self, spec):

        return spec