import torch
import torch.nn as nn
import torchvision.models as models

from src.models.base_model import BaseModel


class Pretrained001(BaseModel):
    def __init__(
        self,
        lr: float = 1e-4
    ):
        super().__init__(lr=lr)
    