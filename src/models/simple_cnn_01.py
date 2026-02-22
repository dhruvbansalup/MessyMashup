import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel

class SimpleCNN01(BaseModel):
    def __init__(self, lr:float=1e-3):
        super().__init__(lr=lr)
       
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Output layer
        self.output_layer = nn.Linear(512, self.num_classes)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)     # (B, 64, 1)
        x = x.squeeze(-1)           # (B, 64)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x