import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super().__init__()
       
        self.block=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
    def forward(self, x):
        x = self.block(x)
        return x

class SimpleCNN02(BaseModel):

    def __init__(self, lr:float=1e-3):
        super().__init__(lr=lr)

        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)

        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_pool(x)
        x = self.fc_layers(x)      
        return x
        