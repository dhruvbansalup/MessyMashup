import torch
import torch.nn as nn
import torchvision.models as models

from src.models.base_model import BaseModel

class Pretrained001(BaseModel):
    def __init__(
        self,
        lr: float = 3e-4,
    ):
        super().__init__(lr=lr)
        
        #Loading Pretrained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # First conv layer modification to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Freeze ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # unfreeze the last block and the fully connected layer for fine-tuning
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # Replacing classification head
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        return self.resnet(x)