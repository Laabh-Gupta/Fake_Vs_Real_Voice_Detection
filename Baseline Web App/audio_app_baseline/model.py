# In audio_app_baseline/model.py

import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    This class defines the Baseline CNN architecture.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        # Input size from training: 64 channels * 16 height * 15 width = 15360
        self.linear_stack = nn.Sequential(
            nn.Linear(15360, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.linear_stack(self.flatten(self.conv_stack(x)))