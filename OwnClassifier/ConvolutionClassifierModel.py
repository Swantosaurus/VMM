import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionClassifierModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvolutionClassifierModel, self).__init__()

        self.featureExtractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 channels 64x64 img

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 channels 32x32 img

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 channels 16x16 img

            # for 256 images
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 128 channels 16x16 img
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(128 * 16 * 16, 512),
            nn.Linear(256 * 16 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.featureExtractor(x))
