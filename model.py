import torch
import torch.nn as nn

from config import DEVICE, SEED


class BasicCNN(nn.Module):
    """4-Block Convolutional Neural Network for Cat vs. Dog Classification."""

    def __init__(self):
        super().__init__()

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.classifier(x)
        return x


def build_model(device=DEVICE, seed=SEED):
    """Instantiates BasicCNN with a fixed random seed and moves it to the target device."""
    torch.manual_seed(seed)
    model = BasicCNN().to(device)
    return model


def load_trained_model(weights_path, device=DEVICE):
    """Loads saved state_dict weights into a BasicCNN instance in eval mode."""
    model = BasicCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    net = build_model()
    print(f"Model initialized on {DEVICE}:\n", net)

