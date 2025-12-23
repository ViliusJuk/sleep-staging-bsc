import torch
import torch.nn as nn

class CNN1DBaseline(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
    nn.Conv1d(1, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
    nn.Conv1d(16, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
    nn.Conv1d(32, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(),
    nn.Dropout(0.2),
    nn.Linear(64, n_classes),  # raw logits
)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # jokio softmax/log_softmax

