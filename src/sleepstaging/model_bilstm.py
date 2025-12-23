import torch
import torch.nn as nn

class CNNBiLSTM(nn.Module):
    """
    Epoch-based: įvestis (B, 1, 3000) -> lengvas CNN laiko sumažinimui -> BiLSTM sekai -> klasifikacija.
    """
    def __init__(self, n_classes: int = 5, lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        # 1) Lengvas temporal feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),  # 3000 -> ~1500
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> ~750

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # -> ~375
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),  # ~187

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # -> ~94
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 2) BiLSTM ant sekos (T' ~ 90–100, input_size=128)
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        # 3) Galva: mean pool per laiką + dropout + linear
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * lstm_hidden, n_classes)  # bidirectional => *2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 3000)
        z = self.cnn(x)                # (B, 128, T')
        z = z.transpose(1, 2)          # (B, T', 128) -> batch_first sekai
        y, _ = self.rnn(z)             # (B, T', 2*hidden)
        y = y.mean(dim=1)              # (B, 2*hidden) – global average per laiką
        logits = self.head(y)          # (B, C) – RAW logits
        return logits

