import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """
    Sequence-aware CNN + BiLSTM model.

    Supports:
      - Epoch mode:    x = (B, 1, 3000)
      - Sequence mode: x = (B, L, 3000), where L = seq_len

    In sequence mode, predicts the CENTER epoch label (L // 2),
    which matches SleepEDFSequenceDataset (y_center).
    """

    def __init__(
        self,
        n_classes: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ---- CNN feature extractor (per epoch) ----
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),  # 3000 -> ~1500
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> ~750

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # -> ~375
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),  # -> ~187

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # -> ~94
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # ---- BiLSTM over epoch sequence ----
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        # ---- Classification head ----
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * lstm_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            - (B, 1, 3000)   epoch mode
            - (B, L, 3000)   sequence mode

        Returns
        -------
        logits : torch.Tensor
            (B, n_classes)
        """

        # ===== Epoch mode (backward compatibility) =====
        if x.ndim == 3 and x.shape[1] == 1:
            # (B, 1, 3000)
            z = self.cnn(x)            # (B, 128, T')
            z = z.mean(dim=-1)         # (B, 128)

            z = z.unsqueeze(1)         # (B, 1, 128)
            y, _ = self.rnn(z)         # (B, 1, 2H)
            h = y[:, 0, :]             # (B, 2H)

            return self.head(h)        # (B, C)

        # ===== Sequence-of-epochs mode =====
        if x.ndim == 3:
            # (B, L, 3000)
            B, L, T = x.shape

            # CNN per epoch
            x = x.reshape(B * L, 1, T)     # (B*L, 1, 3000)
            z = self.cnn(x)                # (B*L, 128, T')
            z = z.mean(dim=-1)             # (B*L, 128)

            # Back to sequence
            z = z.reshape(B, L, 128)       # (B, L, 128)

            # BiLSTM over epoch sequence
            y, _ = self.rnn(z)             # (B, L, 2H)

            center = L // 2
            h = y[:, center, :]            # (B, 2H)

            return self.head(h)             # (B, C)

        raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")

