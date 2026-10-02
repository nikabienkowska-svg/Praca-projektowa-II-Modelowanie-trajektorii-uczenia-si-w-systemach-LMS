"""Implementacja modeli sekwencyjnych LSTM i GRU w PyTorch."""

import torch
import torch.nn as nn


class StudentDropoutLSTM(nn.Module):
    """Model LSTM z warstwą Dropout i klasyfikatorem binarnym
    do przewidywania porzucenia kursu na podstawie trajektorii aktywności.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        mult = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * mult, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor wejściowy kształtu (batch_size, seq_len, input_dim)
            lengths: Rzeczywiste długości sekwencji (opcjonalnie do pack_padded_sequence)
        Returns:
            Logity predykcji (batch_size, 1)
        """
        lstm_out, (hn, _) = self.lstm(x)
        # Pobieramy stan z ostatniego kroku czasowego
        last_hidden = hn[-1]
        logits = self.classifier(last_hidden)
        return logits.squeeze(-1)
