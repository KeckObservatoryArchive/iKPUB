"""
Classification heads for transformer-based models.

All heads take (batch, input_dim) and produce (batch, 1) logits.
"""

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Single hidden layer: input_dim → 256 → 1."""

    def __init__(self, input_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.dense = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DeepMLPHead(nn.Module):
    """Two hidden layers: input_dim → 256 → 64 → 1."""

    def __init__(self, input_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


HEADS = {
    "mlp": MLPHead,
    "deep_mlp": DeepMLPHead,
}
