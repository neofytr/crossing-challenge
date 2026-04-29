from __future__ import annotations

import torch
import torch.nn as nn


class CrossingModel(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8),
        )

        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"CrossingModel: {total:,} parameters")

    def forward(self, x):
        x = self.layer_norm(self.input_proj(x))
        gru_out, _ = self.gru(x)
        last = gru_out[:, -1, :]
        traj = self.traj_head(last).view(-1, 4, 2)
        intent = self.intent_head(last).squeeze(-1)
        return traj, intent
