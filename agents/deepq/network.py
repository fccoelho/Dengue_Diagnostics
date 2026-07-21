"""Rede Q do DQN para o ambiente DengueDiag (map tensor + coordenadas do caso)."""
from __future__ import annotations

import torch
from torch import nn


class DengueNet(nn.Module):
    """Processa obs Dict ``{"map": (B, C, H, W), "case_coords": (B, 2)}`` → Q (B, 6)."""

    def __init__(self, map_shape, action_shape, device="cpu"):
        super().__init__()
        self.device = device

        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_map = torch.zeros(1, *map_shape)
            map_feature_dim = self.map_encoder(dummy_map).shape[1]

        self.coord_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(map_feature_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_shape),
        )

    def forward(self, obs, state=None, info=None):
        device = next(self.parameters()).device
        map_tensor = torch.as_tensor(obs["map"], device=device, dtype=torch.float32)
        coords_tensor = torch.as_tensor(
            obs["case_coords"], device=device, dtype=torch.float32
        )

        if map_tensor.ndim == 3:
            map_tensor = map_tensor.unsqueeze(0)
        if coords_tensor.ndim == 1:
            coords_tensor = coords_tensor.unsqueeze(0)

        map_features = self.map_encoder(map_tensor)
        coord_features = self.coord_encoder(coords_tensor)
        q_values = self.head(torch.cat([map_features, coord_features], dim=1))
        return q_values, state
