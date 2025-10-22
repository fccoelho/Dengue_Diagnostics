# dengue_network.py
import torch
from torch import nn
import numpy as np
from tianshou.data import Batch


class DengueNet(nn.Module):
    """
    Rede que processa a observação Dict: {"map": (B, 4, 400, 400), "case_coords": (B, 2)}
    e retorna 6 Q-Values: (B, 6)
    """

    def __init__(self, map_shape, action_shape, device="cpu"):
        super().__init__()
        self.device = device

        # Encoder CNN para o mapa (4, 400, 400)
        # (Arquitetura de exemplo, pode ser ajustada)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_shape[0], 16, kernel_size=8, stride=4),  # (16, 99, 99)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # (32, 48, 48)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # (64, 46, 46)
            nn.ReLU(),
            nn.Flatten()
        )

        # Calcula o tamanho da saída do map_encoder
        with torch.no_grad():
            dummy_map = torch.zeros(1, *map_shape)
            map_feature_dim = self.map_encoder(dummy_map).shape[1]

        # Encoder MLP para as coordenadas (x, y)
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Head MLP que combina os features e retorna Q-values
        self.head = nn.Sequential(
            nn.Linear(map_feature_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_shape)  # action_shape é 6
        )

    def forward(self, obs, state=None, info={}):
                # Converte para tensores no dispositivo correto
        map_tensor = torch.as_tensor(
            obs["map"], device=self.device, dtype=torch.float32
        )
        coords_tensor = torch.as_tensor(
            obs["case_coords"], device=self.device, dtype=torch.float32
        )

        # Garante que os tensores tenham dimensão de batch
        if map_tensor.ndim == 3:
            map_tensor = map_tensor.unsqueeze(0)
        if coords_tensor.ndim == 1:
            coords_tensor = coords_tensor.unsqueeze(0)

        # Passa pela rede
        map_features = self.map_encoder(map_tensor)
        coord_features = self.coord_encoder(coords_tensor)

        # Concatena os features
        combined_features = torch.cat([map_features, coord_features], dim=1)

        # Gera os Q-values
        q_values = self.head(combined_features)

        return q_values, state