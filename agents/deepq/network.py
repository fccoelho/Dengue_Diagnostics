"""Rede Q do DQN para o ambiente DengueDiag (map tensor + coordenadas do caso)."""
from __future__ import annotations

import torch
from torch import nn


class DengueNet(nn.Module):
    """Processa obs Dict ``{"map": (B, C, H, W), "case_coords": (B, 2)}`` → Q (B, |A|).

    O encoder de mapa termina com ``AdaptiveAvgPool2d`` para fixar a saída em
    ``(64, pooled_size, pooled_size)`` **independentemente do tamanho do mapa**.
    Sem esse pooling, um mapa 400×400 gera ~135 mil features achatadas e uma
    primeira camada ``Linear`` de ~69 M parâmetros (99,98% da rede) — pesada na
    GPU e responsável por checkpoints de ~550 MB. Com ``pooled_size=6`` a rede
    cai para ~1,5 M parâmetros e o checkpoint para poucos MB.

    **Equilíbrio entre os ramos.** O mapa achatado tem 2304 dimensões contra 32
    do contexto, então 96% da capacidade da cabeça ia para um resumo *global* —
    irrelevante para decidir sobre um caso individual. Medido no modelo treinado:
    perturbar o mapa com ruído mudava os Q-values em |ΔQ| ≈ 200–700 (chegando a
    inverter a decisão), enquanto trocar o **diagnóstico do próprio caso** mudava
    em |ΔQ| ≈ 0–0,6 e *nunca* alterava a ação escolhida. Consequência prática: o
    agente ignorava a evidência dos exames que ele mesmo pagou (48,6% das
    conclusões contrariavam o diagnóstico corrente, acertando só 20,7%).

    Por isso o mapa passa por uma projeção (``map_proj_dim``) antes de entrar na
    cabeça, e o ramo de contexto ganhou capacidade equivalente — o objetivo é que
    "o que sei deste caso" concorra de igual para igual com "como está o surto".
    """

    def __init__(
        self,
        map_shape,
        action_shape,
        device="cpu",
        *,
        pooled_size: int = 6,
        context_dim: int = 0,
        map_proj_dim: int = 128,
        context_out_dim: int = 128,
    ):
        super().__init__()
        self.device = device
        # `context_dim > 0` habilita o ramo que consome a evidência sobre o caso
        # atual + a competência do médico (ver CaseByCaseWrapper.context_features).
        self.context_dim = int(context_dim)

        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pooled_size, pooled_size)),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_map = torch.zeros(1, *map_shape)
            map_feature_dim = self.map_encoder(dummy_map).shape[1]

        # Comprime o mapa para que ele não domine a cabeça pela quantidade de
        # dimensões (2304 -> map_proj_dim).
        self.map_proj = nn.Sequential(
            nn.Linear(map_feature_dim, map_proj_dim),
            nn.ReLU(),
        )

        self.coord_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        context_out = 0
        if self.context_dim:
            context_out = context_out_dim
            self.context_encoder = nn.Sequential(
                nn.Linear(self.context_dim, context_out),
                nn.ReLU(),
                nn.Linear(context_out, context_out),
                nn.ReLU(),
            )

        self.head = nn.Sequential(
            nn.Linear(map_proj_dim + 64 + context_out, 512),
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

        map_features = self.map_proj(self.map_encoder(map_tensor))
        coord_features = self.coord_encoder(coords_tensor)
        parts = [map_features, coord_features]

        if self.context_dim:
            ctx = torch.as_tensor(obs["context"], device=device, dtype=torch.float32)
            if ctx.ndim == 1:
                ctx = ctx.unsqueeze(0)
            parts.append(self.context_encoder(ctx))

        q_values = self.head(torch.cat(parts, dim=1))
        return q_values, state
