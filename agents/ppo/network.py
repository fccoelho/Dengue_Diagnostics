"""Ator e crítico do PPO para o ambiente DengueDiag.

**Reaproveitam o tronco do `DengueNet`** (encoder de mapa + coordenadas +
contexto, com o mesmo rebalanceamento `map_proj_dim` / `context_out_dim`). Isso
é deliberado: a comparação DQN × PPO precisa isolar o **algoritmo**, não a
arquitetura. Só as cabeças mudam — o DQN tem uma cabeça de Q-values, aqui há
uma de logits (ator) e uma escalar (crítico).

**Máscara de ação.** O `DiscreteQLearningPolicy` do Tianshou aplica `obs.mask`
sozinho; a `ProbabilisticActorPolicy` (usada pelo PPO) **não** — o `forward`
dela chama o ator e amostra da distribuição, sem consultar máscara nenhuma.
Como o ambiente usa `force_decision_after_tests`, deixar assim removeria em
silêncio a restrição que obriga a concluir depois de dois exames, e o PPO
estaria resolvendo um problema mais fácil que o DQN. Por isso o ator aplica a
máscara aos logits aqui dentro, somando `-inf` às ações proibidas.
"""
from __future__ import annotations

import torch
from torch import nn

from agents.deepq.network import MIN_MAP_SIDE, DengueNet


class DengueTrunk(nn.Module):
    """Encoder compartilhado: mapa + coordenadas + contexto -> vetor de features.

    Constrói-se a partir de um `DengueNet` para garantir, por construção, que o
    tronco é idêntico ao do DQN — inclusive o piso `MIN_MAP_SIDE` e o
    rebalanceamento entre os ramos documentado lá.
    """

    def __init__(
        self,
        map_shape,
        *,
        device="cpu",
        pooled_size: int = 6,
        context_dim: int = 0,
        map_proj_dim: int = 128,
        context_out_dim: int = 128,
    ):
        super().__init__()
        base = DengueNet(
            map_shape,
            1,  # action_shape irrelevante: a cabeça do DQN é descartada
            device=device,
            pooled_size=pooled_size,
            context_dim=context_dim,
            map_proj_dim=map_proj_dim,
            context_out_dim=context_out_dim,
        )
        self.device = device
        self.context_dim = int(context_dim)
        self.map_encoder = base.map_encoder
        self.map_proj = base.map_proj
        self.coord_encoder = base.coord_encoder
        self.context_encoder = getattr(base, "context_encoder", None)
        # Dimensão de saída = entrada da cabeça do DQN, medida no próprio objeto.
        self.output_dim = base.head[0].in_features

    def forward(self, obs):
        device = next(self.parameters()).device
        m = torch.as_tensor(obs["map"], device=device, dtype=torch.float32)
        c = torch.as_tensor(obs["case_coords"], device=device, dtype=torch.float32)
        if m.ndim == 3:
            m = m.unsqueeze(0)
        if c.ndim == 1:
            c = c.unsqueeze(0)

        parts = [self.map_proj(self.map_encoder(m)), self.coord_encoder(c)]
        if self.context_dim:
            ctx = torch.as_tensor(obs["context"], device=device, dtype=torch.float32)
            if ctx.ndim == 1:
                ctx = ctx.unsqueeze(0)
            parts.append(self.context_encoder(ctx))
        return torch.cat(parts, dim=1)


class DengueActor(nn.Module):
    """Tronco -> logits sobre as 7 ações, com `obs.mask` aplicada."""

    def __init__(self, trunk: DengueTrunk, action_shape: int, hidden: int = 128):
        super().__init__()
        self.trunk = trunk
        self.net = nn.Sequential(
            nn.Linear(trunk.output_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(action_shape)),
        )

    def forward(self, obs, state=None, info=None):
        logits = self.net(self.trunk(obs))
        # Ações proibidas recebem -inf: a Categorical lhes dá probabilidade 0.
        if isinstance(obs, dict) or hasattr(obs, "keys"):
            if "mask" in obs:
                mask = torch.as_tensor(
                    obs["mask"], device=logits.device, dtype=torch.bool
                )
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)
                logits = logits.masked_fill(~mask, float("-inf"))
        return logits, state


class DengueCritic(nn.Module):
    """Tronco -> valor escalar do estado."""

    def __init__(self, trunk: DengueTrunk, hidden: int = 128):
        super().__init__()
        self.trunk = trunk
        self.net = nn.Sequential(
            nn.Linear(trunk.output_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, **kwargs):
        return self.net(self.trunk(obs)).squeeze(-1)


def build_actor_critic(
    env,
    *,
    device: str = "cpu",
    pooled_size: int | None = None,
    share_trunk: bool = False,
):
    """Cria (ator, crítico) para o espaço de observação do `env`.

    ``share_trunk=False`` (padrão) dá um encoder a cada um. Compartilhar o
    tronco acopla os gradientes de política e valor — no PPO isso costuma
    exigir ajustar `vf_coef` para que a perda de valor não domine o encoder.
    Mantido como opção, mas não como default.
    """
    map_shape = env.observation_space.spaces["map"].shape
    ctx_space = env.observation_space.spaces.get("context")
    context_dim = int(ctx_space.shape[0]) if ctx_space is not None else 0
    n_acoes = int(env.action_space.n)
    kw = {} if pooled_size is None else {"pooled_size": int(pooled_size)}

    def _trunk():
        return DengueTrunk(
            map_shape, device=device, context_dim=context_dim, **kw
        ).to(device)

    tronco = _trunk()
    ator = DengueActor(tronco, n_acoes).to(device)
    critico = DengueCritic(tronco if share_trunk else _trunk()).to(device)
    return ator, critico


__all__ = [
    "DengueTrunk",
    "DengueActor",
    "DengueCritic",
    "build_actor_critic",
    "MIN_MAP_SIDE",
]
