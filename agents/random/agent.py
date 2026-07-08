"""Agente aleatório (baseline) sobre o ambiente novo.

Para cada caso, escolhe uma ação uniformemente ao acaso entre as 6 possíveis.
Opera sobre o MESMO ambiente usado pelos demais agentes (env bruto + wrappers
`map_tensor` + `case_by_case`), amostrando o `action_space` discreto do wrapper
`CaseByCaseWrapper`.

Como usa a mesma interface e as mesmas métricas (`get_episode_metrics`), seus
resultados são diretamente comparáveis aos do baseline clínico e do DQN/PPO.

Legado: as versões antigas ficaram em `agent_random_old.py` e
`random_agent_old.py`.
"""
from __future__ import annotations

from agents.base import EpisodeRunner


class RandomAgentRunner(EpisodeRunner):
    """Baseline aleatório: ação uniforme por caso."""

    name = "random"

    def choose_action(self, env) -> int:
        return int(env.action_space.sample())  # Discrete(6): ação por caso
