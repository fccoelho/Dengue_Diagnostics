"""Motor de recompensa com custo imediato e desfechos atrasados (delay).

Extração fiel de `DengueDiagnosticsEnv._calc_reward`, encapsulando a fila de
`pending_rewards`. O objetivo é isolar a política de recompensa para permitir,
numa fase seguinte, completar o modelo (bônus final, penalidade por não testar)
e corrigir a semântica do discard sem tocar no laço principal do ambiente.

IMPORTANTE (Fase 1): módulo NOVO, em paralelo ao ambiente legado. A ligação
env -> RewardEngine será feita após validação manual.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

# Custos por ação: [teste_dengue, teste_chik, epi_confirm, nada, confirm, discard]
DEFAULT_COSTS = np.array([1.0, 1.0, 0.5, 0.1, 0.0, 0.0])

CONFIRM = 4
DISCARD = 5


class RewardEngine:
    """Calcula recompensas por passo, agendando desfechos de decisão no futuro.

    Comportamento (idêntico ao ambiente atual):
    - Toda ação paga seu custo imediatamente.
    - Ações de decisão (confirm=4, discard=5) geram bônus/penalidade agendados
      para `t + reward_delay_days`.
    - No fim do episódio, toda a fila pendente é liquidada.
    """

    def __init__(
        self,
        costs=None,
        reward_delay_days: int = 5,
        reward_correct_decision: float = 10.0,
        penalty_incorrect_decision: float = -20.0,
    ):
        self.costs = np.array(costs) if costs is not None else DEFAULT_COSTS.copy()
        self.reward_delay = reward_delay_days
        self.reward_correct_decision = reward_correct_decision
        self.penalty_incorrect_decision = penalty_incorrect_decision
        self.pending_rewards: Dict[int, float] = {}

    def reset(self) -> None:
        """Limpa a fila de recompensas pendentes (chamar em env.reset)."""
        self.pending_rewards = {}

    def compute(
        self,
        action: Iterable[Tuple[int, int]],
        t: int,
        real_cases: pd.DataFrame,
        obs_cases: pd.DataFrame,
        terminated: bool = False,
    ) -> float:
        """Retorna a recompensa do passo atual.

        action: iterável de (case_id, action_id).
        real_cases: dataframe com a doença verdadeira (indexado por case_id).
        obs_cases: dataframe observado (com `agent_diagnosis`).
        """
        immediate_reward = 0.0
        delayed_reward_accum = 0.0

        for case_id, action_id in action:
            immediate_reward -= self.costs[action_id]

            if case_id not in obs_cases.index:
                continue

            true_disease = int(real_cases.loc[case_id, "disease"])
            agent_diagnosis = obs_cases.loc[case_id, "agent_diagnosis"]

            is_correct = False
            is_decision = False

            if action_id == CONFIRM:
                is_decision = True
                if agent_diagnosis == true_disease:
                    is_correct = True
            elif action_id == DISCARD:
                is_decision = True
                discarded = 1 if agent_diagnosis == 0 else 0
                if discarded == true_disease:
                    is_correct = True

            if is_decision:
                if is_correct:
                    delayed_reward_accum += self.reward_correct_decision
                else:
                    delayed_reward_accum += self.penalty_incorrect_decision

        # Agenda o desfecho para t + delay
        if delayed_reward_accum != 0:
            target_t = t + self.reward_delay
            self.pending_rewards[target_t] = (
                self.pending_rewards.get(target_t, 0.0) + delayed_reward_accum
            )

        # Resgata o que venceu hoje
        matured_reward = self.pending_rewards.pop(t, 0.0)

        # Fim do episódio: liquida o restante
        if terminated:
            matured_reward += sum(self.pending_rewards.values())
            self.pending_rewards.clear()

        return immediate_reward + matured_reward
