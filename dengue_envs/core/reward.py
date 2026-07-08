"""Motor de recompensa: custo imediato + desfechos atrasados + placar final.

Este módulo concentra TODA a política de recompensa do ambiente, isolada do
laço de `step()`. O modelo segue a especificação do README e corrige a
semântica do `discard` (antes degenerada).

Estrutura da recompensa (três camadas)
--------------------------------------
1. Custo imediato (pago no mesmo dia): toda ação paga seu custo operacional.
   Modela o gasto de realizar um teste / uma verificação epidemiológica.

2. Desfecho de decisão (pago com atraso de ``reward_delay_days``): as ações
   *decisivas* — ``confirm`` (4) e ``discard`` (5) — só revelam se foram boas
   depois que o resultado laboratorial/epidemiológico "amadurece". Isso modela
   o atraso real entre decidir e saber se a decisão estava certa.

3. Placar final (pago uma única vez no fim do episódio): recompensa por caso
   corretamente classificado (``final_correct_bonus``) e penalidade por caso
   que ficou errado SEM nunca ter sido testado (``penalty_untested_misdiagnosed``).
   Isso implementa o ``episize - misdiagnosed`` e o ``-10`` do README.

Semântica das decisões (corrigida)
----------------------------------
- ``confirm``: o agente afirma que o caso é um arbovírus real com o diagnóstico
  atual (``agent_diagnosis``). Acerto quando ``agent_diagnosis == doença real``.
- ``discard``: o agente afirma que o caso NÃO é dengue/chik (é "outro"). Acerto
  quando a doença real é ``OTHER``. Descartar um caso que era doença de verdade
  é um falso negativo de vigilância (o pior erro) e recebe a penalidade mais
  pesada ``penalty_missed_case``.

Pesos (padrões e justificativa)
-------------------------------
- Testes de laboratório custam mais (1.0) que a confirmação epidemiológica
  (0.5), que custa mais que "não fazer nada" (0.1).
- ``reward_correct_decision`` (+10) recompensa uma decisão correta.
- ``penalty_incorrect_decision`` (-20) pune confirmar errado (falso positivo).
- ``penalty_missed_case`` (-30) pune descartar um doente real (falso negativo),
  mais grave em vigilância epidemiológica.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# Custos por ação: [teste_dengue, teste_chik, epi_confirm, nada, confirm, discard]
DEFAULT_COSTS = np.array([1.0, 1.0, 0.5, 0.1, 0.0, 0.0])

# IDs de ação
TEST_DENGUE = 0
TEST_CHIK = 1
EPI_CONFIRM = 2
DO_NOTHING = 3
CONFIRM = 4
DISCARD = 5

# Rótulo de doença para "outro" (não dengue e não chik)
OTHER = 2


class RewardEngine:
    """Calcula a recompensa por passo em três camadas (imediata, atrasada, final).

    A fila ``pending_rewards`` guarda os desfechos de decisão agendados para o
    futuro (``t + reward_delay_days``). No fim do episódio, tudo que restou na
    fila é liquidado e o placar final é somado.
    """

    def __init__(
        self,
        costs=None,
        reward_delay_days: int = 5,
        reward_correct_decision: float = 10.0,
        penalty_incorrect_decision: float = -20.0,
        penalty_missed_case: float = -30.0,
        final_correct_bonus: float = 1.0,
        penalty_untested_misdiagnosed: float = -10.0,
    ):
        self.costs = np.array(costs) if costs is not None else DEFAULT_COSTS.copy()
        self.reward_delay = reward_delay_days
        self.reward_correct_decision = reward_correct_decision
        self.penalty_incorrect_decision = penalty_incorrect_decision
        self.penalty_missed_case = penalty_missed_case
        self.final_correct_bonus = final_correct_bonus
        self.penalty_untested_misdiagnosed = penalty_untested_misdiagnosed
        self.pending_rewards: Dict[int, float] = {}

    def reset(self) -> None:
        """Limpa a fila de recompensas pendentes (chamar em ``env.reset``)."""
        self.pending_rewards = {}

    def _decision_outcome(
        self, action_id: int, true_disease: int, agent_diagnosis: int
    ) -> Optional[float]:
        """Recompensa (atrasada) de uma ação decisiva, ou ``None`` se não for.

        A avaliação usa a doença VERDADEIRA, nunca o ``agent_diagnosis`` mutado
        pelo ``step`` (o que corrige o bug histórico do ``discard``).
        """
        if action_id == CONFIRM:
            if agent_diagnosis == true_disease:
                return self.reward_correct_decision
            return self.penalty_incorrect_decision

        if action_id == DISCARD:
            if true_disease == OTHER:
                return self.reward_correct_decision
            # Descartou um caso que era doença real: falso negativo de vigilância.
            return self.penalty_missed_case

        return None

    def _terminal_score(
        self, real_cases: pd.DataFrame, obs_cases: pd.DataFrame
    ) -> float:
        """Placar final: bônus por acerto e penalidade por erro não testado.

        Implementa o modelo do README:
        - ``+final_correct_bonus`` por caso cujo ``agent_diagnosis`` bate com a
          verdade (a soma equivale a ``episize - misdiagnosed``);
        - ``penalty_untested_misdiagnosed`` por caso errado que nunca foi testado
          (nem dengue nem chik).
        """
        if obs_cases is None or obs_cases.empty:
            return 0.0

        has_testd = "testd" in obs_cases.columns
        has_testc = "testc" in obs_cases.columns

        total = 0.0
        for idx in obs_cases.index:
            if idx not in real_cases.index:
                continue
            true_disease = int(real_cases.loc[idx, "disease"])
            agent_diagnosis = int(obs_cases.loc[idx, "agent_diagnosis"])

            if agent_diagnosis == true_disease:
                total += self.final_correct_bonus
                continue

            testd = int(obs_cases.loc[idx, "testd"]) if has_testd else 0
            testc = int(obs_cases.loc[idx, "testc"]) if has_testc else 0
            never_tested = (testd == 0) and (testc == 0)
            if never_tested:
                total += self.penalty_untested_misdiagnosed

        return total

    def compute(
        self,
        action: Iterable[Tuple[int, int]],
        t: int,
        real_cases: pd.DataFrame,
        obs_cases: pd.DataFrame,
        terminated: bool = False,
    ) -> float:
        """Retorna a recompensa do passo atual.

        action: iterável de ``(case_id, action_id)``.
        real_cases: dataframe com a doença verdadeira (indexado por case_id).
        obs_cases: dataframe observado (com ``agent_diagnosis`` e ``testd/testc``).
        terminated: se ``True``, liquida a fila pendente e soma o placar final.
        """
        immediate_reward = 0.0
        delayed_reward_accum = 0.0

        for case_id, action_id in action:
            immediate_reward -= self.costs[action_id]

            if obs_cases is None or case_id not in obs_cases.index:
                continue

            true_disease = int(real_cases.loc[case_id, "disease"])
            agent_diagnosis = int(obs_cases.loc[case_id, "agent_diagnosis"])

            outcome = self._decision_outcome(action_id, true_disease, agent_diagnosis)
            if outcome is not None:
                delayed_reward_accum += outcome

        # Agenda o desfecho das decisões para t + delay
        if delayed_reward_accum != 0:
            target_t = t + self.reward_delay
            self.pending_rewards[target_t] = (
                self.pending_rewards.get(target_t, 0.0) + delayed_reward_accum
            )

        # Resgata o que venceu hoje
        matured_reward = self.pending_rewards.pop(t, 0.0)

        # Fim do episódio: liquida o restante da fila e soma o placar final
        if terminated:
            matured_reward += sum(self.pending_rewards.values())
            self.pending_rewards.clear()
            matured_reward += self._terminal_score(real_cases, obs_cases)

        return immediate_reward + matured_reward
