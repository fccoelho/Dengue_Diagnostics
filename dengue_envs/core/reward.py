"""Motor de recompensa: custo imediato + desfechos atrasados + placar final.

Este módulo concentra TODA a política de recompensa do ambiente, isolada do
laço de `step()`. O modelo segue a especificação do README e corrige a
semântica do `discard` (antes degenerada).

Estrutura da recompensa (três camadas)
--------------------------------------
1. Custo imediato (pago no mesmo dia): toda ação paga seu custo operacional.
   Modela o gasto de realizar um teste / uma verificação epidemiológica.

2. Desfecho de decisão (pago com atraso de ``reward_delay_days``): as ações
   *conclusivas* — ``conclude_dengue``/``conclude_chik``/``conclude_other``
   (4/5/6) — só revelam se foram boas depois que o resultado
   laboratorial/epidemiológico "amadurece". Isso modela o atraso real entre
   decidir e saber se a decisão estava certa.

3. Placar final (pago uma única vez no fim do episódio): recompensa por caso
   corretamente classificado (``final_correct_bonus``) e penalidade por caso
   que ficou errado SEM nunca ter sido testado (``penalty_untested_misdiagnosed``).
   Isso implementa o ``episize - misdiagnosed`` e o ``-10`` do README.

Semântica das decisões
----------------------
Há três ações conclusivas, uma para cada classe possível — o agente afirma
explicitamente "este caso É X", em vez de só aceitar (``confirm``) ou negar
(``discard``) o palpite clínico corrente:

- ``conclude_dengue`` / ``conclude_chik`` / ``conclude_other``: a ação em si
  já é a alegação (``claimed = action_id - CONCLUDE_DENGUE``), independente do
  que o diagnóstico clínico dizia. Isso é o que permite o agente **discordar do
  médico usando outra evidência** (ex.: posição espacial) sem precisar gastar um
  exame para revisar o rótulo.
- Acerto (``claimed == doença real``): ``reward_correct_decision``.
- Alegar ``OTHER`` num caso que era doença real: falso negativo de vigilância
  (o pior erro), ``penalty_missed_case``.
- Qualquer outro erro (alegar a arbovirose errada, ou alegar arbovirose num caso
  que era ``OTHER``): ``penalty_incorrect_decision``.

Versão anterior (2 ações: ``confirm``/``discard``): ``confirm`` só podia
aceitar o ``agent_diagnosis`` corrente, então trocar dengue↔chik exigia
testar — mesmo quando a posição espacial já indicava a resposta. E ``confirm``
recompensava também os casos ``OTHER``, o que tornava ``discard`` estritamente
dominado (mesma recompensa quando certo, punição maior quando errado); na
prática nenhum agente usava a ação.

Pesos (padrões e justificativa)
-------------------------------
- Testes de laboratório custam mais (1.0) que a confirmação epidemiológica
  (0.5), que custa mais que "não fazer nada" (0.1). Concluir (qualquer classe)
  não tem custo imediato.
- ``reward_correct_decision`` (+10) recompensa uma decisão correta.
- ``penalty_incorrect_decision`` (-20) pune uma conclusão errada "comum".
- ``penalty_missed_case`` (-30) pune alegar ``OTHER`` num doente real (falso
  negativo), mais grave em vigilância epidemiológica.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# Custos por ação:
# [teste_dengue, teste_chik, epi_confirm, nada, conclude_dengue, conclude_chik, conclude_other]
DEFAULT_COSTS = np.array([1.0, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0])

# IDs de ação
TEST_DENGUE = 0
TEST_CHIK = 1
EPI_CONFIRM = 2
DO_NOTHING = 3
CONCLUDE_DENGUE = 4
CONCLUDE_CHIK = 5
CONCLUDE_OTHER = 6
CONCLUSIVE_ACTIONS = (CONCLUDE_DENGUE, CONCLUDE_CHIK, CONCLUDE_OTHER)

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
        penalty_misdiagnosed: float = -3.0,
        penalty_untested_misdiagnosed: float = 0.0,
        penalty_unresolved: float = -10.0,
        shaping_conclude_bonus: float = 0.0,
    ):
        self.costs = np.array(costs) if costs is not None else DEFAULT_COSTS.copy()
        self.reward_delay = reward_delay_days
        self.reward_correct_decision = reward_correct_decision
        self.penalty_incorrect_decision = penalty_incorrect_decision
        self.penalty_missed_case = penalty_missed_case
        self.final_correct_bonus = final_correct_bonus
        self.penalty_misdiagnosed = penalty_misdiagnosed
        self.penalty_untested_misdiagnosed = penalty_untested_misdiagnosed
        # Caso encerrado sem nenhuma ação conclusiva é um caso não resolvido na
        # vigilância. Sem esta penalidade, não decidir era o refúgio seguro:
        # arriscava apenas -3 no placar, contra -20/-30 de uma decisão errada —
        # e o agente convergia para não agir (79,5% de "nada").
        self.penalty_unresolved = penalty_unresolved
        # Reward shaping: bônus imediato ao concluir um caso que foi investigado.
        # Serve para encurtar o crédito da cadeia investigar -> concluir, cujo
        # desfecho real só chega `reward_delay` dias depois.
        self.shaping_conclude_bonus = shaping_conclude_bonus
        self.pending_rewards: Dict[int, float] = {}

    def reset(self) -> None:
        """Limpa a fila de recompensas pendentes (chamar em ``env.reset``)."""
        self.pending_rewards = {}

    def _decision_outcome(self, action_id: int, true_disease: int) -> Optional[float]:
        """Recompensa (atrasada) de uma ação decisiva, ou ``None`` se não for.

        A alegação (``claimed``) vem diretamente da ação escolhida, não do
        ``agent_diagnosis`` — é o que permite ao agente concluir uma classe
        diferente da que o palpite clínico sugeria, sem depender de teste para
        "editar" o diagnóstico corrente. ``agent_diagnosis`` segue existindo
        (é o que o ``step`` grava em ``obs_cases`` para o placar final e para a
        observação seguinte), mas não é o que decide a recompensa aqui.
        """
        if action_id not in CONCLUSIVE_ACTIONS:
            return None

        claimed = action_id - CONCLUDE_DENGUE  # 0=dengue, 1=chik, 2=other
        if claimed == true_disease:
            return self.reward_correct_decision
        if claimed == OTHER:
            # Alegou "não é arbovirose" num caso que era doença real: falso
            # negativo de vigilância, o pior erro.
            return self.penalty_missed_case
        # Qualquer outro erro: arbovirose errada, ou arbovirose alegada num
        # caso que de fato era OTHER.
        return self.penalty_incorrect_decision

    def _terminal_score(
        self,
        real_cases: pd.DataFrame,
        obs_cases: pd.DataFrame,
        concluded=None,
    ) -> float:
        """Placar final: bônus por acerto e penalidade por erro.

        - ``+final_correct_bonus`` por caso cujo ``agent_diagnosis`` bate com a
          verdade (a soma equivale a ``episize - misdiagnosed``);
        - ``penalty_misdiagnosed`` por caso errado, **independentemente** de ter
          sido testado;
        - ``penalty_untested_misdiagnosed`` (extra) por caso errado que nunca foi
          testado.

        Por que a penalidade principal não depende mais de ter testado: na
        versão anterior o erro só era punido quando o caso **nunca** havia sido
        testado, o que transformava o teste num "passe livre" — pedir exame
        isentava da punição mesmo quando o laudo voltava inconclusivo e não
        mudava nada. O resultado era que testar TUDO virava quase ótimo
        (recompensa -35 contra -396 do DQN treinado), contradizendo a premissa
        de economizar testes. Punindo o erro em si, o exame passa a valer pela
        informação que agrega, e não pelo ato de ter sido pedido.
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

            # Caso que terminou o episódio sem conclusão: fica em aberto na
            # vigilância e paga por isso, independentemente do palpite.
            if concluded is not None and idx not in concluded:
                total += self.penalty_unresolved

            if agent_diagnosis == true_disease:
                total += self.final_correct_bonus
                continue

            # Erro: punido sempre; testar não isenta.
            total += self.penalty_misdiagnosed

            testd = int(obs_cases.loc[idx, "testd"]) if has_testd else 0
            testc = int(obs_cases.loc[idx, "testc"]) if has_testc else 0
            never_tested = (testd == 0) and (testc == 0)
            if never_tested:
                total += self.penalty_untested_misdiagnosed

        return total

    def case_reward(
        self,
        case_id: int,
        action_id: int,
        t: int,
        real_cases: pd.DataFrame,
        obs_cases: pd.DataFrame,
    ) -> float:
        """Recompensa **atribuível a uma única ação sobre um único caso**.

        Cobre o que é consequência direta desta decisão: o custo operacional da
        ação e — quando ``reward_delay_days == 0`` — o desfecho da conclusão.
        Com atraso positivo o desfecho vai para a fila e só será pago em
        ``settle_day`` do dia correspondente, misturado ao de outros casos.

        Esta separação existe para permitir **atribuição de crédito por caso**:
        o ``CaseByCaseWrapper`` decide um caso por passo, mas historicamente
        acumulava as ações do dia e só recebia um escalar agregado no fim dele.
        Medido nesse regime: ~90% da variância da recompensa de um passo vinha
        de decisões tomadas dias antes, sobre **outros** casos — ruído que
        inviabilizava o aprendizado da função Q.
        """
        reward = -self.costs[action_id]

        if obs_cases is None or case_id not in obs_cases.index:
            return reward

        true_disease = int(real_cases.loc[case_id, "disease"])
        outcome = self._decision_outcome(action_id, true_disease)
        if outcome is None:
            return reward

        # Shaping: crédito imediato por concluir um caso que foi de fato
        # investigado. Ajuda quando o desfecho verdadeiro é atrasado.
        if self.shaping_conclude_bonus:
            row = obs_cases.loc[case_id]
            if int(row.get("testd", 0)) or int(row.get("testc", 0)):
                reward += self.shaping_conclude_bonus

        if self.reward_delay <= 0:
            # Sem atraso: o desfecho é atribuível a esta ação, aqui e agora.
            reward += outcome
        else:
            target_t = t + self.reward_delay
            self.pending_rewards[target_t] = (
                self.pending_rewards.get(target_t, 0.0) + outcome
            )

        return reward

    def settle_day(
        self,
        t: int,
        real_cases: pd.DataFrame,
        obs_cases: pd.DataFrame,
        terminated: bool = False,
        concluded=None,
    ) -> float:
        """Recompensa que pertence ao **dia**, não a um caso específico.

        São os desfechos agendados que venceram hoje (de decisões tomadas
        ``reward_delay_days`` atrás) e, no último passo, o placar final.
        """
        matured = self.pending_rewards.pop(t, 0.0)

        if terminated:
            matured += sum(self.pending_rewards.values())
            self.pending_rewards.clear()
            matured += self._terminal_score(real_cases, obs_cases, concluded)

        return matured

    def compute(
        self,
        action: Iterable[Tuple[int, int]],
        t: int,
        real_cases: pd.DataFrame,
        obs_cases: pd.DataFrame,
        terminated: bool = False,
        concluded=None,
    ) -> float:
        """Recompensa de um passo que processa TODAS as ações do dia de uma vez.

        Mantido como composição de ``case_reward`` + ``settle_day`` para que o
        modo agregado (histórico) e o modo por caso compartilhem exatamente a
        mesma lógica — a soma sobre um episódio é idêntica nos dois; o que muda
        é apenas *quando* cada parcela é entregue ao agente.

        action: iterável de ``(case_id, action_id)``.
        real_cases: dataframe com a doença verdadeira (indexado por case_id).
        obs_cases: dataframe observado (com ``agent_diagnosis`` e ``testd/testc``).
        terminated: se ``True``, liquida a fila pendente e soma o placar final.
        concluded: ids dos casos já encerrados por uma ação conclusiva. Usado no
            placar final para penalizar os que ficaram em aberto.
        """
        total = 0.0
        for case_id, action_id in action:
            total += self.case_reward(case_id, action_id, t, real_cases, obs_cases)
        total += self.settle_day(t, real_cases, obs_cases, terminated, concluded)
        return total
