"""Baselines sequenciais: o segundo exame só quando o primeiro não resolve.

Ficam entre o `testonce` (um exame, não resolve os negativos) e o `testtwice`
(dois exames sempre, inclusive quando o primeiro já deu positivo). São a
referência natural para a pergunta "o agente pede exame demais?": um laudo
positivo encerra o caso; negativo ou inconclusivo pede o outro exame; com os
dois em mãos, conclui com o diagnóstico corrente (dengue, chik, ou outro se os
dois deram negativo).

- `sequencial`: sempre começa pela dengue.
- `sequencial_clinico`: começa pela doença que o médico suspeita (dengue se ele
  suspeitar de outra etiologia). Com um médico bom, o primeiro exame acerta mais
  vezes e o segundo é pedido menos.

Ambas são sem estado: a ordem dos exames é recuperada de quais laudos já
existem no caso.
"""
from __future__ import annotations

from agents.base import EpisodeRunner

TEST_DENGUE = 0
TEST_CHIK = 1
CONCLUDE_DENGUE = 4
NOT_TESTED = 0
POSITIVE = 2
CHIK = 1


class SequentialAgentRunner(EpisodeRunner):
    """Dengue primeiro; chik só se a dengue não der positivo."""

    name = "sequencial"

    def _primeiro_exame(self, row) -> int:
        return TEST_DENGUE

    def choose_action(self, env) -> int:
        base = env.unwrapped
        case_id = env.current_case[0]
        if case_id not in base.obs_cases.index:
            return CONCLUDE_DENGUE
        row = base.obs_cases.loc[case_id]
        testd, testc = int(row["testd"]), int(row["testc"])
        concluir = CONCLUDE_DENGUE + int(row["agent_diagnosis"])

        if testd == NOT_TESTED and testc == NOT_TESTED:
            return self._primeiro_exame(row)
        if POSITIVE in (testd, testc):
            return concluir
        if testd == NOT_TESTED:
            return TEST_DENGUE
        if testc == NOT_TESTED:
            return TEST_CHIK
        return concluir


class ClinicalSequentialAgentRunner(SequentialAgentRunner):
    """Começa pela doença que o médico suspeita."""

    name = "sequencial_clinico"

    def _primeiro_exame(self, row) -> int:
        # Sem laudos, `agent_diagnosis` ainda é o palpite do médico.
        return TEST_CHIK if int(row["agent_diagnosis"]) == CHIK else TEST_DENGUE
