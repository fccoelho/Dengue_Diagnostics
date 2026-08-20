"""Baseline "investigar até o fim" — a política fixa mais forte no ambiente v3.

Fluxo: pede o exame de dengue; quando o laudo volta, pede o de chikungunya; com
os dois resultados em mãos, encerra o caso concluindo com o diagnóstico corrente
(ação 4 + ``agent_diagnosis``) — dengue ou chik se algum exame deu positivo,
outro se os dois deram negativo (ou seja, não é arbovirose).

Substitui o `testonce` como referência a partir do momento em que o ambiente
passa a conter casos que não são arbovirose. Com apenas duas doenças, um laudo
negativo para dengue implicava logicamente chikungunya e um único exame bastava
(acurácia 0,960). Com a terceira classe, um negativo é ambíguo e um exame só leva
a 0,756 — são precisos dois para chegar a 0,938.

Números no ambiente v3 (10 seeds, exame a 4,0):

| Política                        | Recompensa | Acurácia | Exames |
|---------------------------------|-----------:|---------:|-------:|
| nada (clínico puro)             |     -304,5 |    0,571 |      0 |
| 1 exame -> confirmar            |     -911,0 |    0,756 |    373 |
| **testtwice (esta)**            | **+208,6** |**0,938** |    746 |
| [oráculo por caso]              |    +2823,0 |    0,977 |    243 |

A distância entre esta política e o oráculo (+2614) é justamente o que se ganha
ao decidir QUAIS casos merecem investigação, em vez de investigar todos.
"""
from __future__ import annotations

from agents.base import EpisodeRunner

TEST_DENGUE = 0
TEST_CHIK = 1
CONCLUDE_DENGUE = 4
NOT_TESTED = 0


class TestTwiceAgentRunner(EpisodeRunner):
    """Investiga com os dois exames e encerra o caso com o diagnóstico corrente."""

    name = "testtwice"

    def choose_action(self, env) -> int:
        base = env.unwrapped
        case_id = env.current_case[0]
        if case_id not in base.obs_cases.index:
            return CONCLUDE_DENGUE
        row = base.obs_cases.loc[case_id]
        if int(row["testd"]) == NOT_TESTED:
            return TEST_DENGUE
        if int(row["testc"]) == NOT_TESTED:
            return TEST_CHIK
        # Dois laudos em mãos: conclui com o diagnóstico corrente (dengue, chik
        # ou outro, se os dois exames deram negativo).
        return CONCLUDE_DENGUE + int(row["agent_diagnosis"])
