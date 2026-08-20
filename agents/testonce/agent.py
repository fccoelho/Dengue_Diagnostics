"""Baseline "testar uma vez e confirmar" — a política fixa mais forte no v2.

Fluxo: na primeira avaliação de um caso, pede o exame (ação 0). Quando o laudo
chega e o caso **retorna**, conclui com o diagnóstico corrente (ação
4 + ``agent_diagnosis``, já atualizado pelo exame), já de posse do resultado.

É a referência que substitui o `testall` no ambiente v2. O `testall` ficou
degenerado desde que os casos passaram a retornar: como ele nunca confirma, o
mesmo caso volta e é testado de novo até o limite de revisitas — pagando o exame
três vezes (~-5648 de recompensa). Não é um teto, é um desperdício.

Números no ambiente v2 (10 seeds, exame custando 7):

| Política              | Recompensa | Acurácia |
|-----------------------|-----------:|---------:|
| confirmall            |     +128,8 |    0,690 |
| **testonce (esta)**   | **+739,2** | **0,960**|
| [oráculo por episódio]|     +995,0 |        — |

Um agente aprendido só agrega valor se superar +739,2 — e o teto realista é
+995,0, alcançável escolhendo por episódio entre confirmar direto (médico bom) e
gastar o exame (médico ruim).
"""
from __future__ import annotations

from agents.base import EpisodeRunner

TEST_DENGUE = 0
CONCLUDE_DENGUE = 4
NOT_TESTED = 0


class TestOnceAgentRunner(EpisodeRunner):
    """Pede um exame por caso e conclui com o diagnóstico corrente quando o
    resultado chega."""

    name = "testonce"

    def choose_action(self, env) -> int:
        base = env.unwrapped
        case_id = env.current_case[0]
        if case_id not in base.obs_cases.index:
            return CONCLUDE_DENGUE
        row = base.obs_cases.loc[case_id]
        if int(row["testd"]) == NOT_TESTED:
            return TEST_DENGUE
        return CONCLUDE_DENGUE + int(row["agent_diagnosis"])
