"""Fila de resultados de laboratório com atraso (turnaround laboratorial).

Ao pedir um teste, a amostra é colhida "hoje" (o resultado já é sorteado, de
forma determinística em relação ao dia do pedido), mas o resultado só fica
disponível para o agente após ``lab_delay_days``. Isso modela o tempo real entre
solicitar um exame e receber o laudo.

O ambiente consome esta fila em cada passo: aplica os resultados que "venceram"
no dia atual e, no fim do episódio, libera todos os pendentes (`flush`).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# Cada entrada pendente: (action_id, case_id, result)
#   action_id: 0 = teste de dengue, 1 = teste de chik
LabEntry = Tuple[int, int, int]


class LabResultQueue:
    """Agenda resultados de laboratório para amadurecerem em ``t + lab_delay``."""

    def __init__(self, lab_delay_days: int = 0):
        self.lab_delay = lab_delay_days
        self.pending: Dict[int, List[LabEntry]] = {}

    def reset(self) -> None:
        """Limpa a fila (chamar em ``env.reset``)."""
        self.pending = {}

    def schedule(self, action_id: int, case_id: int, result: int, t: int) -> None:
        """Agenda um resultado colhido em ``t`` para chegar em ``t + lab_delay``."""
        target_t = t + self.lab_delay
        self.pending.setdefault(target_t, []).append((action_id, case_id, result))

    def pop_matured(self, t: int) -> List[LabEntry]:
        """Retorna (e remove) os resultados que ficam disponíveis no dia ``t``."""
        return self.pending.pop(t, [])

    def flush(self) -> List[LabEntry]:
        """Retorna (e remove) TODOS os resultados pendentes (fim do episódio)."""
        entries = [e for group in self.pending.values() for e in group]
        self.pending.clear()
        return entries

    def has_pending(self, case_id: int) -> bool:
        """Há laudo a caminho para este caso?

        Usado para decidir se um caso ainda **pode voltar** ao agente: um laudo
        pendente agenda revisita ao chegar (`_apply_lab_results`). Sem isso, um
        caso recém-testado seria classificado como abandonado no mesmo passo.
        """
        return any(
            entry[1] == case_id
            for group in self.pending.values()
            for entry in group
        )
