"""Baseline clínico puro (sem ação de investigação).

Este agente sempre escolhe "não fazer nada" (ação 3), ou seja, aceita o
diagnóstico clínico de cada caso sem pedir testes nem confirmar/descartar. Serve
como piso de referência: mede o que o diagnóstico clínico entrega sozinho.

Comparar qualquer agente contra este baseline revela a sua **contribuição
marginal** sobre o clínico. Um agente que não supera o clínico puro não está
agregando valor.
"""
from __future__ import annotations

from agents.base import EpisodeRunner

# Ação "não fazer nada" no CaseByCaseWrapper (aceita o diagnóstico clínico).
DO_NOTHING = 3


class ClinicalOnlyAgentRunner(EpisodeRunner):
    """Baseline que aceita sempre o diagnóstico clínico (ação = do nothing)."""

    name = "clinical"

    def choose_action(self, env) -> int:
        return DO_NOTHING
