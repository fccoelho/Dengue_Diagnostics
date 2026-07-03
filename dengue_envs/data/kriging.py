"""Gerador por superfície de densidade (Kriging) — PLACEHOLDER.

Reservado para a Fase 4 do plano de refatoração. A intenção é estimar a
superfície de densidade de casos por Kriging (sem assumir distribuição
paramétrica) e usá-la tanto para amostrar casos quanto para a confirmação
epidemiológica.

Ainda não implementado: instanciar levanta NotImplementedError de propósito.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


class KrigingDensityGenerator:
    """Distribuição via superfície de densidade Kriging (a implementar)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "KrigingDensityGenerator ainda não foi implementado. "
            "Ponto de extensão previsto para a Fase 4 (ver plano.md)."
        )

    def build_world(self):  # pragma: no cover - placeholder
        raise NotImplementedError

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError
