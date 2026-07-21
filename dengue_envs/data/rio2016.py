"""Gerador baseado em dados reais (Rio de Janeiro, 2016) — PLACEHOLDER.

Reservado para a Fase 4 do plano de refatoração. A intenção é carregar/derivar
uma distribuição espaço-temporal de casos a partir de dados reais de 2016 e
expô-la pela interface `EpidemicGenerator`, sem alterar o ambiente.

Ainda não implementado: instanciar levanta NotImplementedError de propósito,
para deixar explícito que é um ponto de extensão futuro.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


class Rio2016Generator:
    """Distribuição a partir de dados reais de 2016 (a implementar)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Rio2016Generator ainda não foi implementado. "
            "Ponto de extensão previsto para a Fase 4 (ver plano.md)."
        )

    def build_world(self):  # pragma: no cover - placeholder
        raise NotImplementedError

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError
