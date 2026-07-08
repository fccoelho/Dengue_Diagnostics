"""Contratos (interfaces) para geradores de epidemias.

Este módulo define o Protocol que qualquer distribuição epidemiológica deve
seguir para ser plugável no ambiente. A ideia é permitir trocar a fonte de
dados (sintética, dados reais de 2016, superfícies via Kriging, etc.) sem
alterar o `DengueDiagnosticsEnv`.

NOTA (Fase 1): este módulo é NOVO e não substitui `dengue_envs/data/generator.py`.
O `World` legado continua funcionando normalmente. O adaptador
`SyntheticGenerator` (em `synthetic.py`) implementa este Protocol reaproveitando
o `World` existente.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import pandas as pd

# Colunas mínimas esperadas no dataframe de casos produzido por um gerador.
CASE_COLUMNS = ["t", "x", "y", "disease", "testd", "testc", "epiconf"]


@runtime_checkable
class EpidemicGenerator(Protocol):
    """Interface para geradores de casos de epidemia.

    Implementações devem produzir um `casedf` (DataFrame) com, no mínimo, as
    colunas em `CASE_COLUMNS`, e expor um objeto "World-like" com os métodos
    que o ambiente consome (`get_series_up_to_t`, `get_maps_up_to_t`).

    O mapa da epidemia (`dengue_envs.rendering.epidemic_map.plot_epidemic_map`)
    consome diretamente esse ``casedf``/``real_cases``, portanto qualquer
    gerador compatível ganha visualização automática via ``agents/artifacts``.
    """

    size: int
    episize: int
    epilength: int

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        """Gera e retorna o dataframe de casos (casedf)."""
        ...

    def build_world(self):
        """Retorna um objeto World-like consumível pelo ambiente."""
        ...
