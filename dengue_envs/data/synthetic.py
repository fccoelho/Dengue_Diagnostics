"""Gerador sintético de epidemias (SIR + normal truncada).

Adaptador que implementa `EpidemicGenerator` reaproveitando o `World` legado
definido em `dengue_envs/data/generator.py`. Nada do código legado é alterado:
este módulo apenas embrulha o `World` com a interface nova.
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from dengue_envs.data.generator import World


class SyntheticGenerator:
    """Distribuição sintética baseada em curvas SIR e dispersão espacial normal.

    É a distribuição "clássica" já usada no projeto, agora exposta pela
    interface `EpidemicGenerator` para permitir substituição futura por
    distribuições mais realistas sem mudar o ambiente.
    """

    def __init__(
        self,
        size: int = 400,
        episize: int = 150,
        epilength: int = 60,
        dengue_center: Tuple[int, int] = (100, 100),
        chik_center: Tuple[int, int] = (300, 300),
        dengue_radius: int = 90,
        chik_radius: int = 90,
    ):
        self.size = size
        self.episize = episize
        self.epilength = epilength
        self.dengue_center = dengue_center
        self.chik_center = chik_center
        self.dengue_radius = dengue_radius
        self.chik_radius = chik_radius

    def build_world(self) -> World:
        """Cria uma instância nova do `World` legado com os parâmetros atuais."""
        return World(
            self.size,
            self.episize,
            self.epilength,
            self.dengue_center,
            self.chik_center,
            self.dengue_radius,
            self.chik_radius,
        )

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        """Gera o dataframe de casos.

        `seed` é aceito por compatibilidade com a interface. O `World` legado
        usa o RNG global do numpy; a reprodutibilidade determinística completa
        será tratada numa fase posterior (ver plano.md, P2).
        """
        if seed is not None:
            import numpy as np

            np.random.seed(seed)
        world = self.build_world()
        return world.casedf.copy()
