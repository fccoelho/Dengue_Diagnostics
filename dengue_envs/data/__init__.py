"""Camada de dados: geradores de epidemias plugáveis.

Mantém o `World` legado acessível e adiciona a interface `EpidemicGenerator`
com o adaptador `SyntheticGenerator`. Geradores futuros (Rio2016, Kriging)
já têm seus stubs reservados.

Importações legadas continuam válidas, por exemplo:
    from dengue_envs.data.generator import World
"""
from dengue_envs.data.base import CASE_COLUMNS, EpidemicGenerator
from dengue_envs.data.generator import World
from dengue_envs.data.synthetic import SyntheticGenerator

# Funções de Kriging são opcionais (geopandas/pykrige); import explícito:
#   from dengue_envs.data.kriging import load_zikario_cases, intensity_surface_from_points

__all__ = [
    "World",
    "EpidemicGenerator",
    "SyntheticGenerator",
    "CASE_COLUMNS",
]
