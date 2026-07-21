"""Camada de dados: geradores de epidemias plugáveis.

- ``World`` / ``SyntheticGenerator`` — distribuição sintética (SIR + truncnorm)
- ``KrigingDensityGenerator`` — import explícito (evita ciclo com este pacote):

    from dengue_envs.data.kriging_generator import KrigingDensityGenerator
    from dengue_envs.data.kriging import load_zikario_cases, intensity_surface_from_points
"""
from dengue_envs.data.base import CASE_COLUMNS, EpidemicGenerator
from dengue_envs.data.generator import World
from dengue_envs.data.synthetic import SyntheticGenerator

__all__ = [
    "World",
    "EpidemicGenerator",
    "SyntheticGenerator",
    "CASE_COLUMNS",
    "KrigingDensityGenerator",
]


def __getattr__(name: str):
    if name == "KrigingDensityGenerator":
        from dengue_envs.data.kriging_generator import KrigingDensityGenerator

        return KrigingDensityGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
