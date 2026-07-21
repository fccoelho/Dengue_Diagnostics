"""Confirmação epidemiológica baseada em densidade local de casos.

Extração fiel de `DengueDiagnosticsEnv._epi_confirm`. Recebe os mapas de
densidade (`dmap`, `cmap`) explicitamente para ser uma função pura e testável.
"""
from __future__ import annotations

import numpy as np

from dengue_envs.core.clinical import CHIK, DENGUE


def epi_confirm(clinical: int, x: int, y: int, dmap: np.ndarray, cmap: np.ndarray,
                threshold: int = 1) -> int:
    """Retorna confirmação epidemiológica (1) ou não (0).

    A confirmação usa a densidade local de casos da doença suspeita:
    - suspeita de dengue -> olha `dmap`
    - suspeita de chik   -> olha `cmap`
    - "outro"            -> nunca confirma
    """
    x, y = int(x), int(y)
    if clinical == DENGUE:
        return 1 if dmap[x, y] > threshold else 0
    if clinical == CHIK:
        return 1 if cmap[x, y] > threshold else 0
    return 0
