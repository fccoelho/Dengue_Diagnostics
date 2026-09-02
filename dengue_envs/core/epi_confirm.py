"""Confirmação epidemiológica baseada em densidade local de casos.

Função pura, para ser testável fora do ambiente. Recebe os mapas de densidade
(`dmap`, `cmap`) explicitamente.

**Qual mapa passar.** O ambiente passa os mapas de casos *confirmados por
laudo* — o que o agente efetivamente sabe. Passar os mapas do gerador
(`world.get_maps_up_to_t`) entrega a verdade-terreno e é um vazamento: os
mapas do gerador são indexados pela coluna `disease` real, inclusive de casos
que nunca foram notificados.

**Por que o raio existe.** A versão original olhava uma única célula com
`threshold=1`, isto é, exigia ≥2 casos na *mesma* célula. Medido no ambiente
v4 (grid 400×400, ~140 casos de cada arbovirose): a densidade máxima por
célula é 1, então a condição nunca era satisfeita — 5598 chamadas em 5
episódios, todas devolvendo 0. A ação existia, custava, e não informava nada.
Com vizinhança de raio 20 a mesma medição dá 90,7% de cobertura e 92,9% de
acerto na discriminação dengue × chikungunya.
"""
from __future__ import annotations

import numpy as np

from dengue_envs.core.clinical import CHIK, DENGUE


def neighborhood_sum(m: np.ndarray, x: int, y: int, radius: int) -> float:
    """Soma de `m` na vizinhança quadrada de lado ``2*radius + 1`` em (x, y)."""
    if radius <= 0:
        return float(m[x, y])
    size_x, size_y = m.shape
    x0, x1 = max(0, x - radius), min(size_x, x + radius + 1)
    y0, y1 = max(0, y - radius), min(size_y, y + radius + 1)
    return float(m[x0:x1, y0:y1].sum())


def epi_confirm(
    clinical: int,
    x: int,
    y: int,
    dmap: np.ndarray,
    cmap: np.ndarray,
    threshold: int = 1,
    radius: int = 0,
    exclude: tuple = (0.0, 0.0),
) -> int:
    """Retorna confirmação epidemiológica (1) ou não (0).

    Olha a densidade local da doença **suspeita**:
    - suspeita de dengue -> `dmap`
    - suspeita de chik   -> `cmap`
    - "outro"            -> nunca confirma (não há vínculo a estabelecer)

    ``exclude`` é a contribuição do **próprio caso** aos mapas, subtraída antes
    da comparação. Sem isso a ação leria de volta o laudo do próprio caso —
    medido: raio 0 com auto-inclusão dá 99,6% de "acerto", que é o exame já
    pago sendo devolvido com outro nome, não evidência epidemiológica.
    """
    x, y = int(x), int(y)
    if clinical == DENGUE:
        density = neighborhood_sum(dmap, x, y, radius) - float(exclude[0])
    elif clinical == CHIK:
        density = neighborhood_sum(cmap, x, y, radius) - float(exclude[1])
    else:
        return 0
    return 1 if density > threshold else 0
