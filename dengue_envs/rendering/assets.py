"""Localização e carregamento dos assets de imagem da renderização.

Os PNGs usados pelos sprites/legenda ficam em `dengue_envs/rendering/assets/`
(separados do código do ambiente). Este módulo centraliza o caminho e o
carregamento com cache, para que nenhum outro módulo precise saber onde os
arquivos estão nem repetir `pygame.image.load(...)`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import pygame

# Pasta canônica dos assets (ao lado deste arquivo).
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Mapeia a AÇÃO aplicada a um caso -> arquivo de ícone exibido no sprite/legenda.
# (0 testar dengue, 1 testar chik, 2 epi confirm, 3 nada,
#  4 concluir dengue, 5 concluir chik, 6 concluir outro/descartar)
ACTION_ICONS = {
    0: "dengue_test.png",
    1: "chick_test.png",
    2: "epi_test.png",
    3: "no_test.png",
    4: "dengue-checked.png",
    5: "chik-checked.png",
    6: "discard_test.png",
}


def asset_path(name: str) -> str:
    """Caminho absoluto (str) de um asset pelo nome do arquivo."""
    path = ASSETS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Asset de renderização não encontrado: {path}")
    return str(path)


@lru_cache(maxsize=None)
def _load_raw(name: str) -> pygame.Surface:
    """Carrega o asset uma única vez (cacheado). Requer display inicializado."""
    return pygame.image.load(asset_path(name)).convert_alpha()


def load_image(name: str, size: Optional[Tuple[int, int]] = None) -> pygame.Surface:
    """Retorna a superfície do asset, opcionalmente redimensionada.

    O redimensionamento é aplicado a uma cópia do surface cacheado, para não
    corromper o cache.
    """
    surface = _load_raw(name)
    if size is not None:
        return pygame.transform.scale(surface, size)
    return surface


def clear_cache() -> None:
    """Limpa o cache de imagens (útil ao recriar o display)."""
    _load_raw.cache_clear()
