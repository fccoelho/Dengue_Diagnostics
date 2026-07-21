"""Sprites Pygame dos casos (extraídos do ambiente).

`CaseSprite` representa um caso no mapa; ao ser "testado"/decidido, troca seu
ícone conforme a ação aplicada (ver `ACTION_ICONS` em `assets`). `CaseGroup` é
um grupo de sprites por doença.
"""
from __future__ import annotations

from typing import Tuple

import pygame

from dengue_envs.rendering.assets import ACTION_ICONS, load_image

# Tamanho (px) do ícone exibido quando um caso é testado/decidido.
ICON_SIZE = (10, 10)


class CaseSprite(pygame.sprite.Sprite):
    """Sprite de um caso. Começa como um quadrado colorido e vira um ícone."""

    def __init__(
        self,
        case_id: int,
        x: int,
        y: int,
        t: int,
        disease_name: str,
        color: Tuple[int, int, int],
        size: int = 2,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.case_id = case_id
        self.position = (x, y)
        self.disease_name = disease_name
        self.t = t
        self.image = pygame.Surface((size, size))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = (x * scaling_factor, y * scaling_factor)

    def mark_as_tested(self, status: int) -> None:
        """Troca o ícone do sprite conforme a ação aplicada ao caso."""
        icon = ACTION_ICONS.get(int(status))
        if icon is None:
            return
        self.image = load_image(icon, ICON_SIZE)
        self.rect = self.image.get_rect(center=self.rect.center)

    def update(self, *args, **kwargs) -> None:
        pass


class CaseGroup(pygame.sprite.RenderPlain):
    """Grupo de `CaseSprite` de uma mesma doença."""

    def __init__(self, name: str, scaling_factor: float):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.name = name

    @property
    def cases(self):
        return self.sprites()

    def update(self, *args, **kwargs) -> None:
        pass
