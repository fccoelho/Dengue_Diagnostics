"""Camada de renderização (Pygame/matplotlib).

- `PygameRenderer`: janela + desenho do ambiente (isolado do env).
- `CaseSprite` / `CaseGroup`: sprites dos casos.
- `lineplot`: utilitário matplotlib -> PNG em memória.
- `assets`: localização/carregamento dos ícones.
"""
from dengue_envs.rendering.epidemic_map import plot_epidemic_map
from dengue_envs.rendering.pygame_renderer import PygameRenderer, lineplot
from dengue_envs.rendering.sprites import CaseGroup, CaseSprite

__all__ = ["PygameRenderer", "CaseSprite", "CaseGroup", "lineplot", "plot_epidemic_map"]
