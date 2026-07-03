"""Renderização com Pygame (ponto de extração da Fase 1).

Nesta fase, este módulo apenas reexporta o utilitário de plotagem já existente
em `dengue_envs/viz.py`, estabelecendo o local canônico da camada de
renderização. A extração completa da lógica de sprites/desenho que hoje vive
dentro de `DengueDiagnosticsEnv` (CaseSprite, `_render_frame`, etc.) será feita
numa fase posterior, sem alterar o comportamento atual.
"""
from __future__ import annotations

from dengue_envs.viz import lineplot

__all__ = ["lineplot"]
