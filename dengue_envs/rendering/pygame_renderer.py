"""Renderização com Pygame (camada isolada do ambiente).

Toda a lógica de janela/superfícies/sprites/plots vive aqui, no `PygameRenderer`.
O ambiente apenas delega: cria o renderer quando `render_mode != None` e chama
`create_sprites`, `update_sprites` e `render` a cada passo. Isso mantém o
`DengueDiagnosticsEnv` focado na dinâmica do problema, não em desenho.

O utilitário `lineplot` (matplotlib -> PNG em memória) é reexportado aqui por
compatibilidade com quem já importava de `dengue_envs.rendering`.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import pygame

from dengue_envs.rendering.assets import ACTION_ICONS, load_image
from dengue_envs.rendering.sprites import CaseGroup, CaseSprite
from dengue_envs.viz import lineplot

__all__ = ["PygameRenderer", "lineplot"]

# Descrições exibidas na legenda (mesma ordem/semântica de ACTION_ICONS).
LEGEND = [
    (ACTION_ICONS[0], "Dengue Test"),
    (ACTION_ICONS[1], "Chik Test"),
    (ACTION_ICONS[2], "Inconclusive"),
    (ACTION_ICONS[3], "No Test"),
    (ACTION_ICONS[4], "Confirm"),
    (ACTION_ICONS[5], "Discard"),
]

_SCREEN_SIZE = 800
_PLOT_SIZE = (400, 300)


class PygameRenderer:
    """Encapsula a janela Pygame e o desenho do ambiente DengueDiag."""

    def __init__(self, world_size: int, render_fps: int = 10, human: bool = True):
        self.world_size = world_size
        self.render_fps = render_fps
        self.scaling_factor = _SCREEN_SIZE / world_size

        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode(
            size=(_SCREEN_SIZE, _SCREEN_SIZE),
            depth=32,
            flags=pygame.SCALED,
        )
        self.world_surface = pygame.Surface((world_size, world_size))
        self.world_surface.set_colorkey((0, 0, 0))
        self.plot_surface_reward = pygame.Surface(_PLOT_SIZE)
        self.plot_surface_accuracy = pygame.Surface(_PLOT_SIZE)

        self.clock = pygame.time.Clock() if human else None
        self.reset()

    def reset(self) -> None:
        """Recria os grupos de sprites (chamar em `env.reset`)."""
        self.dengue_group = CaseGroup("dengue", self.scaling_factor)
        self.chik_group = CaseGroup("chik", self.scaling_factor)
        self.all_tests = CaseGroup("all", self.scaling_factor)

    def create_sprites(self, cases_df, t: int) -> None:
        """Cria sprites para os casos que surgem no dia `t`."""
        for case in cases_df[cases_df.t == t].itertuples():
            is_dengue = case.disease == 0
            disease = "dengue" if is_dengue else "chik"
            color = (0, 255, 0) if is_dengue else (255, 0, 0)
            # scaling_factor=1: posições em coordenadas do world_surface, que é
            # escalado para a tela pelo flag SCALED.
            sprite = CaseSprite(case.Index, case.x, case.y, case.t, disease, color, 2, 1)
            sprite.add(self.dengue_group if is_dengue else self.chik_group)

    def update_sprites(self, actions: Iterable[Tuple[int, int]]) -> None:
        """Aplica os ícones de ação aos sprites correspondentes."""
        actions = list(actions)
        for group in (self.dengue_group, self.chik_group):
            for sprite in group.sprites():
                for case_id, action_id in actions:
                    if sprite.case_id == case_id:
                        sprite.mark_as_tested(int(action_id))

    def _blit_plot(self, surface: pygame.Surface, stream) -> None:
        image = pygame.transform.scale(
            pygame.image.load(stream, "PNG"), surface.get_rect().size
        )
        surface.blit(image, (0, 0))

    def render(
        self,
        t: int,
        reward_history: Sequence[float],
        accuracy_history: Sequence[float],
        accuracy_label: str = "Multiclass Accuracy",
    ) -> None:
        """Desenha o frame: mapa de casos + gráficos + legenda."""
        pygame.event.pump()

        # Gráficos (recompensa acumulada e acurácia multiclasse).
        steps = range(1, t + 1)
        reward_plot = lineplot(
            steps, reward_history, "Step", "Total Reward", "Total Reward", "plot1"
        )
        accuracy_plot = lineplot(
            steps, accuracy_history, "Step", accuracy_label, accuracy_label, "plot2"
        )
        self._blit_plot(self.plot_surface_reward, reward_plot)
        self._blit_plot(self.plot_surface_accuracy, accuracy_plot)

        # Mapa de casos.
        self.dengue_group.draw(self.world_surface)
        self.chik_group.draw(self.world_surface)

        self.screen.fill((255, 255, 255))

        number_font = pygame.font.SysFont(None, 32)
        timestep_display = number_font.render(
            f"Step {t}", True, (0, 0, 0), (255, 255, 255)
        )
        self.screen.blit(
            timestep_display,
            (int((self.screen.get_width() - timestep_display.get_width()) / 2), 0),
        )

        self.screen.blit(
            self.plot_surface_reward, (0, 500), special_flags=pygame.BLEND_ALPHA_SDL2
        )
        self.screen.blit(
            self.plot_surface_accuracy, (400, 500), special_flags=pygame.BLEND_ALPHA_SDL2
        )
        self.screen.blit(
            self.world_surface, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2
        )

        self._draw_legend()

        pygame.display.update()
        if self.clock is not None:
            self.clock.tick(self.render_fps)

    def _draw_legend(self) -> None:
        legend_x = self.screen.get_width() - 200
        legend_y = 120
        legend_margin = 40
        legend_font = pygame.font.SysFont(None, 18)

        for icon_file, description in LEGEND:
            icon = load_image(icon_file, (10, 10))
            self.screen.blit(icon, (legend_x, legend_y))
            text = legend_font.render(description, True, (0, 0, 0))
            self.screen.blit(text, (legend_x + 40, legend_y))
            legend_y += legend_margin

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()
