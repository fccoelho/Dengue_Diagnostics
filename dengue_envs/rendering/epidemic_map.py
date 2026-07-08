"""Mapa espacial da epidemia (ground truth), independente do gerador de dados.

Funciona com qualquer ``DataFrame`` de casos que siga ``CASE_COLUMNS``
(``t``, ``x``, ``y``, ``disease``, ...): sintético (SIR), Rio 2016, Kriging,
etc. O ambiente chama esta função sobre ``real_cases`` após o ``reset``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# disease: 0 = dengue, 1 = chik, 2 = outro
_DISEASE_STYLE = {
    0: {"color": "green", "marker": ".", "label": "Dengue", "s": 12, "alpha": 0.7},
    1: {"color": "darkorange", "marker": ".", "label": "Chikungunya", "s": 12, "alpha": 0.7},
    2: {"color": "gray", "marker": ".", "label": "Outro", "s": 8, "alpha": 0.4},
}


def plot_epidemic_map(
    cases: pd.DataFrame,
    size: int,
    *,
    dengue_center: Optional[Tuple[int, int]] = None,
    chik_center: Optional[Tuple[int, int]] = None,
    dengue_radius: Optional[int] = None,
    chik_radius: Optional[int] = None,
    title: str = "Mapa da Epidemia (Ground Truth)",
    save_path: Optional[str] = None,
    show: bool = False,
    show_density: bool = True,
):
    """Plota a distribuição espacial VERDADEIRA dos casos.

    - Pontos coloridos por classe (dengue/chik/outro).
    - Camada opcional de densidade (histograma 2D), no estilo do ``World.view``.
    - Círculos tracejados dos focos quando centro/raio são informados
      (simulação sintética); omitidos automaticamente se ``None``.
    """
    if cases is None or cases.empty:
        print("Erro: nao ha casos para plotar o mapa da epidemia.")
        return None

    required = {"x", "y", "disease"}
    if not required.issubset(cases.columns):
        raise ValueError(
            f"DataFrame de casos precisa das colunas {required}, "
            f"recebido: {list(cases.columns)}"
        )

    fig, ax = plt.subplots(figsize=(10, 10))

    if show_density:
        dengue_df = cases[cases.disease == 0]
        chik_df = cases[cases.disease == 1]
        bins = size
        rng = [[0, size], [0, size]]
        if not dengue_df.empty:
            dengue_map = np.histogram2d(dengue_df.x, dengue_df.y, bins=bins, range=rng)[0]
            ax.pcolor(dengue_map.T, cmap="Greens", alpha=0.35)
        if not chik_df.empty:
            chik_map = np.histogram2d(chik_df.x, chik_df.y, bins=bins, range=rng)[0]
            ax.pcolor(chik_map.T, cmap="Blues", alpha=0.30)

    for disease_id, style in _DISEASE_STYLE.items():
        subset = cases[cases.disease == disease_id]
        if subset.empty:
            continue
        ax.scatter(
            subset.x,
            subset.y,
            c=style["color"],
            marker=style["marker"],
            s=style["s"],
            alpha=style["alpha"],
            label=style["label"],
            linewidths=0,
        )

    if dengue_center is not None and dengue_radius is not None:
        ax.add_patch(
            plt.Circle(
                dengue_center,
                dengue_radius,
                color="green",
                fill=False,
                linestyle="--",
                alpha=0.6,
                label="Foco dengue",
            )
        )
    if chik_center is not None and chik_radius is not None:
        ax.add_patch(
            plt.Circle(
                chik_center,
                chik_radius,
                color="darkorange",
                fill=False,
                linestyle="--",
                alpha=0.6,
                label="Foco chik",
            )
        )

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Legenda")
    ax.grid(True, linestyle=":", alpha=0.35)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Mapa da epidemia salvo em: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
