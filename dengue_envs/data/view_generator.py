"""Visualiza e exporta um mapa de casos gerados (synthetic ou kriging).

Cores: **verde = dengue**, **vermelho = chik**.

Uso (da raiz do repositório)::

    poetry run python -m dengue_envs.data.view_generator --generator synthetic
    poetry run python -m dengue_envs.data.view_generator --generator kriging
    poetry run python -m dengue_envs.data.view_generator --generator both --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from dengue_envs.data.kriging_generator import (
    DEFAULT_SURFACES_PATH,
    KrigingDensityGenerator,
    load_kriging_surfaces,
    probability_to_env_grid,
)
from dengue_envs.data.synthetic import SyntheticGenerator
from dengue_envs.rendering.epidemic_map import plot_epidemic_map

DEFAULT_OUT_DIR = Path("results/generator_maps")


def _build_world(
    generator: str,
    *,
    seed: int,
    size: int,
    episize: int,
    epilength: int,
    surfaces_path: Path,
):
    rng = np.random.default_rng(seed)
    if generator == "synthetic":
        # Espalha focos para o mapa não colapsar no canto.
        mid = size // 2
        margin = max(size // 8, 10)
        gen = SyntheticGenerator(
            size=size,
            episize=episize,
            epilength=epilength,
            dengue_center=(margin + size // 10, margin + size // 10),
            chik_center=(size - margin, size - margin),
            dengue_radius=max(size // 5, 20),
            chik_radius=max(size // 5, 20),
        )
        np.random.seed(seed)
        return gen.build_world()

    if generator == "kriging":
        gen = KrigingDensityGenerator(
            size=size,
            episize=episize,
            epilength=epilength,
            surfaces_path=surfaces_path,
        )
        return gen.build_world(random_state=rng)

    raise ValueError(f"Gerador desconhecido: {generator!r}")


def export_generator_map(
    generator: str = "synthetic",
    *,
    seed: int = 42,
    size: int = 400,
    episize: int = 150,
    epilength: int = 60,
    surfaces_path: Union[str, Path] = DEFAULT_SURFACES_PATH,
    output: Optional[Union[str, Path]] = None,
    show: bool = False,
    show_density: bool = True,
) -> Path:
    """Gera um episódio e salva PNG (verde dengue / vermelho chik)."""
    world = _build_world(
        generator,
        seed=seed,
        size=size,
        episize=episize,
        epilength=epilength,
        surfaces_path=Path(surfaces_path),
    )
    if output is None:
        output = DEFAULT_OUT_DIR / f"{generator}_seed_{seed}.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    n_d = int((world.casedf.disease == 0).sum())
    n_c = int((world.casedf.disease == 1).sum())
    title = (
        f"Gerador {generator} (seed={seed}) — "
        f"dengue={n_d} (verde), chik={n_c} (vermelho)"
    )
    plot_epidemic_map(
        world.casedf,
        world.size,
        dengue_center=getattr(world, "dengue_center", None),
        chik_center=getattr(world, "chik_center", None),
        dengue_radius=getattr(world, "dengue_radius", None),
        chik_radius=getattr(world, "chik_radius", None),
        title=title,
        save_path=str(output),
        show=show,
        show_density=show_density,
    )
    print(f"[view_generator] salvo: {output.resolve()}")
    return output


def export_kriging_prob_map(
    *,
    surfaces_path: Union[str, Path] = DEFAULT_SURFACES_PATH,
    size: int = 400,
    output: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Path:
    """Mapa das superfícies P(célula) reamostradas para o grid do env."""
    surfaces = load_kriging_surfaces(surfaces_path)
    pdengue = probability_to_env_grid(surfaces.prob_dengue, size)
    pchik = probability_to_env_grid(surfaces.prob_chik, size)

    if output is None:
        output = DEFAULT_OUT_DIR / "kriging_prob_surfaces.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(pdengue.T, origin="upper", cmap="Greens", aspect="equal")
    axes[0].set_title("P(célula | Dengue)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(pchik.T, origin="upper", cmap="Reds", aspect="equal")
    axes[1].set_title("P(célula | Chikungunya)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Kriging → grid do env (só dengue e chik)", y=1.02)
    plt.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"[view_generator] superfícies: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta mapa de casos do gerador (verde=dengue, vermelho=chik)."
    )
    parser.add_argument(
        "--generator",
        choices=("synthetic", "kriging", "both"),
        default="both",
        help="Qual gerador visualizar (default: both).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--episize", type=int, default=150)
    parser.add_argument("--epilength", type=int, default=60)
    parser.add_argument(
        "--surfaces",
        type=Path,
        default=DEFAULT_SURFACES_PATH,
        help="NPZ de Kriging (só dengue/chik são usados na amostragem).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Pasta de saída (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--prob-map",
        action="store_true",
        help="Também exporta o mapa das superfícies P(célula) do Kriging.",
    )
    parser.add_argument("--show", action="store_true", help="Abre janela matplotlib.")
    parser.add_argument("--no-density", action="store_true", help="Só pontos, sem heatmap.")
    args = parser.parse_args()

    names = ("synthetic", "kriging") if args.generator == "both" else (args.generator,)
    for name in names:
        export_generator_map(
            name,
            seed=args.seed,
            size=args.size,
            episize=args.episize,
            epilength=args.epilength,
            surfaces_path=args.surfaces,
            output=args.output_dir / f"{name}_seed_{args.seed}.png",
            show=args.show,
            show_density=not args.no_density,
        )

    if args.prob_map or "kriging" in names:
        export_kriging_prob_map(
            surfaces_path=args.surfaces,
            size=args.size,
            output=args.output_dir / "kriging_prob_surfaces.png",
            show=args.show,
        )


if __name__ == "__main__":
    main()
