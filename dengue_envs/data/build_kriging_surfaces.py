"""Constrói o ``.npz`` de superfícies Ordinary Kriging (mesmo fluxo do notebook).

Por padrão só **Dengue** e **Chikungunya** (classes do env). Zika não entra.

Uso (da raiz do repositório)::

    poetry run python -m dengue_envs.data.build_kriging_surfaces
    poetry run python -m dengue_envs.data.build_kriging_surfaces --obs-cell 1000 --pred-cell 500

Saída padrão: ``results/kriging/rio_2015_2016_kriging_surfaces.npz``
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dengue_envs.data.kriging_generator import (
    DEFAULT_SURFACES_PATH,
    MODEL_DISEASES,
    build_kriging_surfaces,
    save_kriging_surfaces,
)

_DEFAULT_GPKG = Path(__file__).resolve().parent / "zikario.gpkg"


def main() -> None:
    parser = argparse.ArgumentParser(description="Exportar superfícies Ordinary Kriging.")
    parser.add_argument(
        "--gpkg",
        type=Path,
        default=_DEFAULT_GPKG,
        help="GeoPackage zikario (default: dengue_envs/data/zikario.gpkg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SURFACES_PATH,
        help=f"Arquivo .npz de saída (default: {DEFAULT_SURFACES_PATH}).",
    )
    parser.add_argument("--obs-cell", type=float, default=500.0, help="Célula de agregação (m).")
    parser.add_argument("--pred-cell", type=float, default=500.0, help="Célula de predição (m).")
    parser.add_argument(
        "--variogram",
        default="spherical",
        choices=("spherical", "exponential", "gaussian", "linear"),
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2015, 2016])
    args = parser.parse_args()

    if not args.gpkg.exists():
        raise SystemExit(f"GeoPackage não encontrado: {args.gpkg.resolve()}")

    print(f"[kriging] lendo {args.gpkg}")
    print(f"[kriging] doenças do modelo: {MODEL_DISEASES} (Zika excluída)")
    print(
        f"[kriging] obs_cell={args.obs_cell}m pred_cell={args.pred_cell}m "
        f"variogram={args.variogram}"
    )
    surfaces, payload = build_kriging_surfaces(
        args.gpkg,
        years=tuple(args.years),
        obs_cell_m=args.obs_cell,
        pred_cell_m=args.pred_cell,
        variogram_model=args.variogram,
        diseases=MODEL_DISEASES,
    )
    # Garante que o NPZ do treino não carregue chaves de Zika.
    for key in list(payload.keys()):
        if "zika" in key.lower():
            del payload[key]
    out = save_kriging_surfaces(payload, args.output)
    print(f"[kriging] salvo: {out.resolve()}")
    print(
        f"[kriging] grid dengue {surfaces.prob_dengue.shape} | "
        f"chik {surfaces.prob_chik.shape}"
    )


if __name__ == "__main__":
    main()
