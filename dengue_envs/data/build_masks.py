"""Máscaras de área habitada do Rio e do Recife, no grid de 500 m do kriging.

Por que existem: os casos "outro" (doença febril que não é arbovirose) caíam
uniformes no grid INTEIRO. No Rio a superfície cobre o retângulo todo, mas no
Recife as arboviroses só existem dentro do município (44% do grid) — então
qualquer caso fora do município era "outro" com certeza, e a posição virava um
atalho que não existe no Rio. Com a máscara, os casos "outro" caem só onde
mora gente, e as duas arboviroses também (ver `physical_env_grids`).

As duas cidades seguem a MESMA regra, derivada das notificações: células de
500 m com pelo menos `MIN_CASOS` notificações de qualquer agravo e qualquer
ano, fechadas e com os buracos preenchidos (parques e morros cercados de
cidade ficam dentro; mar, baía e maciços nas bordas, fora). No Recife ela é
ainda cortada pelo polígono oficial do município, que já está nas superfícies.

Uso: `python -m dengue_envs.data.build_masks` → `results/kriging/<cidade>_mask.npz`
e figuras de conferência `results/kriging/<cidade>_mask.png`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from dengue_envs.data.kriging import DEFAULT_RIO_BBOX_LONLAT, project_lonlat
from dengue_envs.data.kriging_generator import DEFAULT_SURFACES_PATH, load_kriging_surfaces

_RAIZ = Path(__file__).resolve().parents[2]
GPKG = Path(__file__).resolve().parent / "zikario.gpkg"
SAIDA = _RAIZ / "results" / "kriging"
MIN_CASOS = 2  # uma notificação isolada pode ser erro de geocodificação


def mascara_de_pontos(x, y, xmin, ymin, cell, ny, nx, min_casos: int = MIN_CASOS) -> np.ndarray:
    """Células (ny, nx) com >= `min_casos` pontos, fechadas e sem buracos."""
    ix = np.floor((np.asarray(x) - xmin) / cell).astype(int)
    iy = np.floor((np.asarray(y) - ymin) / cell).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cont = np.zeros((ny, nx), dtype=int)
    np.add.at(cont, (iy[ok], ix[ok]), 1)
    m = cont >= min_casos
    m = ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=1)
    m = ndimage.binary_fill_holes(m)
    # Fica só com componentes grandes (ilhas de poucas células são ruído).
    rot, n = ndimage.label(m)
    if n > 1:
        tam = ndimage.sum(m, rot, index=np.arange(1, n + 1))
        m = np.isin(rot, 1 + np.flatnonzero(tam >= 10))
    return m


def _pontos_rio():
    import geopandas as gpd

    gdf = gpd.read_file(GPKG, engine="pyogrio")
    lon_min, lat_min, lon_max, lat_max = DEFAULT_RIO_BBOX_LONLAT
    ok = gdf["latitude"].between(lat_min, lat_max) & gdf["longitude"].between(lon_min, lon_max)
    return project_lonlat(gdf.loc[ok, "longitude"].to_numpy(), gdf.loc[ok, "latitude"].to_numpy())


def _pontos_recife():
    from dengue_envs.data.build_recife_surfaces import carrega_geocodificado

    df = carrega_geocodificado()
    df = df[df["x"].notna()]
    return df["x"].to_numpy(), df["y"].to_numpy()


def _figura(s, m, destino: Path, titulo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    with np.errstate(divide="ignore"):
        ax.imshow(np.log(s.prob_dengue + s.prob_chik), origin="lower", cmap="Greys")
    ax.contour(m, levels=[0.5], colors="#eb6834", linewidths=1.2)
    ax.set_title(titulo)
    fig.savefig(destino, dpi=120, bbox_inches="tight")
    plt.close(fig)


def constroi(cidade: str) -> Path:
    if cidade == "rio":
        s = load_kriging_surfaces(_RAIZ / DEFAULT_SURFACES_PATH)
        x, y = _pontos_rio()
        oficial = None
    else:
        caminho = SAIDA / "recife_2016_kriging_surfaces.npz"
        s = load_kriging_surfaces(caminho)
        x, y = _pontos_recife()
        oficial = np.load(caminho)["mask"].astype(bool)
    ny, nx = s.shape
    m = mascara_de_pontos(x, y, s.xmin, s.ymin, s.cell_size, ny, nx)
    if oficial is not None:
        print(f"{cidade}: polígono oficial {oficial.sum()} células; regra das notificações {m.sum()}")
        m &= oficial
    destino = SAIDA / f"{cidade}_mask.npz"
    np.savez_compressed(destino, mask=m, n_pontos=len(x), min_casos=MIN_CASOS,
                        xmin=s.xmin, ymin=s.ymin, cell_size=s.cell_size)
    print(f"{cidade}: {m.sum()} de {m.size} células ({m.mean():.1%}), "
          f"{m.sum() * s.cell_size ** 2 / 1e6:.0f} km²; {len(x)} notificações")
    _figura(s, m, destino.with_suffix(".png"), f"{cidade}: área habitada (laranja) sobre o kriging")
    return destino


def main() -> None:
    for cidade in ("rio", "recife"):
        constroi(cidade)


if __name__ == "__main__":
    main()
