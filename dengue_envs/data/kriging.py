"""Kriging espacial de intensidade / probabilidade sobre um grid.

Fluxo típico (dados de notificação ponto a ponto, como ``zikario.gpkg``):

1. Projetar lon/lat para metros (UTM).
2. Agregar contagens em células de um grid (Kriging em ~100k pontos é inviável).
3. Interpolar a intensidade com Ordinary Kriging (PyKrige).
4. Normalizar a superfície para uma distribuição de probabilidade espacial
   ``P(célula)`` (soma 1) e/ou ``P(doença | célula)``.

Este módulo alimenta o notebook ``kriging_spatial_probability.ipynb`` e o
``KrigingDensityGenerator`` (``kriging_generator.py``), que amostra casos para
treino a partir do ``.npz`` exportado. Uso de intensidade no ``epi_confirm``
ainda é um passo futuro.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

# CRS padrão para o município do Rio (SIRGAS 2000 / UTM zone 23S).
DEFAULT_PROJECTED_CRS = "EPSG:31983"

# Bounding box aproximado do Rio (lon/lat WGS84), alinhado à EDA em analise_dist.
DEFAULT_RIO_BBOX_LONLAT = (-43.80, -23.10, -43.10, -22.70)

DISEASE_MAP = {
    "A90": "Dengue",
    "A920": "Chikungunya",
    "A928": "Zika",
}


@dataclass
class GridSpec:
    """Grade regular em coordenadas projetadas (metros)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    cell_size: float

    @property
    def nx(self) -> int:
        return int(np.ceil((self.xmax - self.xmin) / self.cell_size))

    @property
    def ny(self) -> int:
        return int(np.ceil((self.ymax - self.ymin) / self.cell_size))

    def centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Centros das células: vetores 1D ``xs`` (nx,) e ``ys`` (ny,)."""
        xs = self.xmin + (np.arange(self.nx) + 0.5) * self.cell_size
        ys = self.ymin + (np.arange(self.ny) + 0.5) * self.cell_size
        return xs, ys

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = self.centers()
        return np.meshgrid(xs, ys)


@dataclass
class CountGrid:
    """Contagens agregadas em um grid."""

    grid: GridSpec
    counts: np.ndarray  # shape (ny, nx)
    n_points: int

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def observation_points(
        self,
        *,
        transform: str = "log1p",
        min_count: int = 0,
        include_zeros: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pontos (x, y, z) para o Kriging a partir dos centros das células.

        ``transform``:
          - ``raw``: contagem
          - ``log1p``: log(1 + contagem) — estabiliza caudas pesadas
          - ``rate``: contagem / total (ainda não é probabilidade espacial suave)
        """
        xs, ys = self.grid.centers()
        xx, yy = np.meshgrid(xs, ys)
        zz = self.counts.astype(float)

        if transform == "raw":
            values = zz
        elif transform == "log1p":
            values = np.log1p(zz)
        elif transform == "rate":
            values = zz / max(self.total, 1)
        else:
            raise ValueError(f"transform desconhecido: {transform}")

        if include_zeros:
            mask = zz >= min_count
        else:
            mask = zz > max(min_count, 0)

        return xx[mask], yy[mask], values[mask]


def project_lonlat(
    lon: np.ndarray,
    lat: np.ndarray,
    crs_to: str = DEFAULT_PROJECTED_CRS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converte lon/lat (EPSG:4326) para metros no CRS projetado."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)
    x, y = transformer.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


def bbox_lonlat_to_projected(
    bbox_lonlat: Tuple[float, float, float, float] = DEFAULT_RIO_BBOX_LONLAT,
    crs_to: str = DEFAULT_PROJECTED_CRS,
) -> Tuple[float, float, float, float]:
    """Converte bbox (lon_min, lat_min, lon_max, lat_max) para metros."""
    lon_min, lat_min, lon_max, lat_max = bbox_lonlat
    xs, ys = project_lonlat(
        np.array([lon_min, lon_max, lon_min, lon_max]),
        np.array([lat_min, lat_min, lat_max, lat_max]),
        crs_to=crs_to,
    )
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def aggregate_counts(
    x: np.ndarray,
    y: np.ndarray,
    grid: GridSpec,
) -> CountGrid:
    """Conta pontos por célula do grid (índices fora da grade são ignorados)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ix = np.floor((x - grid.xmin) / grid.cell_size).astype(int)
    iy = np.floor((y - grid.ymin) / grid.cell_size).astype(int)
    inside = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    counts = np.zeros((grid.ny, grid.nx), dtype=np.int64)
    np.add.at(counts, (iy[inside], ix[inside]), 1)
    return CountGrid(grid=grid, counts=counts, n_points=int(inside.sum()))


def ordinary_kriging_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: GridSpec,
    *,
    variogram_model: str = "spherical",
    nlags: int = 15,
    weight: bool = True,
    pseudo_inv: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ordinary Kriging sobre os centros do ``grid``.

    Retorna ``(zhat, sigma)`` com shape ``(ny, nx)``. Valores negativos são
    clipados a 0 (intensidade não-negativa).
    """
    from pykrige.ok import OrdinaryKriging

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if len(x) < 3:
        raise ValueError("Ordinary Kriging precisa de pelo menos 3 pontos de observação.")

    ok = OrdinaryKriging(
        x,
        y,
        z,
        variogram_model=variogram_model,
        nlags=nlags,
        weight=weight,
        verbose=False,
        enable_plotting=False,
        exact_values=False,
        coordinates_type="euclidean",
        pseudo_inv=pseudo_inv,
    )
    xs, ys = grid.centers()
    zhat, sigma = ok.execute("grid", xs, ys)
    zhat = np.asarray(zhat, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    zhat = np.clip(zhat, 0.0, None)
    return zhat, sigma


def normalize_probability(surface: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Converte intensidade ≥ 0 em distribuição espacial ``P(célula)`` (soma 1)."""
    s = np.clip(np.asarray(surface, dtype=float), 0.0, None)
    total = float(s.sum())
    if total <= eps:
        return np.full_like(s, 1.0 / s.size)
    return s / total


def stack_disease_probabilities(
    surfaces: Mapping[str, np.ndarray],
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Dado um mapa doença → intensidade, retorna ``P(doença | célula)``.

    Em cada célula, as probabilidades somam 1 entre as doenças presentes.
    """
    names = list(surfaces.keys())
    stack = np.stack([np.clip(np.asarray(surfaces[n], dtype=float), 0.0, None) for n in names], axis=0)
    denom = stack.sum(axis=0, keepdims=True)
    denom = np.maximum(denom, eps)
    probs = stack / denom
    return {name: probs[i] for i, name in enumerate(names)}


def intensity_surface_from_points(
    x: np.ndarray,
    y: np.ndarray,
    *,
    grid: Optional[GridSpec] = None,
    bbox_xy: Optional[Tuple[float, float, float, float]] = None,
    obs_cell_size: float = 1000.0,
    pred_cell_size: Optional[float] = None,
    transform: str = "log1p",
    variogram_model: str = "spherical",
    include_zero_cells: bool = False,
) -> Tuple[CountGrid, np.ndarray, np.ndarray, GridSpec]:
    """Pipeline completo: agregar → Kriging → superfície no grid de predição.

    Retorna ``(count_grid, intensity, sigma, pred_grid)``.
    A intensidade está na mesma escala do ``transform`` (ex.: log1p); use
    ``normalize_probability`` para obter ``P(x,y)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if bbox_xy is None:
        pad = obs_cell_size
        bbox_xy = (float(x.min() - pad), float(x.max() + pad), float(y.min() - pad), float(y.max() + pad))
    xmin, xmax, ymin, ymax = bbox_xy

    obs_grid = grid or GridSpec(xmin, xmax, ymin, ymax, obs_cell_size)
    counts = aggregate_counts(x, y, obs_grid)
    ox, oy, oz = counts.observation_points(transform=transform, include_zeros=include_zero_cells)

    pred_size = pred_cell_size or obs_cell_size
    pred_grid = GridSpec(xmin, xmax, ymin, ymax, pred_size)
    intensity, sigma = ordinary_kriging_grid(ox, oy, oz, pred_grid, variogram_model=variogram_model)
    return counts, intensity, sigma, pred_grid


def load_zikario_cases(
    path: str,
    *,
    years: Sequence[int] = (2015, 2016),
    bbox_lonlat: Tuple[float, float, float, float] = DEFAULT_RIO_BBOX_LONLAT,
    diseases: Optional[Iterable[str]] = None,
):
    """Carrega e limpa o GeoPackage ``zikario`` (requer geopandas).

    Retorna um GeoDataFrame com colunas ``Doenca``, ``DT_SIN_PRI``, ``x``, ``y``
    (metros, ``DEFAULT_PROJECTED_CRS``).
    """
    import geopandas as gpd
    import pandas as pd

    gdf = gpd.read_file(path, engine="pyogrio")
    gdf["DT_SIN_PRI"] = pd.to_datetime(gdf["DT_SIN_PRI"], errors="coerce")
    gdf["Doenca"] = gdf["ID_AGRAVO"].map(DISEASE_MAP).fillna("Outro")

    lon_min, lat_min, lon_max, lat_max = bbox_lonlat
    mask = (
        gdf["DT_SIN_PRI"].dt.year.isin(list(years))
        & gdf["latitude"].between(lat_min, lat_max)
        & gdf["longitude"].between(lon_min, lon_max)
        & gdf["latitude"].notna()
        & gdf["longitude"].notna()
    )
    if diseases is not None:
        mask &= gdf["Doenca"].isin(list(diseases))

    out = gdf.loc[mask].copy()
    x, y = project_lonlat(out["longitude"].to_numpy(), out["latitude"].to_numpy())
    out["x"] = x
    out["y"] = y
    return out
