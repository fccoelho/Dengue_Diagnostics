"""Gerador de epidemias a partir de superfícies Ordinary Kriging.

Fluxo alinhado ao notebook ``notebooks/kriging_spatial_probability.ipynb``:

1. Carrega (ou constrói) ``.npz`` com ``prob_dengue`` / ``prob_chikungunya``.
2. Redimensiona as probabilidades para o grid do env (``size × size``).
3. Mantém a dinâmica temporal SIR do ``World`` legado.
4. Amostra posições ``(x, y)`` ∝ ``P(célula | doença)`` em vez da truncnorm.

O env não muda: o objeto retornado é World-like
(``casedf``, ``get_series_up_to_t``, ``get_maps_up_to_t``, ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.ndimage import zoom

from dengue_envs.data.kriging import (
    DEFAULT_RIO_BBOX_LONLAT,
    bbox_lonlat_to_projected,
    intensity_surface_from_points,
    load_zikario_cases,
    normalize_probability,
    stack_disease_probabilities,
)

DEFAULT_SURFACES_PATH = Path("results/kriging/rio_2015_2016_kriging_surfaces.npz")

# O env/modelo só trabalha dengue (0) e chik (1) — Zika fica fora do pipeline.
MODEL_DISEASES: Tuple[str, ...] = ("Dengue", "Chikungunya")


@dataclass
class KrigingSurfaces:
    """Superfícies de probabilidade no grid de predição (metros / notebook)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    cell_size: float
    prob_dengue: np.ndarray  # (ny, nx), soma ≈ 1
    prob_chik: np.ndarray
    intensity_dengue: Optional[np.ndarray] = None
    intensity_chik: Optional[np.ndarray] = None
    class_prob_dengue: Optional[np.ndarray] = None
    class_prob_chik: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.prob_dengue.shape


def load_kriging_surfaces(path: Union[str, Path]) -> KrigingSurfaces:
    """Carrega o ``.npz`` exportado pelo notebook / script de build."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Superfícies de Kriging não encontradas: {path.resolve()}\n"
            "Gere com:\n"
            "  poetry run python -m dengue_envs.data.build_kriging_surfaces\n"
            "ou rode notebooks/kriging_spatial_probability.ipynb (célula de export)."
        )
    data = np.load(path, allow_pickle=True)
    files = set(data.files)

    def _req(key: str) -> np.ndarray:
        if key not in files:
            raise KeyError(f"Chave ausente no npz ({path}): {key!r}. Chaves: {sorted(files)}")
        return np.asarray(data[key], dtype=float)

    return KrigingSurfaces(
        xmin=float(data["xmin"]),
        xmax=float(data["xmax"]),
        ymin=float(data["ymin"]),
        ymax=float(data["ymax"]),
        cell_size=float(data["cell_size"]),
        prob_dengue=_req("prob_dengue"),
        prob_chik=_req("prob_chikungunya"),
        intensity_dengue=_req("intensity_dengue") if "intensity_dengue" in files else None,
        intensity_chik=_req("intensity_chikungunya") if "intensity_chikungunya" in files else None,
        class_prob_dengue=_req("class_prob_dengue") if "class_prob_dengue" in files else None,
        class_prob_chik=_req("class_prob_chikungunya") if "class_prob_chikungunya" in files else None,
        meta={
            "path": str(path),
            "variogram_model": (
                str(data["variogram_model"]) if "variogram_model" in files else None
            ),
        },
    )


def build_kriging_surfaces(
    gpkg_path: Union[str, Path],
    *,
    years: Tuple[int, ...] = (2015, 2016),
    obs_cell_m: float = 500.0,
    pred_cell_m: float = 500.0,
    variogram_model: str = "spherical",
    diseases: Tuple[str, ...] = MODEL_DISEASES,
) -> Tuple[KrigingSurfaces, Dict[str, Any]]:
    """Pipeline do notebook: zikario → Ordinary Kriging → probabilidades.

    Por padrão só Dengue e Chikungunya (as classes do env). Zika não entra.
    """
    if not diseases:
        raise ValueError("diseases não pode ser vazio")
    unknown = set(diseases) - {"Dengue", "Chikungunya", "Zika"}
    if unknown:
        raise ValueError(f"Doenças desconhecidas: {sorted(unknown)}")
    if "Zika" in diseases:
        import warnings

        warnings.warn(
            "Zika incluída no Kriging, mas o env/modelo só usa dengue e chik. "
            "Prefira diseases=MODEL_DISEASES.",
            UserWarning,
            stacklevel=2,
        )

    cases = load_zikario_cases(
        str(gpkg_path),
        years=years,
        bbox_lonlat=DEFAULT_RIO_BBOX_LONLAT,
        diseases=diseases,
    )
    xmin, xmax, ymin, ymax = bbox_lonlat_to_projected(DEFAULT_RIO_BBOX_LONLAT)

    results: Dict[str, Dict[str, Any]] = {}
    for disease in diseases:
        sub = cases[cases["Doenca"] == disease]
        if len(sub) < 3:
            raise ValueError(f"Poucos casos para Kriging de {disease}: n={len(sub)}")
        counts, intensity, sigma, pred_grid = intensity_surface_from_points(
            sub["x"].to_numpy(),
            sub["y"].to_numpy(),
            bbox_xy=(xmin, xmax, ymin, ymax),
            obs_cell_size=obs_cell_m,
            pred_cell_size=pred_cell_m,
            transform="log1p",
            variogram_model=variogram_model,
        )
        results[disease] = {
            "counts": counts,
            "intensity": intensity,
            "sigma": sigma,
            "prob": normalize_probability(intensity),
            "pred_grid": pred_grid,
            "n": len(sub),
        }

    class_probs = stack_disease_probabilities(
        {d: results[d]["intensity"] for d in results}
    )
    pred_grid = results["Dengue"]["pred_grid"]
    surfaces = KrigingSurfaces(
        xmin=pred_grid.xmin,
        xmax=pred_grid.xmax,
        ymin=pred_grid.ymin,
        ymax=pred_grid.ymax,
        cell_size=pred_grid.cell_size,
        prob_dengue=results["Dengue"]["prob"],
        prob_chik=results["Chikungunya"]["prob"],
        intensity_dengue=results["Dengue"]["intensity"],
        intensity_chik=results["Chikungunya"]["intensity"],
        class_prob_dengue=class_probs.get("Dengue"),
        class_prob_chik=class_probs.get("Chikungunya"),
        meta={"variogram_model": variogram_model, "obs_cell_size": obs_cell_m},
    )
    payload: Dict[str, Any] = {
        "crs": "EPSG:31983",
        "bbox_lonlat": np.asarray(DEFAULT_RIO_BBOX_LONLAT, dtype=float),
        "xmin": pred_grid.xmin,
        "xmax": pred_grid.xmax,
        "ymin": pred_grid.ymin,
        "ymax": pred_grid.ymax,
        "cell_size": pred_grid.cell_size,
        "variogram_model": variogram_model,
        "obs_cell_size": obs_cell_m,
    }
    for disease, r in results.items():
        key = disease.lower()
        payload[f"intensity_{key}"] = r["intensity"]
        payload[f"prob_{key}"] = r["prob"]
        payload[f"sigma_{key}"] = r["sigma"]
        payload[f"class_prob_{key}"] = class_probs[disease]
    return surfaces, payload


def save_kriging_surfaces(payload: Dict[str, Any], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def probability_to_env_grid(prob: np.ndarray, size: int) -> np.ndarray:
    """Redimensiona ``prob`` (ny, nx) → ``(size, size)`` indexado como ``[x, y]``.

    O notebook usa ``imshow(..., origin='lower')`` com shape ``(ny, nx)``.
    O ``World`` indexa mapas como ``map[x, y]`` (histogram2d). Transpomos e
    reamostramos para o grid discreto do ambiente.
    """
    if size < 2:
        raise ValueError("size deve ser >= 2")
    # (ny, nx) → (x, y) ≈ (nx, ny)
    src = np.asarray(prob, dtype=float).T
    zoom_factors = (size / src.shape[0], size / src.shape[1])
    out = zoom(src, zoom_factors, order=1)
    out = np.asarray(out, dtype=float)
    # Ajuste exato de shape (zoom pode arredondar).
    canvas = np.zeros((size, size), dtype=float)
    hx = min(size, out.shape[0])
    hy = min(size, out.shape[1])
    canvas[:hx, :hy] = out[:hx, :hy]
    canvas = np.clip(canvas, 0.0, None)
    total = float(canvas.sum())
    if total <= 1e-12:
        canvas[:] = 1.0 / canvas.size
    else:
        canvas /= total
    return canvas


def sample_xy_from_prob(
    prob_xy: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Amostra ``n`` células ∝ ``prob_xy[x, y]``; retorna arrays inteiros x, y."""
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    flat = np.asarray(prob_xy, dtype=float).ravel()
    flat = np.clip(flat, 0.0, None)
    s = flat.sum()
    if s <= 0:
        flat = np.ones_like(flat) / flat.size
    else:
        flat = flat / s
    idx = rng.choice(flat.size, size=n, replace=True, p=flat)
    xs, ys = np.unravel_index(idx, prob_xy.shape)
    return xs.astype(int), ys.astype(int)


def _focus_from_prob(prob_xy: np.ndarray) -> Tuple[Tuple[int, int], int]:
    """Centro = célula de maior massa; raio ≈ desvio espacial ponderado."""
    size = prob_xy.shape[0]
    xs = np.arange(size)
    ys = np.arange(size)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    mass = float(prob_xy.sum()) + 1e-12
    cx = int(np.clip(round(float((xx * prob_xy).sum() / mass)), 0, size - 1))
    cy = int(np.clip(round(float((yy * prob_xy).sum() / mass)), 0, size - 1))
    var = float((((xx - cx) ** 2 + (yy - cy) ** 2) * prob_xy).sum() / mass)
    radius = int(np.clip(round(np.sqrt(max(var, 1.0))), 5, size // 2))
    return (cx, cy), radius


def _sir_cumulative(popsize: int, epilength: int, R0: float, I0: int = 10) -> np.ndarray:
    """Curva de incidência acumulada (igual ao ``World`` legado)."""

    def SIR(Y, t, beta, gamma, N):
        S, I, Inc, R = Y
        return [-beta * S * I, beta * S * I - gamma * I, beta * S * I, gamma * I]

    gamma = 0.004
    beta = float(R0) * gamma
    y = odeint(
        SIR,
        [popsize - I0, I0, 0, 0],
        np.arange(0, epilength),
        args=(beta, gamma, popsize),
    )
    return y[:, 2]


class KrigingWorld:
    """World-like: posições via Kriging, tempo via SIR."""

    def __init__(
        self,
        size: int,
        popsize: int,
        epilength: int,
        surfaces: KrigingSurfaces,
        *,
        dengue_r0: float = 1.5,
        chik_r0: float = 1.2,
        random_state: Optional[np.random.Generator] = None,
    ):
        self.size = int(size)
        self.popsize = int(popsize)
        self.epilength = int(epilength)
        self.dengue_r0 = float(dengue_r0)
        self.chik_r0 = float(chik_r0)
        self._rng = random_state or np.random.default_rng()
        self.surfaces = surfaces

        self.prob_dengue = probability_to_env_grid(surfaces.prob_dengue, self.size)
        self.prob_chik = probability_to_env_grid(surfaces.prob_chik, self.size)
        self.dengue_center, self.dengue_radius = _focus_from_prob(self.prob_dengue)
        self.chik_center, self.chik_radius = _focus_from_prob(self.prob_chik)

        self.dengue_curve = _sir_cumulative(self.popsize, self.epilength, self.dengue_r0)
        self.chik_curve = _sir_cumulative(self.popsize, self.epilength, self.chik_r0)

        self.case_series = []
        self.case_dict = {}
        self.dengue_total = 0
        self.chik_total = 0
        self.casedf = None
        self.build_case_series()
        self.build_case_dataframe()

    def _sample(self, disease: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        prob = self.prob_dengue if disease == 0 else self.prob_chik
        return sample_xy_from_prob(prob, n, self._rng)

    def build_case_series(self) -> None:
        self.case_series = []
        self.dengue_total = 0
        self.chik_total = 0
        for t in range(self.epilength):
            dcases_t = int(np.round(self.dengue_curve[t]))
            ccases_t = int(np.round(self.chik_curve[t]))
            if t < 1:
                new_d, new_c = dcases_t, ccases_t
            else:
                new_d = int(np.round(dcases_t - self.dengue_curve[t - 1]))
                new_c = int(np.round(ccases_t - self.chik_curve[t - 1]))
            dx, dy = self._sample(0, new_d)
            cx, cy = self._sample(1, new_c)
            self.dengue_total += new_d
            self.chik_total += new_c
            dengue_cases = [
                {"t": t, "x": int(x), "y": int(y), "disease": 0, "testd": 0, "testc": 0, "epiconf": 0}
                for x, y in zip(dx, dy)
            ]
            chik_cases = [
                {"t": t, "x": int(x), "y": int(y), "disease": 1, "testd": 0, "testc": 0, "epiconf": 0}
                for x, y in zip(cx, cy)
            ]
            self.case_series.append(dengue_cases + chik_cases)

    def build_case_dataframe(self) -> None:
        self.casedf = pd.DataFrame.from_records([c for c in chain(*self.case_series)])
        self.case_dict = self.casedf.to_dict(orient="index")

    def get_series_up_to_t(self, t):
        self.build_case_dataframe()
        return self.casedf[self.casedf.t <= t]

    def get_maps_up_to_t(self, t):
        if self.casedf is None:
            self.build_case_dataframe()
        casedf = self.casedf[self.casedf.t <= t]
        dengue_map = np.histogram2d(
            casedf[casedf.disease == 0].x,
            casedf[casedf.disease == 0].y,
            bins=self.size,
            range=[[0, self.size], [0, self.size]],
        )[0]
        chik_map = np.histogram2d(
            casedf[casedf.disease == 1].x,
            casedf[casedf.disease == 1].y,
            bins=self.size,
            range=[[0, self.size], [0, self.size]],
        )[0]
        return dengue_map, chik_map

    def get_maps_at_t(self, t):
        if self.casedf is None:
            self.build_case_dataframe()
        casedf = self.casedf[self.casedf.t == t]
        dengue_map = np.histogram2d(
            casedf[casedf.disease == 0].x,
            casedf[casedf.disease == 0].y,
            bins=self.size,
            range=[[0, self.size], [0, self.size]],
        )[0]
        chik_map = np.histogram2d(
            casedf[casedf.disease == 1].x,
            casedf[casedf.disease == 1].y,
            bins=self.size,
            range=[[0, self.size], [0, self.size]],
        )[0]
        return dengue_map, chik_map

    def view(self, save_path=None, show=False):
        from dengue_envs.rendering.epidemic_map import plot_epidemic_map

        if self.casedf is None:
            self.build_case_dataframe()
        return plot_epidemic_map(
            self.casedf,
            self.size,
            dengue_center=self.dengue_center,
            chik_center=self.chik_center,
            dengue_radius=self.dengue_radius,
            chik_radius=self.chik_radius,
            save_path=save_path,
            show=show,
        )


class KrigingDensityGenerator:
    """``EpidemicGenerator`` que amostra casos das superfícies de Kriging."""

    def __init__(
        self,
        size: int = 400,
        episize: int = 150,
        epilength: int = 60,
        *,
        surfaces: Optional[KrigingSurfaces] = None,
        surfaces_path: Union[str, Path] = DEFAULT_SURFACES_PATH,
        dengue_r0: float = 1.5,
        chik_r0: float = 1.2,
    ):
        self.size = size
        self.episize = episize
        self.epilength = epilength
        self.dengue_r0 = dengue_r0
        self.chik_r0 = chik_r0
        self.surfaces_path = Path(surfaces_path)
        self._surfaces = surfaces

    @property
    def surfaces(self) -> KrigingSurfaces:
        if self._surfaces is None:
            self._surfaces = load_kriging_surfaces(self.surfaces_path)
        return self._surfaces

    def build_world(
        self,
        *,
        random_state: Optional[np.random.Generator] = None,
        dengue_r0: Optional[float] = None,
        chik_r0: Optional[float] = None,
    ) -> KrigingWorld:
        return KrigingWorld(
            self.size,
            self.episize,
            self.epilength,
            self.surfaces,
            dengue_r0=self.dengue_r0 if dengue_r0 is None else dengue_r0,
            chik_r0=self.chik_r0 if chik_r0 is None else chik_r0,
            random_state=random_state,
        )

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return self.build_world(random_state=rng).casedf.copy()
