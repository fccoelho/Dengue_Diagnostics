"""Poisson Kriging (Goovaerts) para taxas de doença por área.

Referência: Goovaerts (2005/2006) — geostatistical analysis of disease rates.
Taxa observada ``z(v) = d(v) / n(v)``; o risco ``R(v)`` é o alvo. A matriz de
covariância das taxas inclui um termo diagonal de erro Poisson
``m* / n(v_α)``, que reduz o peso de áreas com população pequena.

Este módulo é independente de PyMC (só NumPy/SciPy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist


def spherical_cov(h: np.ndarray, sill: float, range_: float, nugget: float = 0.0) -> np.ndarray:
    """Covariância esférica isotrópica: C(h) = nugget·1_{h=0} + sill·sph(|h|/a)."""
    h = np.asarray(h, dtype=float)
    a = max(float(range_), 1e-12)
    r = h / a
    sph = np.zeros_like(r, dtype=float)
    inside = r <= 1.0
    ri = r[inside]
    sph[inside] = 1.0 - 1.5 * ri + 0.5 * ri**3
    cov = sill * sph
    if nugget:
        cov = np.where(h < 1e-12, cov + nugget, cov)
    return cov


def exponential_cov(h: np.ndarray, sill: float, range_: float, nugget: float = 0.0) -> np.ndarray:
    """Covariância exponencial: C(h) = nugget·1_{0} + sill·exp(-3|h|/a) (range prático)."""
    h = np.asarray(h, dtype=float)
    a = max(float(range_), 1e-12)
    cov = sill * np.exp(-3.0 * h / a)
    if nugget:
        cov = np.where(h < 1e-12, cov + nugget, cov)
    return cov


COV_MODELS = {
    "spherical": spherical_cov,
    "exponential": exponential_cov,
}


@dataclass
class ArealRates:
    """Unidades areais com contagens, exposição (população) e taxas."""

    x: np.ndarray  # centróides (n,)
    y: np.ndarray
    cases: np.ndarray  # d(v)
    population: np.ndarray  # n(v) > 0
    area_id: Optional[np.ndarray] = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.cases = np.asarray(self.cases, dtype=float)
        self.population = np.asarray(self.population, dtype=float)
        if np.any(self.population <= 0):
            raise ValueError("population deve ser > 0 em todas as áreas.")
        if not (len(self.x) == len(self.y) == len(self.cases) == len(self.population)):
            raise ValueError("x, y, cases e population devem ter o mesmo comprimento.")

    @property
    def n_areas(self) -> int:
        return len(self.cases)

    @property
    def rates(self) -> np.ndarray:
        """z(v) = d(v) / n(v)."""
        return self.cases / self.population

    @property
    def mean_rate(self) -> float:
        """Média ponderada pela população (m*)."""
        return float(self.cases.sum() / self.population.sum())

    def coords(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])


@dataclass
class VariogramParams:
    sill: float
    range_: float
    nugget: float = 0.0
    model: Literal["spherical", "exponential"] = "spherical"

    def cov(self, h: np.ndarray) -> np.ndarray:
        return COV_MODELS[self.model](h, self.sill, self.range_, self.nugget)


def empirical_semivariogram(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_lags: int = 12,
    max_dist: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semivariograma empírico (Matheron): retorna (lag_centers, gamma, counts)."""
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    d = cdist(coords, coords)
    iu = np.triu_indices(len(values), k=1)
    dists = d[iu]
    diffs = 0.5 * (values[iu[0]] - values[iu[1]]) ** 2
    if max_dist is None:
        max_dist = float(np.percentile(dists, 50))
    edges = np.linspace(0.0, max_dist, n_lags + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gamma = np.full(n_lags, np.nan)
    counts = np.zeros(n_lags, dtype=int)
    for i in range(n_lags):
        mask = (dists >= edges[i]) & (dists < edges[i + 1])
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            gamma[i] = float(diffs[mask].mean())
    return centers, gamma, counts


def fit_variogram_least_squares(
    lag: np.ndarray,
    gamma: np.ndarray,
    counts: np.ndarray,
    *,
    model: Literal["spherical", "exponential"] = "spherical",
    range_bounds: Optional[Tuple[float, float]] = None,
) -> VariogramParams:
    """Ajuste simples sill/range/nugget por grade (robusto o bastante para exploração)."""
    lag = np.asarray(lag, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    counts = np.asarray(counts, dtype=float)
    ok = np.isfinite(gamma) & (counts > 0)
    lag, gamma, counts = lag[ok], gamma[ok], counts[ok]
    if len(lag) < 2:
        raise ValueError("Semivariograma empírico insuficiente para ajuste.")

    sill0 = float(np.nanmax(gamma))
    nug0 = float(max(gamma[0] * 0.1, 0.0))
    rmin, rmax = range_bounds or (float(lag[0]), float(lag[-1] * 1.5))
    ranges = np.linspace(max(rmin, 1.0), rmax, 25)
    sills = np.linspace(max(sill0 * 0.3, 1e-12), sill0 * 1.5 + 1e-12, 20)
    nuggets = np.linspace(0.0, max(sill0 * 0.5, 1e-12), 10)

    best = None
    best_loss = np.inf
    cov_fn = COV_MODELS[model]
    # gamma_model(h) = (nugget + sill) - C(h)  com C incluindo nugget no zero
    for a in ranges:
        for s in sills:
            for nug in nuggets:
                c0 = nug + s
                gamma_hat = c0 - cov_fn(lag, s, a, nug)
                # no lag>0, C não inclui nugget fora da origem; cov_fn já trata
                w = counts / counts.sum()
                loss = float(np.sum(w * (gamma - gamma_hat) ** 2))
                if loss < best_loss:
                    best_loss = loss
                    best = VariogramParams(sill=s, range_=a, nugget=nug, model=model)
    assert best is not None
    return best


def rate_covariance_matrix(
    areal: ArealRates,
    vparams: VariogramParams,
    *,
    mean_rate: Optional[float] = None,
) -> np.ndarray:
    """C_z(α,β) = C_R(α,β) + δ_αβ · m*/n(α)  (Goovaerts)."""
    mstar = areal.mean_rate if mean_rate is None else float(mean_rate)
    coords = areal.coords()
    h = cdist(coords, coords)
    c_r = vparams.cov(h)
    # No zero, cov_model já soma nugget; o termo Poisson é EXTRA na diagonal
    c_z = c_r.copy()
    diag = np.diag_indices(areal.n_areas)
    c_z[diag] = c_z[diag] + mstar / areal.population
    return c_z


def risk_covariance_points(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    vparams: VariogramParams,
) -> np.ndarray:
    """C_R entre dois conjuntos de coordenadas (sem termo Poisson)."""
    return vparams.cov(cdist(coords_a, coords_b))


@dataclass
class PoissonKrigingResult:
    estimate: np.ndarray
    variance: np.ndarray
    weights_sum: Optional[np.ndarray] = None


def poisson_kriging(
    areal: ArealRates,
    vparams: VariogramParams,
    target_xy: np.ndarray,
    *,
    mean_rate: Optional[float] = None,
    clip_nonnegative: bool = True,
) -> PoissonKrigingResult:
    """Poisson Kriging (area-to-point) nos alvos ``target_xy`` (N, 2).

    Resolve, para cada alvo v0:
        Σ_β λ_β C_z(α,β) + μ = C_R(α, v0)
        Σ_β λ_β = 1
    e
        σ²_PK(v0) = C_R(0) - Σ λ_β C_R(β,v0) - μ
    """
    z = areal.rates
    mstar = areal.mean_rate if mean_rate is None else float(mean_rate)
    c_z = rate_covariance_matrix(areal, vparams, mean_rate=mstar)
    n = areal.n_areas

    # Sistema aumentado (n+1) x (n+1)
    A = np.zeros((n + 1, n + 1), dtype=float)
    A[:n, :n] = c_z
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    A[n, n] = 0.0

    # Regularização leve se necessário
    jitter = 1e-10 * np.trace(c_z) / max(n, 1)
    A[:n, :n] = A[:n, :n] + np.eye(n) * jitter

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)

    coords = areal.coords()
    target_xy = np.asarray(target_xy, dtype=float)
    if target_xy.ndim != 2 or target_xy.shape[1] != 2:
        raise ValueError("target_xy deve ter shape (N, 2).")

    c0 = float(vparams.cov(np.array([0.0]))[0])
    estimates = np.zeros(len(target_xy))
    variances = np.zeros(len(target_xy))
    wsums = np.zeros(len(target_xy))

    c_r_all = risk_covariance_points(coords, target_xy, vparams)  # (n, N)

    for j in range(len(target_xy)):
        rhs = np.zeros(n + 1)
        rhs[:n] = c_r_all[:, j]
        rhs[n] = 1.0
        sol = A_inv @ rhs
        lam, mu = sol[:n], sol[n]
        est = float(lam @ z)
        var = c0 - float(lam @ c_r_all[:, j]) - float(mu)
        estimates[j] = est
        variances[j] = max(var, 0.0)
        wsums[j] = float(lam.sum())

    if clip_nonnegative:
        estimates = np.clip(estimates, 0.0, None)
    return PoissonKrigingResult(estimate=estimates, variance=variances, weights_sum=wsums)


def ordinary_kriging_rates(
    areal: ArealRates,
    vparams: VariogramParams,
    target_xy: np.ndarray,
    *,
    clip_nonnegative: bool = True,
) -> PoissonKrigingResult:
    """Ordinary Kriging das taxas SEM o termo Poisson (baseline de comparação)."""
    # Reusa a mesma matemática com population → ∞ (sem erro de amostragem)
    fake = ArealRates(
        x=areal.x,
        y=areal.y,
        cases=areal.rates * 1e12,  # z igual
        population=np.full(areal.n_areas, 1e12),
    )
    # Mas rates ficam iguais; o termo m*/n → 0
    return poisson_kriging(fake, vparams, target_xy, mean_rate=areal.mean_rate, clip_nonnegative=clip_nonnegative)


def synthetic_population_field(
    x: np.ndarray,
    y: np.ndarray,
    *,
    centers: Tuple[Tuple[float, float], ...] = (),
    base: float = 500.0,
    peak: float = 5000.0,
    lengthscale: float = 8000.0,
) -> np.ndarray:
    """População sintética suave (independente dos casos) para demo do 1/n.

    Sem censo no repositório, este campo evita circularidade (não usar a
    densidade de casos como população).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not centers:
        centers = (
            (float(np.percentile(x, 35)), float(np.percentile(y, 55))),
            (float(np.percentile(x, 70)), float(np.percentile(y, 45))),
        )
    pop = np.full(len(x), base, dtype=float)
    for cx, cy in centers:
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        pop += peak * np.exp(-d2 / (2 * lengthscale**2))
    return pop
