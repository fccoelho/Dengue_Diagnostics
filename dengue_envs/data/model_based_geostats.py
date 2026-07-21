"""Geoestatística baseada em modelo (GLGM) — helpers para exploração em Python.

O padrão INLA-SPDE é tipicamente R-INLA. Em Python o equivalente rigoroso é um
GLGM com campo latente Gaussiano (GP ou GMRF/SPDE-approx) e verossimilhança
Poisson/NB, ajustado por MCMC (PyMC) ou Laplace.

Este módulo fornece:
- montagem de dados areais a partir de pontos;
- precisão GMRF (vizinhaça tipo grade / ICAR intrínseco) como proxy SPDE 2D;
- utilitários de resumo posterior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from dengue_envs.data.kriging import GridSpec, aggregate_counts
from dengue_envs.data.poisson_kriging import ArealRates, synthetic_population_field


@dataclass
class LatticeData:
    """Dados em grade regular para GLGM / Poisson Kriging."""

    grid: GridSpec
    cases: np.ndarray  # (ny, nx)
    population: np.ndarray  # (ny, nx)
    inhabited: np.ndarray  # bool (ny, nx)

    def to_areal(self, min_population: float = 1.0) -> ArealRates:
        """Empilha células habitadas em ``ArealRates`` (centróides)."""
        xs, ys = self.grid.centers()
        XX, YY = np.meshgrid(xs, ys)
        mask = self.inhabited & (self.population >= min_population)
        return ArealRates(
            x=XX[mask],
            y=YY[mask],
            cases=self.cases[mask].astype(float),
            population=self.population[mask].astype(float),
        )

    @property
    def rates(self) -> np.ndarray:
        out = np.full(self.cases.shape, np.nan, dtype=float)
        mask = self.population > 0
        out[mask] = self.cases[mask] / self.population[mask]
        return out


def build_lattice_from_points(
    x: np.ndarray,
    y: np.ndarray,
    grid: GridSpec,
    *,
    population: Optional[np.ndarray] = None,
    min_cases_for_inhabited: int = 0,
) -> LatticeData:
    """Agrega pontos em grade e associa população (sintética se omitida)."""
    counts = aggregate_counts(x, y, grid)
    xs, ys = grid.centers()
    XX, YY = np.meshgrid(xs, ys)
    if population is None:
        pop = synthetic_population_field(XX.ravel(), YY.ravel()).reshape(grid.ny, grid.nx)
    else:
        pop = np.asarray(population, dtype=float)
        if pop.shape != counts.counts.shape:
            raise ValueError("population deve ter shape (ny, nx) do grid.")
    inhabited = counts.counts >= min_cases_for_inhabited
    # células sem nenhum caso ainda podem ser habitadas se pop alta — para
    # mapeamento de risco incluímos células com pop e (opcional) vizinhos.
    # Default: habitada se houve pelo menos 1 caso OU população acima do p20.
    if min_cases_for_inhabited == 0:
        inhabited = (counts.counts > 0) | (pop >= np.percentile(pop, 20))
    return LatticeData(grid=grid, cases=counts.counts.astype(float), population=pop, inhabited=inhabited)


def grid_adjacency_ic_ar(ny: int, nx: int, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Grafo de vizinhança 4-conectada em grade → listas (node_i, node_j) undirected.

    Retorna índices lineares dos nós ativos (mask True) e matriz de adjacência
    esparsa densa Q-prep: lista de pares.
    """
    if mask is None:
        mask = np.ones((ny, nx), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    # map (i,j) -> node id among active
    ids = -np.ones((ny, nx), dtype=int)
    active = np.argwhere(mask)
    for k, (i, j) in enumerate(active):
        ids[i, j] = k
    edges = []
    for i, j in active:
        for di, dj in ((0, 1), (1, 0)):
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx and mask[ni, nj]:
                edges.append((ids[i, j], ids[ni, nj]))
    if not edges:
        return np.array([], dtype=int), np.array([], dtype=int)
    e = np.asarray(edges, dtype=int)
    return e[:, 0], e[:, 1]


def icar_precision(n: int, edge_i: np.ndarray, edge_j: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Precisão ICAR intrínseco: Q = D - W + ε I (própria, full rank).

    Para exploração com N moderado (~100–300). Para N grande, use esparso.
    """
    Q = np.zeros((n, n), dtype=float)
    for i, j in zip(edge_i, edge_j):
        Q[i, j] -= 1.0
        Q[j, i] -= 1.0
        Q[i, i] += 1.0
        Q[j, j] += 1.0
    Q += epsilon * np.eye(n)
    return Q


def reshape_areal_to_grid(
    values: np.ndarray,
    areal: ArealRates,
    grid: GridSpec,
    fill=np.nan,
) -> np.ndarray:
    """Espalha valores areais de volta para a grade (centróides → células)."""
    out = np.full((grid.ny, grid.nx), fill, dtype=float)
    ix = np.floor((areal.x - grid.xmin) / grid.cell_size).astype(int)
    iy = np.floor((areal.y - grid.ymin) / grid.cell_size).astype(int)
    inside = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    out[iy[inside], ix[inside]] = values[inside]
    return out
