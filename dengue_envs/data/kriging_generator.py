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
    mask: Optional[np.ndarray] = None  # (ny, nx) booleano: área habitada

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
        mask=np.asarray(data["mask"], dtype=bool) if "mask" in files else None,
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
    years_by_disease: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Tuple[KrigingSurfaces, Dict[str, Any]]:
    """Pipeline do notebook: zikario → Ordinary Kriging → probabilidades.

    Por padrão só Dengue e Chikungunya (as classes do env). Zika não entra.

    `years_by_disease` sobrescreve `years` para as doenças listadas (ex.:
    ``{"Dengue": (2015,)}``), para montar cenários em que cada doença vem de
    um ano. Existe porque a chikungunya quase não circulou no Rio em 2015
    (70 notificações, contra 14 mil em 2016): uma superfície de chik de 2015
    não é estimável, e um cenário "2015" só é possível trocando a dengue.
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

    anos = {d: tuple((years_by_disease or {}).get(d, years)) for d in diseases}
    cases = load_zikario_cases(
        str(gpkg_path),
        years=sorted(set(chain.from_iterable(anos.values()))),
        bbox_lonlat=DEFAULT_RIO_BBOX_LONLAT,
        diseases=diseases,
    )
    pontos = {}
    for disease in diseases:
        sub = cases[(cases["Doenca"] == disease) & cases["DT_SIN_PRI"].dt.year.isin(anos[disease])]
        pontos[disease] = (sub["x"].to_numpy(), sub["y"].to_numpy())
    surfaces, payload = surfaces_from_points(
        pontos,
        bbox_lonlat_to_projected(DEFAULT_RIO_BBOX_LONLAT),
        obs_cell_m=obs_cell_m,
        pred_cell_m=pred_cell_m,
        variogram_model=variogram_model,
        crs="EPSG:31983",
        bbox_lonlat=DEFAULT_RIO_BBOX_LONLAT,
    )
    for disease in diseases:
        payload[f"years_{disease.lower()}"] = np.asarray(anos[disease], dtype=int)
    return surfaces, payload


def surfaces_from_points(
    points: Dict[str, Tuple[np.ndarray, np.ndarray]],
    bbox_xy: Tuple[float, float, float, float],
    *,
    obs_cell_m: float = 500.0,
    pred_cell_m: float = 500.0,
    variogram_model: str = "spherical",
    crs: str = "EPSG:31983",
    bbox_lonlat: Optional[Tuple[float, float, float, float]] = None,
    mask: Optional[np.ndarray] = None,
) -> Tuple[KrigingSurfaces, Dict[str, Any]]:
    """Pontos projetados (metros) por doença -> superfícies de Kriging + payload do `.npz`.

    `points` mapeia "Dengue"/"Chikungunya" (e opcionalmente outras classes) para
    `(x, y)`. `mask` (booleano, no formato do grid de predição) zera a
    probabilidade fora da área de estudo — por exemplo, o mar e os municípios
    vizinhos —, que de outra forma o Kriging preenche por extrapolação.
    """
    xmin, xmax, ymin, ymax = bbox_xy
    results: Dict[str, Dict[str, Any]] = {}
    for disease, (x, y) in points.items():
        if len(x) < 3:
            raise ValueError(f"Poucos casos para Kriging de {disease}: n={len(x)}")
        counts, intensity, sigma, pred_grid = intensity_surface_from_points(
            x,
            y,
            bbox_xy=(xmin, xmax, ymin, ymax),
            obs_cell_size=obs_cell_m,
            pred_cell_size=pred_cell_m,
            transform="log1p",
            variogram_model=variogram_model,
        )
        if mask is not None:
            if mask.shape != intensity.shape:
                raise ValueError(f"mask {mask.shape} != grid {intensity.shape}")
            intensity = np.where(mask, intensity, 0.0)
        results[disease] = {
            "counts": counts,
            "intensity": intensity,
            "sigma": sigma,
            "prob": normalize_probability(intensity),
            "pred_grid": pred_grid,
            "n": len(x),
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
        "crs": crs,
        "xmin": pred_grid.xmin,
        "xmax": pred_grid.xmax,
        "ymin": pred_grid.ymin,
        "ymax": pred_grid.ymax,
        "cell_size": pred_grid.cell_size,
        "variogram_model": variogram_model,
        "obs_cell_size": obs_cell_m,
    }
    if bbox_lonlat is not None:
        payload["bbox_lonlat"] = np.asarray(bbox_lonlat, dtype=float)
    if mask is not None:
        payload["mask"] = mask.astype(bool)
    for disease, r in results.items():
        key = disease.lower()
        payload[f"n_{key}"] = r["n"]
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


def with_mask(surfaces: "KrigingSurfaces", path: Union[str, Path]) -> "KrigingSurfaces":
    """Troca a máscara de área habitada pela do arquivo (ver `build_masks`)."""
    import dataclasses

    mask = np.asarray(np.load(path)["mask"], dtype=bool)
    if mask.shape != surfaces.shape:
        raise ValueError(f"máscara {path}: {mask.shape} != superfície {surfaces.shape}")
    return dataclasses.replace(surfaces, mask=mask)


def physical_env_grids(
    surfaces: "KrigingSurfaces",
    size: int,
    cell_m: float,
    rng: Optional[np.random.Generator] = None,
    translate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Põe a cidade no grid do env em ESCALA FÍSICA: cada célula tem `cell_m` metros.

    Antes (`probability_to_env_grid`), cada cidade era esticada até ocupar o
    grid inteiro, com escala diferente em x e em y: uma célula do env valia
    ~181 x 113 m no Rio e ~46 x 65 m no Recife. Assim o raio da confirmação
    epidemiológica (20 células) valia ~3 km no Rio e ~1 km no Recife, e a
    densidade de casos por célula mudava de cidade para cidade. Aqui a escala é
    a mesma para todas (isotrópica), e a cidade ocupa só a sua área real.

    Devolve `(prob_dengue, prob_chik, suporte)` indexados como `[x, y]`. O
    suporte é a área habitada (`surfaces.mask`, ou o retângulo inteiro sem
    máscara): as duas arboviroses são zeradas fora dele, e os casos "outro"
    caem só dentro dele (`KrigingWorld._other_cases`). Com `translate`, a
    cidade vai para uma posição sorteada do grid, para que a posição absoluta
    não identifique a cidade; senão, fica centrada.
    """
    mask = surfaces.mask if surfaces.mask is not None else np.ones(surfaces.shape, dtype=bool)
    fator = float(surfaces.cell_size) / float(cell_m)
    sup = zoom(mask.T.astype(float), fator, order=0) > 0.5
    w, h = sup.shape
    if w > size or h > size:
        raise ValueError(f"cidade de {w}x{h} células de {cell_m} m não cabe no grid {size}x{size}")
    if translate and rng is not None:
        ox, oy = int(rng.integers(0, size - w + 1)), int(rng.integers(0, size - h + 1))
    else:
        ox, oy = (size - w) // 2, (size - h) // 2
    suporte = np.zeros((size, size), dtype=bool)
    suporte[ox: ox + w, oy: oy + h] = sup
    saida = []
    for prob in (surfaces.prob_dengue, surfaces.prob_chik):
        p = zoom(np.where(mask, prob, 0.0).T, fator, order=1)[:w, :h]
        tela = np.zeros((size, size), dtype=float)
        tela[ox: ox + p.shape[0], oy: oy + p.shape[1]] = np.clip(p, 0.0, None)
        tela *= suporte
        total = tela.sum()
        saida.append(tela / total if total > 0 else suporte / suporte.sum())
    return saida[0], saida[1], suporte


def place_randomly(
    prob_dengue: np.ndarray,
    prob_chik: np.ndarray,
    rng: np.random.Generator,
    scale_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Recoloca a área ocupada pela cidade numa escala e posição sorteadas do grid.

    Motivo medido (experimento v6): um agente treinado no Recife, cujo contorno
    ocupa metade do grid e fica sempre no mesmo lugar (a menos de 4 espelhos),
    aprendeu a se orientar pela forma da cidade — zerar a posição do caso o
    derruba de +2135 para −4631, e zerar o mapa MELHORA sua transferência para
    o Rio (+464 para +1514). O agente do Rio, cuja superfície ocupa o grid
    inteiro, ignora o espaço. Sortear escala, rotação e posição tira o atalho.

    A mesma transformação vale para as duas doenças (preserva a geometria
    relativa e, portanto, a dificuldade). `scale_range` é o lado maior da área
    ocupada como fração do grid.
    """
    size = prob_dengue.shape[0]
    ocupado = (prob_dengue + prob_chik) > 0
    xs, ys = np.nonzero(ocupado)
    if len(xs) == 0:
        return prob_dengue, prob_chik
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    k = int(rng.integers(0, 4))
    espelho = bool(rng.integers(0, 2))

    def _gira(a):
        a = np.rot90(a[x0:x1, y0:y1], k)
        return np.flipud(a) if espelho else a

    d, c = _gira(prob_dengue), _gira(prob_chik)
    lado = float(rng.uniform(*scale_range)) * size
    fator = lado / max(d.shape)
    alvo = (max(1, int(round(d.shape[0] * fator))), max(1, int(round(d.shape[1] * fator))))
    alvo = (min(alvo[0], size), min(alvo[1], size))
    d = zoom(d, (alvo[0] / d.shape[0], alvo[1] / d.shape[1]), order=1)[: alvo[0], : alvo[1]]
    c = zoom(c, (alvo[0] / c.shape[0], alvo[1] / c.shape[1]), order=1)[: alvo[0], : alvo[1]]
    ox = int(rng.integers(0, size - d.shape[0] + 1))
    oy = int(rng.integers(0, size - d.shape[1] + 1))
    saida = []
    for a in (d, c):
        tela = np.zeros((size, size), dtype=float)
        tela[ox: ox + a.shape[0], oy: oy + a.shape[1]] = np.clip(a, 0.0, None)
        tela /= tela.sum()
        saida.append(tela)
    return saida[0], saida[1]


def augment_surfaces(
    surfaces: "KrigingSurfaces", rng: np.random.Generator
) -> "KrigingSurfaces":
    """Devolve uma cópia das superfícies sob uma transformação rígida aleatória.

    Rotação de 0/90/180/270 graus, espelhamento opcional e deslocamento
    circular pequeno — aplicados **conjuntamente** às duas doenças.

    Por que conjuntamente: a propriedade que define a dificuldade do ambiente é
    a geometria RELATIVA entre os focos de dengue e chik. Medido: no kriging, a
    posição sozinha acerta a doença em 47,1% (contra 93,7% no sintético). Uma
    transformação rígida aplicada às duas superfícies preserva essa razão
    exatamente — muda ONDE as coisas estão, não QUÃO difícil é distingui-las.
    Transformar cada doença por conta própria mudaria a sobreposição e,
    portanto, a dificuldade; seria outro experimento.

    Serve para que o agente não decore a geografia de um único surto: a
    superfície do kriging é carregada uma vez do `.npz` e, sem isto, é sempre a
    mesma cidade em todos os episódios.
    """
    import dataclasses

    # Quatro transformações, todas preservando a RAZÃO DE ASPECTO:
    # identidade, 180 graus, espelho horizontal, espelho vertical.
    #
    # Duas alternativas foram excluídas por argumento geométrico:
    #
    # - Deslocamento circular (`np.roll`): envolve as bordas e parte focos que
    #   ficam perto delas — deixa de ser rígido.
    # - Rotação de 90/270 graus: a superfície NÃO é quadrada (91x145), então
    #   girar troca o aspecto e o redimensionamento para o grid do env
    #   (`probability_to_env_grid`) estica de forma diferente.
    #
    # As quatro restantes sobrevivem ao redimensionamento intactas.
    #
    # Nota sobre a verificação: a métrica "% em que a posição acerta a doença"
    # é ruidosa entre grupos de seeds (medido: 50,7 +- 2,4 no kriging sem
    # perturbação, sobre 4 grupos de 5 seeds). Com a perturbação dá 52,5 +- 1,1
    # — dentro de um desvio, isto é, **sem efeito detectável sobre a
    # dificuldade**, que é o que se queria. Diferenças menores que ~5 pontos
    # nessa métrica não são interpretáveis com poucos seeds.
    op = int(rng.integers(0, 4))

    def _t(a):
        if a is None:
            return None
        if op == 1:
            b = np.rot90(a, 2)
        elif op == 2:
            b = np.fliplr(a)
        elif op == 3:
            b = np.flipud(a)
        else:
            b = a
        return np.ascontiguousarray(b)

    return dataclasses.replace(
        surfaces,
        prob_dengue=_t(surfaces.prob_dengue),
        prob_chik=_t(surfaces.prob_chik),
        intensity_dengue=_t(surfaces.intensity_dengue),
        intensity_chik=_t(surfaces.intensity_chik),
        class_prob_dengue=_t(surfaces.class_prob_dengue),
        class_prob_chik=_t(surfaces.class_prob_chik),
        mask=_t(surfaces.mask),
    )


def transform_surfaces(
    surfaces: "KrigingSurfaces",
    *,
    temperature: float = 1.0,
    clamp_quantiles: Optional[Tuple[float, float]] = None,
    mix_uniform: float = 0.0,
) -> "KrigingSurfaces":
    """Muda QUANTO a posição informa a doença, preservando onde estão os focos.

    Cada superfície do kriging tem focos nítidos, mas os de dengue e chik
    coincidem (correlação 0,85; em 90% das células a razão entre elas fica
    entre 0,68 e 1,42), e é isso que faz a posição acertar a doença em só ~54%
    (`position_bayes_accuracy`). Os três botões, aplicados nesta ordem e
    igualmente às duas doenças:

    - `temperature` (τ): p ∝ p^τ. τ > 1 concentra a massa nos focos e separa
      as doenças (medido: τ=8 -> 0,74; τ=32 -> 0,94, a separação do
      sintético, mas com metade dos casos em 8 de 13.195 células: extremo
      artificial);
      τ < 1 achata; τ = 0 é a uniforme.
    - `clamp_quantiles` (q_lo, q_hi): corta cada superfície nos próprios
      quantis. Cortar o topo achata os focos; o piso tira a cauda.
    - `mix_uniform` (α): (1-α)·p + α·uniforme. α = 1 apaga a geografia.

    Tudo com os valores padrão devolve as superfícies intactas.
    """
    import dataclasses

    if temperature == 1.0 and clamp_quantiles is None and mix_uniform == 0.0:
        return surfaces
    if temperature < 0:
        raise ValueError(f"temperature deve ser >= 0; recebido {temperature}")
    if not 0.0 <= mix_uniform <= 1.0:
        raise ValueError(f"mix_uniform deve estar em [0, 1]; recebido {mix_uniform}")

    def _t(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 0.0, None)
        if temperature != 1.0:
            # Em log, para τ grande não estourar para zero.
            with np.errstate(divide="ignore"):
                logp = np.where(p > 0, np.log(p), -np.inf) * temperature
            p = np.exp(logp - logp[np.isfinite(logp)].max())
        if clamp_quantiles is not None:
            lo, hi = np.quantile(p, clamp_quantiles)
            p = np.clip(p, lo, hi)
        p = normalize_probability(p)
        if mix_uniform:
            p = (1.0 - mix_uniform) * p + mix_uniform / p.size
        return p

    return dataclasses.replace(
        surfaces,
        prob_dengue=_t(surfaces.prob_dengue),
        prob_chik=_t(surfaces.prob_chik),
        meta={**(surfaces.meta or {}), "temperature": temperature,
              "clamp_quantiles": clamp_quantiles, "mix_uniform": mix_uniform},
    )


def position_bayes_accuracy(surfaces: "KrigingSurfaces", size: int = 400) -> float:
    """Acerto do melhor classificador dengue × chik que só vê a posição.

    Com prior igual entre as doenças: 0,5·Σ max(p_dengue, p_chik) no grid do
    env. É a dificuldade espacial do cenário sem ruído de amostragem — 0,5 é
    a posição não dizer nada; 1,0 é a posição resolver o caso.
    """
    d = probability_to_env_grid(surfaces.prob_dengue, size)
    c = probability_to_env_grid(surfaces.prob_chik, size)
    return float(0.5 * np.maximum(d, c).sum())

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
        epi_model: str = "legacy",
        initial_infected_fraction: float = 0.01,
        other_prevalence: float = 0.0,
        random_placement: Optional[Tuple[float, float]] = None,
        physical_cell_m: Optional[float] = None,
        physical_translate: bool = True,
        other_on_support: bool = False,
    ):
        self.size = int(size)
        self.epi_model = epi_model
        if not 0.0 <= other_prevalence < 1.0:
            raise ValueError(f"other_prevalence deve estar em [0,1); recebido {other_prevalence}")
        self.other_prevalence = float(other_prevalence)
        self.initial_infected_fraction = float(initial_infected_fraction)
        self.popsize = int(popsize)
        self.epilength = int(epilength)
        self.dengue_r0 = float(dengue_r0)
        self.chik_r0 = float(chik_r0)
        self._rng = random_state or np.random.default_rng()
        self.surfaces = surfaces

        # Onde caem os casos "outro": o grid inteiro (até o v9) ou só a área
        # habitada. Na escala física é sempre a área habitada.
        self.support = None
        if physical_cell_m is not None:
            self.prob_dengue, self.prob_chik, self.support = physical_env_grids(
                surfaces, self.size, physical_cell_m, self._rng, physical_translate)
        else:
            self.prob_dengue = probability_to_env_grid(surfaces.prob_dengue, self.size)
            self.prob_chik = probability_to_env_grid(surfaces.prob_chik, self.size)
            if other_on_support:
                self.support = (self.prob_dengue + self.prob_chik) > 0
        self._support_idx = np.flatnonzero(self.support) if self.support is not None else None
        if random_placement is not None:
            self.prob_dengue, self.prob_chik = place_randomly(
                self.prob_dengue, self.prob_chik, self._rng, tuple(random_placement))
        self.dengue_center, self.dengue_radius = _focus_from_prob(self.prob_dengue)
        self.chik_center, self.chik_radius = _focus_from_prob(self.prob_chik)

        self.dengue_curve = self._curve(self.dengue_r0, disease=0)
        self.chik_curve = self._curve(self.chik_r0, disease=1)

        self.case_series = []
        self.case_dict = {}
        self.dengue_total = 0
        self.chik_total = 0
        self.casedf = None
        self.build_case_series()
        self.build_case_dataframe()

    def _curve(self, r0: float, disease: int) -> np.ndarray:
        """Mesma escolha de modelo temporal do `World` sintético."""
        from dengue_envs.core.epi_model import CHIK, DENGUE, seir_cumulative_cases

        if self.epi_model == "legacy":
            return _sir_cumulative(self.popsize, self.epilength, r0)
        if self.epi_model != "seir":
            raise ValueError(f"epi_model deve ser 'legacy' ou 'seir'; recebido {self.epi_model!r}")
        params = DENGUE if disease == 0 else CHIK
        return seir_cumulative_cases(
            r0, params, self.popsize, self.epilength, self.initial_infected_fraction
        )

    def _sample(self, disease: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        prob = self.prob_dengue if disease == 0 else self.prob_chik
        return sample_xy_from_prob(prob, n, self._rng)

    def build_case_series(self) -> None:
        self.case_series = []
        self.dengue_total = 0
        self.chik_total = 0
        self.other_total = 0
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
            self.case_series.append(dengue_cases + chik_cases + self._other_cases(t, new_d + new_c))

    def _other_cases(self, t: int, n_arbo: int) -> list:
        """Casos notificados que não são arbovirose — mesmo modelo do `World` sintético.

        Sem eles, um laudo negativo de dengue implica chikungunya: um exame
        sempre basta e descartar nunca é correto. Espaço uniforme (doença
        febril de fundo não segue os focos do vetor) e quantidade proporcional
        aos arbovirais do dia, para a prevalência ficar estável no episódio.
        """
        if self.other_prevalence <= 0 or n_arbo <= 0:
            return []
        n_other = int(np.round(n_arbo * self.other_prevalence / (1.0 - self.other_prevalence)))
        if n_other <= 0:
            return []
        if self._support_idx is not None:
            # Só onde mora gente: fora da cidade não há notificação de nada.
            xs, ys = np.unravel_index(self._rng.choice(self._support_idx, n_other), (self.size, self.size))
        else:
            xs = self._rng.integers(0, self.size, n_other)
            ys = self._rng.integers(0, self.size, n_other)
        self.other_total += n_other
        return [
            {"t": t, "x": int(x), "y": int(y), "disease": 2, "testd": 0, "testc": 0, "epiconf": 0}
            for x, y in zip(xs, ys)
        ]

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
        epi_model: str = "legacy",
        initial_infected_fraction: float = 0.01,
        other_prevalence: float = 0.0,
        random_placement: Optional[Tuple[float, float]] = None,
        physical_cell_m: Optional[float] = None,
        physical_translate: bool = True,
        other_on_support: bool = False,
    ) -> KrigingWorld:
        return KrigingWorld(
            self.size,
            self.episize,
            self.epilength,
            self.surfaces,
            dengue_r0=self.dengue_r0 if dengue_r0 is None else dengue_r0,
            chik_r0=self.chik_r0 if chik_r0 is None else chik_r0,
            random_state=random_state,
            epi_model=epi_model,
            initial_infected_fraction=initial_infected_fraction,
            other_prevalence=other_prevalence,
            random_placement=random_placement,
            physical_cell_m=physical_cell_m,
            physical_translate=physical_translate,
            other_on_support=other_on_support,
        )

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return self.build_world(random_state=rng).casedf.copy()
