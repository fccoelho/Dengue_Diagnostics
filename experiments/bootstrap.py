"""Intervalos de confiança por bootstrap hierárquico e pareado.

Até aqui os resultados eram reportados como média ± desvio entre 3 ou 4 seeds
de treino. Isso esconde duas coisas que um artigo precisa mostrar:

1. **Há duas fontes de variância**, não uma: qual seed treinou o agente, e em
   qual surto ele foi avaliado. O bootstrap reamostra os dois níveis — primeiro
   as seeds de treino, depois os episódios de avaliação (Saravanan et al.,
   2020, "Application of the hierarchical bootstrap to multi-level data").

2. **Os episódios são pareados.** Todo braço é avaliado nas mesmas seeds de
   ambiente, que fixam o mesmo surto e a mesma especificidade do médico.
   Comparar dois braços episódio a episódio remove a variância que vem do
   surto (um médico ruim derruba todo mundo no mesmo episódio). Por isso a
   reamostragem dos episódios é COMPARTILHADA entre braços em cada réplica,
   enquanto as seeds de treino — treinos independentes — são reamostradas
   separadamente por braço.

Estatísticas, na linha de Agarwal et al. (2021), "Deep RL at the Edge of the
Statistical Precipice":

- média e IQM (média interquartil, robusta a episódios catastróficos);
- diferença de médias entre braços;
- P(X > Y): chance de um episódio de X superar o mesmo episódio de Y,
  sobre todos os pares de seeds de treino (empate conta 1/2).

Limitação: com 3 ou 4 seeds de treino o nível superior do bootstrap tem pouca
resolução (3 seeds admitem só 10 reamostras distintas), e o intervalo
percentil tende a ficar estreito demais nesse nível. Os intervalos daqui são
honestos sobre o surto e otimistas sobre a seed; mais seeds de treino são a
única correção real.

Uso: `python -m experiments.bootstrap` (grava em `results/bootstrap/`).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

_RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = _RAIZ / "results"
SAIDA = RESULTADOS / "bootstrap"

N_REPLICAS = 10_000
NIVEL = 0.95

# Métricas do benchmark oficial -> nome curto usado aqui.
METRICAS_BENCHMARK = {
    "Recompensa Total": "recompensa",
    "Acurácia Multiclasse": "acuracia",
    "Testes Realizados": "exames",
}

# Braços do experimento v4: rótulo -> padrão dos diretórios do benchmark.
BRACOS_V4 = {
    "A (GAE)": "bm_ppo_v4a_seir_s*",
    "B (crédito + tempo)": "bm_ppo_v4b_credito_s*",
    "C (crédito)": "bm_ppo_v4c_credito_sem_tempo_s*",
}
BASELINES = ("testonce", "testtwice", "confirmall", "clinical")

# Comparações que o artigo sustenta: (X, Y) lê-se "X contra Y".
COMPARACOES = (
    ("C (crédito)", "A (GAE)"),              # o efeito do crédito
    ("B (crédito + tempo)", "C (crédito)"),  # o efeito das features temporais
    ("C (crédito)", "testonce"),             # distância até a melhor política fixa
    ("B (crédito + tempo)", "testonce"),
    ("A (GAE)", "clinical"),
    ("C (crédito)", "testtwice"),            # no v9 o testtwice é a melhor política fixa
)


# --------------------------------------------------------------------------
# dados
# --------------------------------------------------------------------------

def _le_raw(diretorio: Path) -> pd.DataFrame:
    return pd.read_csv(diretorio / "benchmark_raw.csv", encoding="utf-8-sig")


def _seed_do_diretorio(diretorio: Path) -> int:
    return int(diretorio.name.rsplit("_s", 1)[1])


# Braços do v5: os mesmos A e C, retreinados no kriging_v9 (com casos "outro").
BRACOS_V5 = {
    "A (GAE)": "bm_ppo_v5a_gae_s*",
    "C (crédito)": "bm_ppo_v5c_credito_s*",
}


def carrega_benchmark_v4(raiz: Path = RESULTADOS, bracos: Dict[str, str] = BRACOS_V4,
                         baselines: str = "baseline_v8seir_kriging") -> pd.DataFrame:
    """Tabela longa do benchmark oficial: uma linha por (braço, seed_treino, episódio).

    Os baselines são determinísticos e aparecem repetidos em todo arquivo do
    benchmark; entram uma vez só, lidos de `baseline_v8seir_kriging`, com
    `seed_treino = 0`.
    """
    partes = []
    for braco, padrao in bracos.items():
        dirs = sorted(raiz.glob(padrao))
        if not dirs:
            raise FileNotFoundError(f"nenhum resultado para {braco} em {raiz / padrao}")
        for d in dirs:
            raw = _le_raw(d)
            raw = raw[raw["agent"] == "ppo"].assign(braco=braco, seed_treino=_seed_do_diretorio(d))
            partes.append(raw)
    base = _le_raw(raiz / baselines)
    base = base[base["agent"].isin(BASELINES)]
    partes.append(base.assign(braco=base["agent"], seed_treino=0))

    df = pd.concat(partes, ignore_index=True).rename(columns={"seed": "seed_aval", **METRICAS_BENCHMARK})
    return df[["braco", "seed_treino", "seed_aval", *METRICAS_BENCHMARK.values()]]


def carrega_demo(nome: str, raiz: Path = RESULTADOS) -> pd.DataFrame:
    """Mesma tabela, para um experimento de `robustez` (cache `results/demo/<nome>.csv`)."""
    df = pd.read_csv(raiz / "demo" / f"{nome}.csv")
    rotulos = {"ppo A": "A (GAE)", "ppo C": "C (crédito)"}
    df["braco"] = df["agente"].map(rotulos).fillna(df["agente"])
    # Os baselines rodam uma vez só, sob o rótulo de seed de treino 45.
    df.loc[~df["agente"].str.startswith("ppo"), "seed_treino"] = 0
    return df.rename(columns={"seed": "seed_aval"})[
        ["braco", "seed_treino", "seed_aval", "recompensa", "acuracia", "exames"]
    ]


def matrizes(df: pd.DataFrame, metrica: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Uma matriz (seed_treino × episódio) por braço, com os episódios alinhados.

    O alinhamento é o que torna o pareamento válido: a coluna j de todas as
    matrizes é o MESMO surto. Falha se algum braço não tiver sido avaliado em
    todos os episódios.
    """
    episodios = np.sort(df["seed_aval"].unique())
    saida = {}
    for braco, g in df.groupby("braco", sort=False):
        m = g.pivot_table(index="seed_treino", columns="seed_aval", values=metrica, aggfunc="first")
        m = m.reindex(columns=episodios)
        if m.isna().any().any():
            faltam = sorted(m.columns[m.isna().any()].tolist())
            raise ValueError(f"{braco}: sem resultado para os episódios {faltam}")
        saida[braco] = m.to_numpy(dtype=float)
    return saida, episodios


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------

def _iqm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return trim_mean(x, 0.25, axis=axis)


def _prob_melhora(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """P(X > Y) episódio a episódio, sobre todos os pares de seeds de treino.

    `x`: (..., nx, ne), `y`: (..., ny, ne). Empate conta meio.
    """
    xi = x[..., :, None, :]
    yj = y[..., None, :, :]
    return ((xi > yj) + 0.5 * (xi == yj)).mean(axis=(-3, -2, -1))


def _reamostra(m: np.ndarray, t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """(B, nt, ne) a partir de índices de treino (B, nt) e de episódio (B, ne)."""
    return m[t[:, :, None], e[:, None, :]]


def _intervalo(amostras: np.ndarray, nivel: float = NIVEL) -> Tuple[float, float]:
    a = (1 - nivel) / 2
    lo, hi = np.quantile(amostras, [a, 1 - a])
    return float(lo), float(hi)


def bootstrap(mats: Dict[str, np.ndarray], comparacoes: Iterable[Tuple[str, str]] = (),
              n_replicas: int = N_REPLICAS, seed: int = 0,
              nivel: float = NIVEL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap hierárquico e pareado sobre um conjunto de braços.

    Devolve (por_braco, comparacoes). Cada réplica sorteia UM conjunto de
    episódios, usado por todos os braços, e um conjunto de seeds de treino
    independente para cada braço.
    """
    rng = np.random.default_rng(seed)
    ne = next(iter(mats.values())).shape[1]
    e = rng.integers(0, ne, size=(n_replicas, ne))
    reamostrado = {}
    for braco, m in mats.items():
        t = rng.integers(0, m.shape[0], size=(n_replicas, m.shape[0]))
        reamostrado[braco] = _reamostra(m, t, e)

    linhas = []
    for braco, m in mats.items():
        r = reamostrado[braco]
        media_lo, media_hi = _intervalo(r.mean(axis=(1, 2)), nivel)
        iqm_lo, iqm_hi = _intervalo(_iqm(r.reshape(n_replicas, -1)), nivel)
        linhas.append({
            "braco": braco, "n_treino": m.shape[0], "n_episodios": ne,
            "media": m.mean(), "media_lo": media_lo, "media_hi": media_hi,
            "iqm": float(_iqm(m.ravel())), "iqm_lo": iqm_lo, "iqm_hi": iqm_hi,
            # O desvio entre seeds que era reportado até aqui, para comparação.
            "desvio_seeds": float(m.mean(axis=1).std(ddof=1)) if m.shape[0] > 1 else 0.0,
        })

    comps = []
    for x, y in comparacoes:
        if x not in mats or y not in mats:
            continue
        rx, ry = reamostrado[x], reamostrado[y]
        dif_lo, dif_hi = _intervalo(rx.mean(axis=(1, 2)) - ry.mean(axis=(1, 2)), nivel)
        p_lo, p_hi = _intervalo(_prob_melhora(rx, ry), nivel)
        comps.append({
            "x": x, "y": y,
            "diferenca": mats[x].mean() - mats[y].mean(), "dif_lo": dif_lo, "dif_hi": dif_hi,
            "p_x_melhor": float(_prob_melhora(mats[x], mats[y])), "p_lo": p_lo, "p_hi": p_hi,
        })
    return pd.DataFrame(linhas), pd.DataFrame(comps)


# --------------------------------------------------------------------------
# relatório
# --------------------------------------------------------------------------

def analisa(df: pd.DataFrame, metricas: Sequence[str] = ("recompensa", "acuracia", "exames"),
            comparacoes: Sequence[Tuple[str, str]] = COMPARACOES,
            **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Roda o bootstrap para cada métrica e empilha os resultados."""
    bracos, comps = [], []
    for metrica in metricas:
        mats, _ = matrizes(df, metrica)
        b, c = bootstrap(mats, comparacoes, **kw)
        bracos.append(b.assign(metrica=metrica))
        comps.append(c.assign(metrica=metrica))
    return pd.concat(bracos, ignore_index=True), pd.concat(comps, ignore_index=True)


def _fmt(v: float, casas: int, sinal: bool = False) -> str:
    return f"{v:+.{casas}f}" if sinal else f"{v:.{casas}f}"


def tabela_markdown(bracos: pd.DataFrame, comps: pd.DataFrame, metrica: str = "recompensa") -> str:
    casas = {"recompensa": 0, "acuracia": 3, "exames": 0}[metrica]
    sinal = metrica == "recompensa"
    b = bracos[bracos["metrica"] == metrica].sort_values("media", ascending=False)
    linhas = [f"### {metrica}", "",
              "| braço | seeds | média [IC 95%] | IQM [IC 95%] | desvio entre seeds |",
              "|---|---:|---|---|---:|"]
    for _, r in b.iterrows():
        linhas.append(
            f"| {r.braco} | {r.n_treino} | {_fmt(r.media, casas, sinal)} "
            f"[{_fmt(r.media_lo, casas, sinal)}, {_fmt(r.media_hi, casas, sinal)}] | "
            f"{_fmt(r.iqm, casas, sinal)} [{_fmt(r.iqm_lo, casas, sinal)}, {_fmt(r.iqm_hi, casas, sinal)}] | "
            f"{r.desvio_seeds:.{casas}f} |")
    c = comps[comps["metrica"] == metrica]
    if len(c):
        linhas += ["", "| X contra Y | diferença [IC 95%] | P(X > Y) [IC 95%] |", "|---|---|---|"]
        for _, r in c.iterrows():
            linhas.append(
                f"| {r.x} × {r.y} | {_fmt(r.diferenca, casas, True)} "
                f"[{_fmt(r.dif_lo, casas, True)}, {_fmt(r.dif_hi, casas, True)}] | "
                f"{r.p_x_melhor:.2f} [{r.p_lo:.2f}, {r.p_hi:.2f}] |")
    return "\n".join(linhas)


CONJUNTOS = {
    "benchmark": carrega_benchmark_v4,
    "benchmark_v9": lambda: carrega_benchmark_v4(bracos=BRACOS_V5, baselines="baseline_v9_kriging"),
    "seeds_ineditas": lambda: carrega_demo("seeds_ineditas"),
    # Superfícies de um único ano, mesmas seeds do benchmark oficial.
    "dengue_2015": lambda: carrega_demo("dengue_2015"),
    "rio_2016": lambda: carrega_demo("rio_2016"),
}
# Varredura de dificuldade espacial (robustez.temperatura_espacial).
TEMPERATURAS = (0, 0.5, 1, 2, 4, 8, 16, 32)
CONJUNTOS.update({f"temperatura_{t:g}": (lambda t=t: carrega_demo(f"temperatura_{t:g}"))
                  for t in TEMPERATURAS})


def curva_temperatura(n_replicas: int = N_REPLICAS) -> pd.DataFrame:
    """Uma linha por temperatura: recompensa de cada agente e os dois contrastes.

    Lê os caches que existirem; temperaturas ainda não avaliadas ficam de fora.
    """
    linhas = []
    for t in TEMPERATURAS:
        caminho = RESULTADOS / "demo" / f"temperatura_{t:g}.csv"
        if not caminho.exists():
            continue
        acerto = float(pd.read_csv(caminho)["acerto_posicao"].iloc[0])
        mats, _ = matrizes(carrega_demo(f"temperatura_{t:g}"), "recompensa")
        bracos, comps = bootstrap(mats, [("C (crédito)", "A (GAE)"), ("C (crédito)", "testonce"),
                                         ("C (crédito)", "testtwice")], n_replicas=n_replicas)
        linha = {"temperatura": t, "acerto_posicao": acerto}
        for _, r in bracos.iterrows():
            linha[r.braco] = r.media
            linha[f"{r.braco} lo"] = r.media_lo
            linha[f"{r.braco} hi"] = r.media_hi
        for _, r in comps.iterrows():
            chave = f"{r.x.split()[0]}-{r.y.split()[0]}"
            linha[chave] = r.diferenca
            linha[f"{chave} lo"] = r.dif_lo
            linha[f"{chave} hi"] = r.dif_hi
            linha[f"P {chave}"] = r.p_x_melhor
        linhas.append(linha)
    return pd.DataFrame(linhas)


def main(nomes: Optional[List[str]] = None) -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    for nome in (nomes or list(CONJUNTOS)):
        try:
            df = CONJUNTOS[nome]()
        except FileNotFoundError as e:
            print(f"[{nome}] ignorado: {e}")
            continue
        bracos, comps = analisa(df)
        bracos.to_csv(SAIDA / f"{nome}_bracos.csv", index=False)
        comps.to_csv(SAIDA / f"{nome}_comparacoes.csv", index=False)
        print(f"\n## {nome} ({df['seed_aval'].nunique()} episódios)\n")
        for metrica in ("recompensa", "acuracia", "exames"):
            print(tabela_markdown(bracos, comps, metrica), "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
