"""Figuras do artigo, em inglês, geradas a partir dos resultados em `results/`.

Uso: `python -m experiments.figuras_artigo [nome ...]` — grava em `Artigo/img/`.

`results/` fica fora do git; as figuras vão para `Artigo/img/` (versionado) para
que o artigo compile a partir do repositório. Nenhuma figura aqui mostra casos
individuais: só superfícies agregadas, contagens e métricas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

SAIDA = _RAIZ / "Artigo" / "img"
RES = _RAIZ / "results"

# Paleta categórica (validada): azul, laranja, verde-água; cinzas para referências.
AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
CINZA, CINZA_ESCURO, TEXTO = "#8a8984", "#52514e", "#0b0b0b"
DENGUE, CHIK = AZUL, LARANJA
RAZAO = LinearSegmentedColormap.from_list("chik_dengue", [CHIK, "#f2f1ee", DENGUE])

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left",
    "axes.edgecolor": "#bbbbbb", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#eeeeee", "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _salva(fig, nome: str) -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA / nome)
    plt.close(fig)
    print("->", SAIDA / nome)


# --------------------------------------------------------------------------
# ambiente
# --------------------------------------------------------------------------

def seir() -> None:
    from dengue_envs.core.epi_model import CHIK as P_CHIK, DENGUE as P_DENGUE, final_size, seir_cumulative_cases

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for r0, params, cor, nome in ((1.25, P_DENGUE, DENGUE, "dengue"), (1.70, P_DENGUE, DENGUE, None),
                                  (1.46, P_CHIK, CHIK, "chikungunya"), (1.67, P_CHIK, CHIK, None)):
        c = seir_cumulative_cases(r0, params, 300, 300, initial_infected_fraction=0.01)
        a1.plot(np.diff(c, prepend=0), color=cor, lw=2 if nome else 1.2, ls="-" if nome else "--",
                label=f"{nome}, $R_0$ = {r0}" if nome else f"$R_0$ = {r0}")
    a1.set_xlabel("epidemic day"); a1.set_ylabel("new notified cases per day")
    a1.set_title("Daily incidence at the ends of the $R_0$ range"); a1.legend(fontsize=8)

    r0s = np.linspace(1.05, 2.5, 25)
    a2.plot(r0s, [final_size(r) for r in r0s], color=TEXTO, lw=2, label=r"theory: $z = 1 - e^{-R_0 z}$")
    a2.scatter(r0s, [seir_cumulative_cases(r, P_DENGUE, 10_000, 2000, initial_infected_fraction=1e-4)[-1] / 10_000
                     for r in r0s], s=18, color=DENGUE, zorder=3, label="environment generator")
    a2.set_xlabel("$R_0$"); a2.set_ylabel("final fraction infected")
    a2.set_title("Final epidemic size vs. theory"); a2.legend(fontsize=8)
    _salva(fig, "fig_seir.png")


def _painel_superficie(fig, ax, img, cmap, titulo, norm=None, rotulo="", mascara=None, extent=None):
    if mascara is not None:
        img = np.ma.masked_where(~mascara, img)
    im = ax.imshow(img, origin="lower", cmap=cmap, norm=norm, extent=extent, aspect="equal")
    ax.set_title(titulo); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(rotulo); cb.outline.set_visible(False)


def espacial_rio() -> None:
    from dengue_envs.data.kriging_generator import load_kriging_surfaces, position_bayes_accuracy, transform_surfaces

    s = load_kriging_surfaces(RES / "kriging" / "rio_2015_2016_kriging_surfaces.npz")
    fig, axs = plt.subplots(2, 3, figsize=(15, 6.4), constrained_layout=True)
    for i, (tau, nome) in enumerate(((1, "Rio de Janeiro kriging ($\\tau$ = 1)"),
                                     (32, "Sharpened ($\\tau$ = 32)"))):
        x = transform_surfaces(s, temperature=tau) if tau != 1 else s
        d, c = x.prob_dengue, x.prob_chik
        _painel_superficie(fig, axs[i, 0], d / d.max(), "Blues", "Dengue", rotulo="relative to maximum")
        _painel_superficie(fig, axs[i, 1], c / c.max(), "Oranges", "Chikungunya", rotulo="relative to maximum")
        r = np.log2(d / c)
        lim = max(1.0, float(np.abs(r).max()))
        _painel_superficie(fig, axs[i, 2], r, RAZAO, "Which disease dominates: log$_2$(dengue / chik)",
                           TwoSlopeNorm(0, -lim, lim), "← chik   0 = tie   dengue →")
        axs[i, 0].set_ylabel(f"{nome}\nlocation predicts disease: {position_bayes_accuracy(x):.0%}", fontsize=10)
    _salva(fig, "fig_spatial_rio.png")


def recife_anos() -> None:
    resumo = pd.read_csv(RES / "recife" / "resumo_anos.csv")
    arquivos = sorted((RES / "kriging").glob("recife_20*_kriging_surfaces.npz"))
    n = len(arquivos)
    cols = 4; linhas = int(np.ceil(n / cols))
    fig, axs = plt.subplots(linhas, cols, figsize=(3.6 * cols, 4.0 * linhas), constrained_layout=True, squeeze=False)
    norm = TwoSlopeNorm(0, -2, 2)
    for ax, f in zip(axs.ravel(), arquivos):
        z = np.load(f, allow_pickle=True)
        mascara = z["mask"].astype(bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.ma.masked_where(~mascara, np.log2(z["prob_dengue"] / z["prob_chikungunya"]))
        ano = int(z["ano"])
        acerto = float(resumo.loc[resumo.ano == ano, "acerto_posicao"].iloc[0])
        im = ax.imshow(r, origin="lower", cmap=RAZAO, norm=norm, aspect="equal")
        ax.set_title(f"{ano}: location predicts {acerto:.0%}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    cb = fig.colorbar(im, ax=axs, shrink=0.6)
    cb.set_label("log$_2$(dengue / chik), clipped at ±2  (← chik | dengue →)"); cb.outline.set_visible(False)
    _salva(fig, "fig_recife_years.png")


def recife_resumo() -> None:
    resumo = pd.read_csv(RES / "recife" / "resumo_anos.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.0), constrained_layout=True)
    anos = resumo["ano"].to_numpy()
    larg = 0.27
    for i, (col, cor, nome) in enumerate((("n_dengue", DENGUE, "dengue (confirmed)"),
                                          ("n_chik", CHIK, "chikungunya (confirmed)"),
                                          ("n_descartado", CINZA, "discarded"))):
        a1.bar(np.arange(len(anos)) + (i - 1) * larg, resumo[col], width=larg - 0.03, color=cor, label=nome)
    a1.set_xticks(range(len(anos)), anos, rotation=45)
    a1.set_ylabel("notifications (Recife residents)")
    a1.set_title("Recife notifications by final classification"); a1.legend(fontsize=8)

    a2.plot(anos, resumo["acerto_posicao"], marker="o", color=AZUL, lw=2, label="Recife, by year")
    a2.axhline(0.544, color=LARANJA, ls=":", lw=1.8, label="Rio de Janeiro 2015–16 (training surface)")
    a2.axhline(0.5, color=CINZA, ls="--", lw=1, label="location carries no information")
    a2.set_ylim(0.48, 0.66); a2.set_xticks(anos, anos, rotation=45)
    a2.set_ylabel("Bayes accuracy from location alone")
    a2.set_title("How much location reveals the disease"); a2.legend(fontsize=8, loc="upper right")
    _salva(fig, "fig_recife_summary.png")


# --------------------------------------------------------------------------
# resultados
# --------------------------------------------------------------------------

def bracos_v8() -> None:
    from experiments import bootstrap as B

    df = B.carrega_benchmark_v4()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ordem = [("A (GAE)", "A: standard GAE", LARANJA), ("B (crédito + tempo)", "B: per-case + time", VERDE),
             ("C (crédito)", "C: per-case credit", AZUL)]
    for i, (braco, nome, cor) in enumerate(ordem):
        g = df[df.braco == braco].groupby("seed_treino")[["recompensa", "exames"]].mean()
        a1.bar(i, g.recompensa.mean(), color=cor, width=0.6)
        a1.scatter(np.full(len(g), i), g.recompensa, color="white", edgecolor=TEXTO, zorder=3, s=28)
        a2.bar(i, g.exames.mean(), color=cor, width=0.6)
        a2.text(i, g.exames.mean() + 6, f"{g.exames.mean():.0f}", ha="center", fontsize=9, color=TEXTO)
    base = df[df.braco.isin(["testonce", "clinical"])].groupby("braco")[["recompensa", "exames"]].mean()
    for ax, col in ((a1, "recompensa"), (a2, "exames")):
        ax.axhline(base.loc["testonce", col], color=CINZA_ESCURO, ls="--", lw=1)
        ax.text(2.45, base.loc["testonce", col], " test-once", va="bottom", fontsize=8, color=CINZA_ESCURO)
        ax.set_xticks(range(3), [n for _, n, _ in ordem]); ax.set_xlim(-0.5, 3.0)
    a1.axhline(base.loc["clinical", "recompensa"], color=CINZA, ls=":", lw=1)
    a1.text(2.45, base.loc["clinical", "recompensa"], " clinical", va="bottom", fontsize=8, color=CINZA)
    a1.set_ylabel("episode reward"); a1.set_title("Reward (dots: training seeds)")
    a2.set_ylabel("laboratory tests per episode"); a2.set_title("How much each arm investigates")
    _salva(fig, "fig_arms_twodisease.png")


def politicas_v9() -> None:
    from experiments import bootstrap as B

    df = B.carrega_benchmark_v4(bracos=B.BRACOS_V5, baselines="baseline_v9_kriging")
    seq = pd.read_csv(RES / "baseline_v9_sequencial" / "benchmark_raw.csv", encoding="utf-8-sig")
    seq = seq[seq.agent.str.startswith("sequencial")].rename(columns={"seed": "seed_aval", **B.METRICAS_BENCHMARK})
    df = pd.concat([df, seq.assign(braco=seq.agent, seed_treino=0)[df.columns]], ignore_index=True)
    rec, _ = B.bootstrap(B.matrizes(df, "recompensa")[0], n_replicas=4000)
    exa, _ = B.bootstrap(B.matrizes(df, "exames")[0], n_replicas=4000)
    t = rec.set_index("braco").join(exa.set_index("braco"), rsuffix="_ex")

    nomes = {"C (crédito)": ("C: per-case credit (learned)", AZUL), "A (GAE)": ("A: standard GAE (learned)", LARANJA),
             "sequencial_clinico": ("sequential, clinician-guided", CINZA_ESCURO),
             "sequencial": ("sequential, dengue first", CINZA_ESCURO), "testtwice": ("test-twice", CINZA_ESCURO),
             "testonce": ("test-once", CINZA_ESCURO), "clinical": ("clinical only", CINZA_ESCURO),
             "confirmall": ("confirm-all", CINZA_ESCURO)}
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    for braco, (nome, cor) in nomes.items():
        if braco not in t.index:
            continue
        r = t.loc[braco]
        aprendido = braco in ("C (crédito)", "A (GAE)")
        ax.errorbar(r.media_ex, r.media, yerr=[[r.media - r.media_lo], [r.media_hi - r.media]],
                    xerr=[[r.media_ex - r.media_lo_ex], [r.media_hi_ex - r.media_ex]],
                    fmt="o" if aprendido else "s", ms=9 if aprendido else 6, color=cor, ecolor=cor,
                    elinewidth=1, capsize=0, zorder=3)
        # Deslocamentos à mão: C e o sequencial guiado pelo médico ficam lado a lado.
        dx, dy, ha = {"C (crédito)": (-12, -16, "right"), "sequencial_clinico": (10, 10, "left"),
                      "testtwice": (-12, 8, "right"), "A (GAE)": (14, -14, "left"),
                      "confirmall": (14, 6, "left")}.get(braco, (14, 6, "left"))
        ax.annotate(nome, (r.media_ex, r.media), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=9, color=TEXTO, fontweight="bold" if aprendido else "normal")
    ax.set_xlabel("laboratory tests per episode"); ax.set_ylabel("episode reward")
    ax.set_title("Reward vs. tests with non-arboviral suspects (10 paired outbreaks, 95% CI)")
    _salva(fig, "fig_policies_full.png")


def temperatura() -> None:
    c = pd.read_csv(RES / "bootstrap" / "curva_temperatura.csv").sort_values("acerto_posicao")
    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    ax.axvspan(0.80, 0.96, color="#f2f1ee", zorder=0)
    ax.text(0.805, -950, "sharpened beyond\nany observed city", fontsize=8, color=CINZA_ESCURO, va="bottom")
    for braco, nome, cor in (("C (crédito)", "C: per-case credit", AZUL), ("A (GAE)", "A: standard GAE", LARANJA)):
        ax.fill_between(c.acerto_posicao, c[f"{braco} lo"], c[f"{braco} hi"], color=cor, alpha=0.15, lw=0)
        ax.plot(c.acerto_posicao, c[braco], marker="o", color=cor, lw=2, label=nome)
    ax.plot(c.acerto_posicao, c["testonce"], color=CINZA_ESCURO, ls="--", lw=1.2, label="test-once (fixed)")
    ax.axvspan(0.52, 0.605, color=VERDE, alpha=0.12, zorder=0)
    ax.text(0.5225, -950, "observed range:\nRio and Recife,\nall years", fontsize=8,
            color=CINZA_ESCURO, va="bottom")
    ax.axvline(0.544, color=CINZA, lw=1, ls=":")
    ax.text(0.546, 3050, "training", fontsize=8, color=CINZA_ESCURO)
    ax.set_xlabel("Bayes accuracy of location alone (spatial informativeness)")
    ax.set_ylabel("episode reward")
    ax.set_title("Zero-shot robustness to spatial informativeness (two-disease variant)")
    ax.legend(fontsize=8, loc="center right")
    _salva(fig, "fig_temperature.png")


def transferencia() -> None:
    """Matriz treino × teste: diferença pareada do agente C para a melhor regra fixa, com IC."""
    base = RES / "v6_avaliacao"
    testes = [("teste_rio", "Rio de Janeiro"), ("teste_recife_2016", "Recife 2016*"),
              ("teste_recife_2021", "Recife 2021*"), ("teste_sintetico", "Idealized")]
    treinos = [("rio", "Rio de Janeiro"), ("recife", "Recife"), ("sintetico", "Idealized"),
               ("misto", "Mixture"), ("mistolongo", "Mixture, 2× training")]
    dif, lo, hi = {}, {}, {}
    for t, _ in testes:
        c = pd.read_csv(base / f"comparacoes_{t}.csv")
        c = c[(c.metrica == "recompensa") & (c.y == "sequencial_clinico")]
        for _, r in c.iterrows():
            k = (r.x.split("|")[1], t)
            dif[k], lo[k], hi[k] = r.diferenca, r.dif_lo, r.dif_hi
    treinos = [(k, n) for k, n in treinos if any((k, t) in dif for t, _ in testes)]
    m = np.array([[dif.get((k, t), np.nan) for t, _ in testes] for k, _ in treinos])
    cmap = LinearSegmentedColormap.from_list("dif", ["#e34948", "#f2f1ee", AZUL])
    fig, ax = plt.subplots(figsize=(9.5, 1.1 + 0.85 * len(treinos)), constrained_layout=True)
    im = ax.imshow(m, cmap=cmap, norm=TwoSlopeNorm(0, -1500, 1500), aspect="auto")
    for i, (k, _) in enumerate(treinos):
        for j, (t, _) in enumerate(testes):
            if (k, t) not in dif:
                continue
            sig = lo[(k, t)] > 0 or hi[(k, t)] < 0
            ax.text(j, i, f"{dif[(k, t)]:+.0f}\n[{lo[(k, t)]:+.0f}, {hi[(k, t)]:+.0f}]", ha="center",
                    va="center", fontsize=8.5, color=TEXTO, fontweight="bold" if sig else "normal")
    ax.set_xticks(range(len(testes)), [n for _, n in testes])
    ax.set_yticks(range(len(treinos)), [n for _, n in treinos])
    ax.set_xlabel("evaluated on"); ax.set_ylabel("trained on"); ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("reward of C minus clinician-guided\nsequential rule (paired, 95% CI)"); cb.outline.set_visible(False)
    ax.set_title("Per-case agent vs. the best fixed rule, across training and test geographies")
    _salva(fig, "fig_transfer.png")


FIGURAS = {
    "transferencia": transferencia,
    "seir": seir, "espacial_rio": espacial_rio, "recife_anos": recife_anos, "recife_resumo": recife_resumo,
    "bracos_v8": bracos_v8, "politicas_v9": politicas_v9, "temperatura": temperatura,
}


def main(nomes=None) -> None:
    for n in (nomes or FIGURAS):
        FIGURAS[n]()


if __name__ == "__main__":
    main(sys.argv[1:] or None)
