"""Superfícies de Kriging do Recife, uma por ano, com mapas dos casos.

Uso (da raiz do repositório)::

    python -m dengue_envs.data.build_recife_surfaces            # todos os anos
    python -m dengue_envs.data.build_recife_surfaces --anos 2015 2016

Pré-requisito: os arquivos brutos em `results/recife/raw/` e as bases
geográficas em `results/recife/geo/` (ver `dengue_envs/data/recife.py`).

Saídas:
- `results/kriging/recife_<ano>_kriging_surfaces.npz` — mesmo formato do Rio,
  carregável por `load_kriging_surfaces` (e, portanto, pelo ambiente via
  `surfaces_path`). Traz também a superfície dos casos descartados
  (`prob_descartado`), que ainda não é usada pelo ambiente.
- `results/recife/figuras/` — pontos e superfícies por ano, e o resumo.
- `results/recife/resumo_anos.csv` — contagens, geocodificação e dificuldade.

Casos: confirmados por qualquer critério (laboratorial ou
clínico-epidemiológico), residentes no Recife, com o ano do arquivo do portal.
A probabilidade é zerada fora do município (máscara dos bairros oficiais).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from dengue_envs.data.kriging import GridSpec, intensity_surface_from_points, normalize_probability
from dengue_envs.data.kriging_generator import (
    load_kriging_surfaces,
    position_bayes_accuracy,
    save_kriging_surfaces,
    surfaces_from_points,
)
from dengue_envs.data.recife import CRS_RECIFE, RAIZ, Geocodificador, carrega_notificacoes

CELULA_M = 500.0
SAIDA_NPZ = Path("results/kriging")
FIGURAS = RAIZ / "figuras"
GEOCODIFICADO = RAIZ / "notificacoes_geocodificadas.csv.gz"

# Cores: dengue e chik nas duas primeiras posições da paleta categórica.
COR = {"dengue": "#2a78d6", "chik": "#eb6834", "descartado": "#8a8984"}
ROTULO = {"dengue": "Dengue confirmada", "chik": "Chikungunya confirmada",
          "descartado": "Descartados (não é a doença notificada)"}


def carrega_geocodificado(refaz: bool = False) -> pd.DataFrame:
    if GEOCODIFICADO.exists() and not refaz:
        return pd.read_csv(GEOCODIFICADO, low_memory=False)
    geo = Geocodificador().geocodifica(carrega_notificacoes())
    geo.to_csv(GEOCODIFICADO, index=False)
    return geo


def _grid_e_mascara(bairros) -> tuple:
    import shapely

    xmin, ymin, xmax, ymax = bairros.total_bounds
    bbox = (xmin - CELULA_M, xmax + CELULA_M, ymin - CELULA_M, ymax + CELULA_M)
    grid = GridSpec(*bbox, CELULA_M)
    xx, yy = grid.mesh()
    mascara = shapely.contains_xy(bairros.union_all(), xx, yy)
    return bbox, grid, mascara


def _pontos(df: pd.DataFrame, status: str, ano: int) -> pd.DataFrame:
    sel = (df["status"] == status) & (df["ano_arquivo"] == ano) & df["x"].notna()
    return df.loc[sel, ["x", "y", "nivel"]]


def constroi_ano(df: pd.DataFrame, ano: int, bbox, grid, mascara) -> Dict:
    pontos = {s: _pontos(df, s, ano) for s in ("dengue", "chik", "descartado")}
    surfaces, payload = surfaces_from_points(
        {"Dengue": (pontos["dengue"].x.to_numpy(), pontos["dengue"].y.to_numpy()),
         "Chikungunya": (pontos["chik"].x.to_numpy(), pontos["chik"].y.to_numpy())},
        bbox, obs_cell_m=CELULA_M, pred_cell_m=CELULA_M, crs=CRS_RECIFE, mask=mascara,
    )
    # Casos descartados: a síndrome febril que não é a doença notificada — o
    # "outro" do ambiente, hoje sorteado uniforme no espaço.
    if len(pontos["descartado"]) >= 3:
        _, inten, _, _ = intensity_surface_from_points(
            pontos["descartado"].x.to_numpy(), pontos["descartado"].y.to_numpy(),
            bbox_xy=bbox, obs_cell_size=CELULA_M, pred_cell_size=CELULA_M)
        payload["prob_descartado"] = normalize_probability(np.where(mascara, inten, 0.0))
        payload["n_descartado"] = len(pontos["descartado"])
    payload["ano"] = ano
    payload["fonte"] = "Portal de Dados Abertos do Recife (ODbL); geocodificação por rua/bairro"
    for s, p in pontos.items():
        payload[f"frac_rua_{s}"] = float(p["nivel"].isin(["rua", "rua_vizinha"]).mean()) if len(p) else np.nan

    caminho = save_kriging_surfaces(payload, SAIDA_NPZ / f"recife_{ano}_kriging_surfaces.npz")
    sup = load_kriging_surfaces(caminho)
    d, c = surfaces.prob_dengue, surfaces.prob_chik
    dentro = mascara
    return {
        "ano": ano,
        "n_dengue": len(pontos["dengue"]),
        "n_chik": len(pontos["chik"]),
        "n_descartado": len(pontos["descartado"]),
        "frac_rua_dengue": payload["frac_rua_dengue"],
        "frac_rua_chik": payload["frac_rua_chik"],
        "acerto_posicao": position_bayes_accuracy(sup),
        "correlacao_dengue_chik": float(np.corrcoef(d[dentro], c[dentro])[0, 1]),
        "arquivo": str(caminho),
        "_pontos": pontos,
        "_surfaces": surfaces,
    }


# --------------------------------------------------------------------------
# figuras
# --------------------------------------------------------------------------

def _mapa_razao():
    """Divergente com as cores das doenças: chik (laranja) <- cinza neutro -> dengue (azul)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("chik_dengue", [COR["chik"], "#f2f1ee", COR["dengue"]])


def _estilo():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.edgecolor": "#bbbbbb",
                         "axes.titlelocation": "left"})
    return plt


def figura_pontos(r: Dict, bairros, destino: Path) -> None:
    plt = _estilo()
    fig, axs = plt.subplots(1, 3, figsize=(15, 6.2), constrained_layout=True)
    for ax, s in zip(axs, ("dengue", "chik", "descartado")):
        bairros.boundary.plot(ax=ax, color="#cfcfcf", linewidth=0.5)
        p = r["_pontos"][s]
        ax.scatter(p.x, p.y, s=2, color=COR[s], alpha=0.35, linewidths=0)
        ax.set_title(f"{ROTULO[s]} — n = {len(p):,}".replace(",", "."))
        ax.set_axis_off(); ax.set_aspect("equal")
    fig.suptitle(f"Recife {r['ano']}: casos geocodificados (ponto sorteado na rua ou no bairro de residência)",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(destino, dpi=130); plt.close(fig)


def figura_superficies(r: Dict, mascara, bairros, grid, destino: Path) -> None:
    from matplotlib.colors import TwoSlopeNorm

    plt = _estilo()
    s = r["_surfaces"]
    ext = (grid.xmin, grid.xmin + grid.nx * grid.cell_size, grid.ymin, grid.ymin + grid.ny * grid.cell_size)
    fig, axs = plt.subplots(1, 3, figsize=(15, 6.2), constrained_layout=True)

    def painel(ax, img, titulo, cmap, norm=None, rot=""):
        img = np.ma.masked_where(~mascara, img)
        im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, norm=norm)
        bairros.boundary.plot(ax=ax, color="#9a9a9a", linewidth=0.3)
        ax.set_title(titulo); ax.set_axis_off()
        cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label(rot); cb.outline.set_visible(False)

    painel(axs[0], s.prob_dengue / s.prob_dengue.max(), f"Dengue (n = {r['n_dengue']})", "Blues",
           rot="relativo ao máximo")
    painel(axs[1], s.prob_chik / s.prob_chik.max(), f"Chikungunya (n = {r['n_chik']})", "Oranges",
           rot="relativo ao máximo")
    with np.errstate(divide="ignore", invalid="ignore"):
        razao = np.log2(s.prob_dengue / s.prob_chik)
    lim = float(np.nanmax(np.abs(razao[mascara]))) or 1.0
    painel(axs[2], razao, "Qual doença domina: log₂(dengue / chik)", _mapa_razao(),
           TwoSlopeNorm(0, -lim, lim), rot="← chik   0 = empate   dengue →")
    fig.suptitle(f"Recife {r['ano']}: superfícies de Kriging — a posição acerta a doença em "
                 f"{r['acerto_posicao']:.1%} (correlação dengue × chik {r['correlacao_dengue_chik']:.2f})",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(destino, dpi=130); plt.close(fig)


def figura_resumo(resumo: pd.DataFrame, todos: pd.DataFrame, destino: Path) -> None:
    plt = _estilo()
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    anos = sorted(todos["ano_arquivo"].unique())
    cont = (todos[todos.status.isin(["dengue", "chik", "descartado"])]
            .groupby(["ano_arquivo", "status"]).size().unstack(fill_value=0).reindex(anos))
    larg = 0.27
    for i, s in enumerate(("dengue", "chik", "descartado")):
        if s in cont:
            axs[0].bar(np.arange(len(anos)) + (i - 1) * larg, cont[s], width=larg - 0.03,
                       color=COR[s], label=ROTULO[s])
    axs[0].set_xticks(range(len(anos)), anos, rotation=45)
    axs[0].set_title("Notificações de residentes por ano e classificação final")
    axs[0].legend(frameon=False); axs[0].grid(axis="y", color="#eeeeee"); axs[0].set_axisbelow(True)
    for lado in ("top", "right"):
        axs[0].spines[lado].set_visible(False); axs[1].spines[lado].set_visible(False)

    axs[1].plot(resumo["ano"], resumo["acerto_posicao"], marker="o", color="#2a78d6", linewidth=2)
    axs[1].axhline(0.5, color="#8a8984", linestyle="--", linewidth=1)
    axs[1].text(resumo["ano"].min(), 0.503, "a posição não diz nada", color="#52514e", fontsize=9)
    axs[1].axhline(0.544, color="#eb6834", linestyle=":", linewidth=1.5)
    axs[1].text(resumo["ano"].min(), 0.547, "Rio 2015-16 (referência do ambiente)", color="#52514e", fontsize=9)
    axs[1].set_title("Quanto a posição, sozinha, acerta a doença (Bayes, prior igual)")
    axs[1].set_xticks(resumo["ano"]); axs[1].grid(axis="y", color="#eeeeee")
    fig.savefig(destino, dpi=130); plt.close(fig)


def figura_razoes(resultados: List[Dict], mascara, bairros, grid, destino: Path) -> None:
    from matplotlib.colors import TwoSlopeNorm

    plt = _estilo()
    n = len(resultados)
    cols = min(4, n); linhas = int(np.ceil(n / cols))
    fig, axs = plt.subplots(linhas, cols, figsize=(4 * cols, 4.4 * linhas), constrained_layout=True,
                            squeeze=False)
    ext = (grid.xmin, grid.xmin + grid.nx * grid.cell_size, grid.ymin, grid.ymin + grid.ny * grid.cell_size)
    norm = TwoSlopeNorm(0, -2, 2)
    for ax, r in zip(axs.ravel(), resultados):
        s = r["_surfaces"]
        with np.errstate(divide="ignore", invalid="ignore"):
            razao = np.ma.masked_where(~mascara, np.log2(s.prob_dengue / s.prob_chik))
        im = ax.imshow(razao, origin="lower", extent=ext, cmap=_mapa_razao(), norm=norm)
        bairros.boundary.plot(ax=ax, color="#9a9a9a", linewidth=0.2)
        ax.set_title(f"{r['ano']} — acerto {r['acerto_posicao']:.0%}")
        ax.set_axis_off()
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    cb = fig.colorbar(im, ax=axs, shrink=0.6); cb.set_label("log₂(dengue / chik), cortado em ±2")
    cb.outline.set_visible(False)
    fig.suptitle("Recife: onde cada doença domina, ano a ano", x=0.01, ha="left", fontsize=12)
    fig.savefig(destino, dpi=120); plt.close(fig)


def main(argv=None) -> pd.DataFrame:
    import geopandas as gpd

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anos", type=int, nargs="+", default=None)
    ap.add_argument("--refaz-geocodificacao", action="store_true")
    args = ap.parse_args(argv)

    df = carrega_geocodificado(args.refaz_geocodificacao)
    bairros = gpd.read_file(RAIZ / "geo" / "bairros.geojson").to_crs(CRS_RECIFE)
    bbox, grid, mascara = _grid_e_mascara(bairros)
    # Só anos com as duas doenças: o ambiente precisa das duas superfícies.
    com_chik = set(df.loc[df.status == "chik", "ano_arquivo"])
    anos = args.anos or sorted(com_chik & set(df.loc[df.status == "dengue", "ano_arquivo"]))

    FIGURAS.mkdir(parents=True, exist_ok=True)
    resultados = []
    for ano in anos:
        r = constroi_ano(df, ano, bbox, grid, mascara)
        figura_pontos(r, bairros, FIGURAS / f"pontos_{ano}.png")
        figura_superficies(r, mascara, bairros, grid, FIGURAS / f"superficies_{ano}.png")
        resultados.append(r)
        print(f"[recife {ano}] dengue={r['n_dengue']} chik={r['n_chik']} "
              f"descartados={r['n_descartado']} acerto_posicao={r['acerto_posicao']:.3f} "
              f"corr={r['correlacao_dengue_chik']:.2f}")

    resumo = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in resultados])
    resumo.to_csv(RAIZ / "resumo_anos.csv", index=False)
    figura_resumo(resumo, df, FIGURAS / "resumo.png")
    figura_razoes(resultados, mascara, bairros, grid, FIGURAS / "razao_todos_anos.png")
    return resumo


if __name__ == "__main__":
    main()
