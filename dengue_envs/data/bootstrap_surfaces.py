"""Bootstrap espacial: famílias de superfícies de Kriging plausíveis para cada cidade e ano.

Cada réplica reamostra, com reposição, as notificações de cada doença (mesmo
número de casos) e refaz o Kriging. No Recife, cada réplica também refaz a
geocodificação com outra semente — o ponto sorteado na rua ou no bairro —, de
modo que a incerteza da localização entra junto com a da amostra.

Serve a duas coisas:
1. intervalo de confiança para "quanto a posição revela a doença" (acurácia
   de Bayes da posição) em cada cidade e ano;
2. mais dados para o treino: um cenário pode sortear, a cada episódio, uma das
   réplicas, em vez de ver sempre a mesma superfície.

Uso::

    python -m dengue_envs.data.bootstrap_surfaces --cidade rio --replicas 20
    python -m dengue_envs.data.bootstrap_surfaces --cidade recife --anos 2016 2021 --replicas 20

Saídas: `results/kriging/boot/<cidade>_<ano>_b<NN>.npz` e
`results/kriging/boot/resumo_<cidade>.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from dengue_envs.data.kriging_generator import (
    load_kriging_surfaces,
    position_bayes_accuracy,
    save_kriging_surfaces,
    surfaces_from_points,
)

SAIDA = Path("results/kriging/boot")
CELULA_M = 500.0


def _reamostra(pontos: Dict[str, Tuple[np.ndarray, np.ndarray]], rng: np.random.Generator):
    out = {}
    for d, (x, y) in pontos.items():
        i = rng.integers(0, len(x), len(x))
        out[d] = (x[i], y[i])
    return out


def _metricas(caminho: Path, mascara=None) -> Dict[str, float]:
    s = load_kriging_surfaces(caminho)
    d, c = s.prob_dengue, s.prob_chik
    dentro = mascara if mascara is not None else np.ones_like(d, dtype=bool)
    return {"acerto_posicao": position_bayes_accuracy(s),
            "correlacao": float(np.corrcoef(d[dentro], c[dentro])[0, 1])}


def rio(replicas: int, seed: int = 0) -> pd.DataFrame:
    from dengue_envs.data.kriging import DEFAULT_RIO_BBOX_LONLAT, bbox_lonlat_to_projected, load_zikario_cases

    casos = load_zikario_cases(str(Path(__file__).resolve().parent / "zikario.gpkg"),
                               years=(2015, 2016), diseases=("Dengue", "Chikungunya"))
    pontos = {d: (g["x"].to_numpy(), g["y"].to_numpy()) for d, g in casos.groupby("Doenca")}
    bbox = bbox_lonlat_to_projected(DEFAULT_RIO_BBOX_LONLAT)
    rng = np.random.default_rng(seed)
    linhas = []
    for b in range(replicas):
        _, payload = surfaces_from_points(_reamostra(pontos, rng), bbox, obs_cell_m=CELULA_M,
                                          pred_cell_m=CELULA_M, crs="EPSG:31983",
                                          bbox_lonlat=DEFAULT_RIO_BBOX_LONLAT)
        payload.update({"replica": b, "cidade": "rio", "ano": "2015-2016"})
        caminho = save_kriging_surfaces(payload, SAIDA / f"rio_2015_2016_b{b:02d}.npz")
        linhas.append({"cidade": "rio", "ano": "2015-2016", "replica": b, "arquivo": str(caminho),
                       **_metricas(caminho)})
        print(f"[boot rio] réplica {b}: acerto={linhas[-1]['acerto_posicao']:.3f}", flush=True)
    return pd.DataFrame(linhas)


def recife(anos: Iterable[int], replicas: int, seed: int = 0) -> pd.DataFrame:
    import geopandas as gpd

    from dengue_envs.data.build_recife_surfaces import _grid_e_mascara
    from dengue_envs.data.recife import CRS_RECIFE, RAIZ, Geocodificador, carrega_notificacoes

    notificacoes = carrega_notificacoes()
    geo = Geocodificador()
    bairros = gpd.read_file(RAIZ / "geo" / "bairros.geojson").to_crs(CRS_RECIFE)
    bbox, _, mascara = _grid_e_mascara(bairros)
    rng = np.random.default_rng(seed)
    linhas = []
    for ano in anos:
        base = notificacoes[(notificacoes["ano_arquivo"] == ano)
                            & notificacoes["status"].isin(["dengue", "chik"])].reset_index(drop=True)
        for b in range(replicas):
            # Reamostra as NOTIFICAÇÕES (por doença) e geocodifica de novo com
            # outra semente: incerteza da amostra + incerteza da localização.
            partes = []
            for _, g in base.groupby("status"):
                partes.append(g.iloc[rng.integers(0, len(g), len(g))])
            amostra = geo.geocodifica(pd.concat(partes, ignore_index=True), seed=int(rng.integers(1 << 31)))
            amostra = amostra[amostra["x"].notna()]
            pontos = {nome: (amostra.loc[amostra.status == s, "x"].to_numpy(),
                             amostra.loc[amostra.status == s, "y"].to_numpy())
                      for nome, s in (("Dengue", "dengue"), ("Chikungunya", "chik"))}
            _, payload = surfaces_from_points(pontos, bbox, obs_cell_m=CELULA_M, pred_cell_m=CELULA_M,
                                              crs=CRS_RECIFE, mask=mascara)
            payload.update({"replica": b, "cidade": "recife", "ano": ano})
            caminho = save_kriging_surfaces(payload, SAIDA / f"recife_{ano}_b{b:02d}.npz")
            linhas.append({"cidade": "recife", "ano": ano, "replica": b, "arquivo": str(caminho),
                           **_metricas(caminho, mascara)})
        sub = pd.DataFrame([l for l in linhas if l["ano"] == ano])
        print(f"[boot recife {ano}] acerto={sub.acerto_posicao.mean():.3f} "
              f"[{sub.acerto_posicao.quantile(.025):.3f}, {sub.acerto_posicao.quantile(.975):.3f}]", flush=True)
    return pd.DataFrame(linhas)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cidade", choices=("rio", "recife"), required=True)
    ap.add_argument("--anos", type=int, nargs="+", default=list(range(2015, 2026)))
    ap.add_argument("--replicas", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    SAIDA.mkdir(parents=True, exist_ok=True)
    df = rio(args.replicas, args.seed) if args.cidade == "rio" else recife(args.anos, args.replicas, args.seed)
    destino = SAIDA / f"resumo_{args.cidade}.csv"
    if destino.exists():
        antigo = pd.read_csv(destino)
        chave = ["cidade", "ano", "replica"]
        antigo = antigo[~antigo.set_index(chave).index.isin(df.set_index(chave).index)]
        df = pd.concat([antigo, df], ignore_index=True)
    df.to_csv(destino, index=False)


if __name__ == "__main__":
    main()
