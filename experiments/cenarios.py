"""Gera as configs dos cenários de treino e de teste do experimento v6 (mobilidade entre cenários).

Todos partem do `kriging_v9` (SEIR, casos "outro", mesma economia); só a
geografia muda.

Treino (um agente por cenário, e um na mistura):
- `cenario_sintetico`: dois focos gaussianos (posição acerta 94%).
- `cenario_rio`: a cada episódio, uma das 20 réplicas do bootstrap espacial do
  Rio 2015-16.
- `cenario_recife`: a cada episódio, uma réplica de um ano do Recife, entre os
  anos de TREINO (2017-2020, 2022-2025). 2016 e 2021 — as duas maiores
  epidemias com as duas doenças — ficam fora para teste; 2015 fica fora porque
  tem só 201 chiks confirmadas e é o ano do epicentro da zika.
- `cenario_misto`: um terço de cada um dos três acima.

Teste (superfícies ORIGINAIS, nunca réplicas):
- `teste_sintetico`, `teste_rio` (Rio 2015-16), `teste_recife_2016`,
  `teste_recife_2021` (anos fora do treino do Recife).

Uso: `python -m experiments.cenarios` (grava em `experiments/configs/env/`).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

_RAIZ = Path(__file__).resolve().parents[1]
ENV = _RAIZ / "experiments" / "configs" / "env"
BOOT = "results/kriging/boot"
REPLICAS = 20
ANOS_TREINO_RECIFE = (2017, 2018, 2019, 2020, 2022, 2023, 2024, 2025)
ANOS_TESTE_RECIFE = (2016, 2021)

CABECALHO = """# Gerado por experiments/cenarios.py — não editar à mão.
# Base: kriging_v9 (SEIR, casos "outro"); só a geografia muda.
# {descricao}
"""


def _base() -> dict:
    return yaml.safe_load((ENV / "kriging_v9.yaml").read_text(encoding="utf-8"))


def _kriging(caminho: str, peso: float) -> dict:
    return {"generator": "kriging", "surfaces_path": caminho, "augment_surfaces": True, "weight": peso}


def _rio(peso_total: float = 1.0) -> list:
    return [_kriging(f"{BOOT}/rio_2015_2016_b{b:02d}.npz", peso_total / REPLICAS) for b in range(REPLICAS)]


def _recife(peso_total: float = 1.0) -> list:
    n = len(ANOS_TREINO_RECIFE) * REPLICAS
    return [_kriging(f"{BOOT}/recife_{a}_b{b:02d}.npz", peso_total / n)
            for a in ANOS_TREINO_RECIFE for b in range(REPLICAS)]


def _misto(base: dict, entradas: list) -> dict:
    cfg = copy.deepcopy(base)
    cfg["env"]["generator"] = "mixed"
    cfg["env"]["mix"] = entradas
    cfg["env"].pop("surfaces_path", None)
    return cfg


def cenarios() -> dict:
    base = _base()
    sint = copy.deepcopy(base)
    sint["env"]["generator"] = "synthetic"
    sint["env"].pop("augment_surfaces", None)

    def original(caminho: str) -> dict:
        cfg = copy.deepcopy(base)
        cfg["env"]["surfaces_path"] = caminho
        return cfg

    return {
        "cenario_sintetico": ("Treino e teste: gerador sintético (dois focos gaussianos).", sint),
        "cenario_rio": ("Treino: 20 réplicas do bootstrap espacial do Rio 2015-16.", _misto(base, _rio())),
        "cenario_recife": (f"Treino: réplicas do Recife, anos {list(ANOS_TREINO_RECIFE)}.",
                           _misto(base, _recife())),
        "cenario_misto": ("Treino: 1/3 sintético, 1/3 Rio (réplicas), 1/3 Recife (réplicas).",
                          _misto(base, [{"generator": "synthetic", "weight": 1 / 3}] + _rio(1 / 3) + _recife(1 / 3))),
        "teste_sintetico": ("Teste: gerador sintético.", sint),
        "teste_rio": ("Teste: superfície original do Rio 2015-16.",
                      original("results/kriging/rio_2015_2016_kriging_surfaces.npz")),
        **{f"teste_recife_{a}": (f"Teste: superfície original do Recife {a} (fora do treino).",
                                 original(f"results/kriging/recife_{a}_kriging_surfaces.npz"))
           for a in ANOS_TESTE_RECIFE},
    }


def main() -> None:
    for nome, (descricao, cfg) in cenarios().items():
        texto = CABECALHO.format(descricao=descricao) + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        (ENV / f"{nome}.yaml").write_text(texto, encoding="utf-8")
        print("->", ENV / f"{nome}.yaml")


if __name__ == "__main__":
    main()
