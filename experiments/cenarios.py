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

Escala física (`*_fis`, ver `physical_env_grids`): as duas cidades no mesmo
grid de 200 m por célula, na área real (sem esticar), transladadas para uma
posição sorteada, com os casos "outro" só na área habitada
(`results/kriging/<cidade>_mask.npz`). `cenario_recifesup` só corrige os
casos "outro" (mantém o esticamento): separa as duas causas do atalho.

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


CELULA_FIS_M = 200.0
MASCARA = {"rio": "results/kriging/rio_mask.npz", "recife": "results/kriging/recife_mask.npz"}


def _fis(cfg: dict, cidade: str = None) -> dict:
    """Escala física; a máscara vai em cada entrada do `mix` (é por cidade)."""
    cfg = copy.deepcopy(cfg)
    cfg["env"]["surface_cell_m"] = CELULA_FIS_M
    cfg["env"]["surface_translate"] = True
    for item in cfg["env"].get("mix", []):
        item["surface_mask_path"] = MASCARA["rio" if "/rio_" in item["surfaces_path"] else "recife"]
    if cidade is not None:
        cfg["env"]["surface_mask_path"] = MASCARA[cidade]
    return cfg


def _suporte(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["env"]["kriging_other_on_support"] = True
    return cfg


def _misto(base: dict, entradas: list) -> dict:
    cfg = copy.deepcopy(base)
    cfg["env"]["generator"] = "mixed"
    cfg["env"]["mix"] = entradas
    cfg["env"].pop("surfaces_path", None)
    return cfg


def _com_posicao(cfg: dict, escala=(0.35, 1.0)) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["env"]["surface_random_placement"] = list(escala)
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
        "cenario_recifepos": ("Treino: como cenario_recife, com escala, rotação e posição da cidade sorteadas "
                              "por episódio (tira o atalho de memorizar o contorno).",
                              _com_posicao(_misto(base, _recife()))),
        "cenario_misto": ("Treino: 1/3 sintético, 1/3 Rio (réplicas), 1/3 Recife (réplicas).",
                          _misto(base, [{"generator": "synthetic", "weight": 1 / 3}] + _rio(1 / 3) + _recife(1 / 3))),
        "cenario_riofis": ("Treino: réplicas do Rio em escala física (200 m/célula), transladadas.",
                           _fis(_misto(base, _rio()))),
        "cenario_recifefis": (f"Treino: réplicas do Recife ({list(ANOS_TREINO_RECIFE)}) em escala física.",
                              _fis(_misto(base, _recife()))),
        "cenario_recifesup": ("Treino: como cenario_recife, com os casos 'outro' só dentro do município.",
                              _suporte(_misto(base, _recife()))),
        "teste_sintetico": ("Teste: gerador sintético.", sint),
        "teste_rio_fis": ("Teste: Rio 2015-16 original, em escala física.",
                          _fis(original("results/kriging/rio_2015_2016_kriging_surfaces.npz"), "rio")),
        **{f"teste_recife_{a}_fis": (f"Teste: Recife {a} original, em escala física.",
                                     _fis(original(f"results/kriging/recife_{a}_kriging_surfaces.npz"), "recife"))
           for a in ANOS_TESTE_RECIFE},
        **{f"teste_recife_{a}_sup": (f"Teste: Recife {a} original, casos 'outro' só no município.",
                                     _suporte(original(f"results/kriging/recife_{a}_kriging_surfaces.npz")))
           for a in ANOS_TESTE_RECIFE},
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
