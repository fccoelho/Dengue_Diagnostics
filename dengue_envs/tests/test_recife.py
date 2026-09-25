"""Recife: classificação das notificações, normalização de endereços e máscara das superfícies.

O risco aqui é silencioso: um código de classificação lido errado muda a
doença de milhares de casos, e uma rua que não casa cai no nível de bairro sem
erro nenhum. Os testes travam as regras que as superfícies dependem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dengue_envs.data.recife import classifica, normaliza_texto, nucleo_logradouro


def test_classificacao_da_ficha_de_dengue():
    c = pd.Series(["10", "11.0", "12", "1", "5", "8", None])
    assert classifica("dengue", c).tolist() == [
        "dengue", "dengue", "dengue", "dengue", "descartado", "inconclusivo", "sem_classificacao"]


def test_codigos_de_dengue_na_ficha_de_chik_sao_ambiguos():
    # Fichas de chik de 2015-16 com a escala antiga da dengue, sem sorologia.
    c = pd.Series(["13", "1", "2", "5"])
    assert classifica("chik", c).tolist() == ["chik", "ambiguo", "ambiguo", "descartado"]


@pytest.mark.parametrize("notificado, oficial", [
    ("AV, BEBERIBE", "Avenida Beberibe"),
    ("AV CONS AGUIAR", "Avenida Conselheiro Aguiar"),
    ("R. DR. JOSE MARIA", "Rua Doutor José Maria"),
    ("TV STO AMARO", "Travessa Santo Amaro"),
])
def test_grafias_do_mesmo_logradouro_coincidem(notificado, oficial):
    assert nucleo_logradouro(notificado) == nucleo_logradouro(oficial)


def test_normalizacao_tira_acento_e_pontuacao():
    assert normaliza_texto("  Várzea / Cordeiro ") == "VARZEA CORDEIRO"
    assert normaliza_texto(None) == ""


def test_mascara_zera_a_probabilidade_fora_da_area():
    pytest.importorskip("pykrige")
    from dengue_envs.data.kriging import GridSpec
    from dengue_envs.data.kriging_generator import surfaces_from_points

    rng = np.random.default_rng(0)
    bbox = (0.0, 10_000.0, 0.0, 10_000.0)
    grid = GridSpec(*bbox, 1000.0)
    mascara = np.zeros((grid.ny, grid.nx), dtype=bool)
    mascara[:, :5] = True  # só a metade oeste é "cidade"
    pontos = {d: (rng.uniform(0, 10_000, 200), rng.uniform(0, 10_000, 200))
              for d in ("Dengue", "Chikungunya")}
    sup, payload = surfaces_from_points(pontos, bbox, obs_cell_m=1000.0, pred_cell_m=1000.0,
                                        mask=mascara)
    for p in (sup.prob_dengue, sup.prob_chik):
        assert p[~mascara].sum() == 0.0
        assert p.sum() == pytest.approx(1.0)
    assert payload["mask"].dtype == bool
