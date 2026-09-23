"""Bootstrap hierárquico e pareado: o intervalo mede a variância que diz medir.

O perigo de um intervalo de confiança errado é que ele sempre parece
plausível. Estes testes travam as três propriedades de que as conclusões do
artigo dependem: o pareamento cancela a variância do surto, o nível das seeds
de treino é de fato reamostrado, e o intervalo cobre a média verdadeira na
taxa nominal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments import bootstrap as B


def _tabela(mats):
    linhas = []
    for braco, m in mats.items():
        for i, linha in enumerate(np.atleast_2d(m)):
            for j, v in enumerate(linha):
                linhas.append({"braco": braco, "seed_treino": i, "seed_aval": 100 + j, "recompensa": v})
    return pd.DataFrame(linhas)


def test_pareamento_cancela_a_variancia_do_surto():
    # Episódios com variância enorme, e X sempre exatamente 100 acima de Y no
    # MESMO episódio. Sem pareamento, o intervalo da diferença seria largo;
    # pareado, é exatamente [100, 100].
    y = np.random.default_rng(1).normal(0, 2000, size=(1, 10))
    mats = {"X": y + 100, "Y": y}
    _, comps = B.bootstrap(mats, [("X", "Y")], n_replicas=2000)
    r = comps.iloc[0]
    assert r.diferenca == pytest.approx(100)
    assert r.dif_lo == pytest.approx(100) and r.dif_hi == pytest.approx(100)
    assert r.p_x_melhor == 1.0 and r.p_lo == 1.0


def test_seeds_de_treino_sao_reamostradas():
    # Episódios idênticos, seeds de treino diferentes: toda a variância está
    # no nível superior, e o intervalo precisa refleti-la.
    m = np.array([[0.0] * 10, [10.0] * 10, [20.0] * 10])
    bracos, _ = B.bootstrap({"X": m}, n_replicas=4000)
    r = bracos.iloc[0]
    assert r.media == pytest.approx(10)
    assert r.media_lo < 5 and r.media_hi > 15
    assert r.desvio_seeds == pytest.approx(10)


def test_dados_constantes_dao_intervalo_nulo():
    bracos, _ = B.bootstrap({"X": np.full((3, 10), 7.0)}, n_replicas=500)
    r = bracos.iloc[0]
    assert r.media_lo == r.media_hi == 7.0
    assert r.iqm_lo == r.iqm_hi == 7.0


def test_empate_conta_meio():
    x = np.array([[1.0, 2.0, 3.0]])
    assert float(B._prob_melhora(x, x)) == 0.5
    assert float(B._prob_melhora(x + 1, x)) == 1.0
    assert float(B._prob_melhora(x, x + 1)) == 0.0


def test_cobertura_nominal_no_nivel_dos_episodios():
    # Uma seed, 30 episódios normais: o IC de 95% deve conter a média
    # verdadeira em ~95% das repetições. O percentil sub-cobre um pouco com
    # n=30 (medido em 1000 repetições: 93,5%, igual a um bootstrap ingênuo de
    # referência); com 400 repetições o erro-padrão é ~1,2 ponto.
    rng = np.random.default_rng(7)
    acertos = 0
    n = 400
    for k in range(n):
        m = rng.normal(50, 10, size=(1, 30))
        bracos, _ = B.bootstrap({"X": m}, n_replicas=1000, seed=k)
        acertos += bracos.iloc[0].media_lo <= 50 <= bracos.iloc[0].media_hi
    assert 0.90 <= acertos / n <= 0.97


def test_matrizes_alinham_episodios_e_recusam_buracos():
    df = _tabela({"X": np.array([[1.0, 2.0, 3.0]]), "Y": np.array([[4.0, 5.0, 6.0]])})
    mats, eps = B.matrizes(df.sample(frac=1, random_state=0), "recompensa")
    assert eps.tolist() == [100, 101, 102]
    assert mats["X"].tolist() == [[1.0, 2.0, 3.0]]

    incompleto = df[~((df.braco == "Y") & (df.seed_aval == 101))]
    with pytest.raises(ValueError, match="101"):
        B.matrizes(incompleto, "recompensa")


def test_carrega_o_benchmark_real():
    if not (B.RESULTADOS / "baseline_v8seir_kriging").exists():
        pytest.skip("resultados do v4 ausentes")
    df = B.carrega_benchmark_v4()
    mats, eps = B.matrizes(df, "recompensa")
    assert len(eps) == 10
    assert mats["testonce"].shape == (1, 10)
    assert mats["testonce"].mean() == pytest.approx(2782.1, abs=0.05)
    for braco in B.BRACOS_V4:
        assert mats[braco].shape[0] >= 3
