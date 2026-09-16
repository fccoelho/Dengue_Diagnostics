"""Modelo temporal da epidemia (SEIR com parâmetros de literatura).

O que se trava aqui são propriedades EPIDEMIOLÓGICAS, verificáveis contra a
teoria, e não só o formato das curvas — porque o erro que motivou o módulo era
exatamente um em que o formato parecia razoável e a epidemiologia estava
errada: sem `/N`, um R0 declarado de 1,5 valia 225, e a epidemia inteira cabia
em 9 dias.
"""
from __future__ import annotations

import numpy as np
import pytest

from dengue_envs.core.epi_model import (
    CHIK,
    DENGUE,
    DiseaseParams,
    final_size,
    legacy_sir_cumulative,
    seir_cumulative_cases,
)


def _taxa_de_crescimento(curva, inicio=5, fim=40):
    """r estimado da fase exponencial pela incidência diária."""
    inc = np.diff(curva)
    janela = inc[inicio:fim]
    dias = np.arange(len(janela))
    return np.polyfit(dias, np.log(np.maximum(janela, 1e-12)), 1)[0]


# --- o bug que motivou o módulo ---------------------------------------------

@pytest.mark.parametrize("r0", [1.25, 1.5, 1.7])
def test_effective_r0_equals_declared_r0(r0):
    """Com o `/N` correto, o R0 medido pela taxa de crescimento bate o declarado.

    Para SEIR com estágios exponenciais: R0 = (1 + r·latente)(1 + r·infeccioso).
    """
    # Semente minúscula numa população grande: fase exponencial limpa.
    curva = seir_cumulative_cases(r0, DENGUE, 1e7, 200, initial_infected_fraction=1e-6)
    r = _taxa_de_crescimento(curva, 10, 80)
    r0_medido = (1 + r * DENGUE.latent_days) * (1 + r * DENGUE.infectious_days)
    assert r0_medido == pytest.approx(r0, rel=0.03)


def test_legacy_model_documents_the_bug():
    """O modelo legado comprime a epidemia inteira em ~10 dias.

    Preservado só para reprodução. Este teste existe para que ninguém volte a
    usá-lo achando que R0 = 1,5 significa o que diz.
    """
    curva = legacy_sir_cumulative(popsize=150, epilength=60, r0=1.5)
    inc = np.diff(curva, prepend=0)
    ultimo_dia_com_caso = int(np.nonzero(inc >= 0.5)[0].max())
    assert ultimo_dia_com_caso < 15, "o legado deveria esgotar em ~10 dias"
    assert curva[-1] > 0.9 * 150, "o legado infecta praticamente toda a população"


def test_seir_epidemic_lasts_months_not_days():
    """A mesma população e o mesmo R0 declarado, agora com dinâmica realista."""
    curva = seir_cumulative_cases(1.5, DENGUE, 150, 400, initial_infected_fraction=0.01)
    total = curva[-1]
    d5 = int(np.searchsorted(curva, 0.05 * total))
    d95 = int(np.searchsorted(curva, 0.95 * total))
    assert d95 - d5 > 90, f"epidemia de {d95 - d5} dias — curta demais para dengue"


# --- teoria -----------------------------------------------------------------

@pytest.mark.parametrize("r0", [1.3, 1.56, 1.7, 2.5])
def test_final_size_matches_theory(r0):
    """O total integrado bate a equação do tamanho final z = 1 - exp(-R0 z)."""
    n = 10_000
    curva = seir_cumulative_cases(r0, DENGUE, n, 2000, initial_infected_fraction=1e-4)
    assert curva[-1] / n == pytest.approx(final_size(r0), abs=0.01)


def test_final_size_edge_cases():
    assert final_size(0.9) == 0.0
    assert final_size(1.0) == 0.0
    assert 0.0 < final_size(1.1) < final_size(2.0) < 1.0


def test_generation_time_is_the_sum_of_stages():
    assert DENGUE.generation_time == pytest.approx(16.0)
    assert CHIK.generation_time == pytest.approx(14.0)


def test_longer_generation_time_slows_the_epidemic():
    rapida = DiseaseParams(latent_days=3, infectious_days=3, r0_range=(1.5, 1.5))
    lenta = DiseaseParams(latent_days=11, infectious_days=5, r0_range=(1.5, 1.5))
    a = seir_cumulative_cases(1.5, rapida, 1000, 400)
    b = seir_cumulative_cases(1.5, lenta, 1000, 400)
    assert int(np.diff(a).argmax()) < int(np.diff(b).argmax())


def test_curve_is_monotone_and_starts_near_zero():
    curva = seir_cumulative_cases(1.6, CHIK, 300, 300)
    assert np.all(np.diff(curva) >= 0)
    assert curva[0] == pytest.approx(0.0, abs=1e-9)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="population"):
        seir_cumulative_cases(1.5, DENGUE, 0, 100)
    with pytest.raises(ValueError, match="initial_infected_fraction"):
        seir_cumulative_cases(1.5, DENGUE, 100, 100, initial_infected_fraction=1.5)


# --- parâmetros de literatura -----------------------------------------------

def test_literature_ranges():
    """Faixas ancoradas em Villela et al. 2017 (dengue) e Moreira et al. 2023 (chik)."""
    assert DENGUE.r0_range == (1.25, 1.70)
    assert CHIK.r0_range == (1.46, 1.67)


# --- integração com o ambiente ----------------------------------------------

def _env_seir(**over):
    from dengue_envs.wrappers import make_env

    env = {
        "size": 100, "episize": 300, "epilength": 300, "start_day": 1,
        "lab_delay_days": 5, "reward_delay_days": 0,
        "other_prevalence": 0.25, "epi_model": "seir",
    }
    env.update(over)
    return make_env({"env": env, "wrappers": ["map_tensor", "case_by_case"]})


def test_environment_notifies_cases_over_months():
    env = _env_seir()
    env.reset(seed=4)
    dias = env.unwrapped.world.casedf["t"]
    env.close()
    assert dias.max() - dias.min() > 90


def test_environment_samples_r0_from_literature():
    env = _env_seir(randomize_outbreak=True)
    for s in range(12):
        env.reset(seed=s)
        u = env.unwrapped
        assert DENGUE.r0_range[0] <= u.dengue_r0 <= DENGUE.r0_range[1]
        assert CHIK.r0_range[0] <= u.chik_r0 <= CHIK.r0_range[1]
    env.close()


def test_legacy_remains_the_default():
    """Mudar o default alteraria em silêncio todos os resultados já produzidos."""
    from dengue_envs.wrappers import make_env

    env = make_env({"env": {"size": 60, "episize": 40, "epilength": 10},
                    "wrappers": ["map_tensor", "case_by_case"]})
    assert env.unwrapped.epi_model == "legacy"
    env.close()


def test_unknown_epi_model_is_rejected():
    with pytest.raises(ValueError, match="epi_model"):
        _env_seir(epi_model="sirs")
