"""A confirmação epidemiológica não pode ler a verdade-terreno.

Antes desta mudança, `_epi_confirm` consultava `world.get_maps_up_to_t()` —
histogramas indexados pela coluna `disease` **real**, incluindo casos nunca
notificados. Além de vazar, era inócuo: com `threshold=1` numa única célula de
um grid 400x400, a densidade máxima medida por célula é 1, então a ação
devolvia 0 sempre (5598 chamadas em 5 episódios, todas 0).

Agora ela consulta o mapa que o **agente** construiu, contando apenas casos
confirmados por laudo positivo. Isso é o que torna o exame um investimento:
cada laudo melhora toda `epi_confirm` futura naquela região.
"""
from __future__ import annotations

import numpy as np
import pytest

from dengue_envs.core.clinical import CHIK, DENGUE, POSITIVE
from dengue_envs.core.epi_confirm import epi_confirm, neighborhood_sum
from dengue_envs.wrappers import make_env

_ENV = {
    "size": 60,
    "episize": 40,
    "epilength": 10,
    "start_day": 1,
    "lab_delay_days": 0,
    "reward_delay_days": 0,
    "randomize_outbreak": False,
    "other_prevalence": 0.25,
    "max_case_revisits": 2,
    "epi_radius": 10,
    "epi_threshold": 0,
}


def _cfg(**over):
    env = dict(_ENV)
    env.update(over)
    return {"env": env, "wrappers": ["map_tensor", "case_by_case"],
            "context_features": True, "per_case_reward": True}


# --- a função pura ----------------------------------------------------------

def test_neighborhood_sum_respects_radius():
    m = np.zeros((10, 10))
    m[5, 5] = 1.0
    m[5, 8] = 1.0
    assert neighborhood_sum(m, 5, 5, 0) == 1.0
    assert neighborhood_sum(m, 5, 5, 2) == 1.0     # (5,8) fora do raio 2
    assert neighborhood_sum(m, 5, 5, 3) == 2.0     # entra no raio 3


def test_neighborhood_sum_clips_at_borders():
    m = np.ones((5, 5))
    assert neighborhood_sum(m, 0, 0, 1) == 4.0     # canto: só 2x2 existe


def test_radius_zero_reproduces_historic_behaviour():
    """O default `radius=0` mantém a semântica antiga (célula única)."""
    d = np.zeros((30, 30))
    d[10, 10] = 2.0
    assert epi_confirm(DENGUE, 10, 10, d, d) == 1
    assert epi_confirm(DENGUE, 11, 11, d, d) == 0


def test_exclude_removes_the_cases_own_confirmation():
    """Sem auto-exclusão a ação devolveria o laudo do próprio caso."""
    d = np.zeros((30, 30))
    d[10, 10] = 1.0
    assert epi_confirm(DENGUE, 10, 10, d, d, threshold=0) == 1
    assert epi_confirm(DENGUE, 10, 10, d, d, threshold=0, exclude=(1.0, 0.0)) == 0


def test_other_never_confirms():
    d = np.ones((30, 30)) * 99
    assert epi_confirm(2, 10, 10, d, d, threshold=0) == 0


# --- integração com o ambiente ---------------------------------------------

def test_confirmed_maps_start_empty_and_only_grow_with_positive_labs():
    env = make_env(_cfg())
    env.reset(seed=5)
    u = env.unwrapped
    assert u.confirmed_dmap.sum() == 0.0
    assert u.confirmed_cmap.sum() == 0.0

    terminated = truncated = False
    passos = 0
    while not (terminated or truncated) and passos < 400:
        case_id = env.current_case[0]
        acao = 0 if case_id in u.obs_cases.index else 3
        _obs, _r, terminated, truncated, _i = env.step(acao)
        passos += 1

    total = u.confirmed_dmap.sum() + u.confirmed_cmap.sum()
    assert total > 0, "testando tudo, algum laudo positivo deveria ter saído"
    # O mapa é exatamente o registro de confirmações — um ponto por caso, sem
    # dupla contagem quando o caso é reapresentado (`max_case_revisits`).
    assert total == pytest.approx(len(u._confirmed_by_case))
    # E todo caso registrado tem, de fato, um laudo positivo do agente.
    pos_d = {cid for cid, r in u.testd if r == POSITIVE}
    pos_c = {cid for cid, r in u.testc if r == POSITIVE}
    for case_id, disease in u._confirmed_by_case.items():
        assert case_id in (pos_d if disease == DENGUE else pos_c), (
            f"caso {case_id} entrou no mapa sem laudo positivo correspondente"
        )
    env.close()


def test_confirmed_map_never_exceeds_the_agents_own_evidence():
    """Blindagem contra o vazamento: o mapa do agente ⊆ casos que ele testou."""
    env = make_env(_cfg())
    env.reset(seed=9)
    u = env.unwrapped
    terminated = truncated = False
    passos = 0
    while not (terminated or truncated) and passos < 400:
        case_id = env.current_case[0]
        _obs, _r, terminated, truncated, _i = env.step(
            0 if case_id in u.obs_cases.index else 3
        )
        passos += 1

    confirmados = u.confirmed_dmap.sum() + u.confirmed_cmap.sum()
    testados = len({cid for cid, _ in u.testd} | {cid for cid, _ in u.testc})
    assert confirmados <= testados, (
        f"{confirmados} confirmados para {testados} testados — há informação "
        f"entrando de fora dos laudos do agente"
    )
    env.close()


def test_epi_confirm_is_dead_without_testing():
    """Sem exame não há mapa, então não há confirmação — o incentivo pretendido."""
    env = make_env(_cfg())
    env.reset(seed=11)
    u = env.unwrapped
    terminated = truncated = False
    passos = 0
    while not (terminated or truncated) and passos < 400:
        _obs, _r, terminated, truncated, _i = env.step(2)  # SÓ epi_confirm
        passos += 1
    assert u.confirmed_dmap.sum() == 0.0 and u.confirmed_cmap.sum() == 0.0
    assert all(r == 0 for _cid, r in u.epiconf), (
        "sem nenhum laudo, nenhuma confirmação epidemiológica deveria sair"
    )
    env.close()


def test_epi_confirm_becomes_informative_after_testing():
    """Com exames acumulados a ação passa a devolver 1 — deixa de ser inócua."""
    env = make_env(_cfg(epi_radius=20, epi_threshold=0))
    env.reset(seed=3)
    u = env.unwrapped
    terminated = truncated = False
    passos = 0
    # Testa dengue em tudo; quando o caso volta, pede confirmação epidemiológica.
    while not (terminated or truncated) and passos < 600:
        case_id = env.current_case[0]
        if case_id in u.obs_cases.index and int(u.obs_cases.loc[case_id, "testd"]) == 0:
            acao = 0
        elif case_id in u.obs_cases.index:
            acao = 2
        else:
            acao = 3
        _obs, _r, terminated, truncated, _i = env.step(acao)
        passos += 1

    assert u.confirmed_dmap.sum() > 0, "deveria haver laudos positivos acumulados"
    assert any(r == 1 for _cid, r in u.epiconf), (
        "com mapa construído, alguma confirmação deveria sair positiva"
    )
    env.close()


def test_local_density_excludes_the_case_itself():
    env = make_env(_cfg(epi_radius=5, epi_threshold=0))
    env.reset(seed=7)
    u = env.unwrapped
    terminated = truncated = False
    passos = 0
    while not (terminated or truncated) and passos < 400:
        case_id = env.current_case[0]
        _obs, _r, terminated, truncated, _i = env.step(
            0 if case_id in u.obs_cases.index else 3
        )
        passos += 1

    for case_id, disease in u._confirmed_by_case.items():
        x, y = u.get_case_xy(case_id)
        bruto_d = neighborhood_sum(u.confirmed_dmap, int(x), int(y), u.epi_radius)
        nd, _nc = u.local_confirmed_density(case_id)
        if disease == DENGUE:
            assert nd == pytest.approx(bruto_d - 1.0), (
                "a densidade local deve descontar a própria confirmação"
            )
            break


def test_observation_carries_the_confirmed_maps():
    """Canais 4 e 5 do tensor + 2 posições de contexto (senão a ação é loteria)."""
    env = make_env(_cfg())
    obs, _ = env.reset(seed=13)
    assert obs["map"].shape[0] == 6
    assert obs["context"].shape == (16,)
    assert obs["map"][4].sum() == 0 and obs["map"][5].sum() == 0

    u = env.unwrapped
    terminated = truncated = False
    passos = 0
    while not (terminated or truncated) and passos < 400:
        case_id = env.current_case[0]
        obs, _r, terminated, truncated, _i = env.step(
            0 if case_id in u.obs_cases.index else 3
        )
        passos += 1
    assert obs["map"][4].sum() + obs["map"][5].sum() > 0, (
        "o mapa de confirmados precisa chegar à observação"
    )
    env.close()
