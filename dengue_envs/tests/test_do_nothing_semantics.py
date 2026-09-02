"""Semântica de "não fazer nada".

Regra: **não agir é deixar valer o diagnóstico do médico.** É uma decisão
legítima da vigilância (não investigar este caso), não uma investigação
abandonada — e por isso não paga `penalty_unresolved`. O caso é julgado pelo
acerto do palpite clínico, como qualquer outro.

O que continua sendo punido é abrir investigação e não fechá-la: pedir exame ou
ir a campo (`epi_confirm`) e terminar o episódio sem conclusão.

Motivação medida (§19): com a penalidade caindo sobre TODO caso não concluído,
62-85% dos casos do DQN terminavam "em aberto" e isso respondia pela maior
parte da perda — a inação ficava artificialmente pior que um palpite ruim, e o
custo chegava num bloco só no fim do episódio, sem ser atribuído a decisão
nenhuma.
"""
from __future__ import annotations

import numpy as np
import pytest

from dengue_envs.wrappers import make_env

_ENV = {
    "size": 60,
    "episize": 40,
    "epilength": 8,
    "start_day": 1,
    "lab_delay_days": 1,
    "reward_delay_days": 0,
    "randomize_outbreak": False,
    "other_prevalence": 0.25,
    "max_case_revisits": 2,
    "epi_radius": 10,
    "epi_threshold": 0,
    "penalty_unresolved": -10.0,
}


def _cfg(**over):
    env = dict(_ENV)
    env.update(over)
    return {"env": env, "wrappers": ["map_tensor", "case_by_case"],
            "context_features": True, "per_case_reward": True}


def _roda(cfg, acao, seed=5, limite=600):
    """Aplica sempre a mesma ação; devolve (total, env)."""
    env = make_env(cfg)
    env.reset(seed=seed)
    total = 0.0
    terminated = truncated = False
    i = 0
    while not (terminated or truncated) and i < limite:
        _obs, r, terminated, truncated, _i = env.step(acao)
        total += r
        i += 1
    return total, env


# --- não agir não é abandonar ----------------------------------------------

def test_doing_nothing_never_marks_a_case_as_investigated():
    _total, env = _roda(_cfg(), 3)          # sempre "nada"
    u = env.unwrapped
    assert len(u.investigated_cases) == 0, (
        "não fazer nada não pode abrir investigação em caso nenhum"
    )
    env.close()


def test_doing_nothing_keeps_the_doctors_diagnosis():
    """O caso termina com o palpite clínico — que é o que `agent_diagnosis` já é."""
    _total, env = _roda(_cfg(), 3)
    u = env.unwrapped
    clinico = u.obs_cases["disease"].to_numpy()
    agente = u.obs_cases["agent_diagnosis"].to_numpy()
    assert np.array_equal(clinico, agente), (
        "sem ação do agente, o diagnóstico registrado deve ser o do médico"
    )
    env.close()


def test_doing_nothing_pays_no_unresolved_penalty():
    """A penalidade some ao mudar de -10 para 0 só se ela estivesse sendo cobrada."""
    sem, env_a = _roda(_cfg(penalty_unresolved=-10.0), 3)
    env_a.close()
    zero, env_b = _roda(_cfg(penalty_unresolved=0.0), 3)
    env_b.close()
    assert sem == pytest.approx(zero, abs=1e-6), (
        f"'nada' não deveria pagar penalty_unresolved; {sem} != {zero}"
    )


# --- abrir e não fechar continua sendo punido -------------------------------

def test_testing_without_concluding_is_still_penalised():
    """Gastar exame e não concluir deixa a investigação aberta — isso paga."""
    com, env_a = _roda(_cfg(penalty_unresolved=-10.0), 0)   # testa dengue sempre
    inv = len(env_a.unwrapped.investigated_cases)
    conc = len(env_a.unwrapped.finalized_cases)
    env_a.close()
    sem, env_b = _roda(_cfg(penalty_unresolved=0.0), 0)
    env_b.close()

    assert inv > 0, "testar deve abrir investigação"
    assert conc < inv, "o cenário precisa deixar investigações abertas"
    assert com < sem, (
        f"investigação aberta deveria custar; com={com} sem={sem}"
    )


def test_epi_confirm_also_opens_an_investigation():
    """Ir a campo gasta recurso, então também abre investigação."""
    _total, env = _roda(_cfg(), 2)          # sempre epi_confirm
    assert len(env.unwrapped.investigated_cases) > 0
    env.close()


def test_concluding_closes_almost_every_investigation():
    """Investigar e concluir fecha quase tudo.

    "Quase": casos notificados nos últimos dias têm o laudo a caminho quando o
    episódio encerra, então a investigação fica aberta sem que o agente pudesse
    ter feito diferente. É a cauda que o `settle_days` não cobre, e ela existe
    por construção — o teste trava que seja uma cauda, não a regra.
    """
    env = make_env(_cfg())
    env.reset(seed=8)
    u = env.unwrapped
    terminated = truncated = False
    i = 0
    while not (terminated or truncated) and i < 600:
        case_id = env.current_case[0]
        if case_id in u.obs_cases.index and int(u.obs_cases.loc[case_id, "testd"]) == 0:
            acao = 0
        elif case_id in u.obs_cases.index:
            acao = 4 + int(u.obs_cases.loc[case_id, "agent_diagnosis"])
        else:
            acao = 3
        _obs, _r, terminated, truncated, _i = env.step(acao)
        i += 1

    abertos = u.investigated_cases - u.finalized_cases
    investigados = len(u.investigated_cases)
    assert investigados > 0
    assert len(abertos) / investigados < 0.25, (
        f"{len(abertos)} de {investigados} investigações abertas — deveria ser "
        f"só a cauda do fim do episódio"
    )
    # E as que sobram são mesmo as do fim: todas notificadas depois da mediana.
    # (Não basta olhar o laudo: ele pode ter chegado e o episódio ter encerrado
    # antes de o caso voltar à fila do agente.)
    mediana = float(u.obs_cases["t"].median())
    for cid in abertos:
        assert float(u.obs_cases.loc[cid, "t"]) >= mediana, (
            f"caso {cid} ficou aberto sem ser do fim do episódio"
        )
    env.close()


# --- a economia não muda de lugar ------------------------------------------

def test_invariance_between_reward_modes_still_holds():
    """A mudança é de QUEM paga, não de quanto — a soma segue igual nos 2 modos."""
    for acao in (3, 0, 2, 4):
        cfg_a = _cfg()
        cfg_a["per_case_reward"] = False
        cfg_b = _cfg()
        a, ea = _roda(cfg_a, acao)
        ea.close()
        b, eb = _roda(cfg_b, acao)
        eb.close()
        assert b == pytest.approx(a, abs=1e-6), f"divergiu para ação {acao}"


# --- "nada" é a ação nula: recompensa exatamente zero -----------------------

def test_do_nothing_returns_exactly_zero():
    """Cada passo de "nada" devolve 0.0 — nem custo, nem desfecho, nem penalidade.

    É a ação nula do agente no sentido padrão de RL. Antes custava 0,1: com
    ~373 passos por episódio isso somava uma deriva de ~-37 por episódio,
    constante e sem informação, no alvo de TD da ação mais frequente.
    """
    env = make_env(_cfg())
    env.reset(seed=4)
    terminated = truncated = False
    passos = []
    while not (terminated or truncated) and len(passos) < 400:
        _obs, r, terminated, truncated, _i = env.step(3)
        passos.append(r)
    env.close()

    assert passos, "nenhum passo executado"
    # O último passo carrega o placar final do episódio; os demais são nulos.
    assert all(r == 0.0 for r in passos[:-1]), (
        f"{sum(1 for r in passos[:-1] if r != 0.0)} passos de 'nada' não foram nulos"
    )


def test_do_nothing_cost_is_configurable():
    """O zero é default, não imposição — dá para cobrar por inação se quiser."""
    env = make_env(_cfg(do_nothing_cost=0.5))
    env.reset(seed=4)
    _obs, r, _t, _tr, _i = env.step(3)
    env.close()
    assert r == pytest.approx(-0.5)
