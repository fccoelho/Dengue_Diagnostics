"""Testes dos casos que NÃO são arbovirose (terceira classe).

A peça sutil aqui é a lógica do laudo negativo: com apenas duas doenças, um
negativo para dengue implicava logicamente chikungunya. Com três classes isso
deixa de valer, e um negativo só resolve o caso quando a outra arbovirose também
já foi descartada por exame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dengue_envs.core.clinical import (
    CHIK,
    DENGUE,
    NEGATIVE,
    NOT_TESTED,
    OTHER,
    POSITIVE,
    update_case_status,
)
from dengue_envs.wrappers import make_env

_CFG = {
    "env": {
        "size": 80,
        "episize": 60,
        "epilength": 12,
        "start_day": 1,
        "reward_delay_days": 0,
        "lab_delay_days": 0,
        "randomize_outbreak": False,
        "other_prevalence": 0.25,
    },
    "wrappers": ["map_tensor", "case_by_case"],
}


def _row(testd=NOT_TESTED, testc=NOT_TESTED, diag=DENGUE):
    return pd.DataFrame(
        [{"testd": testd, "testc": testc, "agent_diagnosis": diag}], index=[0]
    )


# --- lógica do laudo negativo -------------------------------------------------

def test_negative_dengue_alone_does_not_conclude_other():
    """Um único negativo NÃO resolve: chik segue possível."""
    obs = _row(diag=DENGUE)
    update_case_status(obs, 0, 0, NEGATIVE)
    assert obs.loc[0, "agent_diagnosis"] == CHIK


def test_two_negatives_conclude_other():
    """Negativo nos DOIS exames => não é arbovirose."""
    obs = _row(diag=DENGUE)
    update_case_status(obs, 0, 0, NEGATIVE)   # dengue descartada -> chik
    update_case_status(obs, 1, 0, NEGATIVE)   # chik descartada  -> outro
    assert obs.loc[0, "agent_diagnosis"] == OTHER


def test_two_negatives_conclude_other_reverse_order():
    """A ordem dos exames não altera a conclusão."""
    obs = _row(diag=CHIK)
    update_case_status(obs, 1, 0, NEGATIVE)
    update_case_status(obs, 0, 0, NEGATIVE)
    assert obs.loc[0, "agent_diagnosis"] == OTHER


def test_positive_still_wins_over_previous_negative():
    """Um positivo define a doença mesmo após um negativo anterior."""
    obs = _row(diag=DENGUE)
    update_case_status(obs, 0, 0, NEGATIVE)   # -> chik
    update_case_status(obs, 1, 0, POSITIVE)   # confirma chik
    assert obs.loc[0, "agent_diagnosis"] == CHIK


# --- gerador ------------------------------------------------------------------

def test_generator_produces_other_cases():
    env = make_env(_CFG)
    env.reset(seed=3)
    truth = env.unwrapped.real_cases["disease"]
    share = (truth == OTHER).mean()
    assert share > 0, "nenhum caso não-arbovirose foi gerado"
    assert 0.15 < share < 0.35, f"prevalência fora do esperado: {share:.2f}"
    env.close()


def test_no_other_cases_when_disabled():
    """Compatibilidade: com prevalência 0 o ambiente volta a ser binário."""
    cfg = {"env": dict(_CFG["env"]), "wrappers": list(_CFG["wrappers"])}
    cfg["env"]["other_prevalence"] = 0.0
    env = make_env(cfg)
    env.reset(seed=3)
    assert (env.unwrapped.real_cases["disease"] == OTHER).sum() == 0
    env.close()


def test_other_cases_are_spatially_uniform():
    """Casos não-arbovirose não se agrupam nos focos do surto."""
    cfg = {"env": dict(_CFG["env"]), "wrappers": list(_CFG["wrappers"])}
    cfg["env"].update({"size": 200, "episize": 400, "epilength": 30})
    env = make_env(cfg)
    env.reset(seed=5)
    rc = env.unwrapped.real_cases
    outros = rc[rc.disease == OTHER]
    arbo = rc[rc.disease != OTHER]
    # A dispersão espacial dos "outros" deve superar a dos arbovirais, que se
    # concentram em torno dos focos.
    assert outros.x.std() > arbo.x.std()
    env.close()


# --- consequência para a decisão ---------------------------------------------
#
# Espaço de ações conclusivas: 4=concluir DENGUE, 5=concluir CHIK,
# 6=concluir OTHER. A alegação vem da própria ação, não do `agent_diagnosis`
# corrente — é o que permite ao agente discordar do palpite clínico usando
# outra evidência (ex.: posição espacial), sem precisar testar para "editar"
# o diagnóstico.

CONCLUDE_DENGUE = 4
CONCLUDE_CHIK = 5
CONCLUDE_OTHER = 6


def test_conclude_other_is_correct_for_true_other_case():
    """Concluir 'outro' deixa de ser sempre errado: agora existe verdade 'outro'."""
    from dengue_envs.core.reward import RewardEngine

    eng = RewardEngine(reward_delay_days=0, reward_correct_decision=10.0)
    real = pd.DataFrame([{"disease": OTHER}], index=[0])
    obs = pd.DataFrame(
        [{"testd": 0, "testc": 0, "agent_diagnosis": DENGUE}], index=[0]
    )
    r = eng.compute([(0, CONCLUDE_OTHER)], t=0, real_cases=real, obs_cases=obs)
    assert r == 10.0


# --- semântica: cada ação conclusiva alega uma classe -------------------------

def _engine():
    from dengue_envs.core.reward import RewardEngine

    return RewardEngine(
        reward_delay_days=0,
        reward_correct_decision=10.0,
        penalty_incorrect_decision=-20.0,
        penalty_missed_case=-30.0,
        final_correct_bonus=0.0,
        penalty_misdiagnosed=0.0,
    )


def _score(action, true_disease):
    real = pd.DataFrame([{"disease": true_disease}], index=[0])
    # agent_diagnosis não importa para a recompensa: a alegação é a ação.
    obs = pd.DataFrame(
        [{"testd": 0, "testc": 0, "agent_diagnosis": DENGUE}], index=[0]
    )
    return _engine().compute([(0, action)], t=0, real_cases=real, obs_cases=obs)


def test_conclude_dengue_rejects_true_other():
    """Concluir DENGUE num caso que era 'outro' é uma alegação errada de
    arbovirose (não é a via correta, mas também não é o pior erro)."""
    assert _score(CONCLUDE_DENGUE, OTHER) == -20.0


def test_conclude_other_is_the_way_to_close_a_non_arbovirus_case():
    """A via correta para encerrar um caso não-arbovirose é concluir OTHER."""
    assert _score(CONCLUDE_OTHER, OTHER) == 10.0


def test_conclude_actions_reward_correct_arbovirus():
    assert _score(CONCLUDE_DENGUE, DENGUE) == 10.0
    assert _score(CONCLUDE_CHIK, CHIK) == 10.0


def test_conclude_other_on_real_disease_is_the_worst_error():
    """Concluir OTHER num caso real é falso negativo de vigilância — pior
    do que confundir dengue com chik."""
    assert _score(CONCLUDE_OTHER, DENGUE) < _score(CONCLUDE_CHIK, DENGUE)


def test_conclude_other_is_not_dominated_anymore():
    """Para um caso OUTRO, concluir OTHER passa a ser estritamente melhor do
    que alegar qualquer arbovirose."""
    assert _score(CONCLUDE_OTHER, OTHER) > _score(CONCLUDE_DENGUE, OTHER)
    assert _score(CONCLUDE_OTHER, OTHER) > _score(CONCLUDE_CHIK, OTHER)
