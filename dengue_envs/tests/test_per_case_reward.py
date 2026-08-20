"""Testes da atribuição de recompensa por caso (`per_case_reward`).

Motivação: no modo agregado (histórico), o `CaseByCaseWrapper` acumula as ações
do dia e só as aplica quando o último caso do dia é decidido. Até lá cada passo
devolve 0.0, e a consequência do dia inteiro cai num único passo — misturada ao
desfecho de dezenas de outros casos. Medido nesse regime: ~90% da variância da
recompensa de um passo vinha de decisões tomadas dias antes, sobre outros casos.

O modo por caso muda **quando** o crédito é entregue, não **quanto**. O teste
central aqui é justamente esse: a recompensa total do episódio tem de ser
idêntica nos dois modos para a mesma sequência de ações.
"""
from __future__ import annotations

import numpy as np
import pytest

from dengue_envs.wrappers import make_env

# Os dois modos somam as mesmas parcelas, mas em ORDEM diferente (por caso vs.
# acumulado por dia), então a igualdade é exata em teoria e aproximada em ponto
# flutuante — divergências observadas ficam na casa de 1e-14.
_TOL = 1e-6

_BASE_ENV = {
    "size": 80,
    "episize": 60,
    "epilength": 12,
    "start_day": 1,
    "lab_delay_days": 2,
    "randomize_outbreak": False,
    "other_prevalence": 0.25,
    "max_case_revisits": 2,
}


def _cfg(per_case: bool, reward_delay: int = 0) -> dict:
    env = dict(_BASE_ENV)
    env["reward_delay_days"] = reward_delay
    return {
        "env": env,
        "wrappers": ["map_tensor", "case_by_case"],
        "per_case_reward": per_case,
    }


def _run(cfg: dict, acoes, seed: int = 7):
    """Roda um episódio aplicando `acoes` ciclicamente; devolve (total, passos)."""
    env = make_env(cfg)
    env.reset(seed=seed)
    total = 0.0
    passos = []
    i = 0
    terminated = truncated = False
    while not (terminated or truncated):
        _obs, r, terminated, truncated, _info = env.step(acoes[i % len(acoes)])
        total += r
        passos.append(r)
        i += 1
    env.close()
    return total, passos


# --- invariância: a soma do episódio não muda -------------------------------

def test_total_reward_is_identical_between_modes():
    """A refatoração redistribui o crédito no tempo, não altera a economia."""
    acoes = [0, 4, 1, 5, 6]  # mistura investigação e conclusões das 3 classes
    total_agregado, _ = _run(_cfg(per_case=False), acoes)
    total_por_caso, _ = _run(_cfg(per_case=True), acoes)
    assert total_por_caso == pytest.approx(total_agregado, abs=_TOL)


def test_total_reward_identical_for_several_policies():
    for acoes in ([3], [4], [6], [0, 4], [0, 1, 6], [2, 5]):
        a = _run(_cfg(per_case=False), acoes)[0]
        b = _run(_cfg(per_case=True), acoes)[0]
        assert b == pytest.approx(a, abs=_TOL), f"divergiu para {acoes}: {b} != {a}"


def test_total_reward_identical_with_reward_delay():
    """Com atraso > 0 o desfecho ainda vai para a fila; a soma segue igual."""
    acoes = [0, 4, 1, 6]
    a = _run(_cfg(per_case=False, reward_delay=3), acoes)[0]
    b = _run(_cfg(per_case=True, reward_delay=3), acoes)[0]
    assert b == pytest.approx(a, abs=_TOL)


# --- atribuição: o crédito chega no passo certo -----------------------------

def test_aggregated_mode_leaves_most_steps_at_zero():
    """Caracteriza o problema que motivou a mudança."""
    _total, passos = _run(_cfg(per_case=False), [4])
    zeros = sum(1 for r in passos if r == 0.0)
    assert zeros / len(passos) > 0.5, (
        "esperado que o modo agregado deixe a maioria dos passos sem sinal"
    )


def test_per_case_mode_pays_on_the_decision_step():
    """No modo por caso, concluir rende no passo da própria conclusão."""
    _total, passos = _run(_cfg(per_case=True), [4])
    zeros = sum(1 for r in passos if r == 0.0)
    assert zeros / len(passos) < 0.1, (
        "no modo por caso quase todo passo conclusivo deve carregar seu desfecho"
    )


def test_per_case_reward_matches_decision_outcome():
    """Concluir corretamente rende ~+reward_correct_decision naquele passo."""
    cfg = _cfg(per_case=True)
    cfg["env"]["reward_correct_decision"] = 10.0
    cfg["env"]["shaping_conclude_bonus"] = 0.0
    env = make_env(cfg)
    env.reset(seed=3)
    u = env.unwrapped

    vistos = []
    terminated = truncated = False
    while not (terminated or truncated) and len(vistos) < 40:
        case_id = env.current_case[0]
        verdade = (
            int(u.real_cases.loc[case_id, "disease"])
            if case_id in u.real_cases.index
            else None
        )
        acao = 4 + verdade if verdade is not None else 3
        _obs, r, terminated, truncated, _i = env.step(acao)
        if verdade is not None:
            vistos.append(r)
    env.close()

    # Concluindo sempre a classe VERDADEIRA, os passos devem ser positivos.
    assert vistos, "nenhum caso avaliado"
    assert np.mean(vistos) > 0, f"média não positiva: {np.mean(vistos)}"


def test_reward_delay_does_not_change_episode_total():
    """O atraso muda QUANDO o desfecho é pago, não QUANTO se ganha no episódio.

    Regressão de um vazamento real: `_advance_empty_days` chamava
    `env.step(())` e **descartava** a recompensa devolvida. Em dias sem casos
    ativos é justamente quando vencem desfechos agendados, então quanto maior o
    `reward_delay_days`, mais recompensa era silenciosamente perdida (medido:
    114 de 444 num único episódio com atraso de 5).
    """
    acoes = [0, 4, 1, 6]

    def cfg_com_atraso(atraso: int) -> dict:
        c = _cfg(per_case=True, reward_delay=atraso)
        # `settle_days` é derivado de max(reward_delay, lab_delay) e define o
        # horizonte; fixá-lo mantém o episódio do mesmo tamanho, isolando a
        # contabilidade da recompensa do comprimento do episódio.
        c["env"]["settle_days"] = 5
        return c

    sem_atraso = _run(cfg_com_atraso(0), acoes)[0]
    for atraso in (1, 3, 5):
        com_atraso = _run(cfg_com_atraso(atraso), acoes)[0]
        assert com_atraso == pytest.approx(sem_atraso, abs=_TOL), (
            f"atraso={atraso} alterou o total do episódio: "
            f"{com_atraso} != {sem_atraso}"
        )


def test_per_case_mode_is_opt_in():
    """Sem a flag, o comportamento histórico é preservado."""
    env = make_env(_cfg(per_case=False))
    assert env.per_case_reward is False
    env.close()
    env = make_env(_cfg(per_case=True))
    assert env.per_case_reward is True
    env.close()
