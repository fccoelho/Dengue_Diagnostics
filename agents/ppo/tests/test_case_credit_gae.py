"""GAE por caso com desconto em dias.

A propriedade central — a razão de existir do `per_case_gae` — é que a
vantagem de uma decisão sobre o caso A **não depende** de quantas decisões
sobre outros pacientes aconteceram entre os passos de A. No GAE padrão ela
depende, e medimos que isso apaga o sinal (correlação -0,019 contra 0,994).
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.ppo.credit import per_case_gae

G, LAM = 0.99, 0.95


def _gae(r, v, day, done, uid, cid):
    return per_case_gae(
        np.array(r, float), np.array(v, float), np.array(day),
        np.array(done, bool), np.array(uid), np.array(cid), G, LAM,
    )


def test_single_case_matches_hand_computation():
    # Caso 7: exame no dia 1 (r=-4), conclusão no dia 6 (r=+10), terminado.
    adv, ret = _gae(r=[-4, 10], v=[1.0, 2.0], day=[1, 6], done=[False, True],
                    uid=[1, 1], cid=[7, 7])
    desc = G ** 5
    delta_1 = 10 + 0 - 2.0                      # último: sem futuro
    delta_0 = -4 + desc * 2.0 - 1.0
    assert adv[1] == pytest.approx(delta_1)
    assert adv[0] == pytest.approx(delta_0 + desc * LAM * delta_1)
    assert ret == pytest.approx(adv + np.array([1.0, 2.0]))


def test_discount_is_by_days_not_by_steps():
    """Dois passos do mesmo caso separados por 6 dias usam gamma**6."""
    adv, _ = _gae(r=[0, 10], v=[0, 0], day=[1, 7], done=[False, True],
                  uid=[1, 1], cid=[3, 3])
    assert adv[0] == pytest.approx((G ** 6) * LAM * 10)


def test_interleaving_other_patients_does_not_change_a_case_advantage():
    """A propriedade que motivou tudo."""
    sozinho, _ = _gae(r=[-4, 10], v=[1.0, 2.0], day=[1, 6],
                      done=[False, True], uid=[1, 1], cid=[7, 7])

    # O mesmo caso 7, agora com 30 decisões sobre outros pacientes no meio,
    # com recompensas grandes e ruidosas — exatamente o ambiente real.
    rng = np.random.default_rng(0)
    outros = 30
    r = [-4] + list(rng.normal(0, 20, outros)) + [10]
    v = [1.0] + list(rng.normal(0, 5, outros)) + [2.0]
    day = [1] + list(np.linspace(1, 6, outros).astype(int)) + [6]
    done = [False] + [True] * outros + [True]
    uid = [1] * (outros + 2)
    cid = [7] + list(range(100, 100 + outros)) + [7]

    intercalado, _ = _gae(r, v, day, done, uid, cid)
    assert intercalado[0] == pytest.approx(sozinho[0])
    assert intercalado[-1] == pytest.approx(sozinho[1])


def test_truncated_case_bootstraps_from_its_own_value():
    """O lote cortou a linha do tempo do caso: bootstrap com o próprio V."""
    adv, _ = _gae(r=[-4], v=[3.0], day=[1], done=[False], uid=[1], cid=[5])
    assert adv[0] == pytest.approx(-4 + G * 3.0 - 3.0)


def test_finished_case_has_no_future():
    adv, _ = _gae(r=[-4], v=[3.0], day=[1], done=[True], uid=[1], cid=[5])
    assert adv[0] == pytest.approx(-4 - 3.0)


def test_same_case_id_in_different_episodes_is_not_merged():
    """ids de caso reiniciam a cada episódio; o `episode_uid` os separa."""
    adv, _ = _gae(r=[-4, 10], v=[0, 0], day=[1, 6], done=[True, True],
                  uid=[111, 222], cid=[7, 7])
    # Se fossem fundidos, o primeiro receberia o +10 descontado do segundo.
    assert adv[0] == pytest.approx(-4)
    assert adv[1] == pytest.approx(10)


def test_steps_without_a_case_are_isolated():
    adv, _ = _gae(r=[5, 0, 7], v=[0, 0, 0], day=[1, 1, 2], done=[False, True, True],
                  uid=[1, 1, 1], cid=[4, 0, 4])
    assert adv[1] == pytest.approx(0.0)           # preenchimento, sozinho
    assert adv[0] == pytest.approx(5 + (G ** 1) * LAM * 7)


def test_returns_are_advantage_plus_value():
    rng = np.random.default_rng(1)
    n = 40
    r, v = rng.normal(0, 5, n), rng.normal(0, 5, n)
    day = np.sort(rng.integers(0, 30, n))
    cid = rng.integers(1, 8, n)
    adv, ret = _gae(r, v, day, rng.random(n) < 0.3, np.ones(n, int), cid)
    assert np.allclose(ret, adv + v)


def test_per_case_ppo_requires_credit_fields():
    """Sem `per_case_reward` o `info` não traz os campos — falha dizendo por quê."""
    import torch
    from tianshou.data import Batch

    from agents.ppo.credit import PerCasePPO

    class _Falso:
        max_batchsize = 8
        gamma, gae_lambda, credit_scale = G, LAM, 1.0

        def critic(self, obs):
            return torch.zeros(len(obs["x"]))

    lote = Batch(obs={"x": np.zeros((3, 1))}, info={"algo": np.zeros(3)})
    with pytest.raises(KeyError, match="per_case_reward"):
        PerCasePPO._add_returns_and_advantages(_Falso(), lote, None, None)
