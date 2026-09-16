"""Decomposição por caso da recompensa e features temporais.

O que se trava aqui é a HONESTIDADE da decomposição. A recompensa do ambiente
não muda — é a métrica do benchmark. O wrapper só passa a informar, em `info`,
quanto de cada passo pertence ao caso decidido (`r_case`), re-temporizando a
parcela do placar final para o passo em que o caso termina.

Se a decomposição inventasse ou perdesse recompensa, o aprendiz otimizaria um
problema diferente do medido. Por isso a identidade central:

    soma(r_case) + parcelas finais dos casos que o episódio fechou em aberto
        == soma(recompensa do ambiente)

Motivação medida (kriging v8): o retorno global que o PPO usa tem correlação
-0,019 com a utilidade de um exame; o retorno do próprio caso, 0,994.
"""
from __future__ import annotations

import numpy as np
import pytest

from dengue_envs.wrappers import make_env

_ENV = {
    "size": 60,
    "episize": 40,
    "epilength": 10,
    "start_day": 1,
    "lab_delay_days": 2,
    "reward_delay_days": 0,
    "randomize_outbreak": False,
    "other_prevalence": 0.25,
    "max_case_revisits": 2,
    "epi_radius": 10,
    "epi_threshold": 0,
    "force_decision_after_tests": True,
}


def _cfg(**over):
    env = dict(_ENV)
    temporal = over.pop("temporal_features", False)
    env.update(over)
    return {
        "env": env,
        "wrappers": ["map_tensor", "case_by_case"],
        "context_features": True,
        "per_case_reward": True,
        "temporal_features": temporal,
    }


def _politica(rng):
    def escolher(env):
        u = env.unwrapped
        cid = env.current_case[0]
        if cid not in u.obs_cases.index:
            return 3
        row = u.obs_cases.loc[cid]
        mask = u.action_mask(cid)
        if int(row["testd"]) == 0 and mask[0] and rng.random() < 0.5:
            return 0
        if rng.random() < 0.3 and mask[3]:
            return 3
        return 4 + int(row["agent_diagnosis"])
    return escolher


def _episodio(cfg, seed, rng_seed=0):
    env = make_env(cfg)
    env.reset(seed=seed)
    escolher = _politica(np.random.default_rng(rng_seed))
    total, soma_rcase, infos = 0.0, 0.0, []
    terminated = truncated = False
    passos = 0
    while not (terminated or truncated) and passos < 2000:
        _o, r, terminated, truncated, info = env.step(escolher(env))
        total += r
        soma_rcase += info["r_case"]
        infos.append(info)
        passos += 1
    return env, total, soma_rcase, infos


def _cauda(env):
    """Parcelas finais dos casos que o episódio encerrou sem atribuir."""
    u = env.unwrapped
    faltam = [c for c in u.obs_cases.index if c not in u._terminal_attributed]
    return sum(
        u.reward_engine.terminal_contribution(
            c, u.real_cases, u.obs_cases,
            concluded=u.finalized_cases, investigated=u.investigated_cases,
        )
        for c in faltam
    ), len(faltam)


# --- a identidade central ---------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_decomposition_reconstructs_the_total_reward(seed):
    env, total, soma_rcase, _ = _episodio(_cfg(), seed)
    cauda, _n = _cauda(env)
    env.close()
    assert soma_rcase + cauda == pytest.approx(total, abs=1e-6), (
        f"a decomposição inventou ou perdeu recompensa: "
        f"r_case={soma_rcase} + cauda={cauda} != total={total}"
    )


def test_decomposition_does_not_change_the_reward():
    """Ligar as features novas não pode alterar a recompensa do ambiente."""
    _e1, total_a, _s, _i = _episodio(_cfg(), seed=7)
    _e1.close()
    _e2, total_b, _s, _i = _episodio(_cfg(temporal_features=True), seed=7)
    _e2.close()
    assert total_a == pytest.approx(total_b, abs=1e-9)


def test_each_case_gets_its_final_share_at_most_once():
    env, _t, _s, infos = _episodio(_cfg(), seed=5)
    feitos = [i["case_id"] for i in infos if i["case_done"] and i["case_id"] != 0]
    env.close()
    # Um caso pode aparecer como "feito" só uma vez com parcela final; o
    # conjunto interno garante isso, e ele precisa bater com a contagem.
    assert len(set(feitos)) == len(env.unwrapped._terminal_attributed & set(feitos))


def test_the_open_tail_is_a_minority():
    """A cauda não atribuída deve ser exceção, não regra."""
    env, _t, _s, _i = _episodio(_cfg(), seed=6)
    _cauda_valor, n_abertos = _cauda(env)
    n_casos = len(env.unwrapped.obs_cases)
    env.close()
    assert n_abertos / max(1, n_casos) < 0.5


# --- info consistente -------------------------------------------------------

def test_every_step_carries_the_credit_fields():
    """O buffer do Tianshou empilha `info`; chave faltando quebra o agrupamento."""
    env, _t, _s, infos = _episodio(_cfg(), seed=8)
    env.close()
    chaves = {"case_id", "day", "r_case", "case_done", "episode_uid"}
    for info in infos:
        assert chaves <= set(info), f"faltou: {chaves - set(info)}"


def test_episode_uid_is_constant_within_and_distinct_across_episodes():
    env = make_env(_cfg())
    uids = []
    for seed in (1, 2):
        env.reset(seed=seed)
        vistos = set()
        terminated = truncated = False
        n = 0
        while not (terminated or truncated) and n < 400:
            _o, _r, terminated, truncated, info = env.step(3)
            vistos.add(info["episode_uid"])
            n += 1
        assert len(vistos) == 1
        uids.append(vistos.pop())
    env.close()
    assert uids[0] != uids[1]


def test_day_never_goes_backwards():
    env, _t, _s, infos = _episodio(_cfg(), seed=9)
    env.close()
    dias = [i["day"] for i in infos]
    assert all(b >= a for a, b in zip(dias, dias[1:]))


# --- features temporais -----------------------------------------------------

def test_temporal_features_extend_the_context():
    env = make_env(_cfg(temporal_features=True))
    obs, _ = env.reset(seed=1)
    assert obs["context"].shape == (20,)
    assert env.observation_space["context"].shape == (20,)
    env.close()


def test_temporal_features_are_off_by_default():
    env = make_env(_cfg())
    obs, _ = env.reset(seed=1)
    assert obs["context"].shape == (16,)
    env.close()


def test_temporal_features_require_context_features():
    cfg = _cfg(temporal_features=True)
    cfg["context_features"] = False
    with pytest.raises(ValueError, match="context_features"):
        make_env(cfg)


def test_temporal_features_stay_in_unit_range():
    env = make_env(_cfg(temporal_features=True))
    env.reset(seed=2)
    terminated = truncated = False
    n = 0
    while not (terminated or truncated) and n < 400:
        obs, _r, terminated, truncated, _i = env.step(0 if n % 2 else 3)
        tempo = obs["context"][16:]
        assert np.all(tempo >= 0.0) and np.all(tempo <= 1.0), tempo
        n += 1
    env.close()


def test_day_fraction_advances_with_the_epidemic():
    env = make_env(_cfg(temporal_features=True))
    obs, _ = env.reset(seed=3)
    inicio = float(obs["context"][16])
    terminated = truncated = False
    n = 0
    while not (terminated or truncated) and n < 400:
        obs, _r, terminated, truncated, _i = env.step(3)
        n += 1
    fim = float(obs["context"][16])
    env.close()
    assert fim > inicio


def test_temporal_features_never_read_the_true_curve():
    """Só contagens de casos NOTIFICADOS — a curva real seria vazamento."""
    import inspect

    from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv

    fonte = inspect.getsource(DengueDiagnosticsEnv.temporal_features)
    for proibido in ("real_cases", "world", "dengue_curve", "chik_curve", "casedf"):
        assert proibido not in fonte, f"temporal_features lê {proibido}"


# --- funciona também com o SEIR ---------------------------------------------

def test_decomposition_holds_under_the_seir_model():
    cfg = _cfg(epi_model="seir", episize=120, epilength=160)
    env, total, soma_rcase, _ = _episodio(cfg, seed=11)
    cauda, _n = _cauda(env)
    env.close()
    assert soma_rcase + cauda == pytest.approx(total, abs=1e-6)
