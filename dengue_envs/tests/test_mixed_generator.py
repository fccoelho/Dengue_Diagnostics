"""Gerador `mixed`: sorteia a distribuição espacial a cada mundo novo.

Motivação medida: o mapa é hoje decorativo — zerar o tensor inteiro muda 0–2%
das decisões do agente, porque dentro de UMA distribuição ele não acrescenta
nada às features do caso. Com duas misturadas, o mapa passa a ser a única forma
de saber em que regime se está, e os regimes pedem políticas opostas (posição
acerta a doença em 93,7% no sintético contra 47,1% no kriging).
"""
from __future__ import annotations

import collections

import pytest

from dengue_envs.wrappers import make_env
from dengue_envs.wrappers.factory import _make_world_builder

_ENV = {
    "size": 60,
    "episize": 40,
    "epilength": 8,
    "start_day": 1,
    "lab_delay_days": 1,
    "reward_delay_days": 0,
    "randomize_outbreak": False,
    "other_prevalence": 0.25,
}


def _cfg(**over):
    env = dict(_ENV)
    env.update(over)
    return {"env": env, "wrappers": ["map_tensor", "case_by_case"],
            "context_features": True, "per_case_reward": True}


def _misto(**over):
    return _cfg(
        generator="mixed",
        mix=[
            {"generator": "synthetic", "weight": 0.5},
            # Duas sintéticas em vez de kriging: o teste não deve depender das
            # superfícies pré-computadas (`results/kriging/*.npz`), que podem
            # não existir num clone limpo.
            {"generator": "synthetic", "weight": 0.5},
        ],
        **over,
    )


def test_mixed_requires_the_mix_key():
    with pytest.raises(ValueError, match="mix"):
        _make_world_builder({"generator": "mixed"})


def test_unknown_generator_lists_the_options():
    with pytest.raises(ValueError, match="mixed"):
        _make_world_builder({"generator": "inexistente"})


def test_mixed_records_which_component_was_drawn():
    """`mixture_component` é o que permite diagnosticar o comportamento por regime."""
    env = make_env(_misto())
    env.reset(seed=0)
    assert env.unwrapped.mixture_component in (0, 1)
    env.close()


def test_mixed_draws_both_components():
    env = make_env(_misto())
    vistos = collections.Counter()
    for s in range(24):
        env.reset(seed=s)
        vistos[env.unwrapped.mixture_component] += 1
    env.close()
    assert set(vistos) == {0, 1}, f"só saiu um componente: {dict(vistos)}"
    # Com pesos iguais, nenhum lado pode dominar grosseiramente.
    assert min(vistos.values()) >= 4, dict(vistos)


def test_weights_are_respected():
    """Peso zero remove a distribuição do sorteio."""
    cfg = _cfg(
        generator="mixed",
        mix=[
            {"generator": "synthetic", "weight": 1.0},
            {"generator": "synthetic", "weight": 0.0},
        ],
    )
    env = make_env(cfg)
    for s in range(10):
        env.reset(seed=s)
        assert env.unwrapped.mixture_component == 0
    env.close()


def test_all_zero_weights_is_an_error():
    with pytest.raises(ValueError, match="peso"):
        _make_world_builder({
            "generator": "mixed",
            "mix": [{"generator": "synthetic", "weight": 0.0}],
        })


def test_draw_is_reproducible_for_a_given_seed():
    """Sem isso, um experimento com `mixed` não é reproduzível."""
    env = make_env(_misto())
    primeira = []
    for s in range(8):
        env.reset(seed=s)
        primeira.append(env.unwrapped.mixture_component)
    segunda = []
    for s in range(8):
        env.reset(seed=s)
        segunda.append(env.unwrapped.mixture_component)
    env.close()
    assert primeira == segunda


def test_episode_runs_end_to_end_under_mixture():
    """O episódio precisa rodar igual, venha o mundo de onde vier."""
    env = make_env(_misto())
    for s in (1, 2, 3):
        obs, _ = env.reset(seed=s)
        u = env.unwrapped
        terminated = truncated = False
        i = 0
        while not (terminated or truncated) and i < 300:
            case_id = env.current_case[0]
            obs, _r, terminated, truncated, _i = env.step(
                0 if case_id in u.obs_cases.index else 3
            )
            i += 1
        assert i > 0 and len(u.obs_cases) > 0
    env.close()


def test_observation_space_is_identical_across_components():
    """Se o espaço mudasse entre componentes, a rede não poderia ser a mesma."""
    misto = make_env(_misto())
    puro = make_env(_cfg())
    assert misto.observation_space["map"].shape == puro.observation_space["map"].shape
    assert misto.observation_space["context"].shape == puro.observation_space["context"].shape
    assert misto.action_space.n == puro.action_space.n
    misto.close()
    puro.close()
