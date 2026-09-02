"""Testes das features de contexto (evidência sobre a competência do médico).

A competência do médico é a variável latente que domina o retorno do episódio.
Estes testes garantem que (a) a estatística observável a estima de fato e
(b) o comportamento antigo continua intacto quando a flag está desligada.
"""
from __future__ import annotations

import numpy as np

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
    },
    "wrappers": ["map_tensor", "case_by_case"],
}


def _cfg(context: bool) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in _CFG.items()}
    cfg["context_features"] = context
    return cfg


def test_context_absent_by_default():
    """Sem a flag, a observação continua exatamente como antes."""
    env = make_env(_CFG)
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"map", "case_coords"}
    assert "context" not in env.observation_space.spaces
    env.close()


def test_context_present_when_enabled():
    """O contexto reúne evidência sobre o médico (2) + atributos do caso (14)."""
    env = make_env(_cfg(True))
    obs, _ = env.reset(seed=0)
    assert "context" in obs
    assert obs["context"].shape == (16,)
    # Sem laudo ainda: taxa neutra (0,5) e evidência zero.
    assert np.allclose(obs["context"][:2], [0.5, 0.0])
    env.close()


def test_case_features_expose_current_case_state():
    """O agente precisa enxergar o caso que está decidindo, não só onde ele está.

    Os atributos existem no mapa, mas o pooling do encoder os dissolve; por isso
    entram explicitamente no vetor de contexto.
    """
    env = make_env(_cfg(True))
    obs, _ = env.reset(seed=0)
    feats = obs["context"][2:]
    assert feats.shape == (14,)
    # One-hot de diagnóstico (3), de cada exame (4+4), o epiconf (1) e a
    # densidade local de confirmados dengue/chik (2).
    assert feats[:3].sum() == 1.0, "diagnóstico clínico deve estar codificado"
    assert feats[3:7].sum() == 1.0, "status do exame de dengue deve estar codificado"
    assert feats[7:11].sum() == 1.0, "status do exame de chik deve estar codificado"
    # Antes de qualquer exame, ambos devem estar em "não testado" (posição 0).
    assert feats[3] == 1.0 and feats[7] == 1.0
    env.close()


def test_case_features_change_after_a_test():
    """Depois de um exame, o vetor do caso reflete o novo status."""
    cfg = _cfg(True)
    cfg["env"]["lab_delay_days"] = 0
    env = make_env(cfg)
    env.reset(seed=1)
    obs, _r, _t, _tr, _i = env.step(0)  # testa dengue
    # Em algum momento do episódio algum caso deixa de estar "não testado".
    visto = obs["context"][3] == 0.0
    for _ in range(60):
        obs, _r, term, trunc, _i = env.step(0)
        visto = visto or obs["context"][3] == 0.0
        if term or trunc:
            break
    assert visto, "o status do exame nunca apareceu nas features do caso"
    env.close()


def test_evidence_accumulates_only_with_informative_results():
    """A força da evidência cresce conforme laudos informativos voltam."""
    env = make_env(_cfg(True))
    env.reset(seed=3)
    terminated = truncated = False
    while not (terminated or truncated):
        _obs, _r, terminated, truncated, _i = env.step(0)  # testa todos
    u = env.unwrapped
    assert u.lab_informative > 0
    assert 0 <= u.lab_agreements <= u.lab_informative
    rate, strength = u.clinical_evidence()
    assert 0.0 <= rate <= 1.0
    assert 0.0 < strength <= 1.0
    env.close()


def test_agreement_rate_tracks_clinical_quality():
    """A estimativa observável acompanha a competência real (latente) do médico."""
    reais, estimados = [], []
    for spec in (0.55, 0.95):
        cfg = _cfg(True)
        cfg["env"]["clinical_specificity"] = spec
        env = make_env(cfg)
        env.reset(seed=7)
        terminated = truncated = False
        while not (terminated or truncated):
            _obs, _r, terminated, truncated, _i = env.step(0)
        rate, _strength = env.unwrapped.clinical_evidence()
        reais.append(spec)
        estimados.append(rate)
        env.close()
    # Médico melhor => maior concordância entre laudo e palpite clínico.
    assert estimados[1] > estimados[0]


def test_counters_reset_between_episodes():
    env = make_env(_cfg(True))
    env.reset(seed=1)
    terminated = truncated = False
    while not (terminated or truncated):
        _obs, _r, terminated, truncated, _i = env.step(0)
    assert env.unwrapped.lab_informative > 0
    obs, _ = env.reset(seed=2)
    assert env.unwrapped.lab_informative == 0
    assert env.unwrapped.lab_agreements == 0
    assert np.allclose(obs["context"][:2], [0.5, 0.0])
    env.close()
