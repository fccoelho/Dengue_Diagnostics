"""Testes do PPO — concentrados no que quebra em silêncio.

O risco maior aqui não é o algoritmo: é a **máscara de ação**. O
`DiscreteQLearningPolicy` do Tianshou aplica `obs.mask` sozinho, mas a
`ProbabilisticActorPolicy` que o PPO usa **não** — o `forward` dela chama o
ator e amostra da distribuição, sem consultar máscara nenhuma. Com
`force_decision_after_tests` ligado, deixar assim removeria em silêncio a
restrição que obriga a concluir depois de dois exames, e o PPO estaria
resolvendo um problema mais fácil que o DQN — invalidando a comparação sem
erro nenhum aparecer.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from agents.ppo.agent import PPOAgent, _infer_pooled_size
from agents.ppo.network import DengueTrunk, build_actor_critic
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
    "force_decision_after_tests": True,
}


def _cfg():
    return {
        "env": dict(_ENV),
        "wrappers": ["map_tensor", "case_by_case"],
        "context_features": True,
        "per_case_reward": True,
    }


def _obs_batch(obs, **over):
    b = {k: np.asarray(v)[None] for k, v in obs.items()}
    b.update(over)
    return b


# --- máscara de ação --------------------------------------------------------

def test_actor_blocks_masked_actions():
    """Ação proibida recebe -inf e nunca é amostrada."""
    env = make_env(_cfg())
    obs, _ = env.reset(seed=1)
    ator, _critico = build_actor_critic(env, device="cpu")

    # Máscara que só permite as conclusivas, como após dois exames.
    b = _obs_batch(obs, mask=np.array([[0, 0, 0, 0, 1, 1, 1]]))
    logits, _ = ator(b)
    assert torch.isinf(logits[0, :4]).all(), "ações proibidas devem ser -inf"

    amostras = Categorical(logits=logits).sample((500,)).numpy().ravel()
    assert set(amostras.tolist()) <= {4, 5, 6}, "vazou ação proibida"
    env.close()


def test_masked_action_is_never_the_argmax():
    """A avaliação usa a moda; ela também precisa respeitar a máscara."""
    env = make_env(_cfg())
    obs, _ = env.reset(seed=2)
    ator, _c = build_actor_critic(env, device="cpu")
    for proibida in range(7):
        mask = np.ones((1, 7), dtype=np.int8)
        mask[0, proibida] = 0
        logits, _ = ator(_obs_batch(obs, mask=mask))
        assert int(logits.argmax()) != proibida
    env.close()


def test_actor_works_without_mask_in_the_observation():
    """Sem `force_decision_after_tests` não há `mask`; não pode quebrar."""
    cfg = _cfg()
    cfg["env"]["force_decision_after_tests"] = False
    env = make_env(cfg)
    obs, _ = env.reset(seed=3)
    assert "mask" not in obs
    ator, _c = build_actor_critic(env, device="cpu")
    logits, _ = ator(_obs_batch(obs))
    assert logits.shape == (1, 7)
    assert torch.isfinite(logits).all()
    env.close()


# --- arquitetura ------------------------------------------------------------

def test_trunk_matches_the_dqn_encoder():
    """A comparação DQN x PPO só é justa se o tronco for o mesmo."""
    from agents.deepq.network import DengueNet

    map_shape = (6, 60, 60)
    dqn = DengueNet(map_shape, 7, context_dim=16)
    trunk = DengueTrunk(map_shape, context_dim=16)
    assert trunk.output_dim == dqn.head[0].in_features
    # Mesmos módulos de encoder, não apenas mesmas dimensões.
    assert type(trunk.map_encoder) is type(dqn.map_encoder)
    assert len(trunk.map_encoder) == len(dqn.map_encoder)


def test_actor_and_critic_shapes():
    env = make_env(_cfg())
    obs, _ = env.reset(seed=4)
    ator, critico = build_actor_critic(env, device="cpu")
    b = _obs_batch(obs)
    logits, _ = ator(b)
    valor = critico(b)
    assert logits.shape == (1, 7)
    assert valor.shape == (1,)
    env.close()


def test_critic_does_not_share_the_trunk_by_default():
    """Compartilhar acopla os gradientes de política e valor; não é o default."""
    env = make_env(_cfg())
    ator, critico = build_actor_critic(env, device="cpu")
    assert ator.trunk is not critico.trunk
    ator2, critico2 = build_actor_critic(env, device="cpu", share_trunk=True)
    assert ator2.trunk is critico2.trunk
    env.close()


# --- checkpoint -------------------------------------------------------------

def test_pooled_size_is_inferred_from_the_checkpoint():
    """Carregar não pode exigir saber com que config se treinou."""
    env = make_env(_cfg())
    ator, critico = build_actor_critic(env, device="cpu", pooled_size=4)
    estado = {f"policy.actor.{k}": v for k, v in ator.state_dict().items()}
    assert _infer_pooled_size(estado) == 4
    env.close()


def test_agent_roundtrip_preserves_behaviour(tmp_path):
    """Salvar e recarregar deve devolver exatamente as mesmas ações."""
    env = make_env(_cfg())
    obs, _ = env.reset(seed=5)
    ator, critico = build_actor_critic(env, device="cpu")

    # Espelha o formato salvo pelo treino: state_dict do algoritmo inteiro.
    estado = {f"policy.actor.{k}": v for k, v in ator.state_dict().items()}
    estado.update({f"critic.{k}": v for k, v in critico.state_dict().items()})
    caminho = tmp_path / "p.pth"
    torch.save(estado, caminho)

    recarregado = PPOAgent.load(caminho, env, device="cpu")
    antes = PPOAgent(ator)
    for _ in range(20):
        assert recarregado.choose_action(obs) == antes.choose_action(obs)
        obs, _r, t, tr, _i = env.step(antes.choose_action(obs))
        if t or tr:
            break
    env.close()


def test_load_fails_loudly_without_actor_weights():
    """Checkpoint de outro formato deve dizer o que está errado."""
    env = make_env(_cfg())
    caminho = None
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        caminho = Path(d) / "ruim.pth"
        torch.save({"model.head.0.weight": torch.zeros(7, 8)}, caminho)
        with pytest.raises(ValueError, match="nenhum peso de ator"):
            PPOAgent.load(caminho, env, device="cpu")
    env.close()


def test_ppo_is_registered_in_the_benchmark():
    """Sem isso o PPO não entra no benchmark com as mesmas seeds dos demais."""
    from experiments.evaluate import AGENT_REGISTRY

    assert AGENT_REGISTRY["ppo"].__name__ == "PPOAgentRunner"
