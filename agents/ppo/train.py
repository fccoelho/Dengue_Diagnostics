"""Treina PPO (Tianshou 2.x) no ambiente DengueDiag via YAML.

Por que PPO, depois de tudo o que o DQN mostrou:

- O DQN converge (com Huber, ±9 entre seeds) para uma política degenerada que
  **nunca compra informação** — 66,7% de inação, zero exames.
- Três ataques falharam pela mesma causa: shaping por potencial (Φ telescopa
  porque ``s'`` já é outro paciente), crédito retroativo (exigiria reescrever
  recompensa passada) e calibrar a economia (dobrar o prêmio de investigar não
  mudou nada — zero exames em 3 seeds).

O PPO ataca isso por outro ângulo: otimiza a política **diretamente**, com
bônus de entropia que sustenta exploração, e estima vantagem por GAE — que
espalha o crédito ao longo da trajetória de forma mais suave que o retorno de
n passos. Não elimina a intercalação de casos, mas não depende de uma função Q
estável sobre ela.

Saídas em ``output_dir``: policy_best.pth, policy_final.pth, logs/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OnPolicyTrainer, OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger

from agents.deepq.train import EnvFactory, _load_yaml, _make_vector_env, _resolve_env_config
from agents.ppo.network import build_actor_critic

_DEFAULT_CONFIG = "experiments/configs/train/ppo_v1_s42.yaml"


def train(config_path: str) -> Path:
    config_path = Path(config_path).resolve()
    cfg = _load_yaml(config_path)
    env_config = _resolve_env_config(cfg, config_path)
    tcfg = cfg.get("train", {})

    seed = int(tcfg.get("seed", 42))
    epochs = int(tcfg.get("epochs", 10))
    step_per_epoch = int(tcfg.get("step_per_epoch", 10000))
    # No on-policy, cada ciclo COLETA um lote e o consome; não há replay buffer
    # de onde reamostrar. `step_per_collect` é o tamanho desse lote.
    step_per_collect = int(tcfg.get("step_per_collect", 2000))
    repeat_per_collect = int(tcfg.get("repeat_per_collect", 10))
    batch_size = int(tcfg.get("batch_size", 256))
    lr = float(tcfg.get("lr", 3e-4))
    gamma = float(tcfg.get("gamma", 0.99))
    gae_lambda = float(tcfg.get("gae_lambda", 0.95))
    eps_clip = float(tcfg.get("eps_clip", 0.2))
    # Entropia é a alavanca de exploração do PPO — o análogo do epsilon, mas
    # que age sobre a POLÍTICA, não sobre a ação escolhida. Dado que o DQN
    # colapsava em "nada", vale começar acima do padrão do Tianshou (0.01).
    ent_coef = float(tcfg.get("ent_coef", 0.01))
    vf_coef = float(tcfg.get("vf_coef", 0.5))
    max_grad_norm = tcfg.get("max_grad_norm", 0.5)
    max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
    num_train_envs = int(tcfg.get("num_train_envs", 8))
    num_test_envs = int(tcfg.get("num_test_envs", 10))
    test_episodes = int(tcfg.get("test_episodes", 20))
    vector_env = str(tcfg.get("vector_env", "dummy")).lower()
    pooled_size = tcfg.get("pooled_size")
    device = tcfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    output_dir = Path(cfg.get("output_dir", "results/ppo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "policy_best.pth"
    final_path = output_dir / "policy_final.pth"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    factory = EnvFactory(env_config)
    train_envs = _make_vector_env(factory, num_train_envs, kind=vector_env)
    test_envs = _make_vector_env(factory, num_test_envs, kind=vector_env)
    train_envs.seed(seed)
    test_envs.seed(seed)

    dummy_env = factory()
    ator, critico = build_actor_critic(
        dummy_env, device=device, pooled_size=pooled_size
    )
    print(f"[ppo] device={device} | epochs={epochs} | ent_coef={ent_coef}")
    print(f"[ppo] lr={lr} | gae_lambda={gae_lambda} | eps_clip={eps_clip}")
    print(f"[ppo] checkpoint -> {best_path.resolve()}")

    policy = DiscreteActorPolicy(
        actor=ator,
        action_space=dummy_env.action_space,
        observation_space=dummy_env.observation_space,
        # Avaliação determinística (moda da distribuição) — o análogo do ε=0
        # que o benchmark do DQN usa, para que a comparação seja justa.
        deterministic_eval=True,
    )
    algorithm = PPO(
        policy=policy,
        critic=critico,
        optim=AdamOptimizerFactory(lr=lr),
        gamma=gamma,
        gae_lambda=gae_lambda,
        eps_clip=eps_clip,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
    )

    # On-policy: o buffer guarda só o lote corrente, então dimensiona por
    # `step_per_collect` — nada a ver com o replay de milhares de transições
    # do DQN.
    train_buffer = VectorReplayBuffer(
        total_size=max(step_per_collect, num_train_envs),
        buffer_num=num_train_envs,
        ignore_obs_next=True,
    )
    train_collector = Collector(algorithm, train_envs, train_buffer)
    test_buffer = VectorReplayBuffer(
        total_size=1000 * num_test_envs,
        buffer_num=num_test_envs,
        ignore_obs_next=True,
    )
    test_collector = Collector(algorithm, test_envs, test_buffer)

    writer = SummaryWriter(str(log_dir))
    logger = TensorboardLogger(writer)

    def save_best_fn(_policy) -> None:
        torch.save(algorithm.state_dict(), best_path)

    params = OnPolicyTrainerParams(
        train_collector=train_collector,
        test_collector=test_collector,
        max_epochs=epochs,
        epoch_num_steps=step_per_epoch,
        collection_step_num_env_steps=step_per_collect,
        update_step_num_repetitions=repeat_per_collect,
        test_step_num_episodes=test_episodes,
        batch_size=batch_size,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
    )
    result = OnPolicyTrainer(algorithm, params).run()
    torch.save(algorithm.state_dict(), final_path)

    print(f"\n[ppo] melhor : {best_path.resolve()}")
    print(f"[ppo] final  : {final_path.resolve()}")
    print(f"[ppo] logs   : {log_dir.resolve()}")
    print(result)
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if any(v is not None for v in (args.epochs, args.seed, args.output_dir)):
        cfg = _load_yaml(Path(args.config).resolve())
        cfg.setdefault("train", {})
        if args.epochs is not None:
            cfg["train"]["epochs"] = args.epochs
        if args.seed is not None:
            cfg["train"]["seed"] = args.seed
        if args.output_dir is not None:
            cfg["output_dir"] = args.output_dir
        tmp = Path(args.config).resolve().parent / "_ppo_override.yaml"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        train(str(tmp))
        tmp.unlink(missing_ok=True)
    else:
        train(args.config)


if __name__ == "__main__":
    main()
