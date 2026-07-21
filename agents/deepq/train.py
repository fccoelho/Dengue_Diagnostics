"""Treina DQN (Tianshou 2.x) no ambiente novo via YAML.

Uso (da raiz do repositório):
    poetry run python agents/deepq/train.py
    poetry run python agents/deepq/train.py --config experiments/configs/train/dqn_delay5.yaml

Saídas em ``output_dir`` (default ``results/dqn/``):
    policy_best.pth, policy_final.pth, tensorboard logs/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tianshou.algorithm.modelfree.dqn import DQN
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger

from agents.deepq.agent import build_policy
from dengue_envs.wrappers import make_env

_DEFAULT_CONFIG = "experiments/configs/train/dqn_delay5.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_env_config(cfg: dict, config_path: Path) -> dict:
    """Aceita ``env_config: ../env/...yaml`` (estilo qlearning) ou ``include`` legado."""
    if "env_config" in cfg:
        env_path = (config_path.parent / cfg["env_config"]).resolve()
        env_cfg = _load_yaml(env_path)
    elif "include" in cfg:
        env_cfg: Dict[str, Any] = {"env": {}, "wrappers": ["map_tensor", "case_by_case"]}
        for rel in cfg["include"]:
            inc = _load_yaml((config_path.parent / rel).resolve())
            if "env" in inc:
                env_cfg["env"].update(inc["env"])
            if "wrappers" in inc:
                env_cfg["wrappers"] = inc["wrappers"]
    else:
        env_cfg = {"env": dict(cfg.get("env", {})), "wrappers": cfg.get("wrappers")}

    if "env" in cfg:
        env_cfg.setdefault("env", {}).update(cfg["env"])
    if "wrappers" in cfg:
        env_cfg["wrappers"] = cfg["wrappers"]
    return env_cfg


def _make_vector_env(factory: Callable[[], Any], n: int, *, kind: str = "dummy"):
    if n <= 0:
        raise ValueError("num_envs deve ser >= 1")
    if kind == "subproc":
        return SubprocVectorEnv([factory for _ in range(n)])
    return DummyVectorEnv([factory for _ in range(n)])


def train(config_path: str) -> Path:
    config_path = Path(config_path).resolve()
    cfg = _load_yaml(config_path)
    env_config = _resolve_env_config(cfg, config_path)
    train_cfg = cfg.get("train", {})

    seed = int(train_cfg.get("seed", 42))
    epochs = int(train_cfg.get("epochs", 50))
    n_step = int(train_cfg.get("n_step", 5))
    batch_size = int(train_cfg.get("batch_size", 64))
    buffer_size = int(train_cfg.get("buffer_size", 20000))
    lr = float(train_cfg.get("lr", 1e-4))
    gamma = float(train_cfg.get("gamma", 0.99))
    target_update_freq = int(train_cfg.get("target_update_freq", 1500))
    step_per_epoch = int(train_cfg.get("step_per_epoch", 10000))
    step_per_collect = int(train_cfg.get("step_per_collect", 1000))
    update_per_step = float(train_cfg.get("update_per_step", 0.1))
    eps_start = float(train_cfg.get("eps_train_start", 1.0))
    eps_final = float(train_cfg.get("eps_train_final", 0.05))
    eps_decay = int(train_cfg.get("eps_train_decay", 50000))
    eps_test = float(train_cfg.get("eps_test", 0.01))
    num_train_envs = int(train_cfg.get("num_train_envs", 2))
    num_test_envs = int(train_cfg.get("num_test_envs", 2))
    vector_env = str(train_cfg.get("vector_env", "dummy")).lower()
    prefill_steps = int(train_cfg.get("prefill_steps", 1000))
    device = train_cfg.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(cfg.get("output_dir", "results/dqn"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "policy_best.pth"
    final_path = output_dir / "policy_final.pth"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    def env_factory():
        return make_env(env_config)

    print(f"[dqn] device={device} | epochs={epochs} | n_step={n_step}")
    print(
        f"[dqn] env reward_delay_days="
        f"{env_config.get('env', {}).get('reward_delay_days')}"
    )
    print(f"[dqn] checkpoint -> {best_path.resolve()}")

    train_envs = _make_vector_env(env_factory, num_train_envs, kind=vector_env)
    test_envs = _make_vector_env(env_factory, num_test_envs, kind=vector_env)
    train_envs.seed(seed)
    test_envs.seed(seed)

    dummy_env = env_factory()
    policy = build_policy(
        dummy_env,
        device=device,
        eps_training=eps_start,
        eps_inference=eps_test,
    )
    dummy_env.close()

    algorithm = DQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=lr),
        gamma=gamma,
        n_step_return_horizon=n_step,
        target_update_freq=target_update_freq,
    )

    buffer = VectorReplayBuffer(
        total_size=buffer_size,
        buffer_num=num_train_envs,
        ignore_obs_next=True,
    )
    train_collector = Collector(
        algorithm, train_envs, buffer, exploration_noise=True
    )
    test_collector = Collector(algorithm, test_envs)

    print(f"[dqn] pré-populando buffer ({prefill_steps} steps)...")
    train_collector.collect(n_step=prefill_steps, reset_before_collect=True)

    writer = SummaryWriter(str(log_dir))
    logger = TensorboardLogger(writer)

    def train_fn(epoch: int, env_step: int) -> None:
        if env_step <= eps_decay:
            eps = eps_start - env_step / eps_decay * (eps_start - eps_final)
        else:
            eps = eps_final
        policy.set_eps_training(eps)

    def test_fn(epoch: int, env_step: int) -> None:
        policy.set_eps_inference(eps_test)

    def save_best_fn(algo) -> None:
        torch.save(algo.policy.state_dict(), best_path)
        print(f"[dqn] melhor policy salva: {best_path}")

    params = OffPolicyTrainerParams(
        max_epochs=epochs,
        epoch_num_steps=step_per_epoch,
        collection_step_num_env_steps=step_per_collect,
        update_step_num_gradient_steps_per_sample=update_per_step,
        batch_size=batch_size,
        train_collector=train_collector,
        test_collector=test_collector,
        test_step_num_episodes=num_test_envs,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )
    trainer = OffPolicyTrainer(algorithm=algorithm, params=params)
    trainer.run()

    torch.save(policy.state_dict(), final_path)
    if not best_path.exists():
        torch.save(policy.state_dict(), best_path)

    train_envs.close()
    test_envs.close()
    writer.close()

    print("[dqn] treino concluído.")
    print(f"[dqn] best : {best_path.resolve()}")
    print(f"[dqn] final: {final_path.resolve()}")
    print(f"[dqn] logs : {log_dir.resolve()}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar DQN (Tianshou) no DengueDiag.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="YAML de treino.")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescreve epochs.")
    parser.add_argument("--output-dir", default=None, help="Sobrescreve output_dir.")
    parser.add_argument("--seed", type=int, default=None, help="Sobrescreve seed.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if any(v is not None for v in (args.epochs, args.output_dir, args.seed)):
        cfg = _load_yaml(config_path)
        cfg.setdefault("train", {})
        if args.epochs is not None:
            cfg["train"]["epochs"] = args.epochs
        if args.seed is not None:
            cfg["train"]["seed"] = args.seed
        if args.output_dir is not None:
            cfg["output_dir"] = args.output_dir
        tmp = config_path.parent / "_dqn_cli_override.yaml"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        config_path = tmp

    os.chdir(_REPO_ROOT)
    train(str(config_path))


if __name__ == "__main__":
    main()
