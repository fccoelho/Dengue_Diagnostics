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
    # Propaga a flag de features de contexto (evidência sobre o médico), venha
    # ela do YAML do ambiente ou do YAML de treino.
    if "context_features" in cfg:
        env_cfg["context_features"] = cfg["context_features"]
    # Escala de recompensa: só existe no treino (ver RewardScaleWrapper).
    if "reward_scale" in cfg:
        env_cfg["reward_scale"] = cfg["reward_scale"]
    return env_cfg


class EnvFactory:
    """Fábrica de ambientes **picklável** — requisito do `SubprocVectorEnv`.

    No Windows o `SubprocVectorEnv` usa `spawn`: a fábrica é serializada e
    reconstruída dentro de cada processo filho. Uma closure definida dentro de
    `main()` não é picklável (`Can't pickle local object`), então a fábrica
    precisa ser um objeto de módulo com estado serializável — aqui, só o dict
    de configuração.
    """

    def __init__(self, env_config: dict):
        self.env_config = env_config

    def __call__(self):
        return make_env(self.env_config)


def _make_vector_env(factory: Callable[[], Any], n: int, *, kind: str = "dummy"):
    if n <= 0:
        raise ValueError("num_envs deve ser >= 1")
    if kind == "subproc":
        return SubprocVectorEnv([factory for _ in range(n)])
    return DummyVectorEnv([factory for _ in range(n)])


def _install_cheap_hasnull(buffer) -> None:
    """Substitui ``buffer.hasnull()`` por uma checagem de NaN leve em memória.

    O ``hasnull()`` padrão do Tianshou é chamado a CADA passo de treino e
    materializa + faz ``deepcopy`` do buffer inteiro para procurar NaN. Com o
    mapa de observação 4x400x400, isso aloca ~2-3x o tamanho do buffer por passo
    e estoura a RAM (ex.: bool (9000,4,400,400) = 5,4 GB só na cópia).

    Como o mapa é ``uint8`` (nunca NaN), basta checar os campos float pequenos —
    ``rew`` e ``obs.case_coords`` (poucos KB) — sem tocar no mapa. Se a estrutura
    do buffer não bater com o esperado, não bloqueia o treino (retorna False).
    """

    def _hasnull() -> bool:
        try:
            if np.isnan(np.asarray(buffer.rew)).any():
                return True
            coords = np.asarray(buffer.obs.case_coords)
            if np.isnan(coords).any():
                return True
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return False

    buffer.hasnull = _hasnull


def train(config_path: str) -> Path:
    config_path = Path(config_path).resolve()
    cfg = _load_yaml(config_path)
    env_config = _resolve_env_config(cfg, config_path)
    train_cfg = cfg.get("train", {})

    seed = int(train_cfg.get("seed", 42))
    epochs = int(train_cfg.get("epochs", 50))
    n_step = int(train_cfg.get("n_step", 5))
    batch_size = int(train_cfg.get("batch_size", 64))
    # RAM do replay buffer ~= buffer_size * bytes_por_obs (mapa uint8).
    # Ex.: mapa 4x400x400 = 640 KB -> 10k = 6,4 GB; 20k = 12,8 GB.
    buffer_size = int(train_cfg.get("buffer_size", 10000))
    lr = float(train_cfg.get("lr", 1e-4))
    gamma = float(train_cfg.get("gamma", 0.99))
    target_update_freq = int(train_cfg.get("target_update_freq", 1500))
    # Perda do erro de TD. `None` = MSE (default do Tianshou), em que o
    # gradiente cresce LINEARMENTE com o erro — um alvo aberrante domina o
    # passo. O Huber satura o gradiente a partir de `delta`, que é o
    # estabilizador usado no DQN da Nature. Medido nesta sessão com MSE: a
    # recompensa de teste oscilava ~6.300 pontos entre avaliações vizinhas
    # depois de 100 mil passos.
    huber_loss_delta = train_cfg.get("huber_loss_delta")
    if huber_loss_delta is not None:
        huber_loss_delta = float(huber_loss_delta)
    # Double DQN já é o default do Tianshou 2.x (`is_double=True`); explicitado
    # aqui para ficar visível na configuração, não escondido num default.
    is_double = bool(train_cfg.get("is_double", True))
    step_per_epoch = int(train_cfg.get("step_per_epoch", 10000))
    step_per_collect = int(train_cfg.get("step_per_collect", 1000))
    update_per_step = float(train_cfg.get("update_per_step", 0.1))
    eps_start = float(train_cfg.get("eps_train_start", 1.0))
    eps_final = float(train_cfg.get("eps_train_final", 0.05))
    eps_decay = int(train_cfg.get("eps_train_decay", 50000))
    eps_test = float(train_cfg.get("eps_test", 0.01))
    num_train_envs = int(train_cfg.get("num_train_envs", 2))
    num_test_envs = int(train_cfg.get("num_test_envs", 2))
    # Nº de episódios por avaliação. NÃO deve ser amarrado a `num_test_envs`:
    # a recompensa varia MUITO com a competência do médico sorteado (de +1992 a
    # -2326 por episódio), então avaliar com 2 episódios mede qual médico calhou,
    # não a qualidade da política — e o `save_best_fn` acaba salvando um sorteio
    # de sorte. Com ~20 episódios o ruído do médico é mediado.
    test_episodes = int(train_cfg.get("test_episodes", 20))
    vector_env = str(train_cfg.get("vector_env", "dummy")).lower()
    prefill_steps = int(train_cfg.get("prefill_steps", 1000))
    # Capacidade por env do buffer de teste (~1 episódio; ver uso abaixo).
    test_episode_capacity = int(train_cfg.get("test_buffer_per_env", 1000))
    device = train_cfg.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda"):
        # Autotune dos algoritmos de convolução: o shape do mapa é fixo durante
        # todo o treino, então o cuDNN escolhe o kernel mais rápido uma vez.
        torch.backends.cudnn.benchmark = True

    output_dir = Path(cfg.get("output_dir", "results/dqn"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "policy_best.pth"
    final_path = output_dir / "policy_final.pth"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    env_factory = EnvFactory(env_config)

    print(f"[dqn] device={device} | epochs={epochs} | n_step={n_step}")
    print(
        f"[dqn] lr={lr} | double={is_double} | "
        f"perda={'huber(%g)' % huber_loss_delta if huber_loss_delta else 'mse'}"
    )
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
    # Estimativa de RAM do replay buffer (obs = mapa uint8; obs_next é
    # reconstruído via ignore_obs_next, então NÃO é duplicado em memória).
    map_space = dummy_env.observation_space.spaces["map"]
    obs_bytes = int(np.prod(map_space.shape)) * np.dtype(map_space.dtype).itemsize
    buffer_gb = buffer_size * obs_bytes / 1e9
    test_buffer_gb = max(1000, test_episode_capacity) * num_test_envs * obs_bytes / 1e9
    print(
        f"[dqn] replay buffer: {buffer_size} transições x {obs_bytes/1024:.0f} KB "
        f"= ~{buffer_gb:.1f} GB | buffer de teste ~{test_buffer_gb:.1f} GB "
        f"| total ~{buffer_gb + test_buffer_gb:.1f} GB (obs em {map_space.dtype})"
    )
    # Resolução espacial que a rede enxerga (lado do AdaptiveAvgPool2d). É o
    # que governa a fidelidade espacial — ver §18 do handoff.
    pooled_size = train_cfg.get("pooled_size")
    policy = build_policy(
        dummy_env,
        device=device,
        eps_training=eps_start,
        eps_inference=eps_test,
        pooled_size=pooled_size,
    )
    dummy_env.close()

    # Currículo: continua o treino a partir de uma policy já treinada (ex.: num
    # ambiente mais simples). Exige o MESMO action_shape/context_dim — que é o
    # caso ao alternar apenas `env_config` mantendo `context_features: true`,
    # já que a rede não lê `mask` (isso é tratado pelo Tianshou fora da rede).
    init_from = train_cfg.get("init_from")
    if init_from:
        init_path = Path(init_from)
        state = torch.load(str(init_path), map_location=device, weights_only=True)
        policy.load_state_dict(state)
        print(f"[dqn] pesos iniciais carregados de: {init_path.resolve()}")

    algorithm = DQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=lr),
        gamma=gamma,
        n_step_return_horizon=n_step,
        target_update_freq=target_update_freq,
        is_double=is_double,
        huber_loss_delta=huber_loss_delta,
    )

    buffer = VectorReplayBuffer(
        total_size=buffer_size,
        buffer_num=num_train_envs,
        ignore_obs_next=True,
    )
    train_collector = Collector(
        algorithm, train_envs, buffer, exploration_noise=True
    )
    # Evita o hasnull() padrão do Tianshou, que a cada passo materializa e
    # deepcopia o buffer inteiro (OOM com o mapa 400x400). Patch no buffer que
    # o trainer realmente usa (o do collector).
    _install_cheap_hasnull(train_collector.buffer)
    # ATENÇÃO: sem um buffer explícito, o Tianshou aloca
    # DEFAULT_BUFFER_MAXSIZE (10 000) * n_envs para o collector de teste — com o
    # mapa 4x400x400 isso são ~12,8 GB só para avaliar (mais que o buffer de
    # treino!). A avaliação só precisa guardar os episódios de teste em curso,
    # então dimensionamos pelo tamanho real de um episódio.
    test_buffer = VectorReplayBuffer(
        total_size=max(1000, test_episode_capacity) * num_test_envs,
        buffer_num=num_test_envs,
        ignore_obs_next=True,
    )
    test_collector = Collector(algorithm, test_envs, test_buffer)

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
        test_step_num_episodes=test_episodes,
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
