"""Agente DQN (Tianshou 2.x) no ambiente novo (map_tensor + case_by_case).

Decide **uma ação por caso** (`Discrete(7)`), no mesmo contrato do random /
Q-Learning. `action_shape` é lido de `env.action_space.n` (ver `build_policy`),
então a rede se adapta automaticamente ao tamanho do espaço de ação.
Checkpoint = ``state_dict`` da ``DiscreteQLearningPolicy``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.data import Batch

from agents.base import EpisodeRunner
from agents.deepq.network import DengueNet

DEFAULT_CHECKPOINT = Path("results/dqn/policy_best.pth")
NUM_ACTIONS = 7


def observation_from_env(env) -> dict:
    """Reconstrói a obs atual do ``CaseByCaseWrapper`` (sem precisar do laço)."""
    wrapper = env
    while wrapper is not None:
        if hasattr(wrapper, "_make_obs") and hasattr(wrapper, "_current_map_obs"):
            if wrapper._current_map_obs is None:
                raise RuntimeError(
                    "Observação do CaseByCaseWrapper ainda não foi inicializada; "
                    "chame env.reset() antes de choose_action."
                )
            return wrapper._make_obs()
        wrapper = getattr(wrapper, "env", None)
    raise TypeError(
        "Ambiente esperado com CaseByCaseWrapper (make_env com map_tensor + case_by_case)."
    )


def build_policy(
    env,
    *,
    device: str,
    eps_training: float = 0.0,
    eps_inference: float = 0.0,
    pooled_size: Optional[int] = None,
) -> DiscreteQLearningPolicy:
    """Cria uma ``DiscreteQLearningPolicy`` + ``DengueNet`` para o espaço do env.

    ``pooled_size`` é o lado do ``AdaptiveAvgPool2d`` do encoder — a resolução
    espacial que a rede **de fato enxerga**, independente do tamanho da
    observação. É o parâmetro que governa a fidelidade espacial: com o bbox do
    Rio (72,3 km), `pooled_size=6` significa células de ~12 km. Para um mapa
    real em que a estrutura fina importe, é este o botão a mexer, não
    `map_size` (ver §18 de REFATORACAO_AMBIENTE.md).
    """
    map_shape = env.observation_space.spaces["map"].shape
    action_shape = env.action_space.n
    # Detecta o ramo de contexto pelo próprio espaço de observação, para que o
    # mesmo código sirva com e sem `context_features`.
    ctx_space = env.observation_space.spaces.get("context")
    context_dim = int(ctx_space.shape[0]) if ctx_space is not None else 0
    kwargs = {} if pooled_size is None else {"pooled_size": int(pooled_size)}
    net = DengueNet(
        map_shape, action_shape, device=device, context_dim=context_dim, **kwargs
    ).to(device)
    return DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        observation_space=env.observation_space,
        eps_training=eps_training,
        eps_inference=eps_inference,
    )


def _infer_pooled_size(state: dict) -> Optional[int]:
    """Deduz o `pooled_size` de um checkpoint pela forma da projeção do mapa."""
    for chave in ("model.map_proj.0.weight", "map_proj.0.weight"):
        peso = state.get(chave)
        if peso is None:
            continue
        lado = int(round((peso.shape[1] / 64) ** 0.5))
        if lado > 0 and 64 * lado * lado == peso.shape[1]:
            return lado
    return None


def load_policy(
    path: Union[str, Path],
    env,
    *,
    device: Optional[str] = None,
) -> DiscreteQLearningPolicy:
    """Carrega pesos ``.pth`` em uma policy pronta para avaliação (ε=0)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Policy DQN não encontrada: {path.resolve()}. "
            "Treine com `poetry run python agents/deepq/train.py`."
        )
    state = torch.load(str(path), map_location=device, weights_only=True)
    # `pooled_size` muda a arquitetura, então precisa bater com o checkpoint.
    # Em vez de exigir que quem carrega saiba disso, inferimos a partir da
    # primeira camada da projeção: in_features = 64 * pooled_size^2.
    pooled = _infer_pooled_size(state)
    policy = build_policy(
        env, device=device, eps_training=0.0, eps_inference=0.0,
        pooled_size=pooled,
    )
    # Aceita state_dict da policy ou só da rede (`model.*`).
    try:
        policy.load_state_dict(state)
    except RuntimeError:
        policy.model.load_state_dict(state)
    policy.eval()
    policy.set_eps_training(0.0)
    policy.set_eps_inference(0.0)
    return policy


class DQNAgent:
    """Wrapper fino sobre a policy para decidir ações a partir do env."""

    def __init__(self, policy: DiscreteQLearningPolicy):
        self.policy = policy

    def choose_action(self, obs: dict) -> int:
        batch = Batch(obs=[obs], info={})
        return int(self.policy(batch).act[0])

    def choose_action_from_env(self, env) -> int:
        return self.choose_action(observation_from_env(env))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env,
        *,
        device: Optional[str] = None,
    ) -> "DQNAgent":
        return cls(load_policy(path, env, device=device))


class DQNAgentRunner(EpisodeRunner):
    """Runner de benchmark: carrega ``.pth`` e age greedy (ε=0)."""

    name = "dqn"

    def __init__(
        self,
        *,
        policy_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.policy_path = Path(policy_path) if policy_path else DEFAULT_CHECKPOINT
        self.device = device
        self._agent: Optional[DQNAgent] = None
        self._bound_env_id: Optional[int] = None

    def _ensure_agent(self, env) -> DQNAgent:
        env_id = id(env)
        if self._agent is None or self._bound_env_id != env_id:
            if self.device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            if self.policy_path.exists():
                self._agent = DQNAgent.load(self.policy_path, env, device=device)
            else:
                print(
                    f"[dqn] aviso: policy ausente em {self.policy_path.resolve()}. "
                    "Usando rede aleatória (comportamento sem treino)."
                )
                policy = build_policy(
                    env, device=device, eps_training=0.0, eps_inference=0.0
                )
                policy.eval()
                self._agent = DQNAgent(policy)
            self._bound_env_id = env_id
        return self._agent

    def choose_action(self, env) -> int:
        return self._ensure_agent(env).choose_action_from_env(env)
