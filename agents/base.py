"""Contrato comum dos agentes + esquema de resultados comparável.

Todos os agentes (random, q-learning, dqn, ppo, ...) devem produzir resultados
no MESMO formato, para que os CSVs gerados pelo benchmark sejam diretamente
comparáveis entre si.

Convenções:
- Todos os agentes são avaliados sobre o MESMO ambiente (env bruto + wrappers
  `map_tensor` + `case_by_case`), construído pela fábrica única
  `dengue_envs.wrappers.make_env`.
- As métricas de episódio vêm de `env.unwrapped.get_episode_metrics()`, que já
  é padronizado no ambiente.
- Cada episódio vira uma linha (dict) com as colunas de `RESULT_COLUMNS`.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Protocol, runtime_checkable

import numpy as np

# Métrica PRIMÁRIA de ranqueamento (a mais fiel ao objetivo real).
PRIMARY_METRIC = "Recompensa Total"

# Colunas de identificação adicionadas a cada linha de resultado.
ID_COLUMNS = ["agent", "seed", "clinical_specificity"]

# Métricas de episódio (devem casar com as chaves de get_episode_metrics()).
# "Recompensa Total" é a primária; as demais são diagnóstico secundário.
METRIC_COLUMNS = [
    "Recompensa Total",
    "Acurácia Multiclasse",
    "Acurácia",
    "Sensibilidade (Dengue)",
    "Especificidade",
    "F1-Score",
    "Precisão",
    "Custo Total de Testes",
    "Testes Realizados",
    "Custo por Acerto",
    "Redução de Testes (%)",
]

# Ordem canônica das colunas do CSV de resultados (mesma para todos os agentes).
RESULT_COLUMNS = ID_COLUMNS + METRIC_COLUMNS

# Uma fábrica de ambiente: recebe um dict de config e devolve o env (com wrappers).
EnvFactory = Callable[..., object]


@runtime_checkable
class AgentRunner(Protocol):
    """Interface que todo agente implementa para ser avaliado no benchmark."""

    name: str

    def evaluate(
        self, make_env: EnvFactory, seeds: List[int], env_config: dict
    ) -> List[Dict]:
        """Roda um episódio por seed e retorna uma linha de métricas por episódio.

        Contrato:
        - constrói o ambiente com `make_env(env_config)` (mesmo env para todos);
        - garante reprodutibilidade por seed;
        - devolve dicts com as chaves de `RESULT_COLUMNS`.
        """
        ...


class EpisodeRunner:
    """Base para agentes sem treino que decidem uma ação por caso.

    Implementa o laço de avaliação padronizado (reprodutível por seed, mesmo
    ambiente, métricas de `get_episode_metrics`). Subclasses só precisam
    definir `name` e `choose_action(env)`.
    """

    name = "base"

    def choose_action(self, env) -> int:  # pragma: no cover - sobrescrito
        raise NotImplementedError

    def evaluate(
        self, make_env: EnvFactory, seeds: List[int], env_config: dict
    ) -> List[Dict]:
        rows: List[Dict] = []
        env = make_env(env_config)
        for seed in seeds:
            seed = int(seed)
            obs, info = env.reset(seed=seed)
            env.action_space.seed(seed)

            terminated = truncated = False
            while not (terminated or truncated):
                action = self.choose_action(env)
                obs, reward, terminated, truncated, info = env.step(action)

            metrics = env.unwrapped.get_episode_metrics()
            row: Dict = {
                "agent": self.name,
                "seed": seed,
                "clinical_specificity": float(env.unwrapped.clinical_specificity),
            }
            row.update(metrics)
            rows.append(row)
            env.close()
        return rows

    def watch(
        self,
        env_config: dict,
        *,
        seed: int = 100,
        make_env=None,
        render_fps: int = 10,
    ) -> Dict:
        """Assiste a UM episódio deste agente com renderização (janela Pygame).

        Usa o mesmo ambiente/wrappers do benchmark, mas com `render_mode="human"`.
        Como o agente decide por `choose_action(env)`, aqui só embrulhamos isso
        no laço de visualização compartilhado.
        """
        from agents.watch import run_watch

        return run_watch(
            env_config,
            act_fn=lambda obs, env: self.choose_action(env),
            seed=seed,
            make_env=make_env,
            render_fps=render_fps,
        )
