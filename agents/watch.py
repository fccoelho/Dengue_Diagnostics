"""Visualização de agentes agindo no ambiente (fora do benchmark).

Fornece um laço de renderização REUTILIZÁVEL por todos os agentes. Cada agente
injeta apenas a sua função de decisão; janela, tratamento de eventos, passos e
resumo do episódio são comuns.

Dois estilos de decisão são aceitos:
- `act_fn(obs, env) -> int`: função pronta (ex.: baselines sem treino);
- `setup(env) -> act_fn`: fábrica que constrói a função a partir do ambiente já
  criado (ex.: DQN, que precisa das shapes do env para montar/-carregar a rede).

O ambiente é sempre construído com `render_mode="human"` e a mesma pilha de
wrappers do benchmark (via `dengue_envs.wrappers.make_env`), então o que você vê
é exatamente o que o agente enxerga na avaliação.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import yaml

from dengue_envs.wrappers import make_env as default_make_env

# Assinatura da função de decisão: recebe a observação atual e o env, devolve a
# ação discreta (0..5) para o caso corrente.
ActFn = Callable[[object, object], int]
SetupFn = Callable[[object], ActFn]


def load_env_config(path: str) -> dict:
    """Carrega um YAML de ambiente (mesmo formato usado pelo benchmark)."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_env(env_config: dict, make_env, render_fps: int):
    """Constrói o env com `render_mode='human'` e ajusta o FPS de exibição."""
    config = dict(env_config)
    env_section = dict(config.get("env", {}))
    env_section["render_mode"] = "human"
    config["env"] = env_section

    env = make_env(config)
    renderer = getattr(env.unwrapped, "renderer", None)
    if renderer is not None:
        renderer.render_fps = render_fps
    return env


def _new_cases_on_day(env, day: int) -> int:
    """Quantos casos foram reportados (novos) no dia `day`."""
    df = getattr(env.unwrapped, "obs_cases", None)
    if df is None or df.empty:
        return 0
    return int((df["t"] == day).sum())


def _print_outbreak_info(env, seed: int) -> None:
    """Imprime parâmetros do surto gerado neste episódio."""
    u = env.unwrapped
    day1 = _new_cases_on_day(env, u.t)
    print(
        f"[watch] seed={seed} | dengue center={u.dengue_center} r0={u.dengue_r0:.2f} | "
        f"chik center={u.chik_center} r0={u.chik_r0:.2f} | "
        f"dia {u.t}: {day1} novos casos"
    )


def _print_summary(steps: int, total_reward: float, metrics: Dict) -> None:
    """Resumo com rótulos ASCII (evita erro de encoding no console Windows)."""
    print("\n=== Resumo do episodio ===")
    print(f"  Passos                : {steps}")
    print(f"  Recompensa (acumulada): {total_reward:.2f}")
    print(f"  Recompensa (env)      : {metrics.get('Recompensa Total', float('nan')):.2f}")
    print(f"  Acuracia multiclasse  : {metrics.get('Acurácia Multiclasse', 0.0):.3f}")
    print(f"  Acuracia binaria      : {metrics.get('Acurácia', 0.0):.3f}")
    print(f"  Testes realizados     : {int(metrics.get('Testes Realizados', 0))}")


def run_watch(
    env_config: dict,
    act_fn: Optional[ActFn] = None,
    *,
    setup: Optional[SetupFn] = None,
    seed: Optional[int] = 100,
    make_env=None,
    render_fps: int = 10,
    max_steps: Optional[int] = None,
) -> Dict:
    """Roda UM episódio com renderização e devolve as métricas finais.

    Forneça `act_fn` OU `setup` (exatamente um dos dois). A mesma `seed` passada
    ao ``env.reset`` reproduz o mesmo surto (focos + curva SIR + posições).
    """
    import pygame

    if (act_fn is None) == (setup is None):
        raise ValueError("Forneça exatamente um entre `act_fn` e `setup`.")

    make_env = make_env or default_make_env

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))

    env = _build_env(env_config, make_env, render_fps)

    if act_fn is None:
        act_fn = setup(env)

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    terminated = truncated = False
    total_reward = 0.0
    steps = 0

    print(f"[watch] iniciando (fps={render_fps}). Feche a janela ou use Ctrl+C para sair.")
    _print_outbreak_info(env, seed)

    # Reporta novos casos quando o dia avança.
    last_day = env.unwrapped.t
    try:
        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("[watch] janela fechada pelo usuario.")
                    terminated = True
            if terminated:
                break

            action = act_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            day = env.unwrapped.t
            if day != last_day:
                print(f"[watch] dia {day}: {_new_cases_on_day(env, day)} novos casos")
                last_day = day

            if max_steps is not None and steps >= max_steps:
                print(f"[watch] max_steps={max_steps} atingido.")
                break
    except KeyboardInterrupt:
        print("\n[watch] interrompido (Ctrl+C).")

    metrics = env.unwrapped.get_episode_metrics()
    _print_summary(steps, total_reward, metrics)
    env.close()
    return metrics
