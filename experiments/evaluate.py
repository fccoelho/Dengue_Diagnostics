"""Benchmark de agentes: avalia um ou mais agentes no MESMO ambiente e salva
resultados comparáveis.

Uso:
    poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml

Saídas (na pasta `output_dir` do YAML), todas com o mesmo esquema de colunas:
    - benchmark_raw.csv    : uma linha por (agente, seed)
    - benchmark_mean.csv   : média das métricas por agente
    - benchmark_std.csv    : desvio-padrão das métricas por agente

Para adicionar novos agentes no futuro, basta registrá-los em AGENT_REGISTRY.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Garante que a raiz do repositório esteja no sys.path para que `import agents`
# funcione mesmo rodando o script diretamente (`python experiments/evaluate.py`).
# O pacote `dengue_envs` é instalado pelo poetry; `agents` não, por isso este bootstrap.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import yaml

from dengue_envs.wrappers import make_env

from agents.base import ID_COLUMNS, PRIMARY_METRIC, RESULT_COLUMNS
from agents.clinical.agent import ClinicalOnlyAgentRunner
from agents.deepq.agent import DQNAgentRunner
from agents.ppo.agent import PPOAgentRunner
from agents.qlearning.agent import QLearningAgentRunner
from agents.random.agent import RandomAgentRunner
from agents.testall.agent import TestAllAgentRunner
from agents.confirmall.agent import ConfirmAllAgentRunner
from agents.testonce.agent import TestOnceAgentRunner
from agents.testtwice.agent import TestTwiceAgentRunner

# Registro de agentes disponíveis (nome -> classe runner).
# Novos algoritmos (ppo, ...) entram aqui conforme migrados.
AGENT_REGISTRY = {
    "clinical": ClinicalOnlyAgentRunner,
    "random": RandomAgentRunner,
    "testall": TestAllAgentRunner,
    "confirmall": ConfirmAllAgentRunner,
    "testonce": TestOnceAgentRunner,
    "testtwice": TestTwiceAgentRunner,
    "qlearning": QLearningAgentRunner,
    "dqn": DQNAgentRunner,
    "ppo": PPOAgentRunner,
}


def _make_runner(name: str, bench: dict):
    """Instancia o runner; agentes com checkpoint usam `checkpoints` no YAML."""
    cls = AGENT_REGISTRY[name]
    checkpoints = bench.get("checkpoints", {})
    if name == "qlearning":
        state_cfg = bench.get("qlearning_state") or bench.get("state")
        return cls(
            q_table_path=checkpoints.get("qlearning"),
            state_config=state_cfg,
        )
    if name in ("dqn", "ppo"):
        return cls(policy_path=checkpoints.get(name))
    return cls()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coloca as colunas na ordem canônica (extras vão para o fim)."""
    ordered = [c for c in RESULT_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in RESULT_COLUMNS]
    return df[ordered + extras]


def _validate_benchmark_config(bench: dict, config_path: Path) -> None:
    """Exige chaves do YAML de benchmark; detecta uso acidental de YAML de treino."""
    missing = [k for k in ("env_config", "agents", "seeds") if k not in bench]
    if not missing:
        return
    hint = ""
    if "train" in bench or "state" in bench and "agents" not in bench:
        hint = (
            "\n\nParece um YAML de *treino* (ex.: qlearning_default.yaml). "
            "Para avaliar agentes use:\n"
            "  poetry run python experiments/evaluate.py "
            "--config experiments/configs/benchmark.yaml"
        )
    raise ValueError(
        f"Config de benchmark inválido: {config_path}\n"
        f"Chaves obrigatórias ausentes: {missing}.{hint}"
    )


def run_benchmark(config_path: str) -> pd.DataFrame:
    config_path = Path(config_path).resolve()
    bench = load_yaml(config_path)
    _validate_benchmark_config(bench, config_path)

    # O caminho do env é relativo ao arquivo de benchmark.
    env_config_path = (config_path.parent / bench["env_config"]).resolve()
    env_config = load_yaml(env_config_path)

    seeds: List[int] = bench["seeds"]
    agent_names: List[str] = bench["agents"]

    # `output_dir` é relativo à raiz do repositório (cwd ao rodar o comando).
    output_dir = Path(bench.get("output_dir", "results/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Por padrão o benchmark salva tudo: CSVs + mapas de confusão + mapas da epidemia.
    save_artifacts = bool(bench.get("save_artifacts", True))

    all_rows: List[Dict] = []
    for name in agent_names:
        if name not in AGENT_REGISTRY:
            raise ValueError(
                f"Agente desconhecido: {name!r}. "
                f"Disponíveis: {sorted(AGENT_REGISTRY)}"
            )
        runner = _make_runner(name, bench)
        print(f"[benchmark] avaliando '{name}' em {len(seeds)} episódios...")
        rows = runner.evaluate(
            make_env,
            seeds,
            env_config,
            artifacts_dir=str(output_dir),
            save_artifacts=save_artifacts,
        )
        all_rows.extend(rows)

    df = _order_columns(pd.DataFrame(all_rows))

    # utf-8-sig para os acentos abrirem corretamente no Excel.
    raw_path = output_dir / "benchmark_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    metric_cols = [c for c in df.columns if c not in ID_COLUMNS]
    mean_df = df.groupby("agent")[metric_cols].mean(numeric_only=True)
    std_df = df.groupby("agent")[metric_cols].std(numeric_only=True)

    # Ranqueamento PRIMÁRIO por recompensa (as demais métricas são diagnóstico
    # secundário). Ordena do melhor (maior recompensa) para o pior.
    if PRIMARY_METRIC in mean_df.columns:
        mean_df = mean_df.sort_values(PRIMARY_METRIC, ascending=False)
        std_df = std_df.reindex(mean_df.index)

    mean_df.to_csv(output_dir / "benchmark_mean.csv", encoding="utf-8-sig")
    std_df.to_csv(output_dir / "benchmark_std.csv", encoding="utf-8-sig")

    print(f"\n[benchmark] resultados salvos em: {output_dir.resolve()}")
    print("  - benchmark_raw.csv")
    print("  - benchmark_mean.csv")
    print("  - benchmark_std.csv")
    if save_artifacts:
        print(f"  - confusion_maps/  (mapas por agente/seed)")
        print(f"  - epidemic_maps/   (mapa ground-truth por seed)")
    else:
        print("  (artefatos visuais omitidos: save_artifacts=false no YAML)")

    _print_ranking(mean_df)
    return df


def _print_ranking(mean_df: pd.DataFrame) -> None:
    """Imprime o ranking por recompensa (primária) + contribuição marginal."""
    if PRIMARY_METRIC not in mean_df.columns:
        print("\nMédia por agente:")
        print(mean_df)
        return

    print(f"\n=== Ranking por {PRIMARY_METRIC} (métrica primária) ===")
    for pos, (agent, reward) in enumerate(mean_df[PRIMARY_METRIC].items(), start=1):
        print(f"  {pos}. {agent:<10} {PRIMARY_METRIC} = {reward:.2f}")

    # Contribuição marginal sobre o baseline clínico puro, se presente.
    if "clinical" in mean_df.index:
        base = mean_df.loc["clinical", PRIMARY_METRIC]
        print(f"\nContribuicao marginal sobre o clinico puro (delta {PRIMARY_METRIC}):")
        for agent in mean_df.index:
            if agent == "clinical":
                continue
            delta = mean_df.loc[agent, PRIMARY_METRIC] - base
            sinal = "acima" if delta >= 0 else "ABAIXO"
            print(f"  {agent:<10} delta = {delta:+.2f} ({sinal} do clinico)")

    print("\nMétricas secundárias (média por agente):")
    print(mean_df)


def main():
    parser = argparse.ArgumentParser(description="Benchmark de agentes.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/benchmark.yaml",
        help="Caminho para o YAML de benchmark.",
    )
    args = parser.parse_args()
    run_benchmark(args.config)


if __name__ == "__main__":
    main()
