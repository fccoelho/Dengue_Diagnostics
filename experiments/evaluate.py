"""CLI de avaliação/benchmark acionado por config YAML.

Sucessor conceitual do `COMPARACAO.py` (que permanece intacto). Roda um ou mais
algoritmos sobre o MESMO ambiente definido no YAML e imprime um comparativo de
recompensa por episódio.

Uso:
    poetry run python -m experiments.evaluate --config experiments/configs/benchmark.yaml
"""
from __future__ import annotations

import argparse
import copy

from experiments.config import load_config
from experiments.runners import get_runner
from experiments.config import build_env_factory


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark do DengueDiagnostics via config YAML")
    parser.add_argument("--config", required=True, help="Caminho para o YAML de benchmark")
    parser.add_argument("--episodes", type=int, default=None, help="Sobrescreve benchmark.episodes")
    parser.add_argument("--seed", type=int, default=None, help="Sobrescreve benchmark.seed")
    return parser.parse_args(argv)


def run_benchmark(config: dict) -> dict:
    """Executa cada algoritmo listado em `agents` e retorna os resultados.

    Estrutura esperada do config:
        env: {...}
        wrappers: [...]
        benchmark:
          episodes: 10
          seed: 42
          agents: [random, dqn, ppo]
    """
    bench = config.get("benchmark", {})
    episodes = bench.get("episodes", 5)
    seed = bench.get("seed", None)
    agents = bench.get("agents", ["random"])

    results = {}
    for algo in agents:
        run_cfg = copy.deepcopy(config)
        run_cfg["train"] = {"algorithm": algo, "episodes": episodes, "seed": seed}
        runner = get_runner(algo)
        env_factory = build_env_factory(run_cfg)
        try:
            results[algo] = runner.run(env_factory, run_cfg)
        except NotImplementedError as exc:
            results[algo] = {"algorithm": algo, "skipped": str(exc)}
    return results


def print_results(results: dict) -> None:
    print("\n=== Benchmark ===")
    print(f"{'algoritmo':<12} {'episódios':>10} {'recompensa média':>18} {'desvio':>10}")
    print("-" * 54)
    for algo, res in results.items():
        if "skipped" in res:
            print(f"{algo:<12} {'--':>10} {'(não integrado)':>18} {'--':>10}")
        else:
            print(
                f"{algo:<12} {res['episodes']:>10} "
                f"{res['mean_reward']:>18.3f} {res['std_reward']:>10.3f}"
            )


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    bench = config.setdefault("benchmark", {})
    if args.episodes is not None:
        bench["episodes"] = args.episodes
    if args.seed is not None:
        bench["seed"] = args.seed

    results = run_benchmark(config)
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
