"""CLI de treino acionado por config YAML.

Uso:
    poetry run python -m experiments.train --config experiments/configs/train/random.yaml

Trocar distribuição ou algoritmo = trocar o YAML (ou usar --config diferente),
sem alterar código.
"""
from __future__ import annotations

import argparse
import json
import os

from experiments.config import load_config
from experiments.runners import available_algorithms, run_from_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Treino do DengueDiagnostics via config YAML")
    parser.add_argument("--config", required=True, help="Caminho para o arquivo YAML de treino")
    parser.add_argument("--episodes", type=int, default=None, help="Sobrescreve train.episodes")
    parser.add_argument("--seed", type=int, default=None, help="Sobrescreve train.seed")
    parser.add_argument("--algorithm", default=None,
                        help=f"Sobrescreve train.algorithm. Opções: {available_algorithms()}")
    parser.add_argument("--output", default=None, help="Caminho .json para salvar os resultados")
    return parser.parse_args(argv)


def apply_overrides(config: dict, args) -> dict:
    train = config.setdefault("train", {})
    if args.algorithm is not None:
        train["algorithm"] = args.algorithm
    if args.episodes is not None:
        train["episodes"] = args.episodes
    if args.seed is not None:
        train["seed"] = args.seed
    return config


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    config = apply_overrides(config, args)

    try:
        result = run_from_config(config)
    except NotImplementedError as exc:
        print(f"[train] {exc}")
        return 0

    print("[train] Resultado:")
    for key, value in result.items():
        if key == "rewards":
            continue
        print(f"  {key}: {value}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"[train] Resultados salvos em {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
