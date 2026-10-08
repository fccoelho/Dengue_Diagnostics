"""Fila de treinos do experimento v6: paralela, com guarda de memória e retomável.

- Roda até `--paralelo` treinos ao mesmo tempo, e só lança um novo se houver
  pelo menos `--ram-livre` GB disponíveis (os demais programas da máquina
  também usam memória).
- Pula treinos concluídos (`policy_final.pth`) e retoma os interrompidos (o
  `agents.ppo.train` continua do último checkpoint de época). Se a máquina
  reiniciar, basta rodar a fila de novo.
- Espera as superfícies de um cenário existirem antes de lançar seus treinos.
- Refaz até 3 vezes um treino que termina com erro.

Uso: `python -m experiments.fila [--paralelo 4] [--ram-livre 5]`.
Imprime uma linha por evento (início, fim, falha) e um resumo a cada 30 min.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import psutil
import yaml

_RAIZ = Path(__file__).resolve().parents[1]
TREINO = _RAIZ / "experiments" / "configs" / "train" / "v6"
SAIDA = _RAIZ / "results" / "ppo_v6"
LOGS = SAIDA / "_logs"

CENARIOS = ("sintetico", "rio", "recife", "misto")
BRACOS = {"C": True, "A": False}  # braço -> per_case_credit
SEEDS = (45, 46, 47)

MODELO = {
    "context_features": True,
    "reward_scale": 0.10,
    "train": {
        "epochs": 10, "step_per_epoch": 30000, "step_per_collect": 2000, "repeat_per_collect": 10,
        "batch_size": 256, "lr": 0.0003, "gamma": 0.99, "gae_lambda": 0.95, "eps_clip": 0.2,
        "ent_coef": 0.02, "vf_coef": 0.5, "max_grad_norm": 0.5, "num_train_envs": 8,
        "num_test_envs": 10, "test_episodes": 20, "vector_env": "dummy",
    },
}


def nome(cenario: str, braco: str, seed: int) -> str:
    return f"{cenario}_{braco}_s{seed}"


# Grupos extras, só braço C: nome -> (cenário de ambiente, épocas, descrição).
EXTRAS = {
    "mistolongo": ("misto", 20, "misto com 20 épocas (600 mil passos): a mistura falhou por falta de treino?"),
    "recifepos": ("recifepos", 10, "Recife com escala e posição da cidade sorteadas: tira o atalho do contorno?"),
    "riofis": ("riofis", 10, "Rio em escala física (200 m/célula), casos 'outro' na área habitada."),
    "recifefis": ("recifefis", 10, "Recife em escala física (200 m/célula), casos 'outro' na área habitada."),
    "recifesup": ("recifesup", 10, "Recife esticado como antes, mas com os casos 'outro' só no município."),
}


def gera_configs_extra(grupo: str) -> List[Path]:
    cenario, epocas, descricao = EXTRAS[grupo]
    TREINO.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for seed in SEEDS:
        cfg = {"env_config": f"../../env/cenario_{cenario}.yaml", **MODELO}
        cfg["train"] = {**MODELO["train"], "seed": seed, "epochs": epocas, "per_case_credit": True}
        cfg["output_dir"] = f"results/ppo_v6/{nome(grupo, 'C', seed)}"
        p = TREINO / f"{nome(grupo, 'C', seed)}.yaml"
        p.write_text(f"# v6 — {descricao} Braço C, seed {seed}. Gerado por experiments/fila.py.\n"
                     + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        caminhos.append(p)
    return caminhos


# v7: matriz completa no ambiente corrigido, com 5 sementes, e a varredura de custo.
# Rio e sintético não mudam com a correção (o Rio ocupa o grid todo; o sintético
# não usa superfícies), então as sementes 45-47 do v6 são reaproveitadas e só
# 48-49 são novas. O Recife corrigido é o `recifesup`; o misto é retreinado.
SEEDS_V7 = (45, 46, 47, 48, 49)
TEST_EPISODES_V7 = 5
V7 = [  # (nome do grupo, cenário de ambiente, épocas, braços, sementes)
    ("sintetico", "sintetico", 10, "CA", SEEDS_V7),
    ("rio", "rio", 10, "CA", SEEDS_V7),
    ("recifesup", "recifesup", 10, "CA", SEEDS_V7),
    ("mistosup", "mistosup", 20, "CA", SEEDS_V7),
    ("riocusto2", "riocusto2", 10, "C", (45, 46, 47)),
    ("riocusto8", "riocusto8", 10, "C", (45, 46, 47)),
]


# Casos do surto proporcionais à população, na escala física (200 m/célula): o Recife
# ganha surtos 4,2x menores. O Rio é a referência (não muda), então as sementes
# 45-47 do `riofis` do v6 valem e só 48-49 são novas.
V_POP = [
    ("recifefispop", "recifefispop", 10, "C", SEEDS_V7),
    ("riofis", "riofis", 10, "C", (48, 49)),
]


def gera_configs_v7(grupos=None) -> List[Path]:
    """Uma config por treino do v7, na ordem: semente primeiro (a 1ª leva cobre tudo)."""
    TREINO.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for seed in SEEDS_V7:
        for grupo, cenario, epocas, bracos, seeds in (V7 if grupos is None else grupos):
            if seed not in seeds:
                continue
            for braco in bracos:
                cfg = {"env_config": f"../../env/cenario_{cenario}.yaml", **MODELO}
                cfg["train"] = {**MODELO["train"], "seed": seed, "epochs": epocas}
                if braco == "C":
                    cfg["train"]["per_case_credit"] = True
                cfg["output_dir"] = f"results/ppo_v6/{nome(grupo, braco, seed)}"
                # A validação de cada época só alimenta o log (a avaliação usa a
                # política final): 5 episódios em vez de 20 cortam ~1/3 do tempo.
                # Treinos já começados ou concluídos mantêm os 20 com que rodaram.
                saida = _RAIZ / cfg["output_dir"]
                if not saida.exists():
                    cfg["train"]["test_episodes"] = TEST_EPISODES_V7
                    # Um ambiente de validação por episódio: menos mundos em memória.
                    cfg["train"]["num_test_envs"] = TEST_EPISODES_V7
                p = TREINO / f"{nome(grupo, braco, seed)}.yaml"
                p.write_text(f"# v7 — grupo {grupo}, braço {braco}, semente {seed}, {epocas} épocas. "
                             "Gerado por experiments/fila.py.\n"
                             + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
                caminhos.append(p)
    return caminhos


def gera_configs_longo() -> List[Path]:
    """Mantido para a fila do misto longo que já está rodando."""
    return gera_configs_extra("mistolongo")


def gera_configs() -> List[Path]:
    """Uma config por (cenário, braço, seed), em ordem de seed: a 1ª leva cobre tudo."""
    TREINO.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for seed in SEEDS:
        for cenario in CENARIOS:
            for braco, credito in BRACOS.items():
                cfg = {"env_config": f"../../env/cenario_{cenario}.yaml", **MODELO}
                cfg["train"] = {**MODELO["train"], "seed": seed}
                if credito:
                    cfg["train"]["per_case_credit"] = True
                cfg["output_dir"] = f"results/ppo_v6/{nome(cenario, braco, seed)}"
                p = TREINO / f"{nome(cenario, braco, seed)}.yaml"
                cabecalho = (f"# v6 — braço {braco} ({'crédito por caso' if credito else 'GAE padrão'}), "
                             f"cenário {cenario}, seed {seed}. Gerado por experiments/fila.py.\n")
                p.write_text(cabecalho + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
                caminhos.append(p)
    return caminhos


def _pronto(cfg_path: Path) -> bool:
    """As superfícies que o cenário usa já existem?"""
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    env_path = (cfg_path.parent / cfg["env_config"]).resolve()
    env = yaml.safe_load(env_path.read_text(encoding="utf-8"))["env"]
    caminhos = [e["surfaces_path"] for e in env.get("mix", []) if "surfaces_path" in e]
    if "surfaces_path" in env:
        caminhos.append(env["surfaces_path"])
    return all((_RAIZ / c).exists() for c in caminhos)


def _concluido(cfg_path: Path) -> bool:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return (_RAIZ / cfg["output_dir"] / "policy_final.pth").exists()


def _log(msg: str) -> None:
    print(f"{time.strftime('%d/%m %H:%M')} {msg}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paralelo", type=int, default=4)
    ap.add_argument("--ram-livre", type=float, default=5.0, help="GB livres exigidos para lançar um treino")
    ap.add_argument("--tentativas", type=int, default=3)
    ap.add_argument("--longo", action="store_true", help="atalho para --grupo mistolongo")
    ap.add_argument("--v7", action="store_true", help="matriz v7 (ambiente corrigido, 5 sementes) e varredura de custo")
    ap.add_argument("--pop", action="store_true", help="casos proporcionais à população (recifefispop, riofis 48-49)")
    ap.add_argument("--grupo", choices=tuple(EXTRAS), nargs="+", default=None,
                    help="roda só grupos extras (braço C), na ordem dada")
    args = ap.parse_args(argv)

    grupos = ["mistolongo"] if args.longo else args.grupo
    if args.v7:
        fila = gera_configs_v7()
    elif args.pop:
        fila = gera_configs_v7(V_POP)
    else:
        fila = [c for g in grupos for c in gera_configs_extra(g)] if grupos else gera_configs()
    LOGS.mkdir(parents=True, exist_ok=True)
    rodando: Dict[Path, subprocess.Popen] = {}
    falhas: Dict[Path, int] = {}
    ambiente = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONIOENCODING": "utf-8"}
    ultimo_resumo = 0.0
    _log(f"fila com {len(fila)} treinos; {sum(_concluido(c) for c in fila)} já concluídos")

    while True:
        # Colhe os que terminaram.
        for cfg, proc in list(rodando.items()):
            codigo = proc.poll()
            if codigo is None:
                continue
            del rodando[cfg]
            if _concluido(cfg):
                _log(f"OK      {cfg.stem}")
            else:
                falhas[cfg] = falhas.get(cfg, 0) + 1
                _log(f"FALHOU  {cfg.stem} (código {codigo}, tentativa {falhas[cfg]}; log em {LOGS / cfg.stem}.log)")

        pendentes = [c for c in fila if not _concluido(c) and c not in rodando
                     and falhas.get(c, 0) < args.tentativas]
        if not pendentes and not rodando:
            desistidos = [c.stem for c in fila if not _concluido(c)]
            _log(f"FIM. concluídos {sum(_concluido(c) for c in fila)}/{len(fila)}"
                 + (f"; desistidos: {desistidos}" if desistidos else ""))
            return

        # Lança o próximo que estiver pronto, se houver vaga e memória.
        livre = psutil.virtual_memory().available / 1e9
        if len(rodando) < args.paralelo and livre >= args.ram_livre:
            for cfg in pendentes:
                if _pronto(cfg):
                    log = open(LOGS / f"{cfg.stem}.log", "a", encoding="utf-8")
                    rodando[cfg] = subprocess.Popen(
                        [sys.executable, "-m", "agents.ppo.train", "--config", str(cfg)],
                        cwd=_RAIZ, stdout=log, stderr=subprocess.STDOUT, env=ambiente)
                    _log(f"INICIO  {cfg.stem} (RAM livre {livre:.1f} GB, rodando {len(rodando)})")
                    time.sleep(90)  # deixa o treino alocar memória antes de medir de novo
                    break

        if time.time() - ultimo_resumo > 1800:
            ultimo_resumo = time.time()
            _log(f"RESUMO  concluídos {sum(_concluido(c) for c in fila)}/{len(fila)}, rodando "
                 f"{[c.stem for c in rodando]}, RAM livre {psutil.virtual_memory().available / 1e9:.1f} GB")
        time.sleep(20)


if __name__ == "__main__":
    main()
