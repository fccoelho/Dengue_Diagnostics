"""Artefatos visuais comuns a todos os agentes (mapas de confusão, epidemia, etc.)."""
from __future__ import annotations

from pathlib import Path

DEFAULT_ARTIFACTS_DIR = Path("results")
DEFAULT_CONFUSION_DIR = DEFAULT_ARTIFACTS_DIR / "confusion_maps"
DEFAULT_EPIDEMIC_DIR = DEFAULT_ARTIFACTS_DIR / "epidemic_maps"


def artifact_dirs(artifacts_root: Path | str | None):
    """Subpastas padrão de artefatos a partir da raiz (ex.: ``results/baseline``)."""
    if artifacts_root is None:
        return DEFAULT_CONFUSION_DIR, DEFAULT_EPIDEMIC_DIR
    root = Path(artifacts_root)
    return root / "confusion_maps", root / "epidemic_maps"


def confusion_map_path(
    agent: str,
    seed: int,
    output_dir: Path | str | None = None,
) -> Path:
    """Caminho canônico do PNG de mapa de confusão para (agente, seed)."""
    base = Path(output_dir) if output_dir is not None else DEFAULT_CONFUSION_DIR
    return base / agent / f"confusao_{agent}_seed_{seed}.png"


def epidemic_map_path(
    seed: int,
    output_dir: Path | str | None = None,
    *,
    agent: str | None = None,
) -> Path:
    """Caminho canônico do PNG de mapa da epidemia (ground truth).

    O surto depende apenas da seed/config do ambiente, não do agente.
    Se ``agent`` for informado, o arquivo fica em subpasta (útil no watch);
    no benchmark usamos só a seed para evitar duplicatas entre agentes.
    """
    base = Path(output_dir) if output_dir is not None else DEFAULT_EPIDEMIC_DIR
    name = f"mapa_epidemia_seed_{seed}.png"
    if agent is not None:
        return base / agent / name
    return base / name


def save_epidemic_map(
    env,
    seed: int,
    *,
    output_dir: Path | str | None = None,
    agent: str | None = None,
    show: bool = False,
    skip_if_exists: bool = True,
) -> Path | None:
    """Salva o mapa da epidemia VERDADEIRA (independente do agente).

    Funciona com qualquer conjunto de dados cujo ``real_cases`` siga o esquema
    padrão (``t``, ``x``, ``y``, ``disease``). Geradores futuros (Rio 2016,
    Kriging, ...) só precisam produzir esse ``DataFrame``.
    """
    path = epidemic_map_path(seed, output_dir, agent=agent)
    if skip_if_exists and path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    title = f"Mapa da Epidemia (seed {seed})"
    env.unwrapped.plot_epidemic_map(title=title, save_path=str(path), show=show)
    print(f"[artifacts] mapa da epidemia salvo em: {path.resolve()}")
    return path


def save_confusion_map(
    env,
    agent: str,
    seed: int,
    *,
    output_dir: Path | str | None = None,
    show: bool = False,
) -> Path:
    """Gera e salva o mapa de confusão espacial ao fim de um episódio.

    Delega o desenho para ``env.unwrapped.plot_confusion_map`` (matplotlib).
    Todos os agentes (benchmark, watch, treino) devem usar esta função para
    manter o mesmo layout/nomenclatura de arquivos.
    """
    path = confusion_map_path(agent, seed, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    title = f"Mapa de Confusao - {agent} (seed {seed})"
    env.unwrapped.plot_confusion_map(title=title, save_path=str(path), show=show)
    print(f"[artifacts] mapa de confusao salvo em: {path.resolve()}")
    return path
