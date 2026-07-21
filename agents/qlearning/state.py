"""Codificação de estado tabular para Q-Learning.

Duas variantes:
- ``rich_v1`` (padrão): features clínicas, decisão, tempo, posição relativa e contexto
  espacial local — tudo **discretizado** (sem case_id, sem x/y brutos, sem mapa inteiro).
- ``compact_v1`` (opcional): ``day|clinical|testd|testc`` — baseline mínimo (~10 estados).

Checkpoints salvos incluem ``state_version`` + ``encoder_config`` para compatibilidade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

STATE_VERSION_COMPACT = "compact_v1"
STATE_VERSION_RICH = "rich_v1"


def _bucket_count(n: int, *, thresholds: Tuple[int, ...] = (0, 1, 3)) -> int:
    """0 -> 0, 1 -> 1, 2-3 -> 2, 4+ -> 3."""
    if n <= thresholds[0]:
        return 0
    if n <= thresholds[1]:
        return 1
    if n <= thresholds[2]:
        return 2
    return 3


def _dist_bucket(dist: float, world_size: int, *, near_frac: float = 0.12, mid_frac: float = 0.30) -> int:
    """0=perto, 1=medio, 2=longe (fracoes do tamanho do mapa)."""
    near = world_size * near_frac
    mid = world_size * mid_frac
    if dist <= near:
        return 0
    if dist <= mid:
        return 1
    return 2


def _quadrant(x: int, y: int, world_size: int) -> int:
    """0=NW, 1=NE, 2=SW, 3=SE (metades do grid)."""
    mid = world_size // 2
    if x < mid and y < mid:
        return 0
    if x >= mid and y < mid:
        return 1
    if x < mid and y >= mid:
        return 2
    return 3


def _day_phase(t: int, epilength: int, *, early_max: int = 5, mid_max: int = 15) -> int:
    """0=inicio, 1=meio, 2=fim do surto, 3=cauda pos-surto."""
    if t <= early_max:
        return 0
    if t <= mid_max:
        return 1
    if t < epilength:
        return 2
    return 3


def _decode_test_status(raw: int) -> int:
    """Canal do mapa (+1): 0=sem caso -> 0 nao testado."""
    if raw <= 0:
        return 0
    return int(raw) - 1  # 0..3


def _decode_clinical(raw: int) -> int:
    if raw <= 0:
        return 2  # other / ausente
    return int(raw) - 1  # 0 dengue, 1 chik, 2 other


@dataclass
class StateEncoder:
    """Discretiza ``CaseByCaseWrapper`` + env bruto em chave de Q-table."""

    version: str = STATE_VERSION_RICH
    local_radius: int = 2
    day_early_max: int = 5
    day_mid_max: int = 15
    focus_near_frac: float = 0.12
    focus_mid_frac: float = 0.30

    def config_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "local_radius": self.local_radius,
            "day_early_max": self.day_early_max,
            "day_mid_max": self.day_mid_max,
            "focus_near_frac": self.focus_near_frac,
            "focus_mid_frac": self.focus_mid_frac,
        }

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]] = None) -> "StateEncoder":
        cfg = cfg or {}
        version = cfg.get("version", STATE_VERSION_RICH)
        if version == STATE_VERSION_COMPACT:
            return cls(version=STATE_VERSION_COMPACT)
        return cls(
            version=STATE_VERSION_RICH,
            local_radius=int(cfg.get("local_radius", 2)),
            day_early_max=int(cfg.get("day_early_max", 5)),
            day_mid_max=int(cfg.get("day_mid_max", 15)),
            focus_near_frac=float(cfg.get("focus_near_frac", 0.12)),
            focus_mid_frac=float(cfg.get("focus_mid_frac", 0.30)),
        )

    def encode(self, env) -> str:
        if self.version == STATE_VERSION_COMPACT:
            return encode_state_compact(env)
        return self._encode_rich(env)

    def _encode_rich(self, env) -> str:
        case_id, x, y = env.current_case
        if case_id == 0:
            return "terminal"

        u = env.unwrapped
        map_tensor = env._current_map_obs
        if map_tensor is None:
            return "unknown"

        clinical_raw = int(map_tensor[0, x, y])
        testd_raw = int(map_tensor[1, x, y])
        testc_raw = int(map_tensor[2, x, y])
        clinical = _decode_clinical(clinical_raw)
        testd = _decode_test_status(testd_raw)
        testc = _decode_test_status(testc_raw)

        row = u.obs_cases.loc[case_id] if case_id in u.obs_cases.index else None
        agent_dx = int(row["agent_diagnosis"]) if row is not None else clinical
        epi = int(row["epiconf"]) if row is not None else 0

        tests_done = (1 if testd > 0 else 0) + (1 if testc > 0 else 0)
        lab_pos = 1 if testd == 2 or testc == 2 else 0
        conflict = 0
        if lab_pos:
            if clinical == 0 and testd != 2:
                conflict = 1
            elif clinical == 1 and testc != 2:
                conflict = 1

        t = int(u.t)
        epilength = int(getattr(u, "epilength", 60))
        dp = _day_phase(t, epilength, early_max=self.day_early_max, mid_max=self.day_mid_max)

        size = int(u.size)
        quad = _quadrant(x, y, size)

        d_center = getattr(u, "dengue_center", (size // 2, size // 2))
        c_center = getattr(u, "chik_center", (size // 2, size // 2))
        dist_d = float(np.hypot(x - d_center[0], y - d_center[1]))
        dist_c = float(np.hypot(x - c_center[0], y - c_center[1]))
        dd = _dist_bucket(dist_d, size, near_frac=self.focus_near_frac, mid_frac=self.focus_mid_frac)
        dc = _dist_bucket(dist_c, size, near_frac=self.focus_near_frac, mid_frac=self.focus_mid_frac)
        closer_d = 1 if dist_d < dist_c else 0

        ld, lc, lpd, lpc, la = self._local_context(map_tensor, x, y, size)

        cases_today = len(u.obs_cases[u.obs_cases.t == t]) if not u.obs_cases.empty else 0
        ct = _bucket_count(cases_today, thresholds=(0, 2, 5))

        n_seen = len(u.obs_cases) if not u.obs_cases.empty else 0
        episize = max(int(getattr(u, "episize", 150)), 1)
        wk = 0 if n_seen <= episize * 0.25 else (1 if n_seen <= episize * 0.6 else 2)

        parts = [
            f"dp={dp}",
            f"cl={clinical}",
            f"adx={agent_dx}",
            f"td={testd}",
            f"tc={testc}",
            f"epi={epi}",
            f"tdn={tests_done}",
            f"lp={lab_pos}",
            f"cf={conflict}",
            f"q={quad}",
            f"dd={dd}",
            f"dc={dc}",
            f"cd={closer_d}",
            f"ld={ld}",
            f"lc={lc}",
            f"lpd={lpd}",
            f"lpc={lpc}",
            f"la={la}",
            f"ct={ct}",
            f"wk={wk}",
        ]
        return "|".join(parts)

    def _local_context(
        self, map_tensor: np.ndarray, x: int, y: int, size: int
    ) -> Tuple[int, int, int, int, int]:
        """Contagens na janela (2r+1)^2 ao redor de (x,y), em buckets 0..3."""
        r = max(self.local_radius, 0)
        x0, x1 = max(0, x - r), min(size, x + r + 1)
        y0, y1 = max(0, y - r), min(size, y + r + 1)

        patch0 = map_tensor[0, x0:x1, y0:y1]
        patch1 = map_tensor[1, x0:x1, y0:y1]
        patch2 = map_tensor[2, x0:x1, y0:y1]
        patch3 = map_tensor[3, x0:x1, y0:y1]

        n_clin_d = int(np.sum(patch0 == 1))
        n_clin_c = int(np.sum(patch0 == 2))
        n_pos_d = int(np.sum(patch1 == 3))  # status 2 -> canal 3
        n_pos_c = int(np.sum(patch2 == 3))
        n_active = int(np.sum(patch3 > 0))

        return (
            _bucket_count(n_clin_d),
            _bucket_count(n_clin_c),
            _bucket_count(n_pos_d),
            _bucket_count(n_pos_c),
            _bucket_count(n_active, thresholds=(0, 1, 2)),
        )


def encode_state_compact(env, *, day_bucket_size: int = 5) -> str:
    """Estado mínimo de 4 campos (``compact_v1``)."""
    case_id, x, y = env.current_case
    if case_id == 0:
        return "terminal"

    map_tensor = env._current_map_obs
    if map_tensor is None:
        return "unknown"

    clinical = int(map_tensor[0, x, y])
    testd = int(map_tensor[1, x, y])
    testc = int(map_tensor[2, x, y])
    day = int(env.unwrapped.t)
    day_bucket = min(day // max(day_bucket_size, 1), 99)
    return f"{day_bucket}|{clinical}|{testd}|{testc}"


# Encoder padrão global (rich).
DEFAULT_ENCODER = StateEncoder.from_config({})


def encode_state_from_env(env, *, encoder: Optional[StateEncoder] = None, day_bucket_size: int = 5) -> str:
    """API estável: usa ``encoder`` se informado; senão rich_v1."""
    enc = encoder or DEFAULT_ENCODER
    if enc.version == STATE_VERSION_COMPACT:
        return encode_state_compact(env, day_bucket_size=day_bucket_size)
    return enc.encode(env)
