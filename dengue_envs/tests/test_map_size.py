"""Resolução da observação de mapa (`map_size`).

Motivação medida: o encoder termina em ``AdaptiveAvgPool2d((6, 6))``, então um
mapa 400×400 é reduzido a 6×6 pela própria rede — a resolução extra é
descartada, mas paga o preço inteiro no caminho dos dados (938 KB por
observação, 5,36 GB de replay buffer, 7,6 s de `gather`+H2D por 100
atualizações contra 0,7 s a 100×100).

O que estes testes travam: reduzir a resolução **preserva a informação que a
observação carrega** (posições agregadas em blocos, contagens somadas) e o
default continua sendo o comportamento histórico.
"""
from __future__ import annotations

import numpy as np
import pytest

from dengue_envs.wrappers import make_env

_ENV = {
    "size": 60,
    "episize": 40,
    "epilength": 8,
    "start_day": 1,
    "lab_delay_days": 0,
    "reward_delay_days": 0,
    "randomize_outbreak": False,
    "other_prevalence": 0.25,
    "epi_radius": 10,
    "epi_threshold": 0,
}


def _cfg(map_size=None):
    cfg = {
        "env": dict(_ENV),
        "wrappers": ["map_tensor", "case_by_case"],
        "context_features": True,
        "per_case_reward": True,
    }
    if map_size is not None:
        cfg["map_size"] = map_size
    return cfg


def test_default_keeps_world_resolution():
    """Sem `map_size`, nada muda — o comportamento histórico é o default."""
    env = make_env(_cfg())
    obs, _ = env.reset(seed=1)
    assert obs["map"].shape == (6, 60, 60)
    env.close()


def test_map_size_reduces_the_observation():
    env = make_env(_cfg(map_size=15))
    obs, _ = env.reset(seed=1)
    assert obs["map"].shape == (6, 15, 15)
    assert env.observation_space["map"].shape == (6, 15, 15)
    env.close()


def test_map_size_must_divide_the_world():
    """Sem divisibilidade as células agregariam blocos de tamanhos diferentes."""
    with pytest.raises(ValueError, match="dividir"):
        make_env(_cfg(map_size=7))


def test_map_size_must_be_within_bounds():
    with pytest.raises(ValueError):
        make_env(_cfg(map_size=0))
    with pytest.raises(ValueError):
        make_env(_cfg(map_size=120))


def _roda(cfg, seed, passos=200):
    env = make_env(cfg)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    terminated = truncated = False
    i = 0
    while not (terminated or truncated) and i < passos:
        case_id = env.current_case[0]
        obs, _r, terminated, truncated, _i = env.step(
            0 if case_id in u.obs_cases.index else 3
        )
        i += 1
    return obs, u


def test_confirmed_counts_are_summed_not_dropped():
    """A agregação dos canais de contagem preserva o total — não descarta casos."""
    obs_full, u_full = _roda(_cfg(), seed=4)
    obs_red, u_red = _roda(_cfg(map_size=15), seed=4)

    # Mesma trajetória (mesma seed, mesma política) => mesmo mapa no ambiente.
    assert u_full.confirmed_dmap.sum() == u_red.confirmed_dmap.sum()
    # E a soma sobrevive à agregação em blocos.
    assert obs_red["map"][4].sum() == pytest.approx(obs_full["map"][4].sum())
    assert obs_red["map"][5].sum() == pytest.approx(obs_full["map"][5].sum())


def test_occupied_cells_map_to_the_right_block():
    """Cada célula ocupada no mapa cheio cai no bloco correspondente do reduzido."""
    obs_full, _ = _roda(_cfg(), seed=6)
    obs_red, _ = _roda(_cfg(map_size=15), seed=6)
    bloco = 60 // 15

    for canal in (0, 1, 2, 3):
        xs, ys = np.nonzero(obs_full["map"][canal])
        for x, y in zip(xs, ys):
            assert obs_red["map"][canal, x // bloco, y // bloco] != 0, (
                f"canal {canal}: célula ({x},{y}) sumiu ao agregar"
            )


def test_reduced_observation_is_much_smaller():
    """O ponto do exercício: o tensor encolhe pelo quadrado do fator."""
    cheio = np.prod(make_env(_cfg()).observation_space["map"].shape)
    reduzido = np.prod(make_env(_cfg(map_size=15)).observation_space["map"].shape)
    assert cheio / reduzido == pytest.approx(16.0)


# --- fidelidade espacial: qual botão realmente governa --------------------

def test_pooled_size_is_the_real_spatial_bottleneck():
    """A rede reduz a mapa a `pooled_size` x `pooled_size`, venha ele de onde vier.

    Com o bbox real do Rio (72,3 km em 400 células = 181 m/célula), `map_size`
    100 dá células de 723 m — mas o `AdaptiveAvgPool2d((6,6))` do encoder
    entrega 12 km/célula à cabeça em QUALQUER resolução de entrada. É por isso
    que reduzir a observação não custa fidelidade: o gargalo espacial é o
    pooling, não o tensor.
    """
    import torch

    from agents.deepq.network import DengueNet

    saidas = set()
    for lado in (400, 100):
        net = DengueNet((6, lado, lado), 7, context_dim=16)
        with torch.no_grad():
            enc = net.map_encoder(torch.zeros(1, 6, lado, lado))
        saidas.add(enc.shape[1])
    assert len(saidas) == 1, (
        f"o encoder deveria entregar a mesma dimensão para toda entrada; {saidas}"
    )


def test_pooled_size_is_configurable():
    """Para um mapa real com estrutura fina, é ESTE o botão a mexer."""
    import torch

    from agents.deepq.network import DengueNet

    fino = DengueNet((6, 100, 100), 7, context_dim=16, pooled_size=12)
    grosso = DengueNet((6, 100, 100), 7, context_dim=16, pooled_size=6)
    with torch.no_grad():
        a = fino.map_encoder(torch.zeros(1, 6, 100, 100))
        b = grosso.map_encoder(torch.zeros(1, 6, 100, 100))
    assert a.shape[1] == 64 * 12 * 12
    assert b.shape[1] == 64 * 6 * 6


def test_checkpoint_roundtrip_preserves_pooled_size():
    """`load_policy` deduz o `pooled_size` do checkpoint, sem precisar do YAML."""
    import tempfile
    from pathlib import Path

    import torch

    from agents.deepq.agent import build_policy, load_policy

    # `map_size` no piso do encoder (ver network.MIN_MAP_SIDE): abaixo de 36 a
    # pilha convolucional não fecha.
    env = make_env(_cfg())
    pol = build_policy(env, device="cpu", pooled_size=4)
    with tempfile.TemporaryDirectory() as d:
        caminho = Path(d) / "p.pth"
        torch.save(pol.state_dict(), caminho)
        recarregada = load_policy(caminho, env, device="cpu")
    # Sem a dedução, `build_policy` usaria o default 6 e o load falharia.
    assert recarregada.model.map_proj[0].in_features == 64 * 4 * 4
    env.close()


def test_map_size_below_the_encoder_floor_fails_loudly():
    """Reduzir a observação tem limite duro, e o erro precisa dizer qual."""
    import pytest as _pytest

    from agents.deepq.agent import build_policy
    from agents.deepq.network import MIN_MAP_SIDE

    assert MIN_MAP_SIDE == 36
    env = make_env(_cfg(map_size=20))          # < 36
    with _pytest.raises(ValueError, match="map_size"):
        build_policy(env, device="cpu")
    env.close()


def test_agents_that_index_the_tensor_survive_a_reduced_map():
    """Regressão: `map_size` quebrou o encoder do q-learning.

    `env.current_case` traz coordenadas do MUNDO (400), mas o tensor passou a
    ser 100x100 — indexar direto estourava o eixo
    (`IndexError: index 167 is out of bounds for axis 1 with size 100`) e só
    apareceu no benchmark, porque nenhum teste cobria q-learning com mapa
    reduzido.
    """
    from agents.qlearning.state import encode_state_from_env

    for map_size in (None, 15):
        env = make_env(_cfg(map_size=map_size))
        env.reset(seed=2)
        u = env.unwrapped
        terminated = truncated = False
        i = 0
        while not (terminated or truncated) and i < 120:
            estado = encode_state_from_env(env)      # não pode estourar
            assert isinstance(estado, str) and estado
            case_id = env.current_case[0]
            _o, _r, terminated, truncated, _i = env.step(
                0 if case_id in u.obs_cases.index else 3
            )
            i += 1
        env.close()


def test_local_context_window_is_not_silently_empty():
    """`_local_context` fatiava com coordenada do mundo: patch vazio, sem erro.

    Diferente do IndexError acima, este degradava em SILÊNCIO — o NumPy devolve
    um array vazio ao fatiar fora do eixo, então todas as contagens locais
    viravam 0 e o estado ficava cego para a vizinhança.
    """
    import numpy as np

    from agents.qlearning.state import StateEncoder

    enc = StateEncoder()
    reduzido = np.zeros((6, 15, 15), dtype=np.uint8)
    reduzido[0, 10, 10] = 1          # um caso clínico de dengue perto do centro
    reduzido[3, 10, 10] = 1
    # (x, y) em coordenadas do MUNDO (60), que caem na célula 10 do tensor 15.
    ld, _lc, _lpd, _lpc, la = enc._local_context(reduzido, 40, 40, 60)
    assert ld > 0 or la > 0, "a janela local não pode vir vazia"
