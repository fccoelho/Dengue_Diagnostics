"""PPO com atribuição de crédito POR CASO e desconto em DIAS de epidemia.

O problema que isto resolve
---------------------------
O ambiente é uma epidemia que evolui dia a dia, e o agente decide um caso por
passo. Os casos se sobrepõem no tempo: no kriging com SEIR de literatura há
~30 decisões sobre OUTROS pacientes entre o exame de um caso e a conclusão
dele. O GAE padrão encadeia passos consecutivos da trajetória, então o valor
de pedir um exame precisa atravessar essas decisões alheias.

Medido (kriging v8), no passo de cada exame, a correlação entre o retorno e o
exame ter levado a um diagnóstico correto:

    retorno global, desconto por passo (GAE padrão)  : -0,019
    retorno do próprio caso, desconto por dia         :  0,994

O sinal existe na trajetória; a definição padrão de retorno o apaga.

O que muda — e o que não muda
-----------------------------
**O ambiente não muda.** A epidemia segue temporal, os casos seguem chegando
pela curva SEIR, os laudos seguem atrasando em dias, e o mapa de confirmados e a
evidência sobre o médico seguem acoplando os casos. A recompensa do ambiente —
a métrica do benchmark — é idêntica.

**Muda só a atribuição de crédito no aprendiz.** Cada transição traz em `info`
o caso decidido, o dia, e a parte da recompensa que pertence àquele caso
(`r_case`, ver `CaseByCaseWrapper.step`). Aqui as transições são agrupadas por
(episódio, caso), e o GAE corre ao longo da linha do tempo de CADA paciente,
descontando por dias decorridos em vez de passos.

É a mesma estrutura da atribuição de crédito em RL multiagente: cada caso como
um agente que compartilha o mundo com os outros.

Limitação conhecida
-------------------
O crédito por caso **não credita o exame pelo seu efeito sobre outros casos**:
um laudo positivo alimenta o mapa de confirmados, que ajuda `epi_confirm` em
casos futuros. Esse benefício é justamente o acoplamento entre casos, e o
retorno por caso não o enxerga. Na prática atual pesa pouco (o agente não usa
`epi_confirm` nem o mapa), mas é uma limitação a declarar.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.data import to_torch_as


def per_case_gae(
    r_case: np.ndarray,
    values: np.ndarray,
    day: np.ndarray,
    case_done: np.ndarray,
    episode_uid: np.ndarray,
    case_id: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vantagens e retornos ao longo da linha do tempo de cada caso.

    Todas as entradas têm uma posição por transição, na ordem do lote. Dentro
    de um mesmo ambiente o lote é cronológico, e um par (episódio, caso)
    pertence a um só ambiente, então a ordem das posições de um grupo é a ordem
    temporal do caso.

    Para as posições j de um caso, com dias d_j:

        desconto_j = gamma ** (d_{j+1} - d_j)
        delta_j    = r_j + desconto_j * V_{j+1} - V_j
        A_j        = delta_j + desconto_j * lambda * A_{j+1}

    No último passo do caso no lote:
      - caso terminado (`case_done`): não há futuro, V_{j+1} = 0;
      - caso ainda aberto (o lote cortou a linha do tempo dele): bootstrap com
        o próprio V_j e um dia de desconto.

    Transições sem caso (`case_id == 0`, passos de preenchimento no fim do
    episódio) viram grupos de uma posição, sem futuro.
    """
    n = len(r_case)
    adv = np.zeros(n, dtype=np.float64)
    r = np.asarray(r_case, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(day, dtype=np.float64)
    feito = np.asarray(case_done, dtype=bool)
    uid = np.asarray(episode_uid, dtype=np.int64)
    cid = np.asarray(case_id, dtype=np.int64)

    grupos: dict = {}
    for pos in range(n):
        chave = (int(uid[pos]), int(cid[pos])) if cid[pos] != 0 else ("solo", pos)
        grupos.setdefault(chave, []).append(pos)

    for posicoes in grupos.values():
        gae_seguinte = 0.0
        for k in range(len(posicoes) - 1, -1, -1):
            p = posicoes[k]
            if k == len(posicoes) - 1:
                if feito[p]:
                    desconto, v_seguinte = 0.0, 0.0
                else:
                    desconto, v_seguinte = gamma, v[p]
                gae_seguinte = 0.0
            else:
                q = posicoes[k + 1]
                desconto = gamma ** max(0.0, d[q] - d[p])
                v_seguinte = v[q]
            delta = r[p] + desconto * v_seguinte - v[p]
            gae = delta + desconto * gae_lambda * gae_seguinte
            adv[p] = gae
            gae_seguinte = gae

    return adv, adv + v


class PerCasePPO(PPO):
    """PPO cujo GAE corre ao longo da linha do tempo de cada caso.

    ``credit_scale`` deve ser o mesmo `reward_scale` aplicado à recompensa do
    ambiente no treino: `r_case` vem em `info`, e o `RewardScaleWrapper` só
    escala a recompensa, não o `info`. Sem isso os alvos do crítico ficariam
    numa escala 10x maior que a pretendida.
    """

    def __init__(self, *args, credit_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.credit_scale = float(credit_scale)

    def _add_returns_and_advantages(self, batch, buffer, indices):
        valores = []
        with torch.no_grad():
            for mini in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                valores.append(self.critic(mini.obs))
        batch.v_s = torch.cat(valores, dim=0).flatten()
        v = batch.v_s.cpu().numpy()

        info = batch.info
        faltam = [k for k in ("r_case", "day", "case_done", "episode_uid", "case_id")
                  if k not in info]
        if faltam:
            raise KeyError(
                f"PerCasePPO precisa de {faltam} em info — use per_case_reward: true "
                "no ambiente (CaseByCaseWrapper)."
            )
        adv, ret = per_case_gae(
            r_case=np.asarray(info["r_case"], dtype=np.float64) * self.credit_scale,
            values=v,
            day=np.asarray(info["day"]),
            case_done=np.asarray(info["case_done"]),
            episode_uid=np.asarray(info["episode_uid"]),
            case_id=np.asarray(info["case_id"]),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        batch.returns = to_torch_as(ret, batch.v_s)
        batch.adv = to_torch_as(adv, batch.v_s)
        return batch
