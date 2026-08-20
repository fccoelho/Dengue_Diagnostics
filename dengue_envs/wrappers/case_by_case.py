"""Wrapper que transforma "N decisões por dia" em "1 decisão por passo".

Versão canônica (Fase 1) do `CaseByCaseWrapper`, idêntica em comportamento à
que vinha sendo usada no treino em `agents/deepq/files/dengue_wrapper.py`.
Aquela cópia legada permanece intacta.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dengue_envs.wrappers.map_tensor import DengueWrapper


class CaseByCaseWrapper(gym.Wrapper):
    """Expõe uma ação discreta por caso, avançando o dia quando os casos acabam."""

    def __init__(
        self,
        env: DengueWrapper,
        context_features: bool = False,
        per_case_reward: bool = False,
    ):
        super().__init__(env)

        self.action_space = spaces.Discrete(7)

        # `per_case_reward` muda QUANDO a recompensa é entregue, não quanto.
        #
        # Modo agregado (padrão, histórico): as ações do dia são acumuladas e
        # só são aplicadas quando o último caso do dia é decidido; até lá cada
        # passo devolve 0.0 e o dia inteiro cai num único passo. Medido nesse
        # regime: ~90% da variância da recompensa de um passo vinha de decisões
        # tomadas dias antes, sobre OUTROS casos — o agente decidia sobre o caso
        # X e recebia o desfecho de dezenas de outros.
        #
        # Modo por caso: a ação é aplicada na hora e o passo devolve a
        # recompensa atribuível àquele caso. A soma sobre o episódio é a mesma;
        # muda apenas a atribuição temporal do crédito.
        self.per_case_reward = bool(per_case_reward)

        # `context_features` acrescenta um vetor com a evidência acumulada sobre a
        # competência do médico (concordância laudo × palpite clínico + força da
        # evidência). Opcional e desligado por padrão para não invalidar
        # checkpoints treinados com a observação antiga.
        self.context_features = bool(context_features)

        spaces_dict = {
            "map": env.observation_space,
            "case_coords": spaces.Box(
                low=0.0, high=1.0, shape=(2,), dtype=np.float32
            ),
        }
        if self.context_features:
            # 2 posições de evidência sobre o médico (nível do episódio) + 12 de
            # atributos do caso atual (nível do caso). As duas partes vão no mesmo
            # vetor porque a rede as consome pelo mesmo ramo denso.
            spaces_dict["context"] = spaces.Box(
                low=0.0, high=1.0, shape=(14,), dtype=np.float32
            )
        # `mask` segue a convenção do Tianshou: 1 = ação permitida. Só entra na
        # observação quando o ambiente de fato restringe alguma ação, para não
        # alterar o espaço (e invalidar checkpoints) sem necessidade.
        self.action_mask = bool(
            getattr(self.unwrapped, "force_decision_after_tests", False)
        )
        if self.action_mask:
            spaces_dict["mask"] = spaces.MultiBinary(7)
        self.observation_space = spaces.Dict(spaces_dict)

        self.active_cases = []
        self.case_iterator = iter(self.active_cases)
        self.pending_actions = []
        self.current_case = (0, 0, 0)
        self._current_map_obs = None

    def _get_active_cases(self):
        """Casos que o agente decide hoje.

        São dois grupos:
        1. os notificados hoje (primeira avaliação);
        2. os **retornados**, cujo laudo chegou e que ainda aguardam decisão —
           é o que permite "testar hoje, decidir de posse do resultado".
        """
        u = self.unwrapped
        df = u.obs_cases
        t = u.t
        cases = []
        seen = set()
        for case in df[df.t == t].itertuples():
            if case.Index in u.finalized_cases:
                continue
            cases.append((case.Index, int(case.x), int(case.y)))
            seen.add(case.Index)

        for case_id in u.take_pending_revisits():
            if case_id in seen or case_id not in df.index:
                continue
            row = df.loc[case_id]
            cases.append((case_id, int(row.x), int(row.y)))
            seen.add(case_id)

        return cases

    def _refresh_active_cases(self):
        self.active_cases = self._get_active_cases()
        self.case_iterator = iter(self.active_cases)

    def _advance_empty_days(self):
        """Pula dias sem casos reportados (comum em start_day=1 ou curvas esparsas).

        Devolve também a recompensa acumulada nesses dias. Um dia sem casos
        ativos não é um dia sem recompensa: é nele que vencem desfechos
        agendados (``reward_delay_days``) e, se for o último, o placar final.
        Antes essa recompensa era **descartada** — um vazamento silencioso que
        crescia com o atraso configurado (medido: 114 de 444 num único episódio
        com atraso de 5 dias).
        """
        terminated, truncated = False, False
        info = {}
        reward_total = 0.0

        while not self.active_cases:
            obs_tensor, reward, terminated, truncated, info = self.env.step(tuple())
            reward_total += reward
            self._current_map_obs = obs_tensor
            if terminated or truncated:
                self.current_case = (0, 0, 0)
                return terminated, truncated, info, reward_total
            self._refresh_active_cases()

        return terminated, truncated, info, reward_total

    def _make_obs(self):
        normalized_x = self.current_case[1] / self.unwrapped.size
        normalized_y = self.current_case[2] / self.unwrapped.size
        obs = {
            "map": self._current_map_obs,
            "case_coords": np.array([normalized_x, normalized_y], dtype=np.float32),
        }
        if self.context_features:
            rate, strength = self.unwrapped.clinical_evidence()
            obs["context"] = np.concatenate(
                [
                    np.array([rate, strength], dtype=np.float32),
                    self.unwrapped.case_features(self.current_case[0]),
                ]
            )
        if self.action_mask:
            obs["mask"] = self.unwrapped.action_mask(self.current_case[0])
        return obs

    def _next_case(self):
        accumulated_reward = 0.0
        last_info = {}

        while True:
            try:
                self.current_case = next(self.case_iterator)
                return self._make_obs(), accumulated_reward, False, False, last_info

            except StopIteration:
                # No modo por caso as ações já foram aplicadas uma a uma; aqui
                # a chamada serve apenas para AVANÇAR O DIA (libera laudos que
                # maturaram, paga o placar final, atualiza o mundo).
                action_tuple = tuple(self.pending_actions)
                self.pending_actions = []

                obs_tensor, reward, terminated, truncated, info = self.env.step(action_tuple)
                accumulated_reward += reward
                self._current_map_obs = obs_tensor
                last_info = info

                if terminated or truncated:
                    self.current_case = (0, 0, 0)
                    return self._make_obs(), accumulated_reward, terminated, truncated, info

                self._refresh_active_cases()
                term, trunc, skip_info, skip_reward = self._advance_empty_days()
                accumulated_reward += skip_reward
                if skip_info:
                    last_info = skip_info
                if term or trunc:
                    self.current_case = (0, 0, 0)
                    return self._make_obs(), accumulated_reward, term, trunc, last_info

    def reset(self, **kwargs):
        obs_tensor, info = self.env.reset(**kwargs)
        self._current_map_obs = obs_tensor
        self.pending_actions = []

        self._refresh_active_cases()
        # No reset a fila de recompensas está vazia, então não há o que perder
        # aqui — o retorno é descartado de propósito.
        self._advance_empty_days()

        try:
            self.current_case = next(self.case_iterator)
        except StopIteration:
            self.current_case = (0, 0, 0)

        return self._make_obs(), info

    def step(self, action: int):
        if self.current_case[0] == 0 and not self.active_cases:
            return self._make_obs(), 0.0, True, False, {}

        case_id = int(self.current_case[0])
        if self.per_case_reward:
            # Aplica agora e cobra agora: a recompensa deste passo é a
            # consequência desta decisão, não um agregado do dia.
            case_r = self.unwrapped.apply_case_action(case_id, int(action))
            obs, day_r, terminated, truncated, info = self._next_case()
            return obs, case_r + day_r, terminated, truncated, info

        self.pending_actions.append((case_id, int(action)))
        return self._next_case()
