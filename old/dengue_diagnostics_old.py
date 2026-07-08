# Basic packages
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional

# Import simulation tools
import gymnasium as gym
import pygame

from dengue_envs.data.generator import World
from dengue_envs.viz import lineplot
from gymnasium import spaces


class DengueDiagnosticsEnv(gym.Env):
    metadata = {"render_modes": ["human", "console"], "render_fps": 1}

    def __init__(
            self,
            size: int = 400,
            episize: int = 150,
            epilength: int = 60,
            reward_delay_days: int = 5, 
            dengue_center=(100, 100),
            chik_center=(300, 300),
            dengue_radius=90,
            chik_radius=90,
            clinical_specificity: Union[float, Tuple[float, float]] = 0.8,
            start_day: int = 1,
            render_mode=None,
    ):
        """

        Args:
            size: Size of the world
            episize: total number of cases in the epidemic
            epilength: length of the epidemic in days
            dengue_center: center of the dengue outbreak
            chik_center: center of the chikungunya outbreak
            dengue_radius: radius of the dengue outbreak
            chik_radius: radius of the chikungunya outbreak
            clinical_specificity: specificity of the clinical diagnosis
            render_mode: render mode
            reward_delay_days: delay in days for the reward to be paid
            start_day: first simulation day when the agent takes actions (default 1)
        """
        super().__init__()
        self.start_day = start_day
        self.t = start_day

        self.reward_delay = reward_delay_days
        self.pending_rewards = {}  
        self.costs = np.array([1.0, 1.0, 0.5, 0.1, 0.0, 0.0])
        self.size = size
        self.episize = episize
        self.epilength = epilength
        self.dengue_center = dengue_center
        self.chik_center = chik_center
        self.dengue_radius = dengue_radius
        self.chik_radius = chik_radius
        self.specificity_setting = clinical_specificity
        if isinstance(self.specificity_setting, tuple):
            low, high = self.specificity_setting
            self.clinical_specificity = np.random.uniform(low, high)
        else:
            self.clinical_specificity = self.specificity_setting
        self.world = World(
            self.size,
            self.episize,
            self.epilength,
            self.dengue_center,
            self.chik_center,
            self.dengue_radius,
            self.chik_radius,
        )

        # Observations are dictionaries as defined below.
        # Data are represented as sequences of cases.
        self.observation_space = spaces.Dict(
            {
                "clinical_diagnostic": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Discrete(self.world.num_cols),  # x coordinate
                            spaces.Discrete(self.world.num_rows),  # y coordinate
                            spaces.Discrete(3),  # Diagnostic: 0: dengue, 1: chik, 2: other
                        )
                    )
                ),  # Clinical diagnosis: 0: dengue, 1: chik, 2: other
                "testd": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Discrete(self.episize),  # case id
                            spaces.Discrete(4)
                            # Dengue testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                        )
                    )
                ),
                "testc": spaces.Sequence(
                    spaces.Tuple((
                        spaces.Discrete(self.episize),  # case id
                        spaces.Discrete(4)
                        # Chikungunya testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                    ))
                ),
                "epiconf": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Discrete(self.episize),  # case id
                            spaces.Discrete(2)  # Epidemiological confirmation: 0: no, 1: yes
                        )
                    )
                ),
                "tnot": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Discrete(self.episize),  # case id
                            spaces.Discrete(self.epilength)  # Day of the clinical diagnosis
                        )
                    )
                ),
            }
        )

        self.real_cases = self.world.casedf.copy()
        # Total number of cases actually generated (can exceed episize due to two
        # overlapping epidemic curves), so case ids range over [0, num_cases).
        self.num_cases = len(self.real_cases)

        # We have 6 actions, corresponding to "test for dengue", "test for chik", "epi confirm", "Do nothing", confirm, discard
        self.action_space = spaces.Sequence(
            spaces.Tuple((spaces.Discrete(self.num_cases), spaces.Discrete(6)))  # case id, action
        )
        self.testd = []
        self.testc = []
        self.epiconf = []
        self.final = []

        self.tcase = []
        self.rewards = []
        self.total_reward = 0
        self.accuracy = []
        self.mean_accuracy_history = []
        self.mape_history = []

        self.obs_cases = pd.DataFrame()
        self._load_episode_state(self.start_day)

        self.obs = {"testd": 0, "testc": 1, "epiconf": 2, "tnot": 3, "nothing": 4, "confirm": 5, "discard": 6,
                    "clinical_diagnostic": 7}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.clock = None  # self.metadata["render_fps"]
        # Initialize rendering
        if self.render_mode is not None:
            self._render_init(mode=self.render_mode)

        self.individual_rewards = [[0]]

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Seed the environment
        Args:
            seed: Seed value

        Returns:
            List of seeds
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_case_id(self, case):
        x = case[0]
        y = case[1]
        return self.real_cases[(self.real_cases.x == x) & (self.real_cases.y == y)].index[0]

    def get_case_xy(self, case_id):
        return self.real_cases.loc[case_id, ["x", "y"]].values

    def _cases_at_t(self, t: int) -> Tuple:
        if self.obs_cases.empty:
            return tuple()
        day_cases = self.obs_cases[self.obs_cases.t == t]
        return tuple((c.x, c.y, c.disease) for c in day_cases.itertuples())

    def _load_episode_state(self, t: int):
        """Load true/observed cases and maps consistently for timestep t.

        Clinical uncertainty is applied only to cases that were not seen before,
        so lab-test results and agent decisions persist across timesteps.
        """
        self.t = t
        self.cases = self.world.get_series_up_to_t(t)
        self._sync_obs_cases()
        self.cases_t = self._cases_at_t(t)
        self.dmap, self.cmap = self.world.get_maps_up_to_t(t)

    def _sync_obs_cases(self):
        """Append newly reported cases while preserving prior tests/decisions."""
        empty_cols = ["t", "x", "y", "disease", "testd", "testc", "epiconf", "agent_diagnosis"]
        if self.cases.empty:
            if self.obs_cases is None or self.obs_cases.empty:
                self.obs_cases = pd.DataFrame(columns=empty_cols)
            return

        if self.obs_cases is None or self.obs_cases.empty:
            known = set()
        else:
            known = set(self.obs_cases.index)

        new_cases = self.cases[~self.cases.index.isin(known)]
        if new_cases.empty:
            return

        new_obs = self._apply_clinical_uncertainty(new_cases)
        if self.obs_cases is None or self.obs_cases.empty:
            self.obs_cases = new_obs
        else:
            self.obs_cases = pd.concat([self.obs_cases, new_obs])

    def _render_init(self, mode="human"):
        """
        Initialize rendering
        """
        if mode == "console":
            return
        pygame.init()
        pygame.display.init()

        # Setting display size
        self.scaling_factor = 800 / self.world.size  # Scaling factor for the display
        self.screen = pygame.display.set_mode(
            size=(800, 800),
            depth=32,
            flags=pygame.SCALED,
        )
        self.world_surface = pygame.Surface((self.world.size, self.world.size))
        self.world_surface.set_colorkey((0, 0, 0))
        self.dengue_group = CaseGroup("dengue", self.scaling_factor)
        self.chik_group = CaseGroup("chik", self.scaling_factor)
        self.all_tests = CaseGroup("all", self.scaling_factor)

        self.plot_surface1 = pygame.Surface((400, 300))
        self.plot_surface2 = pygame.Surface((400, 300))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def _get_obs(self):
        """
        Returns the current observation.
        """
        return {
            "clinical_diagnostic": tuple((c.x, c.y, c.disease) for c in self.obs_cases.itertuples()),
            "testd": tuple((c.Index, c.testd) for c in self.obs_cases.itertuples()),
            "testc": tuple((c.Index, c.testc) for c in self.obs_cases.itertuples()),
            "epiconf": tuple((c.Index, c.epiconf) for c in self.obs_cases.itertuples()),
            "tnot": tuple((c.Index, c.t) for c in self.obs_cases.itertuples()),
        }

    def _apply_clinical_uncertainty(self, cases_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply clinical uncertainty to newly reported cases: the clinical diagnosis
        is subject to misdiagnosis based on the clinical specificity. Sampled once
        per case so the observed diagnosis stays stable for the rest of the episode.
        """
        obs_case_df = cases_df.copy()
        for idx in obs_case_df.index:
            true_disease = obs_case_df.at[idx, "disease"]
            if self.np_random.uniform() < 0.01:
                obs_case_df.at[idx, "disease"] = 2  # Other disease
                continue
            if true_disease == 0:
                if self.np_random.uniform() > self.clinical_specificity:  # Misdiagnosed as chik
                    obs_case_df.at[idx, "disease"] = 1
            elif true_disease == 1:
                if self.np_random.uniform() > self.clinical_specificity:  # Misdiagnosed as dengue
                    obs_case_df.at[idx, "disease"] = 0

        obs_case_df["agent_diagnosis"] = obs_case_df["disease"]

        return obs_case_df

    def _calc_reward(self, true, estimated, action, terminated=False):
        """
        Calcula a recompensa com custos imediatos e agenda as decisões finais (delay).
        """
        PENALTY_INCORRECT_DECISION = -20.0
        REWARD_CORRECT_DECISION = 10.0

        immediate_reward = 0.0
        delayed_reward_accum = 0.0

        for case_id, action_id in action:
            # 1. Aplica o Custo Imediato da Ação (subtraindo)
            immediate_reward -= self.costs[action_id]

            # Ignora ações sobre casos ainda não reportados
            if case_id not in self.obs_cases.index:
                continue

            true_disease = int(self.real_cases.loc[case_id, "disease"])
            agent_diagnosis = self.obs_cases.loc[case_id, "agent_diagnosis"]

            is_correct = False
            is_decision = False

            # 2. Avalia Ações de Decisão (4, 5) para agendar no futuro
            if action_id == 4:  # Confirmar
                is_decision = True
                if agent_diagnosis == true_disease:
                    is_correct = True

            elif action_id == 5:  # Descartar
                is_decision = True
                discarded = 1 if agent_diagnosis == 0 else 0
                if discarded == true_disease:
                    is_correct = True

            # Prepara o valor que será agendado
            if is_decision:
                if is_correct:
                    delayed_reward_accum += REWARD_CORRECT_DECISION
                else:
                    delayed_reward_accum += PENALTY_INCORRECT_DECISION

        # 3. Agenda a recompensa para o futuro (t + delay)
        if delayed_reward_accum != 0:
            target_t = self.t + self.reward_delay
            if target_t not in self.pending_rewards:
                self.pending_rewards[target_t] = 0.0
            self.pending_rewards[target_t] += delayed_reward_accum

        # 4. Resgata as recompensas que "venceram" no timestep atual
        matured_reward = self.pending_rewards.pop(self.t, 0.0)

        # 5. Se for o fim do episódio, força o resgate de tudo que sobrou na fila
        if terminated:
            matured_reward += sum(self.pending_rewards.values())
            self.pending_rewards.clear()

        # Recompensa do step é a soma dos gastos de hoje com os desfechos que venceram hoje
        step_reward = immediate_reward + matured_reward
        self.total_reward += step_reward
        
        self.individual_rewards.append([])
        return step_reward

    def calc_accuracy(self, true, estimated):
        """
        Calcula a acurácia (média da acurácia de Dengue e Chik).
        """
        tpd, fpd, tnd, fnd = 0, 0, 0, 0  # Contadores para Dengue
        tpc, fpc, tnc, fnc = 0, 0, 0, 0  # Contadores para Chikungunya

        # Se não houver dados, não há o que calcular.
        if len(true) == 0:
            self.accuracy.append(0.0)
            self.mean_accuracy_history.append(0.0)
            self.mape_history.append(0.0)
            return 0.0

        for t, e in zip(true, estimated):
            true_label = t['disease']
            est_label = e[2]

            is_dengue = (true_label == 0)
            predicted_dengue = (est_label == 0)

            if is_dengue and predicted_dengue:
                tpd += 1  # Verdadeiro Positivo Dengue
            elif is_dengue and not predicted_dengue:
                fnd += 1  # Falso Negativo Dengue (Real: Dengue, Pred: Chik/Outro)
            elif not is_dengue and predicted_dengue:
                fpd += 1  # Falso Positivo Dengue (Real: Chik/Outro, Pred: Dengue)
            elif not is_dengue and not predicted_dengue:
                tnd += 1  # Verdadeiro Negativo Dengue (Real: Chik/Outro, Pred: Chik/Outro)

            is_chik = (true_label == 1)
            predicted_chik = (est_label == 1)

            if is_chik and predicted_chik:
                tpc += 1  # Verdadeiro Positivo Chik
            elif is_chik and not predicted_chik:
                fnc += 1  # Falso Negativo Chik (Real: Chik, Pred: Dengue/Outro)
            elif not is_chik and predicted_chik:
                fpc += 1  # Falso Positivo Chik (Real: Dengue/Outro, Pred: Chik)
            elif not is_chik and not predicted_chik:
                tnc += 1  # Verdadeiro Negativo Chik (Real: Dengue/Outro, Pred: Dengue/Outro)

        total_cases = len(true)

        accuracy_dengue = (tpd + tnd) / total_cases
        accuracy_chik = (tpc + tnc) / total_cases

        mean_accuracy = (accuracy_dengue + accuracy_chik) / 2

        self.accuracy.append(mean_accuracy)
        self.mean_accuracy_history.append(mean_accuracy)

        true_numdengue = len([c for c in true if c["disease"] == 0])
        estimated_numdengue = len([c for c in estimated if c[2] == 0])
        true_chik = len([c for c in true if c["disease"] == 1])
        estimated_chik = len([c for c in estimated if c[2] == 1])
        true_total = true_numdengue + true_chik
        est_total = estimated_numdengue + estimated_chik

        if true_total > 0:
            mape = np.abs(true_total - est_total) / true_total
        else:
            mape = 0.0

        self.mape_history.append(mape)

        return mean_accuracy

    def _get_info(self):
        """
        Returns the current map of cases for each disease
        """

        return {
            "dengue_grid": self.dmap,
            "chik_grid": self.cmap,
        }

    def _dengue_lab_test(self, case_id):
        """
        Returns the dengue test result for a case, conditioned on the TRUE disease.
        1: Negative, 2: Positive, 3: Inconclusive
        (10% inconclusive, then 90% sensitivity / 90% specificity)
        """
        true_disease = int(self.real_cases.loc[case_id, "disease"])
        if self.np_random.uniform() < 0.1:
            return 3  # Inconclusive
        if true_disease == 0:  # Really dengue -> 90% chance positive
            return 2 if self.np_random.uniform() < 0.9 else 1
        return 1 if self.np_random.uniform() < 0.9 else 2  # Not dengue -> 90% negative

    def _chik_lab_test(self, case_id):
        """
        Returns the chikungunya test result for a case, conditioned on the TRUE disease.
        1: Negative, 2: Positive, 3: Inconclusive
        (10% inconclusive, then 90% sensitivity / 90% specificity)
        """
        true_disease = int(self.real_cases.loc[case_id, "disease"])
        if self.np_random.uniform() < 0.1:
            return 3  # Inconclusive
        if true_disease == 1:  # Really chik -> 90% chance positive
            return 2 if self.np_random.uniform() < 0.9 else 1
        return 1 if self.np_random.uniform() < 0.9 else 2  # Not chik -> 90% negative

    def _update_case_status(self, action, index, result):
        """
        Atualiza o status do caso e o diagnóstico do agente com base
        nos resultados dos testes.
        """
        if action == 0:  # Teste de Dengue
            self.obs_cases.loc[index, "testd"] = result
            if result == 2:  # Positivo para Dengue
                self.obs_cases.loc[index, "agent_diagnosis"] = 0
            elif result == 1:  # Negativo para Dengue
                # Se não é Dengue e a estimativa era Dengue, vira Chik
                if self.obs_cases.loc[index, "agent_diagnosis"] == 0:
                    self.obs_cases.loc[index, "agent_diagnosis"] = 1
            # Se for 3 (Inconclusivo), o agent_diagnosis não muda

        elif action == 1:  # Teste de Chik
            self.obs_cases.loc[index, "testc"] = result
            if result == 2:  # Positivo para Chik
                self.obs_cases.loc[index, "agent_diagnosis"] = 1
            elif result == 1:  # Negativo para Chik
                # Se não é Chik e a estimativa era Chik, vira Dengue
                if self.obs_cases.loc[index, "agent_diagnosis"] == 1:
                    self.obs_cases.loc[index, "agent_diagnosis"] = 0
            # Se for 3 (Inconclusivo), o agent_diagnosis não muda

        elif action == 2:  # Epi confirm
            self.obs_cases.loc[index, "epiconf"] = result
            # Esta ação não altera o 'agent_diagnosis'

    def _epi_confirm(self, case_id):
        """
        Returns the epidemiological confirmation for a case based on the local
        case density of the currently suspected disease.
        1: confirmed by epidemiological evidence, 0: not confirmed
        """
        x, y = self.get_case_xy(case_id)
        x, y = int(x), int(y)
        clinical = int(self.obs_cases.loc[case_id, "disease"])
        if clinical == 0:  # Dengue suspicion
            return 1 if self.dmap[x, y] > 1 else 0
        if clinical == 1:  # Chik suspicion
            return 1 if self.cmap[x, y] > 1 else 0
        return 0

    def reset(self, seed: int = None, options=None, reset_data: bool = False) -> Tuple[Dict, Dict]:
        """
        Resets the environment to the initial state
        Args:
            reset_data: If the world data is supposed to re-created as well. Default is False.

        Returns:

        """
        super().reset(seed=seed)

        if isinstance(self.specificity_setting, tuple):
            # Se for uma tupla (min, max), sorteia um valor uniforme
            low, high = self.specificity_setting
            self.clinical_specificity = self.np_random.uniform(low=low, high=high)
        else:
            # Se for um float, usa esse valor fixo
            self.clinical_specificity = self.specificity_setting

        if reset_data:  # Re-Creates the world if requested
            self.world = World(
                self.size,
                self.episize,
                self.epilength,
                self.dengue_center,
                self.chik_center,
                self.dengue_radius,
                self.chik_radius,
            )
            self.real_cases = self.world.casedf.copy()

        self.testd = []
        self.testc = []
        self.epiconf = []
        self.final = []
        self.rewards = []
        self.accuracy = []
        self.mean_accuracy_history = []
        self.mape_history = []
        self.total_reward = 0
        self.pending_rewards = {}
        self.individual_rewards = [[0]]

        # Clear observed cases so a fresh episode re-samples clinical uncertainty
        self.obs_cases = pd.DataFrame()
        self._load_episode_state(self.start_day)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def get_individual_rewards_at_t(self, t):
        """
        Get the individual rewards at time t
        """
        return self.individual_rewards[t]


    def step(self, action):
        """
        Apply the actions for every case at the current timestep (t)
        and the returns the observation(state at t+1), reward, termination status and info
        action: [list of decisions (2-tuples) for all current cases]: 0: test for dengue, 1: test for chik, 2: epi confirm, 3: Does nothing, 4: Confirm, 5: Discard
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for {self.action_space}")
        # get the current true state (preserving prior tests/decisions)
        self.cases = self.world.get_series_up_to_t(self.t)
        self._sync_obs_cases()
        self.cases_t = self._cases_at_t(self.t)
        # apply the actions
        for case_id, action_id in action:
            # Ignore actions targeting cases that have not been reported yet
            if case_id not in self.obs_cases.index:
                continue
            if action_id == 0:  # Teste de Dengue
                test_result = self._dengue_lab_test(case_id)
                self.testd.append((case_id, test_result))
                self._update_case_status(0, case_id, test_result)
            elif action_id == 1:  # Teste de Chik
                test_result = self._chik_lab_test(case_id)
                self.testc.append((case_id, test_result))
                self._update_case_status(1, case_id, test_result)
            elif action_id == 2:  # Epi confirm
                epi_result = self._epi_confirm(case_id)
                self.epiconf.append((case_id, epi_result))
                self._update_case_status(2, case_id, epi_result)
            elif action_id == 3:  # Do nothing
                # O agent_diagnosis permanece como o palpite clínico
                pass
            elif action_id == 4:  # Confirm
                # Ação decisiva: O agente confirma o 'agent_diagnosis' atual.
                self.final.append(1)
            elif action_id == 5:  # Discard
                # Descarta como falso positivo: diagnóstico do agente vira "Outro" (2)
                self.obs_cases.loc[case_id, "agent_diagnosis"] = 2
                self.final.append(0)

        estimated_for_accuracy = tuple(
            (c.x, c.y, c.agent_diagnosis)
            for c in self.obs_cases.itertuples()
        )

        true_cases = self.cases.to_dict(orient="records")

        self.calc_accuracy(true_cases, estimated_for_accuracy)

        terminated = self.t >= self.epilength + 10
        reward = self._calc_reward(
            self.cases.to_dict(orient="records"),
            estimated_for_accuracy,
            action,
            terminated=terminated 
        )
        # print(f"Reward: {reward} \t Total Reward: {self.total_reward}")

        self.rewards.append(self.total_reward)

        if self.render_mode == "human":
            self._create_sprites()

            self.update_sprites(action)

            self.accuracy_plot = lineplot(
                range(1, self.t + 1), self.mean_accuracy_history, "Step", "Accuracy", "Accuracy", "plot2"
            )

            self.total_reward_plot = lineplot(
                range(1, self.t + 1), self.rewards, "Step", "Total Reward", "Total Reward", "plot1"
            )

            self.plot_surface1.blit(
                pygame.transform.scale(
                    pygame.image.load(self.total_reward_plot, "PNG"), self.plot_surface1.get_rect().size
                ),
                (0, 0),
            )

            self.plot_surface2.blit(
                pygame.transform.scale(
                    pygame.image.load(self.accuracy_plot, "PNG"), self.plot_surface2.get_rect().size
                ),
                (0, 0),
            )

            self.render()

        # Update the timestep
        self.t += 1
        self.dmap, self.cmap = self.world.get_maps_up_to_t(self.t)
        self.cases = self.world.get_series_up_to_t(self.t)
        self._sync_obs_cases()
        observation = self._get_obs()
        info = self._get_info()

        if terminated:
            ep_acc = np.mean(self.mean_accuracy_history) if self.mean_accuracy_history else 0.0
            ep_mape = np.mean(self.mape_history) if self.mape_history else 0.0
            info["episode/accuracy"] = ep_acc
            info["episode/mape"] = ep_mape

        return observation, reward, terminated, False, info

    def update_sprites(self, actions):
        # Update the sprites in the dengue group
        for sprite in self.dengue_group.sprites():
            for id, a in actions:
                if sprite.case_id == id:
                    sprite.mark_as_tested(int(a))

        for sprite in self.chik_group.sprites():
            for id, a in actions:
                if sprite.case_id == id:
                    sprite.mark_as_tested(int(a))

    def render(self):
        """
        Render the environment with a legend on the right side
        """
        pygame.event.pump()

        self.dengue_group.draw(self.world_surface)
        self.chik_group.draw(self.world_surface)

        # Clear the screen
        self.screen.fill((255, 255, 255))

        number_font = pygame.font.SysFont(None, 32)
        timestep_display = number_font.render(
            f"Step {self.t}", True, (0, 0, 0), (255, 255, 255)
        )
        self.screen.blit(
            timestep_display,
            (int((self.screen.get_width() - timestep_display.get_width()) / 2), 0),
        )

        # Blit as superfícies que já foram preparadas no step()
        self.screen.blit(
            self.plot_surface1, (0, 500), special_flags=pygame.BLEND_ALPHA_SDL2
        )

        self.screen.blit(
            self.plot_surface2, (400, 500), special_flags=pygame.BLEND_ALPHA_SDL2
        )

        # Draw the world surface
        self.screen.blit(
            self.world_surface, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2
        )

        # Create the legend on the right side
        legend_x = self.screen.get_width() - 200  # X position of the legend (right side)
        legend_y = 120  # Y position of the first legend item
        legend_margin = 40  # Space between items in the legend

        # Define image mappings and their descriptions
        image_legend = [
            ("dengue_test.png", "Dengue Test"),
            ("chick_test.png", "Chik Test"),
            ("epi_test.png", "Inconclusive"),
            ("no_test.png", "No Test"),
            ("confirm_test.png", "Confirm"),
            ("discard_test.png", "Discard"),
        ]

        # Render the legend
        for image_file, description in image_legend:
            number_font_legend = pygame.font.SysFont(None, 18)

            # Load the image
            image = pygame.image.load(os.path.join(os.path.dirname(__file__), image_file)).convert_alpha()
            image = pygame.transform.scale(image, (10, 10))
            self.screen.blit(image, (legend_x, legend_y))

            # Render the description text
            legend_text = number_font_legend.render(description, True, (0, 0, 0))
            self.screen.blit(legend_text, (legend_x + 40, legend_y))

            # Move to the next item in the legend
            legend_y += legend_margin

        # Update the display
        pygame.display.update()

        # Control the frame rate
        self.clock.tick(10)

    def _create_sprites(self) -> object:
        """
        Create sprites for the cases, based on the contents of self.cases
        """
        for case in self.cases[self.cases.t == self.t].itertuples():
            disease = "dengue" if case.disease == 0 else "chik"
            clr = (0, 255, 0) if disease == "dengue" else (255, 0, 0)
            spr = CaseSprite(case.Index, case.x, case.y, case.t, disease, clr, 2, 1, self)
            if disease == "dengue":
                spr.add(self.dengue_group)
            else:
                spr.add(self.chik_group)

    def plot_confusion_map(self, title="Mapa de Confusão Espacial", save_path=None):
        """
        Gera um mapa espacial colorindo os casos baseados no resultado da classificação
        (TP, TN, FP, FN, Erro de Classe).
        Deve ser chamado AO FINAL de um episódio ou teste.
        """
        if self.obs_cases.empty:
            print("Erro: Não há casos no histórico para plotar.")
            return

        # Listas para guardar as coordenadas de cada categoria
        coords = {
            'TP': [],  # True Positive (Acertou Doença)
            'TN': [],  # True Negative (Acertou "Outro")
            'FP': [],  # False Positive (Disse Doença, era Outro)
            'FN': [],  # False Negative (Disse Outro, era Doença)
            'Misclass': []  # Misclassification (Era Dengue, disse Chik ou vice-versa)
        }

        print("Gerando Mapa de Confusão...")

        # Itera sobre o histórico completo do agente (obs_cases)
        for index, row in self.obs_cases.iterrows():
            # Pega a VERDADE ABSOLUTA (do real_cases, usando o mesmo índice)
            # Nota: Assume-se que os índices de obs_cases correspondem aos de real_cases
            try:
                true_disease = self.real_cases.loc[index, 'disease']
            except KeyError:
                continue  # Caso raro de desalinhamento, ignora

            # Pega o diagnóstico FINAL do agente
            agent_diag = row['agent_diagnosis']
            x, y = row['x'], row['y']

            # Definições de "Doença" (0=Dengue, 1=Chik) vs "Não Doença" (2=Outro)
            is_true_disease = true_disease in [0, 1]
            is_true_other = true_disease == 2
            is_agent_disease = agent_diag in [0, 1]
            is_agent_other = agent_diag == 2

            # Lógica de Classificação
            if is_true_other and is_agent_other:
                coords['TN'].append((x, y))  # Verdadeiro Negativo
            elif is_true_disease and (agent_diag == true_disease):
                coords['TP'].append((x, y))  # Verdadeiro Positivo (Acerto exato)
            elif is_true_other and is_agent_disease:
                coords['FP'].append((x, y))  # Falso Positivo (Alarme Falso)
            elif is_true_disease and is_agent_other:
                coords['FN'].append((x, y))  # Falso Negativo (Omissão)
            elif is_true_disease and is_agent_disease and (agent_diag != true_disease):
                coords['Misclass'].append((x, y))  # Erro de Classificação (Trocou as doenças)

        # --- Plotting com Matplotlib ---
        fig, ax = plt.subplots(figsize=(10, 10))

        # Configura limites e inversão do eixo Y (para combinar com Pygame/Matrizes)
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # Estilos para cada categoria (Cor, Marcador, Legenda)
        styles = {
            'TN': {'color': 'lightgray', 'marker': 'o', 'label': 'True Negative (Acertou "Outro")', 's': 20,
                   'alpha': 0.3},
            'TP': {'color': 'green', 'marker': '^', 'label': 'True Positive (Acertou Doença)', 's': 60},
            'FP': {'color': 'red', 'marker': 'X', 'label': 'False Positive (Alarme Falso)', 's': 60},
            'FN': {'color': 'blue', 'marker': 'v', 'label': 'False Negative (Deixou Passar)', 's': 60},
            'Misclass': {'color': 'orange', 'marker': 's', 'label': 'Misclassification (Trocou Doença)', 's': 50},
        }

        # Plota os pontos de cada categoria
        for category, points in coords.items():
            if points:
                px, py = zip(*points)
                ax.scatter(px, py, **styles[category])

        # Desenha os centros dos focos reais (Ground Truth) para referência visual
        dengue_circle = plt.Circle(self.dengue_center, self.dengue_radius, color='green', fill=False, linestyle='--',
                                   alpha=0.5, label='Raio Dengue Real')
        chik_circle = plt.Circle(self.chik_center, self.chik_radius, color='orange', fill=False, linestyle='--',
                                 alpha=0.5, label='Raio Chik Real')
        ax.add_patch(dengue_circle)
        ax.add_patch(chik_circle)

        # Decoração do Gráfico
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Coordenada X")
        ax.set_ylabel("Coordenada Y")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Legenda")
        ax.grid(True, linestyle=':', alpha=0.4)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Mapa salvo em: {save_path}")

        return fig, ax

    def get_episode_metrics(self):
        """
        Calcula métricas detalhadas (Clínicas e Econômicas) ao final do episódio.
        Deve ser chamado apenas quando done=True.
        """
        # 1. Recuperar Verdade vs Estimativa
        y_true = self.cases['disease'].values  # 0: Dengue, 1: Chik, 2: Outro
        y_pred = self.obs_cases['agent_diagnosis'].values

        # 2. Calcular Matriz de Confusão para Dengue (Classe 0)
        # Consideramos Dengue como "Positivo" e (Chik + Outro) como "Negativo" para estas métricas
        TP = np.sum((y_true == 0) & (y_pred == 0))
        TN = np.sum((y_true != 0) & (y_pred != 0))
        FP = np.sum((y_true != 0) & (y_pred == 0))
        FN = np.sum((y_true == 0) & (y_pred != 0))

        epsilon = 1e-7  # Para evitar divisão por zero

        # 3. Métricas Clínicas
        sensitivity = TP / (TP + FN + epsilon)  # Recall (Dengue)
        specificity = TN / (TN + FP + epsilon)
        precision = TP / (TP + FP + epsilon)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + epsilon)
        accuracy = (TP + TN) / len(y_true)

        # 4. Métricas Econômicas
        # Contar total de testes realizados (listas testd e testc guardam histórico)
        total_tests = len(self.testd) + len(self.testc)
        total_cases = len(y_true)

        # Custo total (baseado nos custos definidos no __init__)
        # Assumindo: Teste=1.0, Confirm/Discard=0.0, DoNothing=0.1
        # Se quiser usar o self.total_reward acumulado, pode usar, mas aqui calculamos custo "bruto" de operação
        test_cost = total_tests * 1.0

        # Custo Médio por Diagnóstico Correto (Total Gasto / Total Acertos)
        total_correct = TP + TN
        cost_per_correct_diagnosis = test_cost / (total_correct + epsilon)

        # Taxa de Redução de Testes (Comparado a testar todos para ambas doenças = 2 testes por pessoa)
        # Cenário base: Testar tudo = 2 * total_cases
        potential_tests = total_cases * 2
        test_reduction_rate = 1 - (total_tests / potential_tests)

        return {
            "Acurácia": accuracy,
            "Sensibilidade (Dengue)": sensitivity,
            "Especificidade": specificity,
            "F1-Score": f1_score,
            "Precisão": precision,
            "Custo Total de Testes": test_cost,
            "Testes Realizados": total_tests,
            "Custo por Acerto": cost_per_correct_diagnosis,
            "Redução de Testes (%)": test_reduction_rate * 100,
            "Recompensa Total": self.total_reward
        }

class CaseSprite(pygame.sprite.Sprite):
    def __init__(
            self,
            id: int,
            x: int,
            y: int,
            t: int,
            disease_name: str,
            color: tuple,
            size: int,
            scaling_factor: float,
            env: DengueDiagnosticsEnv,
    ):
        super().__init__()
        self.case_id = id
        self.image = pygame.Surface((size, size))
        self.position = (x, y)
        self.disease_name = disease_name
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = (x * scaling_factor, y * scaling_factor)
        self.env = env  # And this line

    def mark_as_tested(self, status: int):
        """
        Mark the case as tested
        """
        if status == 0:  # dengue
            self.image = pygame.image.load(
                os.path.join(os.path.dirname(__file__),"dengue_test.png")).convert_alpha()
        elif status == 1:  # chik
            self.image = pygame.image.load(
                os.path.join(os.path.dirname(__file__),"chick_test.png")).convert_alpha()
        elif status == 2:  # inconclusive
            self.image = pygame.image.load(
                os.path.join(os.path.dirname(__file__),"epi_test.png")).convert_alpha()
        elif status == 3:
            self.image = pygame.image.load(
                os.path.join(os.path.dirname(__file__), "no_test.png")).convert_alpha()
        elif status == 4:
            self.image = pygame.image.load(
                os.path.join(os.path.dirname(__file__), "confirm_test.png")).convert_alpha()
        elif status == 5:
            self.image = pygame.image.load(
                os.path.join(os.path.dirname(__file__), "discard_test.png")).convert_alpha()
        self.rect = self.image.get_rect(center=self.rect.center)

    def update(self, *args, **kwargs):
        pass

class CaseGroup(pygame.sprite.RenderPlain):
    def __init__(self, name, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.name = name  # Name of the disease

    @property
    def cases(self):
        return self.sprites()

    def update(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    # Test the environment
    total_time = 60
    env = DengueDiagnosticsEnv(epilength=total_time, size=500, render_mode="human")
    obs = env.reset()

    clock = pygame.time.Clock()

    for t in range(total_time):
        try:
            pygame.event.get()
            action = env.action_space.sample()  # Random action selection
            obs, reward, done, _, info = env.step(action)
            # print(env.get_individual_rewards_at_t(t))
            # print(f"Step: {t}, Reward: {reward}, Done: {done}")

            env.render()
            clock.tick(10)
            # pygame.time.wait(60)
        except:
            pass
    pygame.quit()


# TODO fazer com o delay na recompensa
# TODO avaliar em intervalos maiores
