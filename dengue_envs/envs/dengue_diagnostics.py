# Basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional

# Import simulation tools
import gymnasium as gym
import pygame

from dengue_envs.data.generator import World
from dengue_envs.metrics.episode_metrics import episode_metrics
from dengue_envs.rendering import PygameRenderer, plot_epidemic_map
from gymnasium import spaces

# Núcleo modular (Fase 1): o ambiente agora DELEGA a lógica de clínica,
# sincronização de casos, confirmação epidemiológica e recompensa para
# `dengue_envs.core`. Os métodos internos (`_dengue_lab_test`, `_epi_confirm`,
# etc.) foram mantidos como finas camadas de compatibilidade que apenas chamam
# o núcleo, preservando a API pública e o comportamento validado nos testes.
from dengue_envs.core import (
    ClinicalModel,
    LabResultQueue,
    RewardEngine,
    epi_confirm,
    sync_obs_cases,
    update_case_status,
)


class DengueDiagnosticsEnv(gym.Env):
    metadata = {"render_modes": ["human", "console"], "render_fps": 1}

    def __init__(
            self,
            size: int = 400,
            episize: int = 150,
            epilength: int = 60,
            reward_delay_days: int = 5, 
            dengue_center: Optional[Tuple[int, int]] = None,
            chik_center: Optional[Tuple[int, int]] = None,
            dengue_radius: Optional[int] = None,
            chik_radius: Optional[int] = None,
            randomize_outbreak: bool = True,
            clinical_specificity: Union[float, Tuple[float, float]] = 0.8,
            start_day: int = 1,
            lab_delay_days: Optional[int] = None,
            settle_days: Optional[int] = None,
            reward_correct_decision: float = 10.0,
            penalty_incorrect_decision: float = -20.0,
            penalty_missed_case: float = -30.0,
            final_correct_bonus: float = 1.0,
            penalty_untested_misdiagnosed: float = -10.0,
            render_mode=None,
            world_builder=None,
    ):
        """

        Args:
            size: Size of the world
            episize: total number of cases in the epidemic
            epilength: length of the epidemic in days
            dengue_center: centro do foco de dengue; ``None`` = sortear a cada reset
            chik_center: centro do foco de chik; ``None`` = sortear a cada reset
            dengue_radius / chik_radius: raio do foco; ``None`` = sortear a cada reset
            randomize_outbreak: se True, sorteia R0 de cada curva SIR a cada reset
                (epidemias com contagens diárias diferentes). Se False, usa R0 fixo.
            clinical_specificity: specificity of the clinical diagnosis
            render_mode: render mode
            reward_delay_days: delay in days for the reward to be paid
            start_day: first simulation day when the agent takes actions (default 1)
            lab_delay_days: atraso (dias) até o resultado do laboratório ficar
                disponível. Se None, usa o mesmo valor de reward_delay_days.
            settle_days: dias extras após o último dia da epidemia para os
                atrasos (recompensa/laboratório) amadurecerem antes de encerrar
                o episódio. Se None, usa max(reward_delay_days, lab_delay_days).
            reward_correct_decision: bônus (atrasado) por decisão correta
            penalty_incorrect_decision: penalidade por confirmar errado (falso positivo)
            penalty_missed_case: penalidade por descartar um doente real (falso negativo)
            final_correct_bonus: bônus final por caso corretamente classificado
            penalty_untested_misdiagnosed: penalidade final por caso errado e nunca testado
            world_builder: callable opcional ``(env) -> World-like`` (ex.: Kriging).
                Se None, usa o ``World`` sintético clássico.
        """
        super().__init__()
        self.start_day = start_day
        self.t = start_day
        self._world_builder = world_builder

        self.reward_delay = reward_delay_days
        self.costs = np.array([1.0, 1.0, 0.5, 0.1, 0.0, 0.0])
        # Motor de recompensa (custo imediato + desfechos atrasados + placar
        # final). Detém a fila `pending_rewards`, exposta abaixo como
        # propriedade de leitura.
        self.reward_engine = RewardEngine(
            costs=self.costs,
            reward_delay_days=reward_delay_days,
            reward_correct_decision=reward_correct_decision,
            penalty_incorrect_decision=penalty_incorrect_decision,
            penalty_missed_case=penalty_missed_case,
            final_correct_bonus=final_correct_bonus,
            penalty_untested_misdiagnosed=penalty_untested_misdiagnosed,
        )
        # Atraso do laboratório: se não informado, acompanha o delay da recompensa.
        self.lab_delay_days = (
            reward_delay_days if lab_delay_days is None else lab_delay_days
        )
        # Fila de resultados de laboratório pendentes (turnaround do exame).
        self.lab_queue = LabResultQueue(self.lab_delay_days)
        self.size = size
        self.episize = episize
        self.epilength = epilength
        # Horizonte do episódio: o último dia com casos novos é `epilength - 1`;
        # depois dele damos `settle_days` para os atrasos (recompensa/laboratório)
        # amadurecerem. Substitui o antigo `epilength + 10` arbitrário.
        if settle_days is None:
            settle_days = max(self.reward_delay, self.lab_delay_days)
        self.settle_days = settle_days
        self.horizon = (self.epilength - 1) + self.settle_days
        # Parâmetros de foco: None => amostrados em cada reset (via np_random).
        self._fixed_dengue_center = dengue_center is not None
        self._fixed_chik_center = chik_center is not None
        self._fixed_dengue_radius = dengue_radius is not None
        self._fixed_chik_radius = chik_radius is not None
        self.dengue_center = dengue_center if self._fixed_dengue_center else (0, 0)
        self.chik_center = chik_center if self._fixed_chik_center else (0, 0)
        self.dengue_radius = dengue_radius if self._fixed_dengue_radius else 90
        self.chik_radius = chik_radius if self._fixed_chik_radius else 90
        self.randomize_outbreak = randomize_outbreak
        self.dengue_r0 = 1.5
        self.chik_r0 = 1.2
        # Aceita float (valor fixo) ou (min, max) como tupla/lista (ex.: vindo de YAML).
        self.specificity_setting = (
            tuple(clinical_specificity)
            if isinstance(clinical_specificity, list)
            else clinical_specificity
        )
        self.clinical_specificity = 0.8
        self.clinical_model = ClinicalModel(self.clinical_specificity)
        self.world = None
        self.real_cases = pd.DataFrame()
        self.num_cases = 0

        # Observations are dictionaries as defined below.
        # Data are represented as sequences of cases.
        self.observation_space = spaces.Dict(
            {
                "clinical_diagnostic": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Discrete(self.size),  # x coordinate
                            spaces.Discrete(self.size),  # y coordinate
                            spaces.Discrete(3),  # Diagnostic: 0: dengue, 1: chik, 2: other
                        )
                    )
                ),  # Clinical diagnosis: 0: dengue, 1: chik, 2: other
                "testd": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Discrete(max(self.episize * 2, 1)),  # case id
                            spaces.Discrete(4)
                            # Dengue testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                        )
                    )
                ),
                "testc": spaces.Sequence(
                    spaces.Tuple((
                        spaces.Discrete(max(self.episize * 2, 1)),  # case id
                        spaces.Discrete(4)
                        # Chikungunya testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                    ))
                ),
                "epiconf": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Discrete(max(self.episize * 2, 1)),  # case id
                            spaces.Discrete(2)  # Epidemiological confirmation: 0: no, 1: yes
                        )
                    )
                ),
                "tnot": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Discrete(max(self.episize * 2, 1)),  # case id
                            spaces.Discrete(self.epilength)  # Day of the clinical diagnosis
                        )
                    )
                ),
            }
        )

        self.action_space = spaces.Sequence(
            spaces.Tuple((spaces.Discrete(max(self.episize * 2, 1)), spaces.Discrete(6)))
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
        self.multiclass_accuracy_history = []
        self.mape_history = []

        self.obs_cases = pd.DataFrame()

        self.obs = {"testd": 0, "testc": 1, "epiconf": 2, "tnot": 3, "nothing": 4, "confirm": 5, "discard": 6,
                    "clinical_diagnostic": 7}

        # Primeiro episódio: gera o mundo a partir da seed 0 (reprodutível nos testes).
        self.reset(seed=0)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = None
        # Initialize rendering
        if self.render_mode is not None:
            self._render_init(mode=self.render_mode)

        self.individual_rewards = [[0]]

    @property
    def pending_rewards(self):
        """Fila de recompensas pendentes (mantida pelo `RewardEngine`).

        Exposta como propriedade de leitura para compatibilidade com código que
        inspecionava `env.pending_rewards` no ambiente monolítico antigo.
        """
        return self.reward_engine.pending_rewards

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
        """Append newly reported cases while preserving prior tests/decisions.

        Delega para `dengue_envs.core.case_store.sync_obs_cases`.
        """
        self.obs_cases = sync_obs_cases(
            self.obs_cases, self.cases, self.clinical_model, self.np_random
        )

    def _render_init(self, mode="human"):
        """
        Inicializa a renderização delegando ao `PygameRenderer`.
        """
        if mode == "console":
            self.renderer = None
            return
        self.renderer = PygameRenderer(
            self.world.size,
            render_fps=self.metadata.get("render_fps", 10),
            human=(mode == "human"),
        )

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
        Apply clinical uncertainty to newly reported cases.

        Delega para `ClinicalModel.apply_uncertainty`.
        """
        return self.clinical_model.apply_uncertainty(cases_df, self.np_random)

    def _calc_reward(self, true, estimated, action, terminated=False):
        """
        Calcula a recompensa com custos imediatos e agenda as decisões finais (delay).

        Delega o cálculo para `RewardEngine.compute`, mantendo aqui apenas a
        contabilidade acumulada do episódio (`total_reward`, `individual_rewards`).
        """
        step_reward = self.reward_engine.compute(
            action,
            self.t,
            self.real_cases,
            self.obs_cases,
            terminated=terminated,
        )
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
            self.multiclass_accuracy_history.append(0.0)
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

        # Acurácia MULTICLASSE (3 classes): fração de casos com a classe exata
        # (dengue/chik/outro) correta. É a que alimenta o gráfico de acurácia,
        # pois não mascara a confusão chik<->outro como a binária faz.
        correct_multiclass = sum(
            1 for t, e in zip(true, estimated) if t["disease"] == e[2]
        )
        multiclass_accuracy = correct_multiclass / total_cases

        self.accuracy.append(mean_accuracy)
        self.mean_accuracy_history.append(mean_accuracy)
        self.multiclass_accuracy_history.append(multiclass_accuracy)

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

        Delega para `ClinicalModel.dengue_lab_test`.
        """
        true_disease = int(self.real_cases.loc[case_id, "disease"])
        return self.clinical_model.dengue_lab_test(true_disease, self.np_random)

    def _chik_lab_test(self, case_id):
        """
        Returns the chikungunya test result for a case, conditioned on the TRUE disease.

        Delega para `ClinicalModel.chik_lab_test`.
        """
        true_disease = int(self.real_cases.loc[case_id, "disease"])
        return self.clinical_model.chik_lab_test(true_disease, self.np_random)

    def _update_case_status(self, action, index, result):
        """
        Atualiza o status do caso e o diagnóstico do agente com base nos
        resultados dos testes. Delega para `core.clinical.update_case_status`.
        """
        update_case_status(self.obs_cases, action, index, result)

    def _apply_lab_results(self, entries):
        """Aplica ao `obs_cases` uma lista de resultados de laboratório."""
        for action_id, case_id, result in entries:
            if case_id in self.obs_cases.index:
                self._update_case_status(action_id, case_id, result)

    def _apply_matured_lab_results(self, t):
        """Aplica os resultados de laboratório que ficam disponíveis no dia `t`."""
        self._apply_lab_results(self.lab_queue.pop_matured(t))

    def _flush_lab_results(self):
        """Libera todos os resultados pendentes (fim do episódio)."""
        self._apply_lab_results(self.lab_queue.flush())

    def _epi_confirm(self, case_id):
        """
        Returns the epidemiological confirmation for a case based on the local
        case density of the currently suspected disease.
        1: confirmed by epidemiological evidence, 0: not confirmed

        Delega para `core.epi_confirm.epi_confirm`.
        """
        x, y = self.get_case_xy(case_id)
        clinical = int(self.obs_cases.loc[case_id, "disease"])
        return epi_confirm(clinical, x, y, self.dmap, self.cmap)

    def _sample_outbreak_params(self) -> None:
        """Sorteia focos espaciais e R0 quando não foram fixados na config."""
        margin = min(50, max(10, self.size // 8))
        lo, hi = margin, self.size - margin
        max_radius = max(margin + 1, self.size // 4)
        radius_lo = max(5, margin // 2)
        radius_hi = max(radius_lo + 1, max_radius)

        if not self._fixed_dengue_center:
            self.dengue_center = (
                int(self.np_random.integers(lo, hi)),
                int(self.np_random.integers(lo, hi)),
            )
        if not self._fixed_chik_center:
            self.chik_center = (
                int(self.np_random.integers(lo, hi)),
                int(self.np_random.integers(lo, hi)),
            )
        if not self._fixed_dengue_radius:
            self.dengue_radius = int(self.np_random.integers(radius_lo, radius_hi + 1))
        if not self._fixed_chik_radius:
            self.chik_radius = int(self.np_random.integers(radius_lo, radius_hi + 1))

        if self.randomize_outbreak:
            # Dengue: surto principal (R0 claramente epidêmico, > 1).
            self.dengue_r0 = float(self.np_random.uniform(1.45, 1.85))
            # Chik: sempre menor que dengue (65–88% do R0 da dengue), simulando
            # um surto secundário mais contido.
            chik_frac = float(self.np_random.uniform(0.65, 0.88))
            self.chik_r0 = max(1.12, self.dengue_r0 * chik_frac)
            if self.chik_r0 >= self.dengue_r0:
                self.chik_r0 = self.dengue_r0 - 0.10
        else:
            self.dengue_r0 = 1.5
            self.chik_r0 = 1.2

    def _epidemic_is_valid(self) -> bool:
        """Verifica se o ``World`` gerado tem surtos não triviais e realistas."""
        w = self.world
        # Sempre epidêmico: R0 > 1 e chik claramente menor que dengue.
        if self.dengue_r0 <= 1.05 or self.chik_r0 <= 1.05:
            return False
        if self.chik_r0 >= self.dengue_r0:
            return False
        # Volume mínimo de casos (evita curvas degeneradas).
        if w.dengue_total < 15 or w.chik_total < 8:
            return False
        if w.chik_total >= w.dengue_total:
            return False
        # Pelo menos alguns casos nos primeiros dias ativos do agente (t=1).
        if w.casedf is None or w.casedf.empty:
            return False
        early = w.casedf[w.casedf["t"] <= 1]
        return len(early) >= 5

    def _update_action_space(self) -> None:
        """Atualiza o espaço de ações quando o total de casos muda."""
        self.action_space = spaces.Sequence(
            spaces.Tuple((spaces.Discrete(self.num_cases), spaces.Discrete(6)))
        )

    def _create_world(self) -> None:
        """Gera um novo mundo a partir do RNG atual do episódio.

        Com ``world_builder`` (ex.: Kriging), a espacialidade vem do gerador;
        o SIR (R0) ainda pode ser randomizado em ``_sample_outbreak_params``.
        """
        max_attempts = 25
        for attempt in range(max_attempts):
            self._sample_outbreak_params()
            if self._world_builder is not None:
                self.world = self._world_builder(self)
                if not self._fixed_dengue_center:
                    self.dengue_center = tuple(self.world.dengue_center)
                if not self._fixed_chik_center:
                    self.chik_center = tuple(self.world.chik_center)
                if not self._fixed_dengue_radius:
                    self.dengue_radius = int(self.world.dengue_radius)
                if not self._fixed_chik_radius:
                    self.chik_radius = int(self.world.chik_radius)
            else:
                self.world = World(
                    self.size,
                    self.episize,
                    self.epilength,
                    self.dengue_center,
                    self.chik_center,
                    self.dengue_radius,
                    self.chik_radius,
                    dengue_r0=self.dengue_r0,
                    chik_r0=self.chik_r0,
                    random_state=self.np_random,
                )
            if self._epidemic_is_valid():
                break
            if not self.randomize_outbreak or attempt == max_attempts - 1:
                break
        self.real_cases = self.world.casedf.copy()
        self.num_cases = len(self.real_cases)
        self._update_action_space()

    def reset(self, seed: int = None, options=None, reset_data: bool = False) -> Tuple[Dict, Dict]:
        """
        Resets the environment to the initial state
        Args:
            reset_data: legado; ignorado — o mundo é sempre recriado no reset.
            options: dict opcional; ``regenerate_world=False`` mantém o mundo
                anterior (útil para debug).

        Returns:

        """
        super().reset(seed=seed)
        options = options or {}
        regenerate_world = options.get("regenerate_world", True)

        if isinstance(self.specificity_setting, tuple):
            # Se for uma tupla (min, max), sorteia um valor uniforme
            low, high = self.specificity_setting
            self.clinical_specificity = self.np_random.uniform(low=low, high=high)
        else:
            # Se for um float, usa esse valor fixo
            self.clinical_specificity = self.specificity_setting
        # Recria o modelo clínico com a especificidade (re)amostrada.
        self.clinical_model = ClinicalModel(self.clinical_specificity)

        if regenerate_world or self.world is None:
            self._create_world()

        self.testd = []
        self.testc = []
        self.epiconf = []
        self.final = []
        self.rewards = []
        self.accuracy = []
        self.mean_accuracy_history = []
        self.multiclass_accuracy_history = []
        self.mape_history = []
        self.total_reward = 0
        self.reward_engine.reset()
        self.lab_queue.reset()
        if getattr(self, "renderer", None) is not None:
            self.renderer.reset()
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
            if action_id == 0:  # Teste de Dengue: amostra colhida hoje, laudo atrasa
                test_result = self._dengue_lab_test(case_id)
                self.testd.append((case_id, test_result))
                self.lab_queue.schedule(0, case_id, test_result, self.t)
            elif action_id == 1:  # Teste de Chik: amostra colhida hoje, laudo atrasa
                test_result = self._chik_lab_test(case_id)
                self.testc.append((case_id, test_result))
                self.lab_queue.schedule(1, case_id, test_result, self.t)
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

        terminated = self.t >= self.horizon

        # Resultados de laboratório respeitam o atraso: aplicamos hoje os que
        # amadureceram e, no fim do episódio, liberamos todos os pendentes.
        if terminated:
            self._flush_lab_results()
        else:
            self._apply_matured_lab_results(self.t)

        estimated_for_accuracy = tuple(
            (c.x, c.y, c.agent_diagnosis)
            for c in self.obs_cases.itertuples()
        )

        true_cases = self.cases.to_dict(orient="records")

        self.calc_accuracy(true_cases, estimated_for_accuracy)

        reward = self._calc_reward(
            self.cases.to_dict(orient="records"),
            estimated_for_accuracy,
            action,
            terminated=terminated 
        )
        # print(f"Reward: {reward} \t Total Reward: {self.total_reward}")

        self.rewards.append(self.total_reward)

        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.create_sprites(self.cases, self.t)
            self.renderer.update_sprites(action)
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
        """Delega ao renderer a atualização dos ícones dos sprites."""
        if self.renderer is not None:
            self.renderer.update_sprites(actions)

    def render(self):
        """Renderiza o frame atual, delegando ao `PygameRenderer`.

        O gráfico de acurácia representa a acurácia MULTICLASSE (3 classes).
        """
        if self.renderer is None:
            return
        self.renderer.render(
            self.t,
            self.rewards,
            self.multiclass_accuracy_history,
            accuracy_label="Multiclass Accuracy",
        )

    def _create_sprites(self):
        """Delega ao renderer a criação de sprites para os casos do dia `t`."""
        if self.renderer is not None:
            self.renderer.create_sprites(self.cases, self.t)

    def plot_epidemic_map(self, title="Mapa da Epidemia (Ground Truth)", save_path=None, show=False):
        """Mapa espacial da epidemia VERDADEIRA (``real_cases``).

        Delega para ``dengue_envs.rendering.epidemic_map.plot_epidemic_map``,
        funcionando com qualquer gerador cujo ``casedf`` siga o esquema padrão.
        """
        return plot_epidemic_map(
            self.real_cases,
            self.size,
            dengue_center=self.dengue_center,
            chik_center=self.chik_center,
            dengue_radius=self.dengue_radius,
            chik_radius=self.chik_radius,
            title=title,
            save_path=save_path,
            show=show,
        )

    def plot_confusion_map(self, title="Mapa de Confusão Espacial", save_path=None, show=False):
        """
        Gera um mapa espacial colorindo os casos baseados no resultado da classificação
        (TP, TN, FP, FN, Erro de Classe).
        Deve ser chamado AO FINAL de um episódio ou teste.
        """
        if self.obs_cases.empty:
            print("Erro: Não há casos no histórico para plotar.")
            return None

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

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    def get_episode_metrics(self):
        """
        Calcula métricas detalhadas (clínicas e econômicas) ao final do episódio.
        Deve ser chamado apenas quando done=True.

        Verdade e predição são alinhadas por `case_id` (índice), evitando
        comparações posicionais frágeis. Delega o cálculo para
        `dengue_envs.metrics.episode_metrics`.
        """
        idx = self.obs_cases.index
        # Alinha a doença VERDADEIRA de cada caso observado pelo seu id.
        y_true = self.real_cases.loc[idx, "disease"].to_numpy()
        y_pred = self.obs_cases["agent_diagnosis"].to_numpy()

        total_tests = len(self.testd) + len(self.testc)
        return episode_metrics(y_true, y_pred, total_tests, self.total_reward)


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


# TODO avaliar em intervalos maiores
