# Basic packages
import copy
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

# Statistics tools
import scipy.stats as st
from scipy.integrate import odeint

# Import simulation tools
import gymnasium as gym
import pygame

from gymnasium import spaces


class DengueDiagnosticsEnv(gym.Env):
    def __init__(
        self,
        size: int = 200,
        episize: int = 150,
        epilength: int = 60,
        dengue_center=(30, 30),
        chik_center=(90, 110),
        dengue_radius=10,
        chik_radius=10,
        clinical_specificity=0.8,
    ):
        self.t = 0  # timestep

        self.size = size
        self.episize = episize
        self.epilength = epilength
        self.dengue_center = dengue_center
        self.chik_center = chik_center
        self.dengue_radius = dengue_radius
        self.chik_radius = chik_radius
        self.clinical_specificity = clinical_specificity
    

        self.world = World(
            self.size,
            self.episize,
            self.epilength,
            self.dengue_center,
            self.chik_center,
            self.dengue_radius,
            self.chik_radius,
        )

        self.start_pos = self.world.chik_center  # Starting position
        self.current_pos = (
            self.start_pos
        )  

        # Observations are dictionaries as defined below.
        # Data are represented as sequences of cases.
        self.observation_space = spaces.Dict(
            {
                "clinical": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Discrete(self.world.num_rows),
                            spaces.Discrete(self.world.num_cols),
                            spaces.Discrete(3),
                        )
                    )
                ),  # Clinical diagnosis: 0: dengue, 1: chik, 2: other
                "testd": spaces.Sequence(
                    spaces.Discrete(4)
                ),  # Dengue testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                "testc": spaces.Sequence(
                    spaces.Discrete(4)
                ),  # Chikungunya testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                "epiconf": spaces.Sequence(
                    spaces.Discrete(2)
                ),  # Epidemiological confirmation: 0: no, 1: yes
                "tcase": spaces.Sequence(
                    spaces.Discrete(60)
                ),  # Day of the clinical diagnosis
            }
        )

        # We have 4 actions, corresponding to "test for dengue", "test for chik", "epi confirm", "noop", confirm, discard
        self.action_space = spaces.Sequence(spaces.Tuple(
            (spaces.Discrete(self.size), spaces.Discrete(self.size), spaces.Discrete(6))
        ))
        self.costs = np.array([0.5, 0.5, 0.1, 0.0, 0.0, 0.0])

        # The lists below will be populated by the step() method, as the cases are being generated
        self.cases = []  # True cases
        self.obs_cases = []  # Observed cases

        self.testd = np.zeros(len(self.world.case_series))-1 # Dengue test results -1: not tested, 0: negative, 1: positive, 2: inconclusive
        self.testc = np.zeros(len(self.world.case_series))-1 # Chikungunya test results -1: not tested, 0: negative, 1: positive, 2: inconclusive
        self.epiconf = np.zeros(len(self.world.case_series))-1 # Epidemiological confirmation -1: not checked, 0: not confirmed 1: confirmed
        self.final = np.zeros(len(self.world.case_series)) # Final decision: 0: discarded , 1: confirmed

        self.tcase = []
        self.rewards = []

        # cumulative cases of dengue suspicion
        self.dengue_suspicion = []
        self.chik_suspicion = []

        # cumulative map of cases up to self.t
        self.dmap = self._extract_case_xy(self.world.case_series, disease_code = 0, index = 0)
        self.cmap = self._extract_case_xy(self.world.case_series, disease_code = 1, index = 0)

        # Initialize Pygame
        pygame.init()
        self.nb_pixels = 1000
        self.cell_size = self.nb_pixels / self.world.size

        # Setting display size
        self.screen = pygame.display.set_mode(
            (self.world.num_cols * self.cell_size, self.world.num_rows * self.cell_size)
        )

    def _extract_case_xy(self, series, disease_code, index = None):
        if not index:
            return [case[:-1] for cases in sorted(series.items())[:self.t] for case in cases[1] if case[-1] == disease_code]
        else:
            return [case[:-1] for cases in sorted(series.items())[index] for case in cases[1] if case[-1] == disease_code]

    def _is_valid_position(self, pos):
        row, col = pos

        # If agent goes out of the grid
        if (
            row < 0
            or col < 0
            or row >= self.world.num_rows
            or col >= self.world.num_cols
        ):
            return False

        return True

    def _get_obs(self):
        """
        Returns the current observation.
        """
        obs_cases = self._apply_clinical_uncertainty(self.t)

        return {
            "clinical": obs_cases,
            "testd": [0] * len(obs_cases),
            "testc": [0] * len(obs_cases),
            "t": [np.nan] * len(obs_cases),
        }

    def _apply_clinical_uncertainty(self, t):
        """
        Apply clinical uncertainty to the observations: Observations are subject to misdiagnosis based on the clinical specificity
        """
        obs_case_series = copy.deepcopy(self.world.case_series[t])
        for i, case in enumerate(obs_case_series):
            if self.np_random.uniform()<0.01:
                obs_case_series[i][2] = 2
                continue
            if case[2] == 0:
                if self.np_random.uniform() > self.clinical_specificity:
                    obs_case_series[i][2] = 1
            else:
                if self.np_random.uniform() > self.clinical_specificity:
                    obs_case_series[i][2] = 0

        return obs_case_series

    def _calc_reward(self, true, estimated, action):
        """
        Calculate the reward based on the true count and the actions taken
        """
        if len(estimated) == 0:
            return 0
        errorrate = (len(estimated) - np.sum(estimated == true)) / len(estimated)
        accuracy_reward = 1 if errorrate < 0.15 else 0
        reward = accuracy_reward - 0.1 * self.costs[action[0][-1]]
        return reward

    def _get_info(self):
        """
        Returns the current map of cases for each disease"""
        self.dmap = self._extract_case_xy(self.world.case_series, disease_code = 0)
        self.cmap = self._extract_case_xy(self.world.case_series, disease_code = 1)
        return {
            "dengue_grid": self.dmap,
            "chik_grid": self.cmap,
        }

    def _dengue_lab_test(self, clinical_diag):
        """
        Returns the test result for a dengue case
        1: Negative
        2: Positive
        3: Inconclusive
        """
        if clinical_diag == 3:
            return 1
        if self.np_random.uniform() < 0.1:
            return 3  # Inconclusive
        if self.np_random.uniform() >= 0.9:
            return 1
        else:
            return 2

    def _chik_lab_test(self, clinical_diag):
        """
        Returns the test result for a chikungunya case
        1: Negative
        2: Positive
        3: Inconclusive
        """
        if clinical_diag == 3:
            return 1
        if self.np_random.uniform() < 0.1:
            return 3  # Inconclusive
        if self.np_random.uniform() >= 0.9:
            return 1
        else:
            return 2

    def _epi_confirm(self, case):
        """
        Returns the epidemiological confirmation for a case
        """
        if case[2] == 0:  # Dengue suspicion
            return 1 if self.map[case[0][0], case[0][1]] > 1 else 0
        else:
            return 1 if self.cmap[case[0][0], case[0][1]] > 1 else 0

    def reset(self):
        # Create the world
        self.world = World(
            self.size,
            self.episize,
            self.epilength,
            self.dengue_center,
            self.chik_center,
            self.dengue_radius,
            self.chik_radius,
        )

        self.cases = self.world.case_series[0]
        self.obs_cases = self._apply_clinical_uncertainty(0)

        observation = {
            "clinical": self.obs_cases,
            "testd": [0] * len(self.obs_cases),
            "testc": [0] * len(self.obs_cases),
            "t": [np.nan] * len(self.obs_cases),
        }

        info = self._get_info()

        self.current_pos = (
            self.start_pos
        )  

        return self.current_pos

    def step(self, action):
        """
        Based on the action, does the testing or epidemiological confirmation
        action: [list of decisions for all current cases]: 0: test for dengue, 1: test for chik, 2: epi confirm, 3: Does nothing, 4: Confirm, 5: Discard
        """
        # get the current true state
        cases_series = self.world.case_series[self.t]
        self.cases.extend(cases_series)

        # get the current observation
        observation = self._get_obs()

        # apply the actions
        for i,a in enumerate(action):
            if a == 0: # Dengue test
                self.testd[i] = self._dengue_lab_test()
            elif a == 1: # Chik test
                self.testc[i] = self._chik_lab_test()
            elif a == 2: # Epi confirm
                self.epiconf[i] = self._epi_confirm()
                self.tcase.append(
                    [
                        self.t,
                        0 if not observation["clinical"] else observation["clinical"][-1],
                    ]
                )
            elif a == 3:   # Do nothing
                pass

            elif a == 4:   # Confirm
                self.final[i] = 1
            elif a == 5:   # Discard
                self.final[i] = 0

        self.dengue_positive = self._extract_case_xy(self.world.case_series, disease_code = 0)
        self.chik_positive = self._extract_case_xy(self.world.case_series, disease_code = 1)
        
        self.dengue_suspicion = self._extract_case_xy(self.world.medical_suspicion_series, disease_code = 0)
        self.chik_suspicion = self._extract_case_xy(self.world.medical_suspicion_series, disease_code = 1)

        # An episode is done if timestep is greter than 120
        terminated = self.t > 120
        reward = self._calc_reward(self.cases, observation["clinical"], action)
        self.rewards.append(reward)
        info = self._get_info()

        self.t += 1

        # Reward function
        if np.array_equal(self.current_pos, self.world.dengue_center):
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        return self.current_pos, reward, done, info

    def render(self):
        def draw_rectangles(grid, color):
            for row, col in grid:
                cell_left = row * self.cell_size
                cell_top = col * self.cell_size
                pygame.draw.rect(
                    self.screen,
                    color,
                    (cell_left, cell_top, self.cell_size, self.cell_size),
                )

        # Clear the screen
        self.screen.fill((255, 255, 255))
        draw_rectangles(grid=self.dengue_positive, color=(200, 255, 200))
        draw_rectangles(grid=self.chik_positive, color=(200, 200, 255))
        
        draw_rectangles(grid=self.dengue_suspicion, color=(50, 180, 50))
        draw_rectangles(grid=self.chik_suspicion, color=(50, 50, 180))

        number_font = pygame.font.SysFont( None, 32 )
        number_image = number_font.render(f'Step {self.t}', True, (0,0,0), (255, 255, 255))
        self.screen.blit( number_image, (int((self.nb_pixels - number_image.get_width())/2), 0) )

        pygame.display.update()  # Update the display


class World:
    """
    Initialize random but concentrated distribution of dengue and Chikungunya cases
    """
    def __init__(
        self,
        size: int = 200,
        episize: int = 150,
        epilength: int = 60,
        dengue_center=(30, 30),
        chik_center=(90, 110),
        dengue_radius=10,
        chik_radius=10,
        medical_specificity = 0.95,
        medical_identification_rate = 0.05,
    ):
        """
        size: size of the world
        """
        self.size = size  # World size

        self.num_rows = size  # World represented as a 2D numpy array
        self.num_cols = size

        self.episize = episize  # Size of the epidemic
        self.epilength = epilength  # Length of the epidemic in days

        self.dengue_center = dengue_center
        self.dengue_radius = dengue_radius
        self.chik_center = chik_center
        self.chik_radius = chik_radius

        self.dengue_dist_x = st.distributions.norm(
            self.dengue_center[0], self.dengue_radius
        )
        self.dengue_dist_y = st.distributions.norm(
            self.dengue_center[1], self.dengue_radius
        )

        self.chik_dist_x = st.distributions.norm(self.chik_center[0], self.chik_radius)
        self.chik_dist_y = st.distributions.norm(self.chik_center[1], self.chik_radius)

        self.medical_specificity = medical_specificity
        self.medical_identification_rate = medical_identification_rate

        self.dengue_curve = self._get_epi_curve(R0=2.5)
        self.chik_curve = self._get_epi_curve(R0=1.5)
        
        self.case_series = []
        # Cases per day as a list of lists
        # [[[x1,y1,0], [x2,y2,0], ...], [[x1,y1,0], [x2,y2,0], ...], ...]

        self.medical_suspicion_series = []
        # Cases identified by physicians with possibility for misdiagnosis
        # For self.case_series, an example could be
        # [[[x1,y1,1], [x2,y2,0]], [[x1,y1,0], [x2,y2,1]], ...]

        self.get_daily_cases()

    def _generate_full_outbreak(self):
        dpos = list(map(list, set(map(tuple, self._extract_case_xy(self.case_series, disease_code = 0)))))
        cpos = list(map(list, set(map(tuple, self._extract_case_xy(self.case_series, disease_code = 1)))))
        return dpos, cpos

    def _get_epi_curve(self, R0=2.5):
        """
        Generate an epidemic curve
        returns the Infectious numbers per day
        """

        def SIR(y, t, beta, gamma, N):
            S, I, Inc, R = y
            return [
                -beta * S * I / N,
                beta * S * I / N - gamma * I,
                beta * S * I / N, # Cumulative Incidence
                gamma * I,
            ]

        gamma = 0.004
        beta = R0 * gamma
        y = odeint(
            SIR,
            [self.episize, 1, 0, 0],
            np.arange(0, self.epilength),
            args=(beta, gamma, self.episize),
        )

        return y[:,2]

    def get_daily_cases(self):
        """
        Generate the daily cases based on an epidemic curve
        """
        for t in range(self.epilength):
            total_cases = [case[:-1] for cases in self.case_series for case in cases]
            dcases_x = self.dengue_dist_x.rvs(int(self.dengue_curve[t]))
            dcases_y = self.dengue_dist_y.rvs(int(self.dengue_curve[t]))
            ccases_x = self.chik_dist_x.rvs(int(self.chik_curve[t]))
            ccases_y = self.chik_dist_y.rvs(int(self.chik_curve[t]))

            cases_of_t_day  = [[int(x), int(y), 0] for x, y in zip(dcases_x, dcases_y)]
            cases_of_t_day += [[int(x), int(y), 1] for x, y in zip(ccases_x, ccases_y)]

            for case in cases_of_t_day:
                if case[:-1] in total_cases:
                    cases_of_t_day.remove(case)

            self.case_series.append(cases_of_t_day)

            suspected_cases = []
            if np.random.uniform() < self.medical_identification_rate:
                nb_cases_of_t_day = range(len(cases_of_t_day))
                nb_selected_cases_of_t_day = int(np.ceil(np.random.uniform()*len(cases_of_t_day)))
                selected_cases_nbs = list(np.random.choice(nb_cases_of_t_day, nb_selected_cases_of_t_day, replace=False))
                for selected_cases_nb in selected_cases_nbs:
                    case = copy.deepcopy(cases_of_t_day[selected_cases_nb])
                    if np.random.uniform() < self.medical_specificity:
                        suspected_cases.append(case)
                    else:
                        if case[-1] == 1:
                            case[-1] = 0
                        else:
                            case[-1] = 1
                        suspected_cases.append(case)
                        
            self.medical_suspicion_series.append(suspected_cases)

    def viewer(self):
        dpos, cpos = self._generate_full_outbreak()

        fig, ax = plt.subplots()
        ax.pcolor(dpos, cmap="Greens", alpha=0.5)
        ax.pcolor(cpos, cmap="Blues", alpha=0.5)
        return fig, ax


if __name__ == "__main__":
    # Test the environment
    total_time = 1000
    env = DengueDiagnosticEnv(epilength=total_time)
    obs = env.reset()

    for t in range(total_time):
        pygame.event.get()
        action = env.action_space.sample()  # Random action selection
        obs, reward, done, _ = env.step(action)
        env.render()
        # print('Reward:', reward)
        # print('Done:', done)

        pygame.time.wait(60)