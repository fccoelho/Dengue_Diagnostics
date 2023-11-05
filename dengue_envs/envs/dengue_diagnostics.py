# Basic packages
import copy
import numpy as np
from itertools import chain
from collections import defaultdict

# Import simulation tools
import gymnasium as gym
import pygame

from dengue_envs.data.generator import World
from dengue_envs.viz import lineplot
from gymnasium import spaces


class DengueDiagnosticsEnv(gym.Env):
    metadata = {"render_modes": ["human", "console"], "render_fps": 4}
    def __init__(
        self,
        size: int = 400,
        episize: int = 150,
        epilength: int = 60,
        dengue_center=(100, 100),
        chik_center=(300, 300),
        dengue_radius=90,
        chik_radius=90,
        clinical_specificity=0.8,
        render_mode=None
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
        """
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
        self.current_pos = self.start_pos

        # Observations are dictionaries as defined below.
        # Data are represented as sequences of cases.
        self.observation_space = spaces.Dict(
            {
                "clinical_diagnostic": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Discrete(self.world.num_cols),  # x coordinate
                            spaces.Discrete(self.world.num_rows),   # y coordinate
                            spaces.Discrete(3), # Diagnostic: 0: dengue, 1: chik, 2: other
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
                    spaces.Discrete(self.epilength)
                ),  # Day of the clinical diagnosis
            }
        )

        # We have 4 actions, corresponding to "test for dengue", "test for chik", "epi confirm", "Do nothing", confirm, discard
        self.action_space = spaces.Sequence(
            spaces.Tuple(
                (
                    spaces.Discrete(self.size),
                    spaces.Discrete(self.size),
                    spaces.Discrete(6),
                )
            )
        )
        self.costs = np.array([0.5, 0.5, 0.1, 0.0, 0.0, 0.0])

        # The lists below will be populated by the step() method, as the cases are being generated
        self.cases = []  # True cases
        self.obs_cases = []  # Observed cases

        self.testd = (
            np.zeros(len(self.world.case_series)) - 1
        )  # Dengue test results -1: not tested, 0: negative, 1: positive, 2: inconclusive
        self.testc = (
            np.zeros(len(self.world.case_series)) - 1
        )  # Chikungunya test results -1: not tested, 0: negative, 1: positive, 2: inconclusive
        self.epiconf = (
            np.zeros(len(self.world.case_series)) - 1
        )  # Epidemiological confirmation -1: not checked, 0: not confirmed 1: confirmed
        self.final = np.zeros(
            len(self.world.case_series)
        )  # Final decision: 0: discarded , 1: confirmed

        self.tcase = []
        self.rewards = []

        # cumulative cases of dengue suspicion
        self.dengue_suspicion = []
        self.chik_suspicion = []

        # cumulative map of cases up to self.t
        self.dmap, self.cmap = self.world.get_maps_up_to_t(self.t)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.clock = self.metadata['render_fps']
        # Initialize rendering
        if self.render_mode is not None:
            self._render_init(mode=self.render_mode)

        

    def _render_init(self, mode="human"):
        """
        Initialize rendering
        """
        pygame.init()
        pygame.display.init()

        # Setting display size
        self.scaling_factor = 800/self.world.size  # Scaling factor for the display
        self.screen = pygame.display.set_mode(
            size=(800, 800),
            depth=32,
            flags= pygame.SCALED,
        )
        self.world_surface = pygame.Surface((self.world.size, self.world.size))
        self.world_surface.set_colorkey((0,0,0))
        self.dengue_group = CaseGroup('dengue', self.scaling_factor)
        self.chik_group = CaseGroup('chik', self.scaling_factor)

        self.plot_surface1 = pygame.Surface((400, 300))
        self.plot_surface2 = pygame.Surface((400, 300))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def _get_obs(self):
        """
        Returns the current observation.
        """
        obs_cases = self._apply_clinical_uncertainty(self.t)

        return {
            "clinical_diagnostic": obs_cases,
            "testd": [0] * len(obs_cases),
            "testc": [0] * len(obs_cases),
            "t": [np.nan] * len(obs_cases),
        }
    def _get_info(self):
        pass
    def _apply_clinical_uncertainty(self, t):
        """
        Apply clinical uncertainty to the observations: Observations are subject to misdiagnosis based on the clinical specificity
        """
        obs_case_series = copy.deepcopy(self.world.case_series[t]) # Copy of the true cases
        for i, case in enumerate(obs_case_series):
            if self.np_random.uniform() < 0.01:
                obs_case_series[i]['disease'] = 2 # Other disease
                continue
            if case['disease'] == 0:
                if self.np_random.uniform() > self.clinical_specificity:  # Misdiagnosed as chik
                    obs_case_series[i]['disease'] = 1
            elif case['disease'] == 1:
                if self.np_random.uniform() > self.clinical_specificity: # Misdiagnosed as dengue
                    obs_case_series[i]['disease'] = 0


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
        Returns the current map of cases for each disease
        """
        
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

        self.current_pos = self.start_pos

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
        for i, a in enumerate(action):
            if a == 0:  # Dengue test
                self.testd[i] = self._dengue_lab_test()
            elif a == 1:  # Chik test
                self.testc[i] = self._chik_lab_test()
            elif a == 2:  # Epi confirm
                self.epiconf[i] = self._epi_confirm()
                self.tcase.append(
                    [
                        self.t,
                        0
                        if not observation["clinical"]
                        else observation["clinical"][-1],
                    ]
                )
            elif a == 3:  # Do nothing
                pass

            elif a == 4:  # Confirm
                self.final[i] = 1
            elif a == 5:  # Discard
                self.final[i] = 0

        self.dengue_positive = self.dmap # Array of number of dengue cases per cell
        self.chik_positive = self.cmap # Array of number of chikungunya cases per cell

        

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
        dmap, cmap = self.world.get_maps_at_t(self.t)
        for (x,y),c in np.ndenumerate(dmap):
            for i in range(int(c)):
                spr = CaseSprite(x, y, 'dengue', (0, 255, 0), 2, 1)#self.scaling_factor)
                spr.add(self.dengue_group)
        self.dengue_group.draw(self.world_surface)
        for (x,y),c in np.ndenumerate(cmap):
            for i in range(int(c)):
                spr = CaseSprite(x, y, 'chik', (255, 0, 0), 2, 1)#self.scaling_factor)
                spr.add(self.chik_group)
        self.chik_group.draw(self.world_surface)
        
        # Clear the screen
        self.screen.fill((255, 255, 255))
        

        number_font = pygame.font.SysFont(None, 32)
        timestep_display = number_font.render(
            f"Step {self.t}", True, (0, 0, 0), (255, 255, 255)
        )
        self.screen.blit(
            timestep_display, (int((self.screen.get_width() - timestep_display.get_width()) / 2), 0)
        )
        # Plot learning metrics
        plot1 = lineplot([1,2,3],[1,2,3], 'x', 'y', 'Total Reward')
        plot2 = lineplot([1,2,3],[1,2,3], 'x', 'y', 'Accuracy')
        self.plot_surface1.blit(pygame.transform.scale(pygame.image.load(plot1, 'PNG'),
                                                       self.plot_surface1.get_rect().size), (0,0))
        self.plot_surface2.blit(pygame.transform.scale(pygame.image.load(plot2, 'PNG'),
                                                         self.plot_surface2.get_rect().size), (0,0))
        self.screen.blit(self.plot_surface1,(0, 500), special_flags=pygame.BLEND_ALPHA_SDL2)
        self.screen.blit(self.plot_surface2,(400, 500), special_flags=pygame.BLEND_ALPHA_SDL2)

        # self.screen.blit(csurf,(0,0), special_flags=pygame.BLEND_ALPHA_SDL2)
        self.screen.blit(self.world_surface, (0,0), special_flags=pygame.BLEND_ALPHA_SDL2)
        # self.screen.blit(pygame.transform.scale(self.world_surface, self.screen.get_rect().size), (0,0))
        pygame.display.update()  # Update the display


class CaseSprite(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int, name: str, color: tuple, size: int, scaling_factor: float):
        super().__init__()
        self.image = pygame.Surface((size, size))
        self.position = (x, y)
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = (x * scaling_factor, y * scaling_factor)


    def mark_as_tested(self, status: int):
        """
        Mark the case as tested
        """
        if status == 0:  #dengue
            self.image = pygame.image.load("dengue-checked.png")
        elif status == 1:  #chik
            self.image = pygame.image.load("chik-checked.png")
        elif status == 2:  #inconclusive
            self.image = pygame.image.load("inconclusive.png")

    def update(self, *args, **kwargs):
        pass

class CaseGroup(pygame.sprite.RenderPlain):
    def __init__(self, name, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.name = name # Name of the disease

    @property
    def cases(self):
        return self.sprites()

    def update(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    # Test the environment
    total_time = 360
    env = DengueDiagnosticsEnv(epilength=total_time, size=500, render_mode="human")
    obs = env.reset()

    for t in range(total_time):
        pygame.event.get()
        action = env.action_space.sample()  # Random action selection
        obs, reward, done, _ = env.step(action)
        env.render()
        # print('Reward:', reward)
        # print('Done:', done)

        pygame.time.wait(60)
    pygame.quit()
