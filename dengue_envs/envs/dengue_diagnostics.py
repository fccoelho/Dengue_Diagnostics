# Basic packages
import copy
import time

import numpy as np
import pandas as pd
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
            dengue_center=(100, 100),
            chik_center=(300, 300),
            dengue_radius=90,
            chik_radius=90,
            clinical_specificity=0.8,
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

        # We have 6 actions, corresponding to "test for dengue", "test for chik", "epi confirm", "Do nothing", confirm, discard
        self.action_space = spaces.Sequence(
            spaces.Tuple((spaces.Discrete(self.episize), spaces.Discrete(6))) #case id, action
        )
        self.costs = np.array([0.5, 0.5, 0.1, 0.0, 0.0, 0.0])

        # The lists below will be populated by the step() method, as the cases are being "generated"
        self.cases: pd.DataFrame = self.world.get_series_up_to_t(self.t)  # True cases
        self.obs_cases = self._apply_clinical_uncertainty()  # Observed cases (after applying uncertainty)

        self.testd = []
        self.testc = []
        self.epiconf = []
        self.final = []

        self.tcase = []
        self.rewards = []

        # cumulative map of cases up to self.t
        self.dmap, self.cmap = self.world.get_maps_up_to_t(self.t)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.clock = None  # self.metadata["render_fps"]
        # Initialize rendering
        if self.render_mode is not None:
            self._render_init(mode=self.render_mode)

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
            "tnot": tuple((c.Index, c.t) for c in self.obs_cases.itertuples())
        }

    def _apply_clinical_uncertainty(self):
        """
        Apply clinical uncertainty to the observations: Observations are subject to misdiagnosis based on the clinical specificity
        """
        obs_case_df = copy.deepcopy(self.cases)  # Copy of the true cases
        for case in obs_case_df.iterrows():
            if self.np_random.uniform() < 0.01:
                case[1].disease = 2  # Other disease
                continue
            if case[1].disease == 0:
                if (
                        self.np_random.uniform() > self.clinical_specificity
                ):  # Misdiagnosed as chik
                    case[1].disease = 1
            elif case[1].disease == 1:
                if (
                        self.np_random.uniform() > self.clinical_specificity
                ):  # Misdiagnosed as dengue
                    case[1].disease = 0

        return obs_case_df

    def _calc_reward(self, true, estimated, action):
        """
        Calculate the reward based on the true count and the actions taken
        """
        if len(estimated) == 0:
            return 0
        true_numdengue = len([c for c in true if c["disease"] == 0])
        estimated_numdengue = len([c for c in estimated if c[2] == 0])
        # Mean absolute percentage error
        mape = np.abs(true_numdengue - estimated_numdengue) / max(1, true_numdengue)
        accuracy_reward = 1 if mape < 0.15 else 0
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
        if self.np_random.uniform() < 0.1:  # 90% sensitivity
            return 3  # Inconclusive
        if self.np_random.uniform() >= 0.9:  # 90% specificity
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
        if self.np_random.uniform() < 0.1:  # 90% sensitivity
            return 3  # Inconclusive
        if self.np_random.uniform() >= 0.9:  # 90% specificity
            return 1
        else:
            return 2

    def _update_case_status(self, action):
        pass

    def _epi_confirm(self, case):
        """
        Returns the epidemiological confirmation for a case
        """
        if case[2] == 0:  # Dengue suspicion
            return 1 if self.map[case[0][0], case[0][1]] > 1 else 0
        else:
            return 1 if self.cmap[case[0][0], case[0][1]] > 1 else 0

    def reset(self, seed: int = None, options=None, reset_data: bool = False) -> Tuple[Dict, Dict]:
        """
        Resets the environment to the initial state
        Args:
            reset_data: If the world data is supposed to re-created as well. Default is False.

        Returns:

        """
        super().reset(seed=seed)
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

        self.cases = self.world.get_series_up_to_t(0)
        self.obs_cases = self._apply_clinical_uncertainty()

        observation = self._get_obs()

        info = self._get_info()

        self.t = 1

        return observation, info

    def step(self, action):
        """
        Apply the actions for every case at the current timestep (t)
        and the returns the observation(state at t+1), reward, termination status and info
        action: [list of decisions for all current cases]: 0: test for dengue, 1: test for chik, 2: epi confirm, 3: Does nothing, 4: Confirm, 5: Discard
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for {self.action_space}")
        # get the current true state
        self.cases = self.world.get_series_up_to_t(self.t)
        observation = self._get_obs()

        # apply the actions
        # Fixme: The recording of the actions are not correct.
        obs = {"testd": 0, "testc": 1, "epiconf": 2, "tnot": 3, "nothing": 4, "confirm": 5, "discard": 6, "clinical_diagnostic": 7}
        for a, o in zip(action, observation):
            if obs[o] == 0:  # Dengue test
                self.testd.append((a[0], self._dengue_lab_test(a)))
                print(f"Tested case {a[0]} for dengue")
            elif obs[o] == 1:  # Chik test
                self.testc.append((a[0], self._chik_lab_test(a)))
                print(f"Tested case {a[0]} for chik")
            elif obs[o] == 2:  # Epi confirm
                pass
                # self.epiconf.append(self._epi_confirm(a))
                # self.tcase.append(
                #     [
                #         self.t,
                #         0
                #         if not observation["clinical_diagnostic"]
                #         else observation["clinical_diagnostic"][-1],
                #     ]
                # )
            elif obs[o] == 3:  # Do nothing
                pass
            elif obs[o] == 4:  # Confirm
                self.final.append(1)
            elif obs[o] == 5:  # Discard
                self.final.append(0)

        self.update_sprites()

        # An episode is done if timestep is greter than 120
        terminated = self.t >= self.epilength + 60
        reward = self._calc_reward(
            self.cases.to_dict(orient="records"),
            observation["clinical_diagnostic"],
            action,
        )
        self.rewards.append(reward)
        if self.render_mode == "human":
            self.render()
        # Update the timestep
        self.t += 1
        self.dmap, self.cmap = self.world.get_maps_up_to_t(self.t)
        self.cases = self.world.get_series_up_to_t(self.t)
        self.obs_cases = self._apply_clinical_uncertainty()
        # get the next observation
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def update_sprites(self):
        # Update the sprites in the dengue group
        for sprite in self.dengue_group.sprites():
            for case_id, test_result in self.testd:
                if sprite.case_id == case_id:
                    sprite.mark_as_tested(0)

        # Update the sprites in the chik group
        for sprite in self.chik_group.sprites():
            for case_id, test_result in self.testc:
                if sprite.case_id == case_id:
                    sprite.mark_as_tested(1)

    def render(self):
        """
        Render the environment
        """
        self._create_sprites()
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

        # Plot learning metrics
        plot1 = lineplot(range(1, self.t + 1), self.rewards, "Step", "Total Reward", "Total Reward", "plot1")
        accuracy = [sum(self.rewards[:i+1])/(i+1) for i in range(self.t)]
        plot2 = lineplot(range(1, self.t + 1), accuracy, "Step", "Accuracy", "Total Accuracy", "plot2")

        self.plot_surface1.blit(
            pygame.transform.scale(
                pygame.image.load(plot1, "PNG"), self.plot_surface1.get_rect().size
            ),
            (0, 0),
        )
        self.plot_surface2.blit(
            pygame.transform.scale(
                pygame.image.load(plot2, "PNG"), self.plot_surface2.get_rect().size
            ),
            (0, 1),
        )
        self.screen.blit(
            self.plot_surface1, (0, 500), special_flags=pygame.BLEND_ALPHA_SDL2
        )
        self.screen.blit(
            self.plot_surface2, (400, 500), special_flags=pygame.BLEND_ALPHA_SDL2
        )

        # self.screen.blit(csurf,(0,0), special_flags=pygame.BLEND_ALPHA_SDL2)
        self.screen.blit(
            self.world_surface, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2
        )
        # self.screen.blit(pygame.transform.scale(self.world_surface, self.screen.get_rect().size), (0,0))
        pygame.display.update()
        self.clock.tick()  # Update the elapsed time in the training

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
            self.image = pygame.image.load("dengue-checked.png").convert_alpha()
        elif status == 1:  # chik
            self.image = pygame.image.load("chik-checked.png").convert_alpha()
        elif status == 2:  # inconclusive
            self.image = pygame.image.load("inconclusive.png").convert_alpha()
        self.rect = self.image.get_rect(center=self.rect.center)

    def update(self, *args, **kwargs):
        if self.case_id in self.env.testd:
            self.mark_as_tested(0)
        elif self.case_id in self.env.testc:
            self.mark_as_tested(1)


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
    total_time = 360
    env = DengueDiagnosticsEnv(epilength=total_time, size=500, render_mode="human")
    obs = env.reset()

    for t in range(total_time):
        pygame.event.get()
        action = env.action_space.sample()  # Random action selection
        obs, reward, done, _, info = env.step(action)
        print(f"Reward: {reward}", end="\r")

        # pygame.time.wait(60)
    pygame.quit()
