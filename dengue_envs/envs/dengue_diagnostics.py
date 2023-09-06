import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces, utils
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
import scipy.stats as st
import pygame
from typing import List, Optional, Tuple


class DengueDiagnosticEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str]=None, size: int=50, render_freq=1):
        self.render_mode = render_mode
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the dengue and chikungunya cases locations on a grid.
        # Data are represented as a 2D array of size (size, size) with the number of cases in each cell.
        world = World()
        self._dengue_location, self._chik_location = world.get_grids()
        self.observation_space = spaces.Dict(
            {
                "clinical": Box(low=0, high=1, shape=(size, size), dtype=np.int16), # Clinical diagnosis: 0: dengue, 1: chik
                "testd": Box(low=0, high=3, shape=(size,size), dtype=np.int8), # Dengue testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                "testc": Box(low=0, high=3, shape=(size,size), dtype=np.int8), # Chikungunya testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                "t" : Box(low=0, high=60, shape=(size,size), dtype=np.int8), # Days since clinical diagnosis
            }
        )

        # We have 4 actions, corresponding to "test for dengue", "test for chik", "epi confirm", "noop", "right"
        self.action_space = spaces.Discrete(4)



        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"dengue": self._dengue_location, "chik": self._chik_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class World:
    "Initialize random but concentrated distribution od dengue and Chikungunya cases"
    def __init__(self, size:int=200):
        self.size = size
        self.dpos, self.cpos = self._generate_outbreak()

    def _generate_outbreak(self, dengue_center=(30, 30), chik_center=(90, 110), dengue_radius=10, chik_radius=10):
        xd = st.distributions.norm(dengue_center[0],dengue_radius).rvs(5000)
        yd = st.distributions.norm(dengue_center[1],dengue_radius).rvs(5000)
        dpos, _, _ = np.histogram2d(xd,yd,bins=(range(self.size),range(self.size)))
        xc = st.distributions.norm(chik_center[0],chik_radius).rvs(5000)
        yc = st.distributions.norm(chik_center[1],chik_radius).rvs(5000)
        cpos, _, _ = np.histogram2d(xc,yc,bins=(range(self.size),range(self.size)))
        return dpos, cpos

    def get_grids(self):
        return self.dpos, self.cpos
    def viewer(self):
        fig, ax = plt.subplots()
        ax.pcolor(self.dpos, cmap='Greens', alpha=0.5)
        ax.pcolor(self.cpos, cmap='Blues', alpha=0.5)
        return fig, ax

if __name__== "__main__":
    import matplotlib.pyplot as plt
    w=World()
    fig, ax = w.viewer()
    plt.show()