import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces, utils
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
import scipy.stats as st
from scipy.integrate import odeint
import pygame
from typing import List, Optional, Tuple


class DengueDiagnosticEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str]=None, size: int=50, render_freq=1):
        self.render_mode = render_mode
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.t = 0 # timestep

        # Observations are dictionaries with the dengue and chikungunya cases locations on a grid.
        # Data are represented as a 2D array of size (size, size) with the number of cases in each cell.
        self.world = World()
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
        """
        Returns the current observation.
        """
        return {"dengue": self.world.dengue_series[self.t], "chik": self.world.chik_series[self.t]}

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
        """
        Based on the action, does the testing or epidemiological confirmation
        action: 0: test for dengue, 1: test for chik, 2: epi confirm, 3: Does nothing
        """
        # get the current state
        dengue_cases = self.world.dengue_series[self.t]
        chik_cases = self.world.chik_series[self.t]
        
        # An episode is done if timestep is greter than 120
        terminated = self.t > 120
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
    def __init__(self, size:int=200, episize:int=50, epilength:int=60, dengue_center=(30, 30), chik_center=(90, 110), dengue_radius=10, chik_radius=10):
        """
        size: size of the world
        """
        self.size = size # World size
        self.episize = episize # Size of the epidemic
        self.epilength = epilength # Length of the epidemic in days
        self.dengue_center = dengue_center
        self.dengue_radius = dengue_radius
        self.chik_center = chik_center
        self.chik_radius = chik_radius
        self.dengue_dist = st.distributions.norm(dengue_center[0],dengue_radius)
        self.chik_dist = st.distributions.norm(chik_center[0],chik_radius)
        self.dengue_curve = self._get_epi_curve()
        self.dengue_series = {} # Dengue cases per day as a list of tuples {1: [(x1,y1), (x2,y2), ...], 2: [(x1,y1), (x2,y2), ...], ...)]}
        self.chik_curve = self._get_epi_curve(R0=1.5)
        self.chik_series = {} # Chikungunya cases per day as a list of tuples {1: [(x1,y1), (x2,y2), ...], 2: [(x1,y1), (x2,y2), ...], ...)]}
        self.dsnapshots = {} # Dengue spatial distribution on the world grid
        self.csnapshots = {} # Chikungunya spatial distribution on the world grid
        self.get_daily_cases()

        

    def _generate_full_outbreak(self):
        xd = self.dengue_dist.rvs(self.episize)
        yd = self.chik_dist.rvs(self.episize)
        dpos, _, _ = np.histogram2d(xd,yd,bins=(range(self.size),range(self.size)))
        xc = st.distributions.norm(self.chik_center[0],self.chik_radius).rvs(self.episize)
        yc = st.distributions.norm(self.chik_center[1],self.chik_radius).rvs(self.episize)
        cpos, _, _ = np.histogram2d(xc,yc,bins=(range(self.size),range(self.size)))
        return dpos, cpos
    
    def _get_epi_curve(self, R0=2.5):
        """
        Generate an epidemic curve
        returns the Infectious numbers per day
        """
        def SIR(y, t, beta, gamma):
            S, I, R = y
            return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
        beta, gamma = 0.25, 0.1
        y = odeint(SIR, [self.episize, 1, 0], np.arange(0, self.epilength), args=(beta, gamma))
        return y[:,1]
    
    def get_daily_cases(self):
        """
        Generate the daily cases based on an epidemic curve
        """
        for t in range(self.epilength):
            dcases_x = self.dengue_dist.rvs(int(self.dengue_curve[t]))
            dcases_y = self.dengue_dist.rvs(int(self.dengue_curve[t]))
            dpos, _, _ = np.histogram2d(dcases_x,dcases_y,bins=(range(self.size),range(self.size)))
            ccases_x = self.chik_dist.rvs(int(self.chik_curve[t]))
            ccases_y = self.chik_dist.rvs(int(self.chik_curve[t]))
            cpos, _, _ = np.histogram2d(ccases_x,ccases_y,bins=(range(self.size),range(self.size)))
            self.dengue_series[t] = [(int(x),int(y)) for x, y in zip(dcases_x, dcases_y)]
            self.chik_series[t] = [(int(x),int(y)) for x, y in zip(ccases_x, ccases_y)]
            self.dsnapshots[t] = dpos
            self.csnapshots[t] = cpos
    
        
    def viewer(self):
        dpos, cpos = self._generate_full_outbreak()
        fig, ax = plt.subplots()
        ax.pcolor(dpos, cmap='Greens', alpha=0.5)
        ax.pcolor(cpos, cmap='Blues', alpha=0.5)
        return fig, ax

if __name__== "__main__":
    import matplotlib.pyplot as plt
    w=World()
    fig, ax = w.viewer()
    plt.show()