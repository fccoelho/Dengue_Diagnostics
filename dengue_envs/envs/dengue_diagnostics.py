import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces, utils
from gymnasium.spaces import Box
import scipy.stats as st
from scipy.integrate import odeint
import pygame
from typing import List, Optional, Tuple
from collections import defaultdict
from copy import copy


class DengueDiagnosticEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str]=None, size: int=50, clinical_specificity=0.8, render_freq=1):
        self.render_mode = render_mode
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.t = 0 # timestep
        self.clinical_specificity = clinical_specificity

        # Observations are dictionaries as defined below.
        # Data are represented as sequences of cases.
        self.world = World()
    
        self.observation_space = spaces.Dict(
            {
                "clinical": spaces.Sequence(Box(low=0,high=self.size, shape=(1,3))), # Clinical diagnosis: 0: dengue, 1: chik
                "testd": spaces.Sequence(spaces.Discrete(4)) , # Dengue testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                "testc": spaces.Sequence(spaces.Discrete(4)), # Chikungunya testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                "epiconf": spaces.Sequence(spaces.Discrete(2)), # Epidemiological confirmation: 0: no, 1: yes
                "tcase" : spaces.Sequence(spaces.Discrete(60)), # Day of the clinical diagnosis
            }
        )

        # We have 4 actions, corresponding to "test for dengue", "test for chik", "epi confirm", "noop"
        self.action_space = spaces.Discrete(4)
        self.costs = np.array([0.5, 0.5, 0.1, 0.0])
        # The lists below will be populated by the step() method, as the cases are being generated
        self.cases = [] # True cases
        self.obs_cases = [] # Observed cases
        self.testd = np.zeros(self.world.epilength)
        self.testc = np.zeros(self.world.epilength)
        self.epiconf = np.zeros(self.world.epilength)
        self.tcase = []
        self.rewards = []

        # cumulative map of cases up to self.t
        self.dmap = self.world.dsnapshots[0]
        self.cmap = self.world.csnapshots[0]

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
        obs_cases = self._apply_clinical_uncertainty(self.t)

        return {"clinical": obs_cases, "testd": [0]*len(obs_cases), "testc": [0]*len(obs_cases), "t": [np.nan]*len(obs_cases)}

    def _apply_clinical_uncertainty(self, t):
        """
        Apply clinical uncertainty to the observations: Observations are subject to misdiagnosis based on the clinical specificity
        """
        obs_case_series = copy(self.world.case_series[t])
        for i, case in enumerate(self.world.case_series[t]):
            if case[2] == 0:
                if self.np_random.uniform() > self.clinical_specificity:
                    obs_case_series[t][i][2] = 1
            else:
                if self.np_random.uniform() > self.clinical_specificity:
                    obs_case_series[t][i][2] = 0

        
        return obs_case_series
    
    def _calc_reward(self, true, estimated, actions):
        """
        Calculate the reward based on the true count and the actions taken
        """
        errorrate = (len(estimated)- (estimated == true).sum())/len(estimated)
        accuracy_reward = 1 if errorrate < 0.15 else 0
        reward = accuracy_reward - 0.1*np.array([self.cost[a] for a in actions]).sum()
        return reward
        

    def _get_info(self):
        """
        Returns the current map of cases for each disease"""
        self.dmap = self.world.dsnapshots[0]
        self.cmap = self.world.csnapshots[0]
        for t in range(1, self.t):
            self.dmap += self.world.dsnapshots[t]
            self.cmap += self.world.csnapshots[t]
        return {
            "dengue_grid": self.dmap,
            "chik_grid": self.cmap,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Create the world
        self.world = World()
        self.cases = self.world.case_series[0]
        dengue_location = self.world.dsnapshots[0]
        chik_location = self.world.csnapshots[0]
        obs_cases = self._apply_clinical_uncertainty(0)
        self.obs_cases = obs_cases

        

        observation = {"clinical": obs_cases, "testd": [0]*len(obs_cases), "testc": [0]*len(obs_cases), "t": [np.nan]*len(obs_cases)}
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Based on the action, does the testing or epidemiological confirmation
        action: [list of decisions for all current cases]: 0: test for dengue, 1: test for chik, 2: epi confirm, 3: Does nothing
        """
        # get the current true state
        cases_series = self.world.case_series[self.t]
        self.cases.extend(cases_series)

        # get the current observation
        observation = self._get_obs()        
        # Apply actions
        for i,a in enumerate(action):
            if a == 0:
                self.testd.append(self._dengue_lab_test(case))
            elif a == 1:
                self.testc.append(self._chik_lab_test)
            elif a == 2:
                self.tcase.append(self.t, observation['clinical'][i])
            else:
                pass
        
        
        # An episode is done if timestep is greter than 120
        terminated = self.t > 120
        reward = self._calc_reward(self.cases, observation["clinical"], action)
        self.rewards.append(reward)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.t += 1

        return observation, reward, terminated, False, info
    
    def _dengue_lab_test(self):
        """
        Returns the test result for a dengue case
        """
        if self.np_random.uniform() <0.1:
            return 3 # Inconclusive
        if self.np_random.uniform() >= 0.9:
            return 1
        else:
            return 2
        
    def _chik_lab_test(self):
        """
        Returns the test result for a chikungunya case
        """
        if self.np_random.uniform() <0.1:
            return 3 # Inconclusive
        if self.np_random.uniform() >= 0.9:
            return 1
        else:
            return 2
        
    def _epi_confirm(self, case):
        """
        Returns the epidemiological confirmation for a case
        """
        if case[2] == 0: # Dengue suspicion
            return 1 if self.map[case[0][0],case[0][1]] > 1 else 0
        else:
            return 1 if self.cmap[case[0][0],case[0][1]] > 1 else 0


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
    def __init__(self, size:int=200, episize:int=150, epilength:int=60, dengue_center=(30, 30), chik_center=(90, 110), dengue_radius=10, chik_radius=10):
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
        self.chik_curve = self._get_epi_curve(R0=1.5)
        self.case_series = defaultdict(list) # Cases per day as a list of tuples {1: [(x1,y1,0), (x2,y2,0), ...], 2: [(x1,y1,0), (x2,y2,0), ...], ...]}
        
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
        gamma = 0.1
        beta = R0*gamma
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
            self.case_series[t].extend([(int(x),int(y),0) for x, y in zip(dcases_x, dcases_y)])
            self.case_series[t].extend([(int(x),int(y),1) for x, y in zip(ccases_x, ccases_y)])
            
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
    env = DengueDiagnosticEnv()
    obs, info = env.reset()
    print(obs, info)
    # w=World()
    # fig, ax = w.viewer()
    # plt.show()