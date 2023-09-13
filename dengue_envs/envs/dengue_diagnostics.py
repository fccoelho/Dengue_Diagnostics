# Basic packages
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


class DengueDiagnosticEnv(gym.Env):
    def __init__(self, size:int=200, episize:int=150, epilength:int=60, 
                 dengue_center=(30, 30), chik_center=(90, 110), 
                 dengue_radius=10, chik_radius=10, 
                 clinical_specificity=0.8):

        self.t = 0 # timestep

        self.size = size
        self.episize = episize 
        self.epilength = epilength
        self.dengue_center = dengue_center
        self.chik_center = chik_center
        self.dengue_radius = dengue_radius
        self.chik_radius = chik_radius
        self.clinical_specificity = clinical_specificity

        self.world = World(self.size, self.episize, self.epilength, self.dengue_center,
                           self.chik_center, self.dengue_radius, self.chik_radius)

        self.start_pos = self.world.chik_center # Starting position
        self.current_pos = self.start_pos # Starting position is current positon of agent        
        self.hist_pos = []

        # Observations are dictionaries as defined below.
        # Data are represented as sequences of cases.
        self.observation_space = spaces.Dict(
            {
                "clinical": spaces.Sequence(spaces.Tuple((spaces.Discrete(self.world.num_rows), 
                                            spaces.Discrete(self.world.num_cols), 
                                            spaces.Discrete(2)))), # Clinical diagnosis: 0: dengue, 1: chik

                "testd": spaces.Sequence(spaces.Discrete(4)) , # Dengue testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                "testc": spaces.Sequence(spaces.Discrete(4)), # Chikungunya testing status: 0: not tested, 1: negative, 2: positive, 3: inconclusive
                "epiconf": spaces.Sequence(spaces.Discrete(2)), # Epidemiological confirmation: 0: no, 1: yes
                "tcase" : spaces.Sequence(spaces.Discrete(60)), # Day of the clinical diagnosis
            }
        )

        # We have 4 actions, corresponding to "test for dengue", "test for chik", "epi confirm", "noop"
        self.action_space = spaces.Tuple((spaces.Discrete(self.size), spaces.Discrete(self.size), spaces.Discrete(4)))
        self.costs = np.array([0.5, 0.5, 0.1, 0.0])

        # The lists below will be populated by the step() method, as the cases are being generated
        self.cases = [] # True cases
        self.obs_cases = [] # Observed cases

        self.testd = []
        self.testc = []
        self.epiconf = []

        self.tcase = []
        self.rewards = []

        # cumulative map of cases up to self.t
        self.dmap = self.world.dsnapshots[0]
        self.cmap = self.world.csnapshots[0]

        # Initialize Pygame
        pygame.init()
        self.cell_size = 1000/self.world.size

        # Setting display size
        self.screen = pygame.display.set_mode((self.world.num_cols * self.cell_size, self.world.num_rows * self.cell_size))

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.world.num_rows or col >= self.world.num_cols:
            return False
        
        return True
    
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
        obs_case_series = self.world.case_series[t][:]
        for i, case in enumerate(self.world.case_series[t]):
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
        errorrate = (len(estimated)- np.sum(estimated == true))/len(estimated)
        accuracy_reward = 1 if errorrate < 0.15 else 0
        reward = accuracy_reward - 0.1*self.costs[action[-1]]
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

    def reset(self):
        
        # Create the world
        self.world = World(self.size, self.episize, self.epilength, self.dengue_center,
                           self.chik_center, self.dengue_radius, self.chik_radius)

        self.cases = self.world.case_series[0]
        self.obs_cases = self._apply_clinical_uncertainty(0)

        observation = {"clinical": self.obs_cases, 
                       "testd": [0]*len(self.obs_cases), 
                       "testc": [0]*len(self.obs_cases), 
                       "t": [np.nan]*len(self.obs_cases)}
        
        info = self._get_info()

        self.current_pos = self.start_pos # Starting position is current positon of agent        
        self.hist_pos = []

        return self.current_pos

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

        # Move the agent based on the selected action
        if action[0] == 0:  # Up
            self.testd.append(self._dengue_lab_test())
        elif action == 1:  # Down
            self.testc.append(self._chik_lab_test())
        elif action == 2:  # Left
            new_pos[1] -= 1
            self.tcase.append([self.t, 0 if not observation['clinical'] else observation['clinical'][-1]])
        elif action == 3:  # Right
            new_pos[1] += 1
        new_pos = np.array(action[:-1])

        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos
            self.hist_pos.append(list(new_pos))
            self.hist_pos = list(map(list, set(map(tuple, self.hist_pos))))

        self.world.current_time += 1
        dengue_positive = np.where(self.world.dsnapshots[self.world.current_time])
        self.dengue_positive = [list(arg) for arg in list(zip(dengue_positive[0], dengue_positive[1]))]
        chik_positive = np.where(self.world.csnapshots[self.world.current_time])
        self.chik_positive = [list(arg) for arg in list(zip(chik_positive[0], chik_positive[1]))]

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
                pygame.draw.rect(self.screen, color, (cell_left, cell_top, self.cell_size, self.cell_size))

        # Clear the screen
        self.screen.fill((255, 255, 255))
        draw_rectangles(grid = self.dengue_positive, color = (200, 255, 200))
        draw_rectangles(grid = self.chik_positive, color = (200, 200, 255))
        draw_rectangles(grid = self.hist_pos, color = (127, 127, 127))
        
        inspected_dengue = set(map(tuple, self.dengue_positive))
        inspected_dengue = list(inspected_dengue.intersection(set(map(tuple, self.hist_pos))))

        inspected_chik = set(map(tuple, self.chik_positive))
        inspected_chik = list(inspected_chik.intersection(set(map(tuple, self.hist_pos))))

        draw_rectangles(grid = inspected_dengue, color = (0, 255, 0))
        draw_rectangles(grid = inspected_chik, color = (0, 0, 255))


        pygame.draw.rect(self.screen, (200, 0, 0), (self.start_pos[0]*self.cell_size,
                                                    self.start_pos[1]*self.cell_size, 
                                                    self.cell_size, self.cell_size))
        
        pygame.draw.rect(self.screen, (0, 0, 0), (self.current_pos[0]*self.cell_size,
                                                  self.current_pos[1]*self.cell_size, 
                                                  self.cell_size, self.cell_size))

        pygame.display.update()  # Update the display


class World:
    "Initialize random but concentrated distribution od dengue and Chikungunya cases"
    def __init__(self, size:int=200, episize:int=150, epilength:int=60, 
                 dengue_center=(30, 30), chik_center=(90, 110), 
                 dengue_radius=10, chik_radius=10):
        """
        size: size of the world
        """
        self.size = size # World size

        self.num_rows = size # World represented as a 2D numpy array
        self.num_cols = size

        self.episize = episize # Size of the epidemic
        self.epilength = epilength # Length of the epidemic in days

        self.dengue_center = dengue_center
        self.dengue_radius = dengue_radius
        self.chik_center = chik_center
        self.chik_radius = chik_radius

        self.dengue_dist_x = st.distributions.norm(self.dengue_center[0], self.dengue_radius)
        self.dengue_dist_y = st.distributions.norm(self.dengue_center[1], self.dengue_radius)

        self.chik_dist_x = st.distributions.norm(self.chik_center[0], self.chik_radius)
        self.chik_dist_y = st.distributions.norm(self.chik_center[1], self.chik_radius)

        self.dengue_curve = self._get_epi_curve(R0=2.5)
        self.chik_curve = self._get_epi_curve(R0=1.5)
        self.case_series = defaultdict(list) 
        # Cases per day as a list of lists 
        #{1: [[x1,y1,0], [x2,y2,0], ...], 2: [[x1,y1,0], [x2,y2,0], ...], ...}
        
        self.dsnapshots = {} # Dengue spatial distribution on the world grid
        self.csnapshots = {} # Chikungunya spatial distribution on the world grid
        self.get_daily_cases()

        self.current_time = 0 # Starting time

        dengue_positive = np.where(self.dsnapshots[self.current_time])
        self.dengue_positive = [list(arg) for arg in list(zip(dengue_positive[0], dengue_positive[1]))]

        chik_positive = np.where(self.csnapshots[self.current_time])
        self.chik_positive = [list(arg) for arg in list(zip(chik_positive[0], chik_positive[1]))]

    def _generate_full_outbreak(self):
        dpos = self.dsnapshots[self.epilength-1] 
        cpos = self.csnapshots[self.epilength-1]
        return dpos, cpos
    
    def _get_epi_curve(self, R0=2.5):
        """
        Generate an epidemic curve
        returns the Infectious numbers per day
        """
        def SIR(y, t, beta, gamma, N):
            S, I, Inc, R = y
            return [-beta*S*I/N,
                    beta*S*I/N-gamma*I,
                    beta*S*I/N
                    gamma*I
                    ]
        gamma = 0.004
        beta = R0*gamma
        y = odeint(SIR, [self.episize, 1,0, 0], np.arange(0, self.epilength), args=(beta, gamma, self.episize))
        incidence = np.diff(y[:,1]) # Daily incidence
        return incidence
    
    def get_daily_cases(self):
        """
        Generate the daily cases based on an epidemic curve
        """
        for t in range(self.epilength):
            # if t == 0:
            #     dcases_x = self.dengue_dist_x.rvs(int(self.dengue_curve[t]))
            #     dcases_y = self.dengue_dist_y.rvs(int(self.dengue_curve[t]))
            #     ccases_x = self.chik_dist_x.rvs(int(self.chik_curve[t]))
            #     ccases_y = self.chik_dist_y.rvs(int(self.chik_curve[t]))
            # else:
            #     dcases_x = self.dengue_dist_x.rvs(int(self.dengue_curve[t]-self.dengue_curve[t-1]))
            #     dcases_y = self.dengue_dist_y.rvs(int(self.dengue_curve[t]-self.dengue_curve[t-1]))
            #     ccases_x = self.chik_dist_x.rvs(int(self.chik_curve[t]-self.chik_curve[t-1]))
            #     ccases_y = self.chik_dist_y.rvs(int(self.chik_curve[t]-self.chik_curve[t-1]))

            dcases_x = self.dengue_dist_x.rvs(int(self.dengue_curve[t]))
            dcases_y = self.dengue_dist_y.rvs(int(self.dengue_curve[t]))
            ccases_x = self.chik_dist_x.rvs(int(self.chik_curve[t]))
            ccases_y = self.chik_dist_y.rvs(int(self.chik_curve[t]))

            dpos, _, _ = np.histogram2d(dcases_x,dcases_y,bins=(range(self.size),range(self.size)))
            cpos, _, _ = np.histogram2d(ccases_x,ccases_y,bins=(range(self.size),range(self.size)))
            
            self.case_series[t].extend([[int(x),int(y),0] for x, y in zip(dcases_x, dcases_y)])
            self.case_series[t].extend([[int(x),int(y),1] for x, y in zip(ccases_x, ccases_y)])
            
            if t == 0:
                self.dsnapshots[t] = dpos
                self.csnapshots[t] = cpos
            else:
                self.dsnapshots[t] = dpos + self.dsnapshots[t-1]
                self.csnapshots[t] = cpos + self.csnapshots[t-1]
    
    def viewer(self):
        dpos, cpos = self._generate_full_outbreak()
        
        fig, ax = plt.subplots()
        ax.pcolor(dpos, cmap='Greens', alpha=0.5)
        ax.pcolor(cpos, cmap='Blues', alpha=0.5)
        return fig, ax

if __name__== "__main__":
    # Test the environment
    total_time = 1000
    env = DengueDiagnosticEnv(epilength = total_time)
    obs = env.reset()

    for t in range(total_time-1):
        pygame.event.get()
        action = env.action_space.sample()  # Random action selection
        obs, reward, done, _ = env.step(action)
        env.render()
        # print('Reward:', reward)
        # print('Done:', done)

        pygame.time.wait(60)