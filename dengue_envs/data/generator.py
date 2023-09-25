import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from itertools import chain


class World:
    """
    Initialize random but concentrated distribution of dengue and Chikungunya cases
    """

    def __init__(
        self,
        size: int = 200,
        popsize: int = 150,
        epilength: int = 60,
        dengue_center=(30, 30),
        chik_center=(90, 110),
        dengue_radius=10,
        chik_radius=10,
        medical_specificity=0.95,
        medical_identification_rate=0.05,
    ):
        """
        size: size of the world
        """
        self.size = size  # World size

        self.num_rows = size  # World represented as a 2D numpy array
        self.num_cols = size

        self.popsize = popsize  # Size of the epidemic
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

        # Cumulative Incidence curves
        self.dengue_curve = self._get_epi_curve(R0=2.5)
        self.chik_curve = self._get_epi_curve(R0=1.5)

        self.case_series = []
        # Cases per day as a list of lists
        # [[[x1,y1,0], [x2,y2,0], ...], [[x1,y1,0], [x2,y2,0], ...], ...]
        self.dengue_total = 0
        self.chik_total = 0
        self.get_daily_cases()




    def _get_epi_curve(self, I0=10, R0=2.5):
        """
        Generate an epidemic curve
        returns the Infectious numbers per day
        """

        def SIR(Y, t, beta, gamma, N):
            S, I, Inc, R = Y
            return [
                -beta * S * I,
                beta * S * I - gamma * I,
                beta * S * I,  # Cumulative Incidence
                gamma * I,
            ]

        gamma = 0.004
        beta = R0 * gamma
        y = odeint(
            SIR,
            [self.popsize - I0, I0, 0, 0],
            np.arange(0, self.epilength),
            args=(beta, gamma, self.popsize),
        )

        return y[:, 2]

    def get_daily_cases(self):
        """
        Generate the daily cases based on an epidemic curve
        """
        for t in range(self.epilength):
            dcases_t = int(np.round(self.dengue_curve[t]))
            ccases_t = int(np.round(self.chik_curve[t]))

            if t < 1:
                dcases_x = self.dengue_dist_x.rvs(dcases_t)
                dcases_y = self.dengue_dist_y.rvs(dcases_t)
                ccases_x = self.chik_dist_x.rvs(ccases_t)
                ccases_y = self.chik_dist_y.rvs(ccases_t)
                self.dengue_total += dcases_t
                self.chik_total += ccases_t
            else:
                new_d = int(np.round(dcases_t - self.dengue_curve[t - 1]))
                new_c = int(np.round(ccases_t - self.chik_curve[t - 1]))
                dcases_x = self.dengue_dist_x.rvs(new_d) # New cases on day t, because curve is cumulative
                dcases_y = self.dengue_dist_y.rvs(new_d)
                ccases_x = self.chik_dist_x.rvs(new_c)
                ccases_y = self.chik_dist_y.rvs(new_c)
                self.dengue_total += new_d
                self.chik_total += new_c
            dengue_cases = [{'t': t, 'x':x, 'y':y, 'v':0} for x, y in zip(dcases_x, dcases_y)]
            chik_cases = [{'t': t, 'x':x, 'y':y, 'v':1} for x, y in zip(ccases_x, ccases_y)]
            self.case_series.append(dengue_cases + chik_cases)

    def view(self):
        casedf = pd.DataFrame.from_records([c for c in chain(*self.case_series)])

        dengue_map = np.histogram2d(
            casedf[casedf.v == 0].x, casedf[casedf.v == 0].y, bins=self.size
        )[0]
        chik_map = np.histogram2d(
            casedf[casedf.v == 1].x, casedf[casedf.v == 1].y, bins=self.size
        )[0]

        fig, ax = plt.subplots()
        ax.pcolor(dengue_map, cmap="Greens", alpha=0.5)
        ax.pcolor(chik_map, cmap="Blues", alpha=0.5)
        return fig, ax