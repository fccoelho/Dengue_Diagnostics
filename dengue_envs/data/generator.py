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
    ):
        """
        size: size of the world
        popsize: population size
        epilength: length of the epidemic in days
        dengue_center: center of the dengue outbreak
        chik_center: center of the chikungunya outbreak
        dengue_radius: radius of the dengue outbreak
        chik_radius: radius of the chikungunya outbreak
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
        # a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
        self.dengue_dist_x = st.distributions.truncnorm(a=(0-self.dengue_center[0])/self.dengue_radius,
                                                        b=(size-1 - self.dengue_center[0])/self.dengue_radius,
                                                        loc=self.dengue_center[0],
                                                        scale=self.dengue_radius
        )
        self.dengue_dist_y = st.distributions.truncnorm(a=(0-self.dengue_center[1])/self.dengue_radius,
                                                        b=(size-1 - self.dengue_center[1])/self.dengue_radius,
                                                        loc=self.dengue_center[1],
                                                        scale=self.dengue_radius
        )

        self.chik_dist_x = st.distributions.truncnorm(a=(0-self.chik_center[0])/self.chik_radius,
                                                      b=(size-1 - self.chik_center[0])/self.chik_radius,
                                                      loc=self.chik_center[0],
                                                      scale=self.chik_radius
        )
        self.chik_dist_y = st.distributions.truncnorm(a=(0-self.chik_center[1])/self.chik_radius,
                                                      b=(size-1 - self.chik_center[1])/self.chik_radius,
                                                      loc=self.chik_center[1],
                                                      scale=self.chik_radius
        )

        # Cumulative Incidence curves
        self.dengue_curve = self._get_epi_curve(R0=1.5)
        self.chik_curve = self._get_epi_curve(R0=1.2)

        self.case_series = []
        # Cases per day as a list of lists
        # [[[x1,y1,0], [x2,y2,0], ...], [[x1,y1,0], [x2,y2,0], ...], ...]
        self.case_dict = {} # Cases per ID
        self.dengue_total = 0
        self.chik_total = 0
        self.casedf = None  # Case dataframe with incremental id numeric ids
        self.build_case_series()
        self.build_case_dataframe()
        # print(self.case_dict)

    def _get_epi_curve(self, I0=10, R0=1.5):
        """
        Generate an epidemic curve
        returns the Infectious numbers per day
        :param I0: Initial number of infectious
        :param R0: Basic Reproductive Number
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

    def build_case_series(self):
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
                dcases_x = self.dengue_dist_x.rvs(
                    new_d
                )  # New cases on day t, because curve is cumulative
                dcases_y = self.dengue_dist_y.rvs(new_d)
                ccases_x = self.chik_dist_x.rvs(new_c)
                ccases_y = self.chik_dist_y.rvs(new_c)
                self.dengue_total += new_d
                self.chik_total += new_c
            dengue_cases = [
                {"t": t, "x": int(x), "y": int(y), "disease": 0, "testd": 0, "testc": 0,"epiconf": 0}
                for x, y in zip(dcases_x, dcases_y)
            ]
            chik_cases = [
                {"t": t, "x": int(x), "y": int(y), "disease": 1, "testd": 0, "testc": 0,"epiconf": 0}
                for x, y in zip(ccases_x, ccases_y)
            ]
            self.case_series.append(dengue_cases + chik_cases)

    def build_case_dataframe(self):
        """
        Build a dataframe of cases
        """
        self.casedf = pd.DataFrame.from_records([c for c in chain(*self.case_series)])
        self.case_dict = self.casedf.to_dict(orient="index")

    def get_series_up_to_t(self, t):
        """
        Get a list of cases up to time t as a dataframe
        """
        # if t > self.casedf['t'].max():
        self.build_case_dataframe()
        casedf = self.casedf
        casedf = casedf[casedf.t <= t]
        return casedf

    def get_maps_up_to_t(self, t):
        """
        Get a dengue and chik maps of the world up to time t represented as an array with the number of cases per cell of the spatial grid
        """
        if self.casedf is None:
            self.build_case_dataframe()
        casedf = self.casedf
        casedf = casedf[casedf.t <= t]
        dengue_map = np.histogram2d(
            casedf[casedf.disease == 0].x,
            casedf[casedf.disease == 0].y,
            bins=self.size,
            range=[[0, self.size], [0, self.size]],
        )[0]
        chik_map = np.histogram2d(
            casedf[casedf.disease == 1].x,
            casedf[casedf.disease == 1].y,
            bins=self.size,
            range=[[0, self.size], [0, self.size]],
        )[0]
        return dengue_map, chik_map

    def get_maps_at_t(self, t):
        """
        Get a dengue and chik maps of the world at time t represented as an array with the number of cases per cell of the spatial grid
        """
        if self.casedf is None:
            self.build_case_dataframe()
        casedf = self.casedf
        casedf = casedf[casedf.t == t]
        dengue_map = np.histogram2d(
            casedf[casedf.disease == 0].x,
            casedf[casedf.disease == 0].y,
            bins=self.size,
            range=[[0, self.size], [0, self.size]],
        )[0]
        chik_map = np.histogram2d(
            casedf[casedf.disease == 1].x,
            casedf[casedf.disease == 1].y,
            bins=self.size,
            range=[[0, self.size], [0, self.size]],
        )[0]
        return dengue_map, chik_map

    def view(self):
        if self.casedf is None:
            self.build_case_dataframe()

        casedf = self.casedf
        dengue_map = np.histogram2d(
            casedf[casedf.disease == 0].x,
            casedf[casedf.disease == 0].y,
            bins=self.size,
            range=[[0, self.size], [0, self.size]],
        )[0]
        chik_map = np.histogram2d(
            casedf[casedf.disease == 1].x,
            casedf[casedf.disease == 1].y,
            bins=self.size,
            range=[[0, self.size], [0, self.size]],
        )[0]

        fig, ax = plt.subplots()
        ax.pcolor(dengue_map, cmap="Greens", alpha=0.5)
        ax.pcolor(chik_map, cmap="Blues", alpha=0.5)
        return fig, ax
