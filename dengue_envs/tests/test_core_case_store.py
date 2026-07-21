import unittest

import numpy as np
import pandas as pd

from dengue_envs.core.case_store import OBS_COLUMNS, sync_obs_cases
from dengue_envs.core.clinical import ClinicalModel


def _cases(index, disease):
    return pd.DataFrame(
        {
            "t": [0] * len(index),
            "x": [10 + i for i in range(len(index))],
            "y": [10 + i for i in range(len(index))],
            "disease": disease,
            "testd": [0] * len(index),
            "testc": [0] * len(index),
            "epiconf": [0] * len(index),
        },
        index=index,
    )


class CaseStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.model = ClinicalModel(clinical_specificity=1.0, other_prob=0.0)
        self.rng = np.random.default_rng(0)

    def test_empty_cases_returns_empty_frame(self):
        obs = sync_obs_cases(pd.DataFrame(), pd.DataFrame(), self.model, self.rng)
        self.assertListEqual(list(obs.columns), OBS_COLUMNS)
        self.assertTrue(obs.empty)

    def test_new_cases_get_agent_diagnosis(self):
        cases = _cases([0, 1], [0, 1])
        obs = sync_obs_cases(pd.DataFrame(), cases, self.model, self.rng)
        self.assertIn("agent_diagnosis", obs.columns)
        self.assertEqual(len(obs), 2)

    def test_existing_cases_are_preserved(self):
        cases = _cases([0, 1], [0, 1])
        obs = sync_obs_cases(pd.DataFrame(), cases, self.model, self.rng)
        # Simula uma decisão do agente já registrada.
        obs.loc[0, "testd"] = 2
        obs.loc[0, "agent_diagnosis"] = 1
        # Sincroniza de novo com os MESMOS casos -> nada muda.
        obs2 = sync_obs_cases(obs, cases, self.model, self.rng)
        self.assertEqual(obs2.loc[0, "testd"], 2)
        self.assertEqual(obs2.loc[0, "agent_diagnosis"], 1)

    def test_only_new_cases_are_appended(self):
        cases_t0 = _cases([0, 1], [0, 1])
        obs = sync_obs_cases(pd.DataFrame(), cases_t0, self.model, self.rng)
        cases_t1 = _cases([0, 1, 2], [0, 1, 0])
        obs2 = sync_obs_cases(obs, cases_t1, self.model, self.rng)
        self.assertEqual(len(obs2), 3)
        self.assertIn(2, obs2.index)


if __name__ == "__main__":
    unittest.main()
