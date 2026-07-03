import unittest

import numpy as np
import pandas as pd

from dengue_envs.core.clinical import (
    CHIK,
    DENGUE,
    INCONCLUSIVE,
    NEGATIVE,
    POSITIVE,
    ClinicalModel,
    update_case_status,
)


def _cases_df():
    return pd.DataFrame(
        {
            "t": [0, 0],
            "x": [10, 20],
            "y": [10, 20],
            "disease": [0, 1],
            "testd": [0, 0],
            "testc": [0, 0],
            "epiconf": [0, 0],
        },
        index=[0, 1],
    )


class ClinicalModelTestCase(unittest.TestCase):
    def test_perfect_specificity_keeps_diagnosis(self):
        model = ClinicalModel(clinical_specificity=1.0, other_prob=0.0)
        rng = np.random.default_rng(0)
        obs = model.apply_uncertainty(_cases_df(), rng)
        self.assertListEqual(list(obs["disease"]), [0, 1])
        self.assertListEqual(list(obs["agent_diagnosis"]), [0, 1])

    def test_apply_uncertainty_does_not_mutate_input(self):
        model = ClinicalModel(clinical_specificity=0.5)
        rng = np.random.default_rng(1)
        original = _cases_df()
        _ = model.apply_uncertainty(original, rng)
        self.assertNotIn("agent_diagnosis", original.columns)

    def test_dengue_lab_test_deterministic_positive(self):
        model = ClinicalModel(clinical_specificity=0.8, sensitivity=1.0, inconclusive_prob=0.0)
        rng = np.random.default_rng(0)
        self.assertEqual(model.dengue_lab_test(DENGUE, rng), POSITIVE)

    def test_dengue_lab_test_deterministic_negative(self):
        model = ClinicalModel(clinical_specificity=0.8, specificity=1.0, inconclusive_prob=0.0)
        rng = np.random.default_rng(0)
        self.assertEqual(model.dengue_lab_test(CHIK, rng), NEGATIVE)

    def test_lab_test_can_be_inconclusive(self):
        model = ClinicalModel(clinical_specificity=0.8, inconclusive_prob=1.0)
        rng = np.random.default_rng(0)
        self.assertEqual(model.dengue_lab_test(DENGUE, rng), INCONCLUSIVE)
        self.assertEqual(model.chik_lab_test(CHIK, rng), INCONCLUSIVE)

    def test_update_case_status_dengue_positive(self):
        obs = _cases_df()
        obs["agent_diagnosis"] = obs["disease"]
        obs.loc[1, "agent_diagnosis"] = 1
        update_case_status(obs, action=0, index=1, result=POSITIVE)
        self.assertEqual(obs.loc[1, "testd"], POSITIVE)
        self.assertEqual(obs.loc[1, "agent_diagnosis"], DENGUE)

    def test_update_case_status_dengue_negative_flips_to_chik(self):
        obs = _cases_df()
        obs["agent_diagnosis"] = [0, 0]  # ambos suspeitos de dengue
        update_case_status(obs, action=0, index=0, result=NEGATIVE)
        self.assertEqual(obs.loc[0, "agent_diagnosis"], CHIK)

    def test_update_case_status_epiconf_does_not_change_diagnosis(self):
        obs = _cases_df()
        obs["agent_diagnosis"] = obs["disease"]
        update_case_status(obs, action=2, index=0, result=1)
        self.assertEqual(obs.loc[0, "epiconf"], 1)
        self.assertEqual(obs.loc[0, "agent_diagnosis"], 0)


if __name__ == "__main__":
    unittest.main()
