from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_sentence_lead_attribution_pilot_v1 as pilot  # noqa: E402


@dataclass
class Group:
    group_id: str
    target_ids: tuple[str, ...]


@dataclass
class Analysis:
    groups: tuple[Group, ...]
    group_features: dict


class SentenceLeadPilotTests(unittest.TestCase):
    def test_candidate_requires_contrast_and_preserves_neutral_rows(self) -> None:
        records = [
            {"REFNO": "1", "RESULTNO": "a"},
            {"REFNO": "1", "RESULTNO": "b"},
            {"REFNO": "1", "RESULTNO": "c"},
            {"REFNO": "2", "RESULTNO": "d"},
        ]
        analyses = {
            "1": Analysis(
                (Group("ga", ("a",)), Group("gb", ("b",)), Group("gc", ("c",))),
                {
                    "ga": {"candidate_signal": 1.0},
                    "gb": {"candidate_signal": 0.0},
                    "gc": {"candidate_signal": None},
                },
            ),
            "2": Analysis(
                (Group("gd", ("d",)),), {"gd": {"candidate_signal": 1.0}}
            ),
        }
        baseline = np.asarray([0.2, 0.8, 0.4, 0.2])
        candidate, changed = pilot.candidate_probability(
            baseline, records, analyses, 0.6
        )
        np.testing.assert_allclose(candidate, [0.68, 0.32, 0.4, 0.2])
        self.assertEqual(changed.tolist(), [True, True, False, False])

    def test_mcnemar_exact_is_two_sided(self) -> None:
        labels = np.asarray([1, 1, 0, 0])
        baseline = np.asarray([0.1, 0.1, 0.1, 0.1])
        candidate = np.asarray([0.9, 0.9, 0.9, 0.1])
        result = pilot.mcnemar_exact(labels, baseline, candidate)
        self.assertEqual(result["baseline_wrong_candidate_right"], 2)
        self.assertEqual(result["baseline_right_candidate_wrong"], 1)
        self.assertEqual(result["two_sided_exact_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
