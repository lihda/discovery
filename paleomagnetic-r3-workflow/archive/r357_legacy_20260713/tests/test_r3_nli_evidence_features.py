from __future__ import annotations

import re
import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_nli_evidence_features as evidence


class R3NliEvidenceFeatureTest(unittest.TestCase):
    def test_ocr_line_break_does_not_truncate_method_sentence(self) -> None:
        text = (
            "METHODS\nWe measured\nthermo-magnetic curves on representative "
            "specimens using a recording balance to determine the Curie point. "
            "The resulting curves were reproducible.\n\nDISCUSSION"
        )
        concepts = [
            {
                "category": "method_taxonomy",
                "concept_id": "thermomagnetic_curve",
                "patterns": [re.compile(r"thermo[- ]?magnetic", re.I)],
            }
        ]

        passages, audit = evidence.retrieve_passages(text, concepts, maximum=2)

        self.assertEqual(audit["raw_pattern_matches"], 1)
        self.assertIn("We measured thermo-magnetic curves", passages[0]["text"])
        self.assertIn("recording balance", passages[0]["text"])

    def test_dynamic_hypothesis_uses_expert_method_concept(self) -> None:
        hypotheses = evidence.passage_hypotheses(
            {"concepts": ["magnetite", "three_axis_irm"]}
        )

        self.assertEqual(len(hypotheses), len(evidence.HYPOTHESIS_NAMES))
        self.assertIn("Three-axis IRM", hypotheses[1])
        self.assertEqual(
            hypotheses[2], "The passage identifies a remanence carrier."
        )

    def test_aggregate_features_pool_each_paper_without_identifiers(self) -> None:
        scores = np.asarray(
            [
                [0.9, 0.8, 0.7],
                [0.3, 0.4, 0.5],
                [0.2, 0.1, 0.6],
            ],
            dtype=np.float32,
        )
        rows = [
            {
                "method_hits": 1,
                "diagnostic_hits": 0,
                "carrier_hits": 1,
                "own_study_hits": 1,
                "result_hits": 1,
                "retrieval_score": 9,
                "citation_heavy": False,
            },
            {
                "method_hits": 0,
                "diagnostic_hits": 1,
                "carrier_hits": 0,
                "own_study_hits": 0,
                "result_hits": 1,
                "retrieval_score": 5,
                "citation_heavy": True,
            },
            {
                "method_hits": 0,
                "diagnostic_hits": 0,
                "carrier_hits": 1,
                "own_study_hits": 0,
                "result_hits": 0,
                "retrieval_score": 4,
                "citation_heavy": False,
            },
        ]

        matrix, names = evidence.aggregate_features(scores, [0, 0, 1], rows, 2)

        self.assertEqual(matrix.shape, (2, len(names)))
        self.assertAlmostEqual(
            matrix[0, names.index("own_study_generic__max")], 0.9, places=6
        )
        self.assertEqual(
            matrix[0, names.index("retrieval__passage_count")], 2.0
        )
        self.assertFalse(any("REFNO" in name or "RESULTNO" in name for name in names))

    def test_builder_exposes_no_labels_or_targets_default(self) -> None:
        self.assertFalse(hasattr(evidence, "DEFAULT_TARGETS"))
        self.assertFalse(hasattr(evidence, "DEFAULT_LABELS"))


if __name__ == "__main__":
    unittest.main()
