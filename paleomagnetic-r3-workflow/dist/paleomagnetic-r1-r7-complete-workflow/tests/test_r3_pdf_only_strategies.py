from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as experiment


class R3PdfOnlyStrategyKnowledgeTest(unittest.TestCase):
    @unittest.skipUnless(
        experiment.np is not None,
        "optional R3 text-model dependencies are not installed",
    )
    def test_inverse_target_multiplicity_weights_equalize_each_target(self) -> None:
        records = [
            {"REFNO": "1"},
            {"REFNO": "1"},
            {"REFNO": "1"},
            {"REFNO": "2"},
        ]
        canonical_targets = ["alpha", "alpha", "beta", "gamma"]

        weights = experiment.inverse_target_multiplicity_weights(
            experiment.np.asarray([0, 1, 2, 3]),
            records,
            canonical_targets,
        )

        self.assertEqual(weights.tolist(), [0.5, 0.5, 1.0, 1.0])

    def test_positive_kb_runtime_pattern_interface_is_explicit(self) -> None:
        concepts, audit = experiment.load_knowledge_patterns(
            ROOT / "data/calibration/r3_rockmag_text_distillation_v1.json",
            ROOT / "data/calibration/r3_positive_rockmag_expert_kb_v2.json",
        )

        self.assertEqual(audit["positive_kb_status"], "loaded")
        self.assertEqual(
            audit["positive_kb_pattern_field"], "runtime_original_text_patterns"
        )
        self.assertGreaterEqual(audit["runtime_pattern_count"], 66)
        self.assertTrue(
            any(
                concept["source"] == "positive_rockmag_expert_kb"
                for concept in concepts
            )
        )

    def test_frozen_positive_kb_hash(self) -> None:
        path = ROOT / "data/calibration/r3_positive_rockmag_expert_kb_v2.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "b0fc28866fe6702938a7cc02fcf7b87b68abc9e3b4e391a53a53de1615b5e9ef",
        )


if __name__ == "__main__":
    unittest.main()
