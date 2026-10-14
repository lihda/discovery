from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from experiment_r3_own_study_events import (  # noqa: E402
    classify_heading,
    event_signals,
    sectionize,
)


class R3OwnStudyEventTests(unittest.TestCase):
    def test_sectionize_tracks_method_results_and_references(self) -> None:
        pages = [
            {
                "page": "1",
                "section_hints": [],
                "text": (
                    "Summary\nThe samples contain magnetite.\n"
                    "Laboratory procedure\nIRM acquisition curves were measured.\n"
                    "Rock magnetic results\nThe curves show saturation at 300 mT."
                ),
            },
            {
                "page": "2",
                "section_hints": [],
                "text": "References\nSmith et al. 1999. Magnetic methods.",
            },
        ]
        segments = sectionize(pages)
        self.assertEqual(
            [segment["section"] for segment in segments],
            ["abstract", "methods", "results", "references"],
        )

    def test_own_execution_context_is_strong(self) -> None:
        signals = event_signals(
            "In this study, we measured IRM acquisition curves for all samples; "
            "the curves show saturation at 300 mT.",
            "results",
            True,
        )
        self.assertTrue(signals["first_person"])
        self.assertTrue(signals["quantitative"])
        self.assertTrue(signals["own_strong"])
        self.assertFalse(signals["external"])

    def test_reference_context_is_external_not_own(self) -> None:
        signals = event_signals(
            "Smith et al. (1999) reported Curie temperature measurements.",
            "references",
            False,
        )
        self.assertTrue(signals["citation"])
        self.assertTrue(signals["external"])
        self.assertFalse(signals["own_strong"])

    def test_negated_future_measurement_is_not_own(self) -> None:
        signals = event_signals(
            "No hysteresis measurement was performed; it should be measured in future work.",
            "discussion",
            False,
        )
        self.assertTrue(signals["method_negation"])
        self.assertTrue(signals["future"])
        self.assertFalse(signals["own_strong"])

    def test_heading_classifier_ignores_long_body_sentence(self) -> None:
        self.assertEqual(classify_heading("3 Laboratory procedure"), "methods")
        self.assertEqual(classify_heading("Discussion"), "discussion")
        self.assertEqual(classify_heading("References"), "references")
        self.assertIsNone(
            classify_heading(
                "Results from previous studies were discussed in a long explanatory sentence "
                "that is clearly body prose rather than a section heading and continues further."
            )
        )


if __name__ == "__main__":
    unittest.main()
