from __future__ import annotations

import inspect
from pathlib import Path
import re
import tempfile
import unittest

from scripts import experiment_r3_component_manifest as experiment


class R3ComponentManifestTests(unittest.TestCase):
    def test_scope_extraction_preserves_table_caption_and_paragraph(self) -> None:
        pages = [
            {
                "page": "3",
                "text": (
                    "Alpha Formation samples were collected at three sites and "
                    "were subjected to IRM acquisition experiments.\n\n"
                    "Figure 2. Thermal demagnetization of Alpha Formation specimens.\n"
                    "| Site | Component | Mineral |\n"
                    "| AF-1 | H | magnetite |\n"
                ),
            }
        ]

        scopes = experiment.extract_scopes(pages)
        kinds = {scope["kind"] for scope in scopes}

        self.assertIn("paragraph", kinds)
        self.assertIn("figure_or_table_caption", kinds)
        self.assertIn("table_row", kinds)
        self.assertTrue(all(scope["text_sha256"] for scope in scopes))

    def test_component_codes_are_case_sensitive_and_descriptions_are_normalized(self) -> None:
        self.assertEqual(
            experiment.extract_components(
                "The magnetization of these samples was measured and ratios were calculated."
            ),
            [],
        )
        components = experiment.extract_components(
            "The high-temperature component A was isolated; component B is secondary. ChRM remains."
        )
        self.assertEqual(set(components), {"high_temperature", "a", "b", "chrm"})

    def test_sample_and_component_bridges_open_only_source_derived_gate(self) -> None:
        patterns = [
            {
                "category": "method",
                "concept_id": "irm_acquisition",
                "pattern": re.compile(r"\bIRM\b", re.I),
            },
            {
                "category": "carrier",
                "concept_id": "magnetite",
                "pattern": re.compile(r"\bmagnetite\b", re.I),
            },
        ]
        identity = [
            {"REFNO": "10", "RESULTNO": "100", "ROCKNAME": "Alpha Formation"}
        ]
        pages = [
            {
                "page": "1",
                "text": (
                    "Alpha Formation samples from site AF-1 contain component H.\n\n"
                    "For sample AF-1, IRM acquisition did not saturate and magnetite "
                    "was identified as the carrier of component H."
                ),
            }
        ]

        paper = experiment.build_paper_manifest("10", identity, pages, patterns)
        target = paper["targets"][0]

        self.assertIn("AF-1", target["sample_ids"])
        self.assertIn("h", target["components"])
        self.assertTrue(target["gate"]["sample_bridge"])
        self.assertTrue(target["gate"]["component_bridge"])
        self.assertTrue(target["gate"]["eligible_for_article_propagation"])
        self.assertIn("method:irm_acquisition", target["bridged_concepts"])
        self.assertIn("carrier:magnetite", target["bridged_concepts"])

    def test_freeze_api_cannot_accept_a_label_source(self) -> None:
        parameters = inspect.signature(experiment.freeze_manifest).parameters
        self.assertNotIn("targets_path", parameters)
        self.assertNotIn("labels_path", parameters)
        self.assertNotIn("labels", parameters)

    @unittest.skipIf(experiment.np is None, "numpy is optional for extraction-only tests")
    def test_oof_npz_contains_only_strategy_probabilities_and_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oof.npz"
            experiment.atomic_strategy_npz(
                path,
                {"target_local": experiment.np.asarray([0.2, 0.8])},
            )
            with experiment.np.load(path) as artifact:
                self.assertEqual(
                    set(artifact.files),
                    {
                        "target_local__probability",
                        "target_local__decision",
                    },
                )
                self.assertFalse(any("label" in key or "id" in key for key in artifact.files))
                self.assertEqual(artifact["target_local__decision"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
