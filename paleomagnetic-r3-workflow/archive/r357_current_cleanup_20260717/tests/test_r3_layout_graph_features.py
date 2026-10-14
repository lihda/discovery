import re
import unittest

import numpy as np

from scripts.r3_layout_graph_features import build_layout_graph_target_features
from scripts.r3_paleomag_scope_features import target_profile
from scripts.r3_trace_page_graph_features import Concept


class LayoutGraphFeatureTests(unittest.TestCase):
    def setUp(self):
        self.concepts = (
            Concept("irm", "method", (re.compile(r"\bIRM\b", re.I),)),
            Concept("identified", "diagnostic", (re.compile(r"identif(?:y|ies|ied)", re.I),)),
            Concept("magnetite", "carrier", (re.compile(r"\bmagnetite\b", re.I),)),
            Concept("carrier_of", "bridge", (re.compile(r"carrier\s+of", re.I),)),
        )
        self.targets = {
            "10": target_profile("Alpha Formation"),
            "11": target_profile("Beta Formation"),
        }

    def feature(self, result, key, name):
        row = result["target_keys"].index(key)
        column = result["feature_names"].index(name)
        return float(result["matrix"][row, column])

    def test_same_line_relations_form_exclusive_complete_chain(self):
        page = {
            "page": 1,
            "blocks": [
                {
                    "id": "results",
                    "confidence": 0.9,
                    "lines": [
                        {
                            "id": "alpha",
                            "text": (
                                "Alpha Formation IRM results identify magnetite as the "
                                "carrier of the ChRM."
                            ),
                        },
                        {"id": "beta", "text": "Beta Formation was thermally demagnetized."},
                    ],
                }
            ],
        }
        result = build_layout_graph_target_features(page, self.targets, self.concepts)
        for category in ("method", "diagnostic", "carrier", "bridge"):
            self.assertGreater(
                self.feature(result, "10", f"layout_{category}_same_line_mass"), 0.0
            )
            self.assertEqual(
                self.feature(result, "11", f"layout_{category}_same_line_mass"), 0.0
            )
        self.assertEqual(
            self.feature(result, "10", "layout_exclusive_complete_chain"), 1.0
        )
        self.assertEqual(
            self.feature(
                result, "10", "layout_exclusive_complete_chain_with_diagnostic"
            ),
            1.0,
        )
        self.assertAlmostEqual(self.feature(result, "10", "layout_confidence"), 0.9)

    def test_parent_block_relation_is_weaker_but_unique(self):
        page = {
            "blocks": [
                {
                    "id": "alpha-block",
                    "lines": [
                        {"text": "Alpha Formation samples were selected."},
                        {"text": "IRM measurements were then acquired."},
                    ],
                },
                {"id": "beta-block", "text": "Beta Formation directions."},
            ]
        }
        result = build_layout_graph_target_features(page, self.targets, self.concepts)
        self.assertGreater(
            self.feature(result, "10", "layout_method_same_block_mass"), 0.0
        )
        self.assertEqual(
            self.feature(result, "10", "layout_method_same_line_mass"), 0.0
        )
        self.assertGreater(
            self.feature(result, "10", "layout_unique_sibling_margin_mean"), 0.0
        )

    def test_explicit_same_line_multi_target_evidence_is_shared(self):
        page = {
            "blocks": [
                {
                    "lines": [
                        {
                            "text": (
                                "Alpha Formation and Beta Formation samples were measured by IRM."
                            )
                        }
                    ]
                }
            ]
        }
        result = build_layout_graph_target_features(page, self.targets, self.concepts)
        for key in self.targets:
            self.assertGreater(
                self.feature(result, key, "layout_method_same_line_mass"), 0.0
            )
            self.assertGreater(
                self.feature(result, key, "layout_shared_evidence_mass"), 0.0
            )
            self.assertEqual(
                self.feature(result, key, "layout_exclusive_evidence_mass"), 0.0
            )
            self.assertEqual(
                self.feature(result, key, "layout_assignment_entropy_mean"), 1.0
            )
        self.assertEqual(result["assignments"][0]["status"], "shared")

    def test_parent_block_near_tie_is_ambiguous_not_shared(self):
        page = {
            "blocks": [
                {
                    "id": "mixed-block",
                    "lines": [
                        {"text": "Alpha Formation samples."},
                        {"text": "Beta Formation samples."},
                        {"text": "IRM acquisition curves were measured."},
                    ],
                }
            ]
        }
        result = build_layout_graph_target_features(page, self.targets, self.concepts)
        for key in self.targets:
            self.assertGreater(
                self.feature(result, key, "layout_ambiguous_evidence_mass"), 0.0
            )
            self.assertEqual(self.feature(result, key, "layout_evidence_mass"), 0.0)
            self.assertEqual(
                self.feature(result, key, "layout_has_ambiguous_evidence"), 1.0
            )
        self.assertEqual(result["assignments"][0]["status"], "ambiguous")

    def test_same_table_row_builds_target_specific_chain(self):
        page = {
            "tables": [
                {
                    "id": "rockmag",
                    "confidence": 0.85,
                    "rows": [
                        [
                            "Alpha Formation",
                            "IRM",
                            "identifies magnetite",
                            "carrier of ChRM",
                        ],
                        ["Beta Formation", "thermal demagnetization only"],
                    ],
                }
            ]
        }
        result = build_layout_graph_target_features(page, self.targets, self.concepts)
        for category in ("method", "diagnostic", "carrier", "bridge"):
            self.assertGreater(
                self.feature(result, "10", f"layout_{category}_same_table_row_mass"),
                0.0,
            )
            self.assertEqual(
                self.feature(result, "11", f"layout_{category}_same_table_row_mass"),
                0.0,
            )
        self.assertEqual(
            self.feature(result, "10", "layout_exclusive_complete_chain"), 1.0
        )

    def test_caption_relation_is_distinct_from_block_and_line(self):
        page = {
            "captions": [
                {
                    "id": "fig-2",
                    "confidence": 0.75,
                    "text": "Alpha Formation IRM identifies magnetite, carrier of ChRM.",
                }
            ]
        }
        result = build_layout_graph_target_features(page, self.targets, self.concepts)
        self.assertGreater(
            self.feature(result, "10", "layout_method_same_caption_mass"), 0.0
        )
        self.assertEqual(
            self.feature(result, "10", "layout_method_same_block_mass"), 0.0
        )
        self.assertEqual(
            self.feature(result, "10", "layout_method_same_line_mass"), 0.0
        )

    def test_evidence_mass_share_entropy_and_confidence(self):
        page = {
            "blocks": [
                {
                    "confidence": 0.8,
                    "lines": [
                        {
                            "text": (
                                "Alpha Formation IRM identifies magnetite as carrier of ChRM."
                            )
                        }
                    ],
                },
                {
                    "confidence": 0.6,
                    "lines": [{"text": "Beta Formation IRM experiment."}],
                },
            ]
        }
        result = build_layout_graph_target_features(page, self.targets, self.concepts)
        alpha_share = self.feature(result, "10", "layout_evidence_share")
        beta_share = self.feature(result, "11", "layout_evidence_share")
        self.assertAlmostEqual(alpha_share + beta_share, 1.0)
        self.assertGreater(alpha_share, beta_share)
        self.assertAlmostEqual(
            self.feature(result, "10", "layout_evidence_category_entropy"), 1.0
        )
        self.assertAlmostEqual(self.feature(result, "10", "layout_confidence"), 0.8)
        self.assertAlmostEqual(self.feature(result, "11", "layout_confidence"), 0.6)

    def test_duplicate_canonical_identity_is_unresolved_and_never_split_by_key(self):
        duplicate_targets = {
            "RESULTNO-10": target_profile("Alpha Formation"),
            "RESULTNO-12": target_profile("Alpha Formation"),
        }
        page = {
            "blocks": [
                {
                    "lines": [
                        {
                            "text": (
                                "Alpha Formation IRM identifies magnetite as carrier of ChRM."
                            )
                        }
                    ]
                }
            ]
        }
        result = build_layout_graph_target_features(
            page, duplicate_targets, self.concepts
        )
        self.assertTrue(np.array_equal(result["matrix"][0], result["matrix"][1]))
        for key in duplicate_targets:
            self.assertEqual(
                result["target_metadata"][key]["resolution_status"],
                "unresolved_duplicate_identity",
            )
            self.assertEqual(
                self.feature(result, key, "layout_unresolved_duplicate_identity"), 1.0
            )
            self.assertEqual(self.feature(result, key, "layout_evidence_mass"), 0.0)
            self.assertGreater(
                self.feature(result, key, "layout_unresolved_evidence_mass"), 0.0
            )
        self.assertTrue(
            all(
                assignment["status"] == "unresolved_duplicate_identity"
                and assignment["target_keys"] == []
                for assignment in result["assignments"]
            )
        )
        self.assertFalse(result["audit"]["result_keys_used_for_assignment"])

    def test_shared_event_can_resolve_sibling_but_not_duplicate_result_rows(self):
        targets = {
            "alpha-1": target_profile("Alpha Formation"),
            "alpha-2": target_profile("Alpha Formation"),
            "beta": target_profile("Beta Formation"),
        }
        page = {
            "blocks": [
                {
                    "lines": [
                        {
                            "text": (
                                "Alpha Formation and Beta Formation samples underwent IRM."
                            )
                        }
                    ]
                }
            ]
        }
        result = build_layout_graph_target_features(page, targets, self.concepts)
        assignment = result["assignments"][0]
        self.assertEqual(
            assignment["status"], "shared_with_unresolved_duplicate_identity"
        )
        self.assertEqual(assignment["target_keys"], ["beta"])
        for key in ("alpha-1", "alpha-2"):
            self.assertEqual(self.feature(result, key, "layout_evidence_mass"), 0.0)
            self.assertGreater(
                self.feature(result, key, "layout_unresolved_evidence_mass"), 0.0
            )
        self.assertGreater(
            self.feature(result, "beta", "layout_shared_evidence_mass"), 0.0
        )

    def test_rejects_outcome_fields_and_invalid_confidence(self):
        with self.assertRaisesRegex(ValueError, "forbidden outcome field"):
            build_layout_graph_target_features(
                {"R3": 1, "blocks": []}, self.targets, self.concepts
            )
        with self.assertRaisesRegex(ValueError, "must be in"):
            build_layout_graph_target_features(
                {"blocks": [{"text": "Alpha Formation IRM", "confidence": 1.2}]},
                self.targets,
                self.concepts,
            )

    def test_feature_order_and_assignments_are_deterministic(self):
        page = {
            "blocks": [
                {
                    "lines": [
                        {"text": "Alpha Formation IRM identifies magnetite as carrier of ChRM."}
                    ]
                }
            ]
        }
        first = build_layout_graph_target_features(page, self.targets, self.concepts)
        second = build_layout_graph_target_features(page, self.targets, self.concepts)
        self.assertEqual(first["feature_names"], second["feature_names"])
        self.assertEqual(first["assignments"], second["assignments"])
        self.assertTrue(np.array_equal(first["matrix"], second["matrix"]))


if __name__ == "__main__":
    unittest.main()
