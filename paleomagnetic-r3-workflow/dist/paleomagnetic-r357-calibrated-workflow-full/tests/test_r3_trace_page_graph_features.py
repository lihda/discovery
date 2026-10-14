import json
from pathlib import Path
import tempfile
import unittest

from scripts.r3_trace_page_graph_features import (
    build_trace_page_graph_features,
    load_trace_concepts,
)


class TracePageGraphFeatureTests(unittest.TestCase):
    def setUp(self):
        self.identity = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "Alpha Formation"},
            {"REFNO": "1", "RESULTNO": "11", "ROCKNAME": "Beta Formation"},
        ]
        self.cache = {
            "1": {
                "REFNO": "1",
                "pages": [
                    {"page": 1, "text": "Alpha Formation samples were investigated by IRM acquisition."},
                    {"page": 2, "text": "Thermomagnetic curves identify magnetite in Alpha Formation."},
                    {"page": 3, "text": "Magnetite is the carrier of the primary ChRM in Alpha Formation."},
                    {"page": 4, "text": "Beta Formation directions were thermally demagnetized only."},
                ],
            }
        }
        payload = {
            "concepts": [
                {"concept_id": "irm", "category": "method", "search_patterns": [r"\bIRM\b"]},
                {"concept_id": "thermomagnetic", "category": "method", "search_patterns": ["thermomagnetic"]},
                {"concept_id": "magnetite", "category": "carrier", "search_patterns": ["magnetite"]},
                {"concept_id": "carrier_result", "category": "diagnostic", "search_patterns": ["identify"]},
                {"concept_id": "carrier_bridge", "category": "bridge", "search_patterns": ["carrier of", "carries"]},
            ]
        }
        self.temporary = tempfile.TemporaryDirectory()
        path = Path(self.temporary.name) / "concepts.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.concepts = load_trace_concepts(path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_physical_page_chain_is_target_specific(self):
        result = build_trace_page_graph_features(self.identity, self.cache, self.concepts)
        names = result["feature_names"]
        matrix = result["matrix"]
        complete = names.index("trace_chain_complete")
        connected = names.index("trace_chain_connected")
        score = names.index("trace_chain_score")
        self.assertEqual(matrix[0, complete], 1.0)
        self.assertEqual(matrix[0, connected], 1.0)
        self.assertGreater(matrix[0, score], matrix[1, score])

    def test_rejects_label_column(self):
        contaminated = [dict(self.identity[0], R3=1), self.identity[1]]
        with self.assertRaisesRegex(ValueError, "forbidden fields"):
            build_trace_page_graph_features(contaminated, self.cache, self.concepts)

    def test_feature_order_is_deterministic(self):
        first = build_trace_page_graph_features(self.identity, self.cache, self.concepts)
        second = build_trace_page_graph_features(self.identity, self.cache, self.concepts)
        self.assertEqual(first["feature_names"], second["feature_names"])
        self.assertTrue((first["matrix"] == second["matrix"]).all())

    def test_component_qualifier_separates_rows_from_same_material(self):
        identity = [
            {"REFNO": "2", "RESULTNO": "20", "ROCKNAME": "Gamma Formation, A component"},
            {"REFNO": "2", "RESULTNO": "21", "ROCKNAME": "Gamma Formation, B component"},
        ]
        cache = {
            "2": {
                "REFNO": "2",
                "pages": [
                    {"page": 1, "text": "Samples came from the Gamma Formation."},
                    {"page": 2, "text": "IRM acquisition of the A component was measured."},
                    {"page": 3, "text": "Magnetite carries the primary A component remanence."},
                    {"page": 4, "text": "The B component was thermally demagnetized."},
                ],
            }
        }
        result = build_trace_page_graph_features(identity, cache, self.concepts)
        names = result["feature_names"]
        matrix = result["matrix"]
        evidence = names.index("qualifier_evidence_page_count_log")
        exclusive = names.index("qualifier_exclusive_page_count_log")
        event_mass = names.index("event_method_irm_target_mass_count_log")
        chain = names.index("trace_chain_score")
        self.assertGreater(matrix[0, evidence], matrix[1, evidence])
        self.assertGreater(matrix[0, exclusive], 0.0)
        self.assertGreater(matrix[0, event_mass], matrix[1, event_mass])
        self.assertGreater(matrix[0, chain], matrix[1, chain])

    def test_audit_marks_indistinguishable_duplicate_identity_rows(self):
        identity = [self.identity[0], dict(self.identity[0], RESULTNO="12")]
        cache = {"1": self.cache["1"]}
        result = build_trace_page_graph_features(identity, cache, self.concepts)
        self.assertEqual(result["paper_audit"][0]["indistinguishable_identity_rows"], 1)
        self.assertTrue((result["matrix"][0] == result["matrix"][1]).all())

    def test_frozen_bridge_concept_replaces_fallback_bridge_counter(self):
        path = Path(self.temporary.name) / "bridge.json"
        path.write_text(
            json.dumps(
                {
                    "concepts": [
                        {
                            "concept_id": "method",
                            "category": "method",
                            "search_patterns": [r"\bIRM\b"],
                        },
                        {
                            "concept_id": "carrier",
                            "category": "carrier",
                            "search_patterns": ["magnetite"],
                        },
                        {
                            "concept_id": "diagnostic",
                            "category": "diagnostic",
                            "search_patterns": ["identify"],
                        },
                        {
                            "concept_id": "bridge",
                            "category": "bridge",
                            "search_patterns": ["carrier of"],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        result = build_trace_page_graph_features(
            self.identity, self.cache, load_trace_concepts(path)
        )
        names = result["feature_names"]
        bridge = names.index("query_bridge_available")
        self.assertEqual(result["matrix"][0, bridge], 1.0)

    def test_event_distance_uses_original_coordinates_after_text_folding(self):
        prefix = "Fig. 1.....      \n\n\n" * 100
        identity = [
            {"REFNO": "3", "RESULTNO": "30", "ROCKNAME": "Alpha Formation"}
        ]
        cache = {
            "3": {
                "REFNO": "3",
                "pages": [
                    {
                        "page": 1,
                        "text": prefix
                        + "Alpha Formation samples underwent IRM acquisition. "
                        + "Magnetite carries the remanence.",
                    }
                ],
            }
        }
        result = build_trace_page_graph_features(identity, cache, self.concepts)
        names = result["feature_names"]
        matrix = result["matrix"]
        for suffix in (
            "occurrence_count_log",
            "target_mass_count_log",
            "mean_distance_score",
        ):
            with self.subTest(suffix=suffix):
                feature = names.index(f"event_method_irm_{suffix}")
                self.assertGreater(matrix[0, feature], 0.0)

    def test_restricted_coreference_events_enter_features_and_context(self):
        cache = {
            "1": {
                "REFNO": "1",
                "pages": [
                    {
                        "page": 1,
                        "text": (
                            "Alpha Formation specimens were measured by IRM. "
                            "These specimens contain magnetite that carries remanence."
                        ),
                    }
                ],
            }
        }
        result = build_trace_page_graph_features(self.identity, cache, self.concepts)
        names = result["feature_names"]
        assigned = names.index("coref_assigned_event_count_log")
        self.assertGreater(result["matrix"][0, assigned], 0.0)
        self.assertEqual(result["matrix"][1, assigned], 0.0)
        self.assertNotIn("[TARGET_BOUND_ATOMIC_EVENTS]", result["target_context_documents"][0])

    def test_fuzzy_anchor_drives_page_relevance_and_event_binding(self):
        identity = [
            {
                "REFNO": "4",
                "RESULTNO": "40",
                "ROCKNAME": "Indian Harbour Dykes",
            }
        ]
        cache = {
            "4": {
                "REFNO": "4",
                "pages": [
                    {
                        "page": 1,
                        "text": (
                            "Indian Harbor specimens were measured by IRM. "
                            "Magnetite carries the stable remanence."
                        ),
                    }
                ],
            }
        }
        result = build_trace_page_graph_features(identity, cache, self.concepts)
        names = result["feature_names"]
        fuzzy = names.index("anchor_fuzzy_count_log")
        method = names.index("query_method_available")
        event = names.index("event_method_irm_target_mass_count_log")
        self.assertGreater(result["matrix"][0, fuzzy], 0.0)
        self.assertEqual(result["matrix"][0, method], 1.0)
        self.assertGreater(result["matrix"][0, event], 0.0)
        self.assertEqual(result["row_metadata"][0]["fuzzy_anchor_pages"], 1)


if __name__ == "__main__":
    unittest.main()
