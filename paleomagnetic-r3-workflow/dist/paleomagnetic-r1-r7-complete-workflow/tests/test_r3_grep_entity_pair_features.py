import unittest

from scripts.r3_grep_entity_pair_features import build_grep_entity_pair_features


class GrepEntityPairFeatureTests(unittest.TestCase):
    def setUp(self):
        self.identity = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "Alpha Formation"},
            {"REFNO": "1", "RESULTNO": "11", "ROCKNAME": "Beta Formation"},
        ]
        self.cache = {
            "1": {
                "REFNO": "1",
                "source": {"source_quality_flags": []},
                "pages": [
                    {
                        "page": 1,
                        "source": "synthetic",
                        "section_hints": ["Rock magnetism"],
                        "text": (
                            "Alpha Formation samples were measured using IRM acquisition "
                            "and hysteresis experiments. The curves did not saturate at "
                            "1 T. Magnetite carries the primary remanence of the Alpha "
                            "Formation.\n\nBeta Formation directions were measured, but no "
                            "carrier study was reported for those samples."
                        ),
                    }
                ],
            }
        }

    def test_target_evidence_mass_is_not_copied_to_sibling(self):
        result = build_grep_entity_pair_features(self.identity, self.cache)
        names = result["feature_names"]
        matrix = result["matrix"]

        self.assertEqual(matrix.shape[0], 2)
        self.assertEqual(len(names), matrix.shape[1])
        global_index = names.index("graph_bridge_global_relation_present")
        mass_index = names.index("graph_bridge_mass_share")
        margin_index = names.index(
            "contrast_graph_bridge_mass_share_minus_sibling_max"
        )
        self.assertEqual(matrix[0, global_index], matrix[1, global_index])
        self.assertGreater(matrix[0, mass_index], matrix[1, mass_index])
        self.assertGreater(matrix[0, margin_index], 0.0)
        self.assertLess(matrix[1, margin_index], 0.0)

    def test_identity_rejects_labels_and_source_fields(self):
        contaminated = [dict(self.identity[0], R3=1), self.identity[1]]
        with self.assertRaisesRegex(ValueError, "forbidden fields"):
            build_grep_entity_pair_features(contaminated, self.cache)

    def test_feature_order_is_deterministic(self):
        first = build_grep_entity_pair_features(self.identity, self.cache)
        second = build_grep_entity_pair_features(self.identity, self.cache)
        self.assertEqual(first["feature_names"], second["feature_names"])
        self.assertTrue((first["matrix"] == second["matrix"]).all())


if __name__ == "__main__":
    unittest.main()
