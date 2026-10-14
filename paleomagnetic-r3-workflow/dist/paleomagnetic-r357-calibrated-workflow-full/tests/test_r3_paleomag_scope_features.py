import unittest

from scripts.r3_paleomag_scope_features import (
    build_paleomag_scope_features,
    resolve_document_target_anchors,
    target_profile,
)


class PaleomagScopeFeatureTests(unittest.TestCase):
    def setUp(self):
        self.identity = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "Alpha Formation"},
            {"REFNO": "1", "RESULTNO": "11", "ROCKNAME": "Beta Formation"},
        ]
        self.cache = {
            "1": {
                "REFNO": "1",
                "pages": [
                    {
                        "page": 1,
                        "text": (
                            "Results--Alpha Formation Thermomagnetic analysis. "
                            "IRM acquisition saturates at 300 mT and magnetite is the "
                            "principal carrier of the ChRM.\n\n"
                            "Results--Beta Formation. Samples only underwent thermal "
                            "demagnetization and were shown on Zijderveld diagrams."
                        ),
                    }
                ],
            }
        }

    def test_target_specific_rockmag_context_exceeds_demag_only_sibling(self):
        result = build_paleomag_scope_features(self.identity, self.cache)
        names = result["feature_names"]
        matrix = result["matrix"]
        rockmag = names.index("local_rockmag_count_log")
        complete = names.index("local_complete_chain_context_count_log")
        demag_only = names.index("local_demag_only_context_count_log")
        self.assertGreater(matrix[0, rockmag], matrix[1, rockmag])
        self.assertGreater(matrix[0, complete], matrix[1, complete])
        self.assertGreater(matrix[1, demag_only], matrix[0, demag_only])

    def test_rejects_identity_label_field(self):
        contaminated = [dict(self.identity[0], R3=1), self.identity[1]]
        with self.assertRaisesRegex(ValueError, "forbidden fields"):
            build_paleomag_scope_features(contaminated, self.cache)

    def test_feature_order_is_deterministic(self):
        first = build_paleomag_scope_features(self.identity, self.cache)
        second = build_paleomag_scope_features(self.identity, self.cache)
        self.assertEqual(first["feature_names"], second["feature_names"])
        self.assertTrue((first["matrix"] == second["matrix"]).all())

    def test_fuzzy_anchor_joins_redbeds_compound_in_one_page(self):
        text = "The Devonian red beds contain a stable remanence."
        result = resolve_document_target_anchors(
            [text], {"redbeds": target_profile("redbeds")}
        )
        page = result["redbeds"][1]
        folded = "the devonian red beds contain a stable remanence"
        expected = (folded.index("red beds") + folded.index("red beds") + len("red beds")) // 2
        self.assertEqual(page["fuzzy"], [expected])
        self.assertEqual(page["fuzzy_score"], 1.0)
        self.assertFalse(page["ambiguous"])

    def test_fuzzy_anchor_joins_target_compound_tokens(self):
        text = "These redbeds preserve the characteristic component."
        result = resolve_document_target_anchors(
            [text], {"red beds": target_profile("red beds")}
        )
        self.assertEqual(len(result["red beds"][1]["fuzzy"]), 1)
        self.assertEqual(result["red beds"][1]["fuzzy_score"], 1.0)

    def test_fuzzy_anchor_accepts_harbour_harbor_spelling(self):
        text = "Samples from Indian Harbor were thermally treated."
        result = resolve_document_target_anchors(
            [text], {"target": target_profile("Indian Harbour Dykes")}
        )
        page = result["target"][1]
        self.assertEqual(len(page["fuzzy"]), 1)
        self.assertGreaterEqual(page["fuzzy_score"], 0.90)

    def test_overlapping_sibling_near_tie_is_unassigned(self):
        text = "The Harbor specimens yielded stable directions."
        result = resolve_document_target_anchors(
            [text],
            {
                "british": target_profile("Harbour"),
                "ocr_variant": target_profile("Harborr"),
            },
        )
        self.assertEqual(result["british"][1]["fuzzy"], [])
        self.assertEqual(result["ocr_variant"][1]["fuzzy"], [])
        self.assertTrue(result["british"][1]["ambiguous"])
        self.assertTrue(result["ocr_variant"][1]["ambiguous"])

    def test_reference_page_never_receives_fuzzy_anchor(self):
        result = resolve_document_target_anchors(
            {
                1: "Methods contain no named unit.",
                7: "Smith studied Indian Harbour Dykes and Indian Harbor.",
            },
            {"target": target_profile("Indian Harbour Dykes")},
            reference_pages={7},
        )
        self.assertEqual(result["target"][7]["exact"], [])
        self.assertEqual(result["target"][7]["cluster"], [])
        self.assertEqual(result["target"][7]["fuzzy"], [])
        self.assertEqual(result["target"][7]["fuzzy_score"], 0.0)

    def test_compound_match_cannot_cross_a_page_boundary(self):
        result = resolve_document_target_anchors(
            ["The unit is red", "beds with stable remanence."],
            {"target": target_profile("redbeds")},
        )
        self.assertEqual(result["target"][1]["fuzzy"], [])
        self.assertEqual(result["target"][2]["fuzzy"], [])

    def test_clear_sibling_fuzzy_leader_is_retained(self):
        text = "redbeds mudstond doleritf rhyolitf were sampled"
        result = resolve_document_target_anchors(
            [text],
            {
                "leader": target_profile("red beds"),
                "trailing_sibling": target_profile(
                    "redbeds mudstone dolerite rhyolite"
                ),
            },
        )
        self.assertEqual(len(result["leader"][1]["fuzzy"]), 1)
        self.assertEqual(result["trailing_sibling"][1]["fuzzy"], [])
        self.assertFalse(result["leader"][1]["ambiguous"])

    def test_existing_document_exact_anchor_disables_all_fuzzy_fallback(self):
        result = resolve_document_target_anchors(
            {
                1: "Indian Harbour Dykes were sampled.",
                2: "Indian Harbor was later discussed.",
            },
            {"target": target_profile("Indian Harbour Dykes")},
        )
        self.assertTrue(result["target"][1]["exact"])
        self.assertEqual(result["target"][2]["fuzzy"], [])

    def test_existing_document_cluster_anchor_disables_all_fuzzy_fallback(self):
        result = resolve_document_target_anchors(
            {
                1: "Alpha samples from the Dolerite were measured.",
                2: "Alpha Harborr Doleritf was discussed later.",
            },
            {"target": target_profile("Alpha Harbor Dolerite")},
        )
        self.assertTrue(result["target"][1]["cluster"])
        self.assertEqual(result["target"][2]["fuzzy"], [])

    def test_folded_coordinates_are_stable_for_unicode_input(self):
        text = "Caf\u00e9: RED-BEDS occur here."
        profile = target_profile("redbeds")
        first = resolve_document_target_anchors([text], {"target": profile})
        second = resolve_document_target_anchors([text], {"target": profile})
        self.assertEqual(first, second)
        folded = "cafe red beds occur here"
        expected = (
            folded.index("red beds")
            + folded.index("red beds")
            + len("red beds")
        ) // 2
        self.assertEqual(first["target"][1]["fuzzy"], [expected])


if __name__ == "__main__":
    unittest.main()
