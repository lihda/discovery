import math
import unittest

from scripts.r3_paleomag_entity_graph import FEATURE_NAMES, build_entity_graph


class PaleomagEntityGraphTests(unittest.TestCase):
    def test_exact_alias_and_probability_schema(self):
        result = build_entity_graph(
            {"alpha": "Alpha Formation", "beta": "Beta Group"},
            [{"chunk_id": "s1", "kind": "sentence", "text": "Alpha Fm samples were collected."}],
        )
        assignment = result["assignments"][0]
        self.assertEqual(assignment["top_target"], "alpha")
        self.assertEqual(assignment["status"], "assigned_alias")
        total = sum(assignment["target_probabilities"].values())
        total += assignment["global_mass"] + assignment["unresolved_mass"]
        self.assertAlmostEqual(total, 1.0)
        self.assertEqual(result["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(tuple(result["target_features"]["alpha"]), FEATURE_NAMES)

        inventory = build_entity_graph(
            {"site": "Gamma Site", "lithology": "Shadli basalts"}, []
        )["inventories"]
        self.assertTrue(any(alias["text"] == "gamma" for alias in inventory["site"]["aliases"]))
        self.assertTrue(any(alias["text"] == "shadli" for alias in inventory["lithology"]["aliases"]))

    def test_generic_lithology_cannot_uniquely_bind(self):
        result = build_entity_graph(
            {"basalt": "Basalt"},
            [{"chunk_id": "p1", "kind": "paragraph", "text": "Basalt samples were measured."}],
        )
        assignment = result["assignments"][0]
        self.assertIsNone(assignment["top_target"])
        self.assertEqual(assignment["status"], "unresolved_alias_conflict")
        self.assertEqual(assignment["target_scores"]["basalt"], 0.0)

    def test_parenthetical_abbreviation_is_kept_only_when_mutual_unique(self):
        accepted = build_entity_graph(
            {"alpha": "Alpha Formation", "beta": "Beta Formation"},
            [
                {"chunk_id": "s1", "kind": "sentence", "text": "Alpha Formation (AF) was sampled."},
                {"chunk_id": "s2", "kind": "sentence", "text": "BF (Beta Formation) was sampled."},
                {"chunk_id": "s3", "kind": "sentence", "text": "AF contains red beds."},
            ],
        )
        alpha_aliases = accepted["inventories"]["alpha"]["aliases"]
        self.assertTrue(any(alias["text"] == "af" for alias in alpha_aliases))
        self.assertEqual(accepted["assignments"][2]["top_target"], "alpha")

        rejected = build_entity_graph(
            {"alpha": "Alpha Formation", "alpine": "Alpine Formation"},
            [
                {"chunk_id": "s1", "kind": "sentence", "text": "Alpha Formation (AF) was sampled."},
                {"chunk_id": "s2", "kind": "sentence", "text": "Alpine Formation (AF) was sampled."},
            ],
        )
        self.assertFalse(
            any(
                alias["text"] == "af"
                for inventory in rejected["inventories"].values()
                for alias in inventory["aliases"]
            )
        )

    def test_cross_entity_alias_conflict_fails_closed(self):
        result = build_entity_graph(
            {"formation": "Alpha Formation", "group": "Alpha Group"},
            [{"chunk_id": "s1", "kind": "sentence", "text": "Alpha samples were analyzed."}],
        )
        assignment = result["assignments"][0]
        self.assertEqual(assignment["status"], "unresolved_alias_conflict")
        self.assertIsNone(assignment["top_target"])
        self.assertEqual(set(assignment["target_scores"].values()), {0.0})
        self.assertIn("alpha", result["audit"]["alias_conflicts"])

    def test_composite_geological_identity_yields_unique_subunit_aliases(self):
        result = build_entity_graph(
            {
                "burakovka": "Burakovka Intrusion (Aganozero Block)",
                "moncha": "Dolerite Dykes, Moncha Pluton",
            },
            [
                {
                    "chunk_id": "s1",
                    "kind": "sentence",
                    "text": "The Aganozero Block contains the sampled section.",
                },
                {
                    "chunk_id": "s2",
                    "kind": "sentence",
                    "text": "Results from the Moncha Pluton are shown below.",
                },
            ],
        )
        self.assertEqual(result["assignments"][0]["top_target"], "burakovka")
        self.assertEqual(result["assignments"][1]["top_target"], "moncha")
        self.assertTrue(
            any(
                alias["text"] == "aganozero block"
                for alias in result["inventories"]["burakovka"]["aliases"]
            )
        )

    def test_ocr_diacritic_and_intervening_lithology_cluster_are_resolved(self):
        result = build_entity_graph(
            {
                "kopparnas": "Kopparnäs Dykes",
                "matozero_dyke": "Matozero Dykes",
                "matozero_sill": "Matozero Sill",
            },
            [
                {
                    "chunk_id": "s1",
                    "kind": "sentence",
                    "text": "Kopparna¨s diabase dykes were sampled.",
                },
                {
                    "chunk_id": "s2",
                    "kind": "sentence",
                    "text": "The Matozero doleritic dykes carry four components.",
                },
            ],
        )
        self.assertEqual(result["assignments"][0]["top_target"], "kopparnas")
        self.assertEqual(result["assignments"][1]["top_target"], "matozero_dyke")
        self.assertTrue(
            result["assignments"][1]["signals"]["matozero_dyke"][
                "token_cluster_match"
            ]
        )

    def test_component_and_polarity_mentions_disambiguate_siblings(self):
        result = build_entity_graph(
            {
                "gamma_a": "Gamma Formation, A component",
                "gamma_b": "Gamma Formation, B component",
                "delta_n": "Delta lavas, normal component",
                "delta_r": "Delta lavas, reversed component",
            },
            [
                {"chunk_id": "s1", "kind": "sentence", "text": "The A component of Gamma Formation was isolated."},
                {"chunk_id": "s2", "kind": "sentence", "text": "The reversed component of Delta lavas is stable."},
            ],
        )
        self.assertEqual(result["assignments"][0]["top_target"], "gamma_a")
        self.assertEqual(result["assignments"][1]["top_target"], "delta_r")
        self.assertEqual(result["assignments"][0]["status"], "assigned_qualified")
        self.assertGreater(
            result["assignments"][0]["swap_delta"]["gamma_a"], 0.0
        )

    def test_duplicate_canonical_identity_is_never_pseudo_distinguished(self):
        result = build_entity_graph(
            {"row_1": "Alpha Fm.", "row_2": "Alpha Formation"},
            [{"chunk_id": "s1", "kind": "sentence", "text": "Alpha Formation was sampled."}],
        )
        assignment = result["assignments"][0]
        self.assertEqual(assignment["status"], "duplicate_canonical_identity")
        self.assertIsNone(assignment["top_target"])
        self.assertEqual(assignment["target_scores"]["row_1"], assignment["target_scores"]["row_2"])
        self.assertEqual(assignment["target_probabilities"]["row_1"], assignment["target_probabilities"]["row_2"])
        self.assertEqual(assignment["swap_delta"]["row_1"], 0.0)
        self.assertEqual(result["target_features"]["row_1"], result["target_features"]["row_2"])

    def test_all_component_and_polarity_forms_are_detected(self):
        targets = {
            "a": "U, A component", "b": "U, B component", "c": "U, C component",
            "htc": "V, HTC", "ltc": "V, LTC", "chrm": "W, ChRM",
            "n": "X, N", "r": "X, R",
        }
        texts = (
            ("a", "A component of U"), ("b", "component B of U"),
            ("c", "C magnetization of U"), ("htc", "HTC of V"),
            ("ltc", "low-temperature component of V"),
            ("chrm", "ChRM of W"), ("n", "normal polarity of X"),
            ("r", "R-polarity of X"),
        )
        result = build_entity_graph(
            targets,
            [
                {"chunk_id": f"s{index}", "kind": "sentence", "text": text}
                for index, (_, text) in enumerate(texts)
            ],
        )
        self.assertEqual(
            [assignment["top_target"] for assignment in result["assignments"]],
            [target_id for target_id, _ in texts],
        )
        self.assertTrue(all(math.isfinite(row["entropy"]) for row in result["assignments"]))

    def test_input_boundary_rejects_evaluation_fields(self):
        with self.assertRaisesRegex(ValueError, "forbidden evaluation field"):
            build_entity_graph(
                {"alpha": "Alpha Formation"},
                [{"chunk_id": "s1", "kind": "sentence", "text": "Alpha", "labels": 1}],
            )
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            build_entity_graph(
                {"alpha": "Alpha Formation"},
                [{"chunk_id": "s1", "kind": "sentence", "text": "Alpha", "page": 1}],
            )


if __name__ == "__main__":
    unittest.main()
