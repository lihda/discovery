import hashlib
import unittest
from unittest import mock

import scripts.paper_text_retrieval as retrieval_module
from scripts.paper_text_retrieval import PaperTextRetriever, retrieve_descriptions


def page(number, text, *section_hints):
    return {
        "page": number,
        "source": "synthetic",
        "text": text,
        "section_hints": list(section_hints),
    }


class PaperTextRetrievalTests(unittest.TestCase):
    def test_r3_formal_criterion_methods_are_retrieved(self):
        row = {
            "pages": [
                page(
                    2,
                    "Alpha Formation samples were studied by low-temperature treatment of IRM, "
                    "AMS and anisotropy of anhysteretic remanence. Polished thin sections and "
                    "mineral separation identify magnetite as the carrier of a primary remanence.",
                    "Rock magnetism",
                )
            ]
        }

        result = retrieve_descriptions(row, "Alpha Formation", False, "R3")
        subtypes = {hit["subtype"] for hit in result["categories"]["method"]}

        self.assertIn("low_temperature_irm_experiment", subtypes)
        self.assertIn("magnetic_fabric_experiment", subtypes)
        self.assertIn("mineralogical_observation", subtypes)
        self.assertTrue(result["categories"]["bridge"])

    def test_r3_diagnostic_results_are_independent_auditable_hits(self):
        row = {
            "pages": [
                page(
                    1,
                    "Alpha Formation IRM acquisition curves were measured. "
                    "The curves did not saturate up to 1 T. The Curie temperature was "
                    "580 C. The Hcr/Hc=2.79 and the samples fall in the PSD field.",
                    "Rock magnetism",
                )
            ]
        }

        result = retrieve_descriptions(row, "Alpha Formation", False, "R3")
        diagnostic = result["categories"]["diagnostic"]

        self.assertTrue(diagnostic)
        self.assertTrue(all(hit["category"] == "diagnostic" for hit in diagnostic))
        self.assertTrue(all(hit["claim_text_sha256"] for hit in diagnostic))
        self.assertTrue(
            {
                "saturation_behavior",
                "curie_temperature",
                "hysteresis_ratio_result",
                "domain_state",
            }.issubset({hit["subtype"] for hit in diagnostic})
        )

    def test_r3_diagnostics_require_reported_results(self):
        positives = (
            (
                "Alpha Formation IRM curves do not saturate until 2 T.",
                "saturation_behavior",
                "2 T",
            ),
            (
                "The Curie temperature of the Alpha Formation was 580 C.",
                "curie_temperature",
                "580 C",
            ),
            (
                "Alpha Formation samples yielded Hcr/Hc=2.79.",
                "hysteresis_ratio_result",
                "2.79",
            ),
            (
                "Alpha Formation samples fall in the PSD field.",
                "domain_state",
                "PSD field",
            ),
        )
        for text, subtype, result_text in positives:
            with self.subTest(subtype=subtype):
                result = retrieve_descriptions(
                    {"pages": [page(1, text)]},
                    "Alpha Formation",
                    False,
                    "R3",
                )
                hits = [
                    hit
                    for hit in result["categories"]["diagnostic"]
                    if hit["subtype"] == subtype
                ]
                self.assertTrue(hits)
                self.assertIn(result_text, hits[0]["matched_text"])

    def test_r3_diagnostic_vocabulary_without_a_result_is_rejected(self):
        negatives = (
            "Alpha Formation saturation was not measured.",
            "Thermomagnetic measurements of Alpha Formation were made to determine "
            "the Curie temperature of 580 C.",
            "Alpha Formation hysteresis ratios were plotted to obtain information "
            "about domain state.",
            "Alpha Formation was discussed. The Verwey transition of magnetite "
            "occurs near 120 K.",
        )
        for text in negatives:
            with self.subTest(text=text):
                result = retrieve_descriptions(
                    {"pages": [page(1, text)]},
                    "Alpha Formation",
                    False,
                    "R3",
                )
                self.assertEqual(result["categories"]["diagnostic"], [])

        background = retrieve_descriptions(
            {
                "pages": [
                    page(
                        1,
                        "Alpha Formation was discussed. The Verwey transition of magnetite "
                        "occurs near 120 K.",
                    )
                ]
            },
            "Alpha Formation",
            False,
            "R3",
        )
        self.assertFalse(
            any(
                hit["subtype"] == "low_temperature_irm_experiment"
                for hit in background["categories"]["method"]
            )
        )

    def test_reusable_retriever_matches_standalone_api_and_scans_once(self):
        row = {
            "pages": [
                page(
                    1,
                    "Alpha Formation has no structural correction. "
                    "Beta Formation was tilt-corrected. IRM measurements show that "
                    "hematite is the carrier of remanence in both units.",
                )
            ]
        }
        standalone_alpha = retrieve_descriptions(row, "Alpha Formation", True, "R5")
        standalone_beta = retrieve_descriptions(row, "Beta Formation", True, "R5")
        retriever = PaperTextRetriever(row)

        with mock.patch.object(
            retrieval_module,
            "_scan_pattern_matches",
            wraps=retrieval_module._scan_pattern_matches,
        ) as scan:
            cached_alpha = retriever.retrieve("Alpha Formation", True, "R5")
            cached_beta = retriever.retrieve("Beta Formation", True, "r5")
            repeated_alpha = retriever.retrieve("Alpha Formation", True, "R5")

            self.assertEqual(scan.call_count, 1)
            self.assertEqual(cached_alpha, standalone_alpha)
            self.assertEqual(cached_beta, standalone_beta)
            self.assertEqual(repeated_alpha, cached_alpha)

            retriever.retrieve("Alpha Formation", True, "R3")
            retriever.retrieve("Beta Formation", True, "R3")
            self.assertEqual(scan.call_count, 2)

        info = retriever.cache_info()
        self.assertEqual(info["cached_indicators"], ["R3", "R5"])
        self.assertEqual(info["indicator_scan_counts"], {"R3": 1, "R5": 1})
        self.assertEqual(info["cached_target_profiles"], 2)

    def test_cached_matches_do_not_reuse_mutable_result_dicts(self):
        row = {
            "pages": [
                page(
                    1,
                    "Gamma Formation was tilt-corrected and restored to paleohorizontal.",
                )
            ]
        }
        retriever = PaperTextRetriever(row)

        first = retriever.retrieve("Gamma Formation", False, "R5")
        expected = retriever.retrieve("Gamma Formation", False, "R5")
        first["hits"][0]["matched_text"] = "caller mutation"
        first["categories"]["rescue"].clear()
        repeated = retriever.retrieve("Gamma Formation", False, "R5")

        self.assertEqual(repeated, expected)
        self.assertNotEqual(repeated["hits"][0]["matched_text"], "caller mutation")

    def test_multi_target_evidence_is_ranked_by_local_binding(self):
        row = {
            "pages": [
                page(
                    1,
                    "Results\n"
                    "The Alpha Formation has no structural correction. "
                    "This makes its paleohorizontal control uncertain.\n\n"
                    "The Beta Formation was tilt-corrected and restored to paleohorizontal.",
                    "Results",
                )
            ]
        }

        alpha = retrieve_descriptions(row, "Alpha Formation", True, "R5")
        beta = retrieve_descriptions(row, "Beta Formation", True, "R5")

        self.assertEqual(alpha["target_anchor"]["status"], "exact")
        self.assertEqual(beta["target_anchor"]["status"], "exact")
        self.assertEqual(alpha["hits"][0]["category"], "risk")
        self.assertEqual(beta["hits"][0]["category"], "rescue")
        self.assertTrue(alpha["hits"][0]["local_target"])
        self.assertTrue(beta["hits"][0]["local_target"])

    def test_multi_target_paragraph_scope_promotes_only_exclusive_target_evidence(self):
        row = {
            "pages": [
                page(
                    1,
                    "Alpha Formation samples were selected for rock-magnetic analysis. "
                    "IRM acquisition curves did not saturate at 1 T. Magnetite was "
                    "identified as the remanence carrier.\n\n"
                    "Beta Formation samples were thermally demagnetized only.",
                )
            ]
        }

        result = retrieve_descriptions(
            row,
            "Alpha Formation",
            True,
            "R3",
            competing_targets=["Beta Formation"],
        )
        scoped_hits = [
            hit
            for hit in result["hits"]
            if hit["target_anchor"] == "target_same_paragraph"
        ]

        self.assertEqual(
            result["scope_binding"]["status"],
            "enabled_with_competing_target_exclusion",
        )
        self.assertTrue(scoped_hits)
        self.assertTrue(all(hit["local_target"] for hit in scoped_hits))
        self.assertEqual({hit["scope_id"] for hit in scoped_hits}, {"page:1:paragraph:1"})
        self.assertTrue(all(hit["scope_target_exclusive"] for hit in scoped_hits))

    def test_promoted_claim_retains_late_raw_match_and_hashes_saved_text(self):
        long_paragraph = (
            "Alpha Formation samples were selected. "
            + "background " * 125
            + "IRM acquisition curves did not saturate at 1 T."
        )
        self.assertGreater(len(long_paragraph), 1200)
        row = {
            "pages": [
                page(
                    1,
                    long_paragraph + "\n\nBeta Formation samples were demagnetized.",
                )
            ]
        }

        result = retrieve_descriptions(
            row,
            "Alpha Formation",
            True,
            "R3",
            competing_targets=["Beta Formation"],
        )
        method = next(
            hit
            for hit in result["categories"]["method"]
            if hit["target_anchor"] == "target_same_paragraph"
            and "IRM" in hit["matched_text"]
        )

        self.assertEqual(len(method["claim_text"]), 1200)
        self.assertIn(method["matched_text"], method["claim_text"])
        self.assertEqual(
            method["claim_text_sha256"],
            hashlib.sha256(method["claim_text"].encode("utf-8")).hexdigest(),
        )

    def test_multi_target_paragraph_scope_rejects_competing_target(self):
        row = {
            "pages": [
                page(
                    1,
                    "Alpha Formation and Beta Formation samples were compared. "
                    "IRM acquisition curves did not saturate at 1 T and magnetite was identified.",
                )
            ]
        }

        result = retrieve_descriptions(
            row,
            "Alpha Formation",
            True,
            "R3",
            competing_targets=["Beta Formation"],
        )
        method = next(
            hit for hit in result["categories"]["method"] if "IRM" in hit["matched_text"]
        )

        self.assertFalse(method["local_target"])
        self.assertNotEqual(method["target_anchor"], "target_same_paragraph")
        self.assertIn("beta formation", method["scope_competing_targets"])

    def test_same_sentence_competing_target_is_not_local(self):
        row = {
            "pages": [
                page(
                    1,
                    "Unlike the Alpha Formation, the Beta Formation was tested with IRM "
                    "acquisition curves that identify magnetite as the remanence carrier.",
                )
            ]
        }

        result = retrieve_descriptions(
            row,
            "Alpha Formation",
            True,
            "R3",
            competing_targets=["Beta Formation"],
        )
        method = next(
            hit for hit in result["categories"]["method"] if "IRM" in hit["matched_text"]
        )

        self.assertFalse(method["local_target"])
        self.assertNotEqual(method["target_anchor"], "target_same_sentence")
        self.assertIn("beta formation", method["scope_competing_targets"])
        self.assertEqual(method["claim_text"], method["matched_text"])
        self.assertNotIn("Beta Formation", method["claim_text"])

    def test_multi_target_table_row_scope_is_auditable(self):
        row = {
            "pages": [
                page(
                    3,
                    "Unit\tMethod\tInterpretation\n"
                    "Alpha Formation\tIRM acquisition\tMagnetite carrier of remanence\n"
                    "Beta Formation\tIRM acquisition\tHematite carrier of remanence",
                )
            ]
        }

        result = retrieve_descriptions(
            row,
            "Alpha Formation",
            True,
            "R3",
            competing_targets=["Beta Formation"],
        )
        method = next(
            hit
            for hit in result["categories"]["method"]
            if hit["scope_id"] == "page:3:table_row:2"
        )
        sibling_method = next(
            hit
            for hit in result["categories"]["method"]
            if hit["scope_id"] == "page:3:table_row:3"
        )

        self.assertEqual(method["target_anchor"], "target_same_table_row")
        self.assertEqual(method["scope_kind"], "table_row")
        self.assertEqual(method["scope_id"], "page:3:table_row:2")
        self.assertTrue(method["local_target"])
        self.assertFalse(sibling_method["local_target"])
        self.assertIn("beta formation", sibling_method["scope_competing_targets"])

        sentence_only = retrieve_descriptions(
            row,
            "Alpha Formation",
            True,
            "R3",
            competing_targets=["Beta Formation"],
            allow_structured_scope=False,
        )
        sentence_only_method = next(
            hit
            for hit in sentence_only["categories"]["method"]
            if hit["scope_id"] == "page:3:table_row:2"
        )
        self.assertEqual(sentence_only["scope_binding"]["status"], "disabled_by_policy")
        self.assertFalse(sentence_only_method["local_target"])
        self.assertNotEqual(
            sentence_only_method["target_anchor"],
            "target_same_table_row",
        )

    def test_multi_target_scope_promotion_requires_competing_target_universe(self):
        row = {
            "pages": [
                page(
                    1,
                    "Alpha Formation samples were selected. IRM acquisition curves did not "
                    "saturate at 1 T.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Alpha Formation", True, "R3")
        method = next(
            hit for hit in result["categories"]["method"] if "IRM" in hit["matched_text"]
        )

        self.assertEqual(
            result["scope_binding"]["status"],
            "disabled_without_competing_target_universe",
        )
        self.assertFalse(method["local_target"])
        self.assertNotEqual(method["target_anchor"], "target_same_paragraph")

    def test_short_page_without_paragraph_boundaries_is_not_promoted(self):
        row = {
            "pages": [
                page(
                    1,
                    "Alpha Formation samples were selected. IRM acquisition curves did not "
                    "saturate at 1 T.",
                )
            ]
        }

        result = retrieve_descriptions(
            row,
            "Alpha Formation",
            True,
            "R3",
            competing_targets=["Beta Formation"],
        )
        method = next(
            hit for hit in result["categories"]["method"] if "IRM" in hit["matched_text"]
        )

        self.assertFalse(method["scope_bounded"])
        self.assertFalse(method["local_target"])
        self.assertNotEqual(method["target_anchor"], "target_same_paragraph")

    def test_exact_target_forward_discourse_scope_binds_two_following_sentences(self):
        row = {
            "pages": [
                page(
                    1,
                    "Alpha Formation samples were selected for magnetic measurements. "
                    "IRM acquisition curves were measured. The curves did not saturate "
                    "up to 1 T. Beta Formation samples were thermally demagnetized.",
                )
            ]
        }

        result = retrieve_descriptions(
            row,
            "Alpha Formation",
            True,
            "R3",
            competing_targets=["Beta Formation"],
            allow_discourse_scope=True,
        )
        method = next(
            hit for hit in result["categories"]["method"] if "IRM" in hit["matched_text"]
        )
        diagnostic = next(
            hit
            for hit in result["categories"]["diagnostic"]
            if hit["subtype"] == "saturation_behavior"
        )

        self.assertTrue(method["local_target"])
        self.assertTrue(diagnostic["local_target"])
        self.assertEqual(method["target_anchor"], "target_same_discourse")
        self.assertEqual(diagnostic["target_anchor"], "target_same_discourse")
        self.assertEqual(method["scope_id"], diagnostic["scope_id"])
        self.assertEqual(method["scope_kind"], "discourse")
        self.assertNotIn("saturate", method["claim_text"])
        self.assertNotIn("IRM", diagnostic["claim_text"])

    def test_discourse_scope_accepts_unambiguous_token_variants(self):
        cases = (
            (
                "Group YA dykes",
                "Group YB dykes",
                "YA group dykes samples were selected. "
                "IRM acquisition curves were measured.",
            ),
            (
                "Obuasi Greenstone",
                "Kumasi Greenstone",
                "The greenstone body at Obuasi was sampled. "
                "IRM acquisition curves were measured.",
            ),
        )
        for target, sibling, text in cases:
            with self.subTest(target=target):
                baseline = retrieve_descriptions(
                    {"pages": [page(1, text)]},
                    target,
                    True,
                    "R3",
                    competing_targets=[sibling],
                )
                enhanced = retrieve_descriptions(
                    {"pages": [page(1, text)]},
                    target,
                    True,
                    "R3",
                    competing_targets=[sibling],
                    allow_discourse_scope=True,
                )
                baseline_method = next(
                    hit for hit in baseline["categories"]["method"] if "IRM" in hit["matched_text"]
                )
                enhanced_method = next(
                    hit for hit in enhanced["categories"]["method"] if "IRM" in hit["matched_text"]
                )

                self.assertFalse(baseline_method["local_target"])
                self.assertTrue(enhanced_method["local_target"])
                self.assertEqual(enhanced_method["target_anchor"], "target_same_discourse")
                self.assertEqual(enhanced_method["scope_kind"], "discourse")

    def test_discourse_scope_uses_strict_material_aliases(self):
        targets = (
            "Alpha Ridge Basalts, N directions",
            "Alpha Ridge Basalts normal component",
            "Alpha Ridge Basalts A magnetization",
            "Alpha Ridge Basalts Combined",
        )
        text = (
            "Alpha Ridge Basalts samples were selected. "
            "IRM acquisition curves were measured."
        )
        for target in targets:
            with self.subTest(target=target):
                baseline = retrieve_descriptions(
                    {"pages": [page(1, text)]},
                    target,
                    True,
                    "R3",
                    competing_targets=["Beta Plain Dykes"],
                )
                enhanced = retrieve_descriptions(
                    {"pages": [page(1, text)]},
                    target,
                    True,
                    "R3",
                    competing_targets=["Beta Plain Dykes"],
                    allow_discourse_scope=True,
                )
                baseline_method = next(
                    hit for hit in baseline["categories"]["method"] if "IRM" in hit["matched_text"]
                )
                enhanced_method = next(
                    hit for hit in enhanced["categories"]["method"] if "IRM" in hit["matched_text"]
                )

                self.assertFalse(baseline_method["local_target"])
                self.assertTrue(enhanced_method["local_target"])
                self.assertEqual(enhanced_method["target_anchor"], "target_same_discourse")

    def test_material_alias_rejects_conflicts_and_multi_material_targets(self):
        cases = (
            (
                "Alpha Ridge Basalts N directions",
                ["Alpha Ridge Basalts S directions"],
                "Alpha Ridge Basalts samples were selected. IRM curves were measured.",
                "direction_sibling",
            ),
            (
                "Alpha Ridge Basalts Combined",
                ["Alpha Ridge Basalts"],
                "Alpha Ridge Basalts samples were selected. IRM curves were measured.",
                "base_target_sibling",
            ),
            (
                "Alpha Ridge Basalts and Beta Plain Dykes Combined",
                ["Gamma Hills Sills"],
                "Alpha Ridge Basalts and Beta Plain Dykes samples were selected. "
                "IRM curves were measured.",
                "conjoined_materials",
            ),
            (
                "Alpha Ridge Basalts, Beta Plain Dykes Combined",
                ["Gamma Hills Sills"],
                "Alpha Ridge Basalts, Beta Plain Dykes samples were selected. "
                "IRM curves were measured.",
                "comma_material_list",
            ),
            (
                "Alpha Basalts Combined",
                ["Beta Plain Dykes"],
                "Alpha Basalts samples were selected. IRM curves were measured.",
                "one_specific_material_token",
            ),
        )
        for target, siblings, text, case in cases:
            with self.subTest(case=case):
                result = retrieve_descriptions(
                    {"pages": [page(1, text)]},
                    target,
                    True,
                    "R3",
                    competing_targets=siblings,
                    allow_discourse_scope=True,
                )
                method = next(
                    hit for hit in result["categories"]["method"] if "IRM" in hit["matched_text"]
                )

                self.assertFalse(method["local_target"])
                self.assertNotEqual(method["target_anchor"], "target_same_discourse")

    def test_exclusive_target_heading_binds_a_bounded_target_section(self):
        positive_text = (
            "Obuasi Greenstone\n"
            "IRM acquisition curves were measured. "
            "The curves do not saturate until 2 T.\n"
            "Kumasi Greenstone\n"
            "Thermal demagnetization was performed."
        )
        positive = retrieve_descriptions(
            {"pages": [page(1, positive_text)]},
            "Obuasi Greenstone",
            True,
            "R3",
            competing_targets=["Kumasi Greenstone"],
            allow_discourse_scope=True,
        )
        method = next(
            hit for hit in positive["categories"]["method"] if "IRM" in hit["matched_text"]
        )
        diagnostic = next(
            hit
            for hit in positive["categories"]["diagnostic"]
            if hit["subtype"] == "saturation_behavior"
        )

        self.assertTrue(method["local_target"])
        self.assertTrue(diagnostic["local_target"])
        self.assertEqual(method["scope_id"], diagnostic["scope_id"])
        self.assertEqual(method["target_anchor"], "target_same_sentence")
        self.assertEqual(diagnostic["target_anchor"], "target_same_section")
        self.assertEqual(method["scope_kind"], "target_section")

        third_sentence = retrieve_descriptions(
            {
                "pages": [
                    page(
                        1,
                        "Obuasi Greenstone\nContext was recorded. "
                        "Additional context was recorded. IRM acquisition curves were measured.",
                    )
                ]
            },
            "Obuasi Greenstone",
            True,
            "R3",
            competing_targets=["Kumasi Greenstone"],
            allow_discourse_scope=True,
        )
        late_method = next(
            hit
            for hit in third_sentence["categories"]["method"]
            if "IRM" in hit["matched_text"]
        )
        self.assertTrue(late_method["local_target"])
        self.assertEqual(late_method["target_anchor"], "target_same_section")

        too_late = retrieve_descriptions(
            {
                "pages": [
                    page(
                        1,
                        "Obuasi Greenstone\n"
                        + " ".join(
                            f"Context sentence {index} was recorded."
                            for index in range(1, 9)
                        )
                        + " IRM acquisition curves were measured.",
                    )
                ]
            },
            "Obuasi Greenstone",
            True,
            "R3",
            competing_targets=["Kumasi Greenstone"],
            allow_discourse_scope=True,
        )
        too_late_method = next(
            hit
            for hit in too_late["categories"]["method"]
            if "IRM" in hit["matched_text"]
        )
        self.assertFalse(too_late_method["local_target"])
        self.assertNotEqual(too_late_method["target_anchor"], "target_same_section")

    def test_variant_discourse_scope_rejects_unsafe_anchors(self):
        cases = (
            (
                "Group YA dykes",
                "Group YB dykes",
                "YA group dykes samples were selected. Group YB dykes samples were described. "
                "IRM acquisition curves were measured.",
                "sibling_collision",
            ),
            (
                "Obuasi Greenstone",
                "Kumasi Greenstone",
                "Obuasi samples were selected. IRM acquisition curves were measured.",
                "single_target_token",
            ),
            (
                "Obuasi Greenstone",
                "Kumasi Greenstone",
                "The Obuasi locality contains unrelated descriptive words that place the "
                "greenstone token beyond the local alias window. IRM curves were measured.",
                "nonlocal_token_pair",
            ),
            (
                "Obuasi Greenstone",
                "Kumasi Greenstone",
                "The greenstone body at Obuasi was sampled.\nOther Unit\n"
                "IRM acquisition curves were measured.",
                "cross_heading",
            ),
            (
                "Obuasi Greenstone",
                "Kumasi Greenstone",
                "References\nThe greenstone body at Obuasi was sampled. "
                "IRM acquisition curves were measured.",
                "references",
            ),
        )
        for target, sibling, text, case in cases:
            with self.subTest(case=case):
                result = retrieve_descriptions(
                    {"pages": [page(1, text)]},
                    target,
                    True,
                    "R3",
                    competing_targets=[sibling],
                    allow_discourse_scope=True,
                )
                method = next(
                    hit for hit in result["categories"]["method"] if "IRM" in hit["matched_text"]
                )
                self.assertFalse(method["local_target"])
                self.assertNotEqual(method["target_anchor"], "target_same_discourse")

    def test_generic_target_cannot_open_a_discourse_scope(self):
        result = retrieve_descriptions(
            {
                "pages": [
                    page(
                        1,
                        "Dykes samples were selected. IRM acquisition curves were measured.",
                    )
                ]
            },
            "Dykes",
            True,
            "R3",
            competing_targets=["Basalts"],
            allow_discourse_scope=True,
        )
        method = next(
            hit for hit in result["categories"]["method"] if "IRM" in hit["matched_text"]
        )

        self.assertFalse(method["local_target"])
        self.assertNotEqual(method["target_anchor"], "target_same_discourse")

    def test_forward_discourse_scope_stops_at_sibling_or_sentence_limit(self):
        cases = (
            (
                "Alpha Formation samples were measured using IRM curves. "
                "Beta Formation samples were selected. The curves did not saturate up to 1 T.",
                "sibling",
            ),
            (
                "Alpha Formation samples were selected for magnetic measurements. "
                "Context was recorded. Additional context was recorded. "
                "IRM curves did not saturate up to 1 T.",
                "sentence_limit",
            ),
        )
        for text, case in cases:
            with self.subTest(case=case):
                result = retrieve_descriptions(
                    {"pages": [page(1, text)]},
                    "Alpha Formation",
                    True,
                    "R3",
                    competing_targets=["Beta Formation"],
                    allow_discourse_scope=True,
                )
                diagnostic = next(
                    hit
                    for hit in result["categories"]["diagnostic"]
                    if hit["subtype"] == "saturation_behavior"
                )
                self.assertFalse(diagnostic["local_target"])
                self.assertNotEqual(diagnostic["target_anchor"], "target_same_discourse")

    def test_scientific_decimal_and_figure_abbreviation_do_not_hide_sibling(self):
        texts = (
            "Alpha Formation samples had IRM at 1.0 T whereas Beta Formation samples did not.",
            "Alpha Formation IRM is shown in Fig. 3 whereas Beta Formation has the result.",
            "Alpha Formation IRM follows Smith, D.J. whereas Beta Formation has the result.",
        )
        for text in texts:
            with self.subTest(text=text):
                result = retrieve_descriptions(
                    {"pages": [page(1, text)]},
                    "Alpha Formation",
                    True,
                    "R3",
                    competing_targets=["Beta Formation"],
                    allow_discourse_scope=True,
                )
                method = next(
                    hit
                    for hit in result["categories"]["method"]
                    if "IRM" in hit["matched_text"]
                )
                self.assertFalse(method["local_target"])
                self.assertIn("beta formation", method["scope_competing_targets"])

    def test_sentence_closure_accepts_terminal_quotation_mark(self):
        text = 'Alpha Formation samples were described as "stable." IRM was measured.'
        spans = retrieval_module._sentence_spans(text)

        self.assertEqual(len(spans), 2)
        self.assertTrue(retrieval_module._sentence_is_reliably_closed(text, spans[0]))

    def test_reference_hits_are_flagged_and_ranked_below_body_text(self):
        row = {
            "pages": [
                page(
                    1,
                    "Magnetic mineralogy\n"
                    "In the Delta Formation, IRM acquisition identifies hematite.\n"
                    "References\n"
                    "Delta Formation study: magnetite is the carrier of remanence (Smith, 1990).",
                )
            ]
        }

        result = retrieve_descriptions(row, "Delta Formation", False, "R3")
        reference_hits = [hit for hit in result["hits"] if hit["reference_noise"]]
        body_hits = [hit for hit in result["hits"] if not hit["reference_noise"]]

        self.assertTrue(reference_hits)
        self.assertTrue(body_hits)
        self.assertTrue(all(hit["section"] == "References" for hit in reference_hits))
        self.assertGreater(
            min(hit["target_relevance"] for hit in body_hits),
            max(hit["target_relevance"] for hit in reference_hits),
        )

    def test_doi_bibliography_ratio_title_is_not_a_diagnostic_result(self):
        row = {
            "pages": [
                page(
                    13,
                    "Alpha Formation results occur in the left column. Although References\n"
                    "previously discussed text is interleaved here. Dunlop, D.J., 2002. "
                    "Theory and application of the Day plot (Mrs/Ms versus Hcr/Hc), 1: "
                    "theoretical curves. J. Geophys. Res. 107 (B3), 2056. "
                    "https://doi .org /10 .1029 /2001JB000486.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Alpha Formation", False, "R3")
        self.assertFalse(
            any(
                hit["subtype"] == "hysteresis_ratio_result"
                for hit in result["categories"]["diagnostic"]
            )
        )
        day_plot = [
            hit
            for hit in result["categories"]["method"]
            if hit["subtype"] == "domain_state_experiment"
        ]
        self.assertTrue(day_plot)
        self.assertTrue(all(hit["reference_noise"] for hit in day_plot))

    def test_r3_chain_prefers_same_then_adjacent_pages(self):
        row = {
            "pages": [
                page(
                    1,
                    "Summary\nThe Epsilon Formation was studied by IRM acquisition experiments.",
                ),
                page(
                    2,
                    "Rock magnetism\nFor the Epsilon Formation, hematite is the predominant "
                    "carrier of remanence in the characteristic component.",
                    "Rock magnetism",
                ),
                page(
                    7,
                    "The Epsilon Formation also underwent thermomagnetic measurements.",
                ),
            ]
        }

        result = retrieve_descriptions(row, "Epsilon Formation", False, "R3")
        chain = result["best_chain"]

        self.assertTrue(chain["complete"])
        self.assertTrue(chain["connected"])
        self.assertEqual(chain["connection"], "adjacent_pages")
        self.assertEqual(chain["pages"], [1, 2])
        self.assertEqual(set(chain["hits"]), {"method", "carrier", "bridge"})

    def test_carried_out_is_not_a_carrier_bridge(self):
        row = {
            "pages": [
                page(
                    1,
                    "Methods\nThermal demagnetization of the Zeta Formation was carried "
                    "out to isolate the stable magnetization.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Zeta Formation", False, "R3")

        self.assertTrue(result["categories"]["method"])
        self.assertEqual(result["categories"]["bridge"], [])
        self.assertFalse(result["best_chain"]["complete"])

    def test_drm_instrument_model_is_not_an_origin_bridge(self):
        row = {
            "pages": [
                page(
                    1,
                    "Jan Lake Granite remanent magnetizations were measured with a "
                    "Canadian Thin Films Model DRM-420 cryogenic magnetometer.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Jan Lake Granite", False, "R3")

        origin_hits = [
            hit
            for hit in result["categories"]["bridge"]
            if hit["subtype"] == "origin_bridge"
        ]
        self.assertEqual(origin_hits, [])

    def test_primary_drm_remains_an_origin_bridge(self):
        row = {
            "pages": [
                page(
                    1,
                    "IRM measurements of the Lambda Formation show that hematite carries "
                    "a primary DRM.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Lambda Formation", False, "R3")

        self.assertIn(
            "origin_bridge",
            {hit["subtype"] for hit in result["categories"]["bridge"]},
        )

    def test_negated_carrier_mentions_are_not_positive_evidence(self):
        statements = (
            "Neither magnetite nor hematite was identified in these samples.",
            "No magnetite was detected in these samples.",
            "Hematite was not identified in these samples.",
            "Maghemite is not present in these samples.",
        )
        for statement in statements:
            with self.subTest(statement=statement):
                row = {
                    "pages": [
                        page(
                            1,
                            "The Theta Formation was examined with IRM. " + statement,
                        )
                    ]
                }

                result = retrieve_descriptions(row, "Theta Formation", False, "R3")

                self.assertTrue(result["categories"]["method"])
                self.assertEqual(result["categories"]["carrier"], [])
                self.assertFalse(result["best_chain"]["complete"])

    def test_positive_carrier_mention_remains_available(self):
        row = {
            "pages": [
                page(
                    1,
                    "IRM measurements of the Iota Formation identify magnetite as the "
                    "main carrier of characteristic remanence.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Iota Formation", False, "R3")

        self.assertTrue(result["categories"]["carrier"])
        self.assertTrue(result["best_chain"]["complete"])

    def test_embedded_r3_scores_are_not_retrieved_as_evidence(self):
        row = {
            "pages": [
                page(
                    2,
                    "Alpha Formation has insufficient carrier evidence (R3: 0). "
                    "Beta Formation has identified magnetic carriers (R3=1).",
                )
            ]
        }

        alpha = retrieve_descriptions(row, "Alpha Formation", True, "R3")
        beta = retrieve_descriptions(row, "Beta Formation", True, "R3")

        self.assertEqual(alpha["categories"]["verdict"], [])
        self.assertEqual(beta["categories"]["verdict"], [])

    def test_r5_returns_target_bound_risk_and_rescue(self):
        row = {
            "pages": [
                page(
                    3,
                    "Structural results\nThe Eta Formation was not structurally corrected. "
                    "Nevertheless, a positive fold test was obtained for the Eta Formation.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Eta Formation", False, "R5")

        self.assertTrue(result["categories"]["risk"])
        self.assertTrue(result["categories"]["rescue"])
        self.assertEqual(result["categories"]["risk"][0]["page"], 3)
        self.assertEqual(result["categories"]["rescue"][0]["page"], 3)
        self.assertIn("quote", result["categories"]["risk"][0])
        self.assertIn("section", result["categories"]["rescue"][0])

    def test_embedded_r5_scores_are_ignored_but_semantic_evidence_remains(self):
        row = {
            "pages": [
                page(
                    1,
                    "Jalokoski target\n(R5: 0). Possible tilting could not be ruled out.",
                ),
                page(
                    2,
                    "Control target\nR5=1; no tilting has occurred since intrusion.",
                ),
            ]
        }

        risk = retrieve_descriptions(row, "Jalokoski target", True, "R5")
        rescue = retrieve_descriptions(row, "Control target", True, "R5")

        self.assertTrue(
            any(hit["subtype"] == "tilting_not_excluded" for hit in risk["categories"]["risk"])
        )
        self.assertTrue(any(hit["subtype"] == "no_tilting" for hit in rescue["categories"]["rescue"]))
        self.assertFalse(
            any(hit["subtype"] == "direct_r5_verdict" for hit in risk["categories"]["risk"])
        )
        self.assertFalse(
            any(hit["subtype"] == "direct_r5_verdict" for hit in rescue["categories"]["rescue"])
        )

    def test_rotation_negation_is_rescue_not_risk(self):
        row = {
            "pages": [
                page(
                    1,
                    "Kappa Formation\nNo block rotation occurred; the observations argue "
                    "against vertical-axis rotation, and any rotation is unlikely.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Kappa Formation", False, "R5")

        self.assertTrue(result["categories"]["rescue"])
        self.assertTrue(
            all(hit["subtype"] != "rotation" for hit in result["categories"]["risk"])
        )
        self.assertTrue(
            any(hit["subtype"] == "rotation_negated" for hit in result["categories"]["rescue"])
        )

    def test_unicode_target_uses_nfkd_ascii_fallback(self):
        row = {
            "pages": [
                page(
                    4,
                    "Rock magnetism\nThermomagnetic analysis of the Sao Tome Formation "
                    "shows magnetite as the carrier of remanence.",
                )
            ]
        }

        result = retrieve_descriptions(
            row,
            "S\u00e3o Tom\u00e9 Formation",
            False,
            "R3",
        )

        self.assertEqual(result["target_anchor"]["status"], "exact")
        self.assertEqual(result["target_anchor"]["pages"], [4])
        self.assertTrue(result["categories"]["method"])
        self.assertTrue(result["categories"]["bridge"])

    def test_generic_target_is_explicitly_downweighted(self):
        row = {"pages": [page(1, "Basalt was tilt-corrected before averaging.")]}

        result = retrieve_descriptions(row, "Basalt", False, "R5")

        self.assertEqual(result["target_anchor"]["status"], "generic_only")
        self.assertTrue(result["target_anchor"]["common_target"])
        self.assertLess(result["hits"][0]["binding"], 0.7)

    def test_generic_plural_target_binds_singular_in_same_sentence(self):
        row = {
            "pages": [
                page(
                    1,
                    "The Eocene dyke formations were not tilt-corrected because their "
                    "positions are not known with certainty.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Dykes", True, "R5")
        risk = next(
            hit
            for hit in result["categories"]["risk"]
            if hit["subtype"] == "unknown_structural_attitude"
        )

        self.assertEqual(result["target_anchor"]["status"], "generic_only")
        self.assertTrue(risk["local_target"])

    def test_unperformed_contact_test_is_not_rescue(self):
        row = {
            "pages": [
                page(
                    1,
                    "Jalokoski Gabbro: it was not possible to perform baked contact tests.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Jalokoski Gabbro", False, "R5")

        self.assertFalse(
            any(hit["subtype"] == "field_test" for hit in result["categories"]["rescue"])
        )

    def test_post_kinematic_remanence_does_not_require_structural_correction(self):
        row = {
            "pages": [
                page(
                    1,
                    "Komati Formation: no structural correction has been made because "
                    "the characteristic direction is a post-kinematic remanence acquired "
                    "during unroofing and cooling.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Komati Formation", False, "R5")

        self.assertFalse(
            any(
                hit["subtype"] == "missing_structural_correction"
                for hit in result["categories"]["risk"]
            )
        )
        self.assertTrue(
            any(
                hit["subtype"] == "correction_not_needed"
                for hit in result["categories"]["rescue"]
            )
        )

    def test_post_folding_secondary_component_is_risk_not_rescue(self):
        row = {
            "pages": [
                page(
                    1,
                    "Omega Formation contains a post-folding secondary remanence component "
                    "that overprints the folded strata.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Omega Formation", False, "R5")

        self.assertTrue(
            any(
                hit["subtype"] == "post_folding_secondary"
                for hit in result["categories"]["risk"]
            )
        )
        self.assertFalse(
            any(
                hit["subtype"] == "correction_not_needed"
                for hit in result["categories"]["rescue"]
            )
        )

    def test_uncorrected_orthogonal_projection_caption_is_not_r5_risk(self):
        row = {
            "pages": [
                page(
                    3,
                    "Figure 4. Orthogonal projections (uncorrected for tilt) of alternating "
                    "field and thermal demagnetization vector endpoints for the Kidd Creek samples.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Kidd Creek", False, "R5")

        self.assertFalse(
            any(
                hit["subtype"] == "missing_structural_correction"
                for hit in result["categories"]["risk"]
            )
        )

        trailing_caption = {
            "pages": [
                page(
                    4,
                    "Directions (uncorrected for tilt) are shown as solid circles. "
                    "Figure 5. Equal-area plots of site-mean directions for Kidd Creek.",
                )
            ]
        }
        trailing_result = retrieve_descriptions(trailing_caption, "Kidd Creek", False, "R5")
        self.assertFalse(
            any(
                hit["subtype"] == "missing_structural_correction"
                for hit in trailing_result["categories"]["risk"]
            )
        )

    def test_negligible_tilt_and_undeformed_target_are_rescue_evidence(self):
        row = {
            "pages": [
                page(
                    2,
                    "The Pikes Peak Granite is undeformed. No structural correction was "
                    "applied because the maximum tilt is less than 5 degrees.",
                )
            ]
        }

        result = retrieve_descriptions(row, "Pikes Peak Granite", False, "R5")

        self.assertFalse(
            any(
                hit["subtype"] == "missing_structural_correction"
                for hit in result["categories"]["risk"]
            )
        )
        self.assertTrue(
            any(
                hit["subtype"] == "correction_not_needed"
                for hit in result["categories"]["rescue"]
            )
        )

        ocr_row = {
            "pages": [
                page(
                    3,
                    "No structural correction was applied to the Pikes Peak locality. "
                    "The maximum tilt correction that could be applied to all results "
                    "is less than5‡.",
                )
            ]
        }
        ocr_result = retrieve_descriptions(ocr_row, "Pikes Peak", False, "R5")
        self.assertFalse(
            any(
                hit["subtype"] == "missing_structural_correction"
                for hit in ocr_result["categories"]["risk"]
            )
        )


if __name__ == "__main__":
    unittest.main()
