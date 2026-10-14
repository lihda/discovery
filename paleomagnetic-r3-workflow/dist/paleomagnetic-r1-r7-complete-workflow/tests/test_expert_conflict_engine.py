import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.expert_conflict_engine import (
    DEFAULT_KNOWLEDGE_BASE,
    ExpertConflictEngine,
    KnowledgeBaseValidationError,
    load_knowledge_base,
    validate_knowledge_base,
)


def page(number, text):
    return {
        "page": number,
        "source": "synthetic_original_paper",
        "text": text,
        "section_hints": ["Results"],
    }


class ExpertConflictEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge = load_knowledge_base(DEFAULT_KNOWLEDGE_BASE)
        cls.engine = ExpertConflictEngine.from_path(DEFAULT_KNOWLEDGE_BASE)

    def test_default_knowledge_base_is_bilingual_and_label_free(self):
        self.assertFalse(self.knowledge["provenance"]["label_supervision"])
        self.assertFalse(self.knowledge["provenance"]["contains_row_answer_mapping"])
        self.assertEqual(
            {rule["indicator"] for rule in self.knowledge["rules"]},
            {"R3", "R5", "R7"},
        )
        self.assertTrue(
            all(rule["claim"]["en"] and rule["claim"]["zh"] for rule in self.knowledge["rules"])
        )
        serialized = json.dumps(self.knowledge, ensure_ascii=False).casefold()
        self.assertNotIn("resultno", serialized)
        self.assertNotIn("refno", serialized)

    def test_default_rules_define_semantic_target_binding_thresholds(self):
        expected = {
            "R3.METHOD.CARRIER_DIAGNOSTIC": 0.48,
            "R3.CARRIER.NAMED_MINERAL": 0.48,
            "R3.BRIDGE.CARRIER_REMANENCE": 0.48,
            "R3.CARRIER.EXPLICIT_ABSENCE": 0.78,
            "R5.TEMP.POST_FOLD_SECONDARY": 0.78,
            "R5.TEMP.POST_KINEMATIC_PRIMARY_INTRUSION": 0.70,
            "R5.RISK.ALLOCHTHONOUS_TERRANE": 0.78,
            "R5.RISK.NON_STRATIFIED_TARGET": 0.78,
            "R5.RISK.INTERNAL_VERTICAL_AXIS_ROTATION": 0.78,
            "R5.RISK.PALEOHORIZONTAL_UNKNOWN": 0.78,
            "R5.RISK.UNCORRECTED_DEFORMATION": 0.78,
            "R5.RISK.TILTING_NOT_EXCLUDED": 0.78,
            "R5.RISK.POST_ACQUISITION_ROTATION": 0.78,
            "R5.RISK.INCLINATION_FLATTENING_CORRECTION": 0.78,
            "R5.RESCUE.STRUCTURAL_CORRECTION": 0.70,
            "R5.RESCUE.UNDEFORMED_OR_NEGLIGIBLE_TILT": 0.70,
            "R5.RESCUE.POSITIVE_FIELD_TEST": 0.70,
            "R5.SUPPORT.RIGID_CRATON_SINCE_MAGNETIZATION": 0.70,
            "R5.EXCEPTION.FLATTENING_INTERCALATED_VOLCANIC": 0.78,
            "R5.EXCEPTION.FLATTENING_UNCORRECTED_SEDIMENTARY": 0.78,
            "R7.RISK.YOUNGER_POLE_MATCH": 0.82,
            "R7.RISK.REMAGNETIZATION": 0.82,
            "R7.FORMAL.COMPLETE_A95_NONOVERLAP": 0.86,
            "R7.FORMAL.FIELD_OLDER_THAN_YOUNGER": 0.86,
            "R7.SUPPORT.YOUNGER_POLE_MISMATCH": 0.82,
            "R7.SUPPORT.AGE_APPROPRIATE_APWP_MATCH": 0.82,
            "R7.RESCUE.PRIMARY_MAGNETIZATION": 0.72,
            "R7.RESCUE.PRIMARY_FIELD_TEST": 0.72,
        }
        actual = {
            rule["rule_id"]: rule["evidence"]["minimum_target_binding"]
            for rule in self.knowledge["rules"]
        }

        self.assertEqual(actual, expected)

    def test_r5_policy_gives_undefeated_conflicts_priority_over_support(self):
        policy = self.knowledge["indicators"]["R5"]["decision_policy"]
        self.assertTrue(policy["conflict_overrides_support"])
        self.assertEqual(
            policy["required_support_sets"],
            [["cratonic_rigidity"], ["temporal_rescue"]],
        )

    def test_r3_policy_requires_method_carrier_and_significance(self):
        policy = self.knowledge["indicators"]["R3"]["decision_policy"]
        self.assertEqual(
            policy["required_support_groups"],
            ["method", "carrier", "bridge"],
        )

    def test_r7_policy_requires_a_formal_decision_path(self):
        policy = self.knowledge["indicators"]["R7"]["decision_policy"]
        self.assertEqual(policy["required_support_groups"], ["formal_r7_decision"])
        self.assertTrue(policy["conflict_overrides_support"])

    def test_validation_rejects_identifier_and_label_leakage(self):
        labelled = copy.deepcopy(self.knowledge)
        labelled["provenance"]["label_supervision"] = True
        with self.assertRaisesRegex(KnowledgeBaseValidationError, "label_supervision"):
            validate_knowledge_base(labelled)

        answer_map = copy.deepcopy(self.knowledge)
        answer_map["rules"][0]["RESULTNO"] = {"123": 1}
        with self.assertRaisesRegex(KnowledgeBaseValidationError, "forbidden identifier"):
            validate_knowledge_base(answer_map)

        textual_identifier = copy.deepcopy(self.knowledge)
        textual_identifier["title"]["en"] = "Lookup by REFNO"
        with self.assertRaisesRegex(KnowledgeBaseValidationError, "identifier reference"):
            validate_knowledge_base(textual_identifier)

    def test_validation_rejects_invalid_target_binding_threshold(self):
        invalid_type = copy.deepcopy(self.knowledge)
        invalid_type["rules"][0]["evidence"]["minimum_target_binding"] = "0.48"
        with self.assertRaisesRegex(KnowledgeBaseValidationError, "must be numeric"):
            validate_knowledge_base(invalid_type)

        invalid_range = copy.deepcopy(self.knowledge)
        invalid_range["rules"][0]["evidence"]["minimum_target_binding"] = 1.01
        with self.assertRaisesRegex(KnowledgeBaseValidationError, r"must be in \[0, 1\]"):
            validate_knowledge_base(invalid_range)

        distance_without_exact = copy.deepcopy(self.knowledge)
        rule = distance_without_exact["rules"][0]
        rule["evidence"]["maximum_target_distance_chars"] = 100
        with self.assertRaisesRegex(
            KnowledgeBaseValidationError, "requires require_exact_target_in_claim"
        ):
            validate_knowledge_base(distance_without_exact)

    def test_validation_checks_defeasible_group_references_and_policy(self):
        missing_group = copy.deepcopy(self.knowledge)
        flattening = next(
            rule
            for rule in missing_group["rules"]
            if rule["rule_id"] == "R5.RISK.INCLINATION_FLATTENING_CORRECTION"
        )
        flattening["unless_support_groups"] = ["nonexistent_exception_group"]
        with self.assertRaisesRegex(
            KnowledgeBaseValidationError, "references missing SUPPORT groups"
        ):
            validate_knowledge_base(missing_group)

        invalid_policy = copy.deepcopy(self.knowledge)
        next(
            rule
            for rule in invalid_policy["rules"]
            if rule["rule_id"] == "R5.RISK.INCLINATION_FLATTENING_CORRECTION"
        )["unless_support_policy"] = "SOMETIMES"
        with self.assertRaisesRegex(KnowledgeBaseValidationError, "must be ANY or ALL"):
            validate_knowledge_base(invalid_policy)

        support_rule = copy.deepcopy(self.knowledge)
        next(
            rule
            for rule in support_rule["rules"]
            if rule["rule_id"] == "R5.EXCEPTION.FLATTENING_INTERCALATED_VOLCANIC"
        )["unless_support_groups"] = ["flattening_sedimentary_corroboration"]
        with self.assertRaisesRegex(KnowledgeBaseValidationError, "only valid for CONFLICT"):
            validate_knowledge_base(support_rule)

    def test_required_support_sets_are_backward_compatible_alternatives(self):
        knowledge = copy.deepcopy(self.knowledge)
        policy = knowledge["indicators"]["R3"]["decision_policy"]
        policy["required_support_groups"] = []
        policy["required_support_sets"] = [
            ["method", "carrier", "bridge"],
            ["method", "bridge"],
        ]
        engine = ExpertConflictEngine(knowledge)

        method_only = engine.evaluate(
            "R3",
            "Alpha Formation",
            retrieval={
                "hits": [
                    {
                        "indicator": "R3",
                        "category": "method",
                        "subtype": "rock_magnetic_experiment",
                        "quote": "IRM experiments were performed on the Alpha Formation.",
                        "binding": 1.0,
                    }
                ]
            },
        )
        self.assertEqual(method_only["relation"], "SUPPORT")
        self.assertEqual(method_only["decision"]["outcome"], "UNKNOWN")
        self.assertIsNone(method_only["decision"]["satisfied_support_set"])

        complete = engine.evaluate(
            "R3",
            "Alpha Formation",
            pages=[
                page(
                    3,
                    "IRM experiments on the Alpha Formation show that the characteristic "
                    "remanence is carried by magnetite.",
                )
            ],
        )
        self.assertEqual(complete["decision"]["outcome"], "POSITIVE")
        self.assertEqual(
            complete["decision"]["satisfied_support_set"],
            ["bridge", "carrier", "method"],
        )

    def test_validation_allows_review_only_unknown_with_null_value(self):
        review_rule = next(
            rule
            for rule in self.knowledge["rules"]
            if rule["rule_id"] == "R5.TEMP.POST_FOLD_SECONDARY"
        )
        self.assertEqual(review_rule["effect"], {"relation": "UNKNOWN", "value": None})
        validate_knowledge_base(self.knowledge)

        invalid_unknown = copy.deepcopy(self.knowledge)
        next(
            rule
            for rule in invalid_unknown["rules"]
            if rule["rule_id"] == "R5.TEMP.POST_FOLD_SECONDARY"
        )["effect"]["value"] = 0
        with self.assertRaisesRegex(KnowledgeBaseValidationError, "inconsistent"):
            validate_knowledge_base(invalid_unknown)

    def test_loading_rejects_leaky_artifact_on_disk(self):
        leaky = copy.deepcopy(self.knowledge)
        leaky["provenance"]["label_supervision"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaky.json"
            path.write_text(json.dumps(leaky), encoding="utf-8")
            with self.assertRaises(KnowledgeBaseValidationError):
                load_knowledge_base(path)

    def test_r3_method_retrieval_hit_is_support_with_provenance(self):
        retrieval = {
            "hits": [
                {
                    "indicator": "R3",
                    "category": "method",
                    "subtype": "rock_magnetic_experiment",
                    "matched_text": "IRM acquisition",
                    "quote": (
                        "IRM acquisition experiments were performed on specimens from "
                        "the Alpha Formation."
                    ),
                    "page": 4,
                    "section": "Rock magnetism",
                    "source": "page_cache",
                    "binding": 1.0,
                    "target_anchor": "target_same_sentence",
                    "reference_noise": False,
                }
            ]
        }

        result = self.engine.evaluate("R3", "Alpha Formation", retrieval=retrieval)

        self.assertEqual(result["relation"], "SUPPORT")
        self.assertIsNone(result["decision"]["recommended_value"])
        self.assertEqual(
            result["decision"]["missing_support_groups"],
            ["bridge", "carrier"],
        )
        method = next(
            relation
            for relation in result["relations"]
            if relation["rule_id"] == "R3.METHOD.CARRIER_DIAGNOSTIC"
        )
        self.assertEqual(method["relation"], "SUPPORT")
        self.assertEqual(method["evidence"]["page"], 4)
        self.assertEqual(len(method["evidence"]["quote_sha256"]), 64)
        self.assertEqual(len(method["evidence"]["page_text_sha256"]), 64)

    def test_r3_origin_word_without_carrier_significance_is_incomplete(self):
        result = self.engine.evaluate(
            "R3",
            "Alpha Formation",
            pages=[
                page(
                    4,
                    "IRM experiments were performed on the Alpha Formation. "
                    "A primary remanence was isolated, and magnetite occurs in the unit.",
                )
            ],
        )

        self.assertIsNone(result["decision"]["recommended_value"])
        self.assertIn("bridge", result["decision"]["missing_support_groups"])

    def test_complete_r3_method_and_carrier_supports_positive_decision(self):
        result = self.engine.evaluate(
            "R3",
            "Beta Formation",
            pages=[
                page(
                    2,
                    "IRM and hysteresis experiments on the Beta Formation identify "
                    "magnetite as the principal carrier of characteristic remanence.",
                )
            ],
        )

        self.assertEqual(result["relation"], "SUPPORT")
        self.assertEqual(result["decision"]["recommended_value"], 1)
        self.assertEqual(result["decision"]["outcome"], "POSITIVE")
        self.assertEqual(result["decision"]["missing_support_groups"], [])
        self.assertTrue({"method", "carrier", "bridge"}.issubset(result["decision"]["support_groups"]))

    def test_r5_post_folding_secondary_is_review_only_unknown(self):
        result = self.engine.evaluate(
            "R5",
            "Gamma Formation",
            pages=[
                page(
                    7,
                    "The Gamma Formation carries a secondary post-folding "
                    "magnetization component.",
                )
            ],
        )

        self.assertEqual(result["relation"], "UNKNOWN")
        self.assertIsNone(result["decision"]["recommended_value"])
        self.assertEqual(result["decision"]["outcome"], "UNKNOWN")
        self.assertNotIn("TEMPORAL_ORDER", result["conflict_types"])
        relation = next(
            item for item in result["relations"] if item["rule_id"] == "R5.TEMP.POST_FOLD_SECONDARY"
        )
        self.assertEqual(relation["relation"], "UNKNOWN")
        self.assertIsNone(relation["decision_value"])
        self.assertEqual(relation["claim"]["temporal"]["relation"], "AFTER")
        self.assertTrue(relation["evidence"]["claim_scope"]["passed"])
        self.assertIn("secondary post-folding", relation["evidence"]["quote"])

    def test_r5_post_kinematic_primary_intrusion_is_rescue(self):
        result = self.engine.evaluate(
            "R5",
            "Delta Intrusion",
            pages=[
                page(
                    3,
                    "The post-kinematic Delta Intrusion carries a primary TRM "
                    "acquired during cooling.",
                )
            ],
        )

        self.assertEqual(result["relation"], "SUPPORT")
        self.assertEqual(result["decision"]["recommended_value"], 1)
        self.assertNotIn("R5.TEMP.POST_FOLD_SECONDARY", {item["rule_id"] for item in result["relations"]})
        self.assertIn(
            "R5.TEMP.POST_KINEMATIC_PRIMARY_INTRUSION",
            {item["rule_id"] for item in result["relations"]},
        )

    def test_r5_post_deformation_intrusion_support_does_not_require_primary_trm_wording(self):
        result = self.engine.evaluate(
            "R5",
            "Delta Intrusion",
            pages=[
                page(
                    3,
                    "The Delta Intrusion was emplaced after the last deformational event.",
                )
            ],
        )

        self.assertEqual(result["relation"], "SUPPORT")
        self.assertEqual(result["decision"]["recommended_value"], 1)
        self.assertIn(
            "R5.TEMP.POST_KINEMATIC_PRIMARY_INTRUSION",
            {item["rule_id"] for item in result["relations"] if item["relation"] == "SUPPORT"},
        )

    def test_r5_allochthonous_and_parautochthonous_targets_conflict(self):
        cases = [
            "The Alpha Terrane is interpreted as allochthonous.",
            "The Alpha Terrane is considered parautochthonous.",
        ]
        for text in cases:
            with self.subTest(text=text):
                result = self.engine.evaluate(
                    "R5", "Alpha Terrane", pages=[page(4, text)]
                )
                self.assertEqual(result["relation"], "CONFLICT")
                self.assertEqual(result["decision"]["recommended_value"], 0)
                self.assertIn(
                    "R5.RISK.ALLOCHTHONOUS_TERRANE",
                    {
                        item["rule_id"]
                        for item in result["relations"]
                        if item["relation"] == "CONFLICT"
                    },
                )

        negated = self.engine.evaluate(
            "R5",
            "Alpha Terrane",
            pages=[page(4, "The Alpha Terrane is not allochthonous.")],
        )
        self.assertNotEqual(negated["relation"], "CONFLICT")

    def test_r5_hard_conflict_does_not_transfer_from_a_named_neighbor(self):
        result = self.engine.evaluate(
            "R5",
            "Tamworth Belt Ignimbrites, Rouchel Block",
            pages=[
                page(
                    4,
                    "Ignimbrites within the Rouchel, Gresford and Myall blocks occur "
                    "in the southern Tamworth Belt. The Hastings block is an "
                    "allochthonous part of the Tamworth Belt.",
                )
            ],
            multi_target=True,
        )

        self.assertNotEqual(result["relation"], "CONFLICT")
        relation = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R5.RISK.ALLOCHTHONOUS_TERRANE"
        )
        self.assertEqual(relation["relation"], "UNKNOWN")
        self.assertIn(
            relation["evidence"]["claim_scope"]["reason"],
            {"cross_sentence_match", "exact_target_not_in_same_claim"},
        )

    def test_r5_hard_conflict_requires_local_target_proximity(self):
        filler = "regional background context " * 12
        result = self.engine.evaluate(
            "R5",
            "Alpha Formation",
            pages=[
                page(
                    4,
                    "The Alpha Formation is described here, "
                    + filler
                    + "while the reported direction uses a flattening correction.",
                )
            ],
        )

        relation = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R5.RISK.INCLINATION_FLATTENING_CORRECTION"
        )
        self.assertEqual(relation["relation"], "UNKNOWN")
        self.assertEqual(
            relation["evidence"]["claim_scope"]["reason"],
            "target_too_far_from_rule_match",
        )

    def test_r5_undefeated_internal_rotation_overrides_ordinary_support(self):
        result = self.engine.evaluate(
            "R5",
            "Alpha Block",
            pages=[
                page(
                    5,
                    "The Alpha Block underwent internal vertical-axis rotation. "
                    "A positive fold test was obtained for the Alpha Block.",
                )
            ],
        )

        self.assertEqual(result["relation"], "CONFLICT")
        self.assertEqual(result["decision"]["recommended_value"], 0)
        self.assertEqual(result["evidence_status"], "BOTH")
        self.assertEqual(result["decision"]["policy_decision"], 0)
        self.assertEqual(result["rule_status_summary"]["active_conflict"], 1)
        self.assertGreaterEqual(result["rule_status_summary"]["active_support"], 1)
        self.assertTrue(result["decision"]["conflict_overrides_support"])
        self.assertIn(
            "R5.RISK.INTERNAL_VERTICAL_AXIS_ROTATION",
            {item["rule_id"] for item in result["relations"] if item["relation"] == "CONFLICT"},
        )
        self.assertIn(
            "R5.RESCUE.POSITIVE_FIELD_TEST",
            {item["rule_id"] for item in result["relations"] if item["relation"] == "SUPPORT"},
        )

    def test_r5_ordinary_structural_correction_does_not_replace_rigid_frame(self):
        result = self.engine.evaluate(
            "R5",
            "Alpha Formation",
            pages=[page(5, "The Alpha Formation direction was corrected for bedding tilt.")],
        )

        self.assertEqual(result["relation"], "SUPPORT")
        self.assertIsNone(result["decision"]["recommended_value"])
        self.assertEqual(result["decision"]["outcome"], "UNKNOWN")
        self.assertEqual(result["evidence_status"], "SUPPORTED_ONLY")
        self.assertEqual(result["decision"]["policy_decision"], "ABSTAIN")
        self.assertIsNone(result["decision"]["satisfied_support_set"])

    def test_r5_non_stratified_target_is_conflict_unless_intrusion_is_post_deformation(self):
        conflict = self.engine.evaluate(
            "R5",
            "Alpha Complex",
            pages=[page(5, "The Alpha Complex consists of non-stratified rocks.")],
        )
        self.assertEqual(conflict["relation"], "CONFLICT")
        self.assertEqual(conflict["decision"]["recommended_value"], 0)

        rescued = self.engine.evaluate(
            "R5",
            "Alpha Intrusion",
            pages=[
                page(
                    5,
                    "The non-stratified Alpha Intrusion was emplaced after the last "
                    "deformational event.",
                )
            ],
        )
        self.assertEqual(rescued["relation"], "SUPPORT")
        self.assertEqual(rescued["decision"]["recommended_value"], 1)
        risk = next(
            item
            for item in rescued["relations"]
            if item["rule_id"] == "R5.RISK.NON_STRATIFIED_TARGET"
        )
        support = next(
            item
            for item in rescued["relations"]
            if item["rule_id"] == "R5.TEMP.POST_KINEMATIC_PRIMARY_INTRUSION"
        )
        self.assertEqual(risk["relation"], "UNKNOWN")
        self.assertEqual(risk["defeasible"]["status"], "DEFEATED")
        self.assertEqual(risk["defeasible"]["matched_support_groups"], ["temporal_rescue"])
        self.assertEqual(
            risk["defeasible"]["exception_relations"][0]["relation_id"],
            support["relation_id"],
        )

    def test_r5_post_deformation_intrusion_defeats_formal_setting_conflicts(self):
        cases = (
            "The Alpha Intrusion lies in an allochthonous terrane and was emplaced "
            "after the last deformational event.",
            "The Alpha Intrusion underwent internal vertical-axis rotation before it was "
            "emplaced after the last deformational event.",
        )
        for text in cases:
            with self.subTest(text=text):
                result = self.engine.evaluate(
                    "R5", "Alpha Intrusion", pages=[page(5, text)]
                )
                self.assertEqual(result["decision"]["recommended_value"], 1)
                formal_risks = [
                    item
                    for item in result["relations"]
                    if item["rule_id"]
                    in {
                        "R5.RISK.ALLOCHTHONOUS_TERRANE",
                        "R5.RISK.INTERNAL_VERTICAL_AXIS_ROTATION",
                    }
                ]
                self.assertTrue(formal_risks)
                self.assertTrue(
                    all(item["defeasible"]["status"] == "DEFEATED" for item in formal_risks)
                )

    def test_r5_rigid_craton_since_magnetization_supports_criterion(self):
        result = self.engine.evaluate(
            "R5",
            "Alpha Block",
            pages=[
                page(
                    5,
                    "The Alpha Block remained a rigid part of the Laurentian craton "
                    "since its magnetization was acquired.",
                )
            ],
        )

        self.assertEqual(result["relation"], "SUPPORT")
        self.assertEqual(result["decision"]["recommended_value"], 1)
        relation = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R5.SUPPORT.RIGID_CRATON_SINCE_MAGNETIZATION"
        )
        self.assertEqual(relation["claim"]["temporal"]["relation"], "SINCE")

    def test_r5_flattening_correction_is_conflict_by_default(self):
        result = self.engine.evaluate(
            "R5",
            "Alpha Formation",
            pages=[
                page(
                    6,
                    "The Alpha Formation paleomagnetic direction was corrected for "
                    "inclination flattening.",
                )
            ],
        )

        self.assertEqual(result["relation"], "CONFLICT")
        self.assertEqual(result["decision"]["recommended_value"], 0)
        relation = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R5.RISK.INCLINATION_FLATTENING_CORRECTION"
        )
        self.assertEqual(relation["defeasible"]["status"], "ACTIVE")
        self.assertEqual(relation["defeasible"]["exception_relations"], [])

    def test_r5_flattening_volcanic_exception_defeats_conflict_with_lineage(self):
        result = self.engine.evaluate(
            "R5",
            "Alpha Formation",
            pages=[
                page(
                    6,
                    "The Alpha Formation paleomagnetic direction was corrected for "
                    "inclination flattening. Intercalated volcanic rocks in the Alpha "
                    "Formation have a similar R-value and corroborate the inclination. "
                    "The Alpha Formation remained a rigid part of the Laurentian craton "
                    "since its magnetization was acquired.",
                )
            ],
        )

        self.assertEqual(result["relation"], "SUPPORT")
        self.assertEqual(result["decision"]["recommended_value"], 1)
        risk = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R5.RISK.INCLINATION_FLATTENING_CORRECTION"
        )
        exception = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R5.EXCEPTION.FLATTENING_INTERCALATED_VOLCANIC"
        )
        self.assertEqual(risk["relation"], "UNKNOWN")
        self.assertEqual(risk["defeasible"]["status"], "DEFEATED")
        self.assertEqual(
            risk["defeasible"]["matched_support_groups"],
            ["flattening_volcanic_corroboration"],
        )
        self.assertEqual(
            risk["defeasible"]["exception_relations"][0]["relation_id"],
            exception["relation_id"],
        )
        self.assertEqual(exception["defeasible_effects"]["effect"], "DEFEATS")
        self.assertEqual(
            exception["defeasible_effects"]["relations"][0]["relation_id"],
            risk["relation_id"],
        )

    def test_r5_flattening_same_sequence_sedimentary_exception_defeats_conflict(self):
        result = self.engine.evaluate(
            "R5",
            "Alpha Formation",
            pages=[
                page(
                    6,
                    "The Alpha Formation direction uses a flattening correction. "
                    "Other sedimentary rocks within the same Alpha Formation sequence "
                    "do not require an inclination-flattening correction and corroborate "
                    "the inclination.",
                )
            ],
        )

        self.assertEqual(result["relation"], "SUPPORT")
        risk = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R5.RISK.INCLINATION_FLATTENING_CORRECTION"
        )
        self.assertEqual(risk["defeasible"]["status"], "DEFEATED")
        self.assertEqual(
            risk["defeasible"]["matched_support_groups"],
            ["flattening_sedimentary_corroboration"],
        )

    def test_r5_flattening_exception_is_narrow_and_target_bound(self):
        missing_r_value = self.engine.evaluate(
            "R5",
            "Alpha Formation",
            pages=[
                page(
                    7,
                    "The Alpha Formation direction uses a flattening correction. "
                    "Intercalated volcanic rocks occur in the Alpha Formation.",
                )
            ],
        )
        self.assertEqual(missing_r_value["relation"], "CONFLICT")

        off_target = self.engine.evaluate(
            "R5",
            "Alpha Formation",
            pages=[
                page(
                    7,
                    "The Alpha Formation direction uses a flattening correction. "
                    "Intercalated volcanic rocks in the Beta Formation have a similar "
                    "R-value and corroborate its inclination.",
                )
            ],
            multi_target=True,
        )
        self.assertEqual(off_target["relation"], "CONFLICT")

    def test_r5_flattening_any_and_all_exception_policies_are_generic(self):
        knowledge = copy.deepcopy(self.knowledge)
        flattening = next(
            rule
            for rule in knowledge["rules"]
            if rule["rule_id"] == "R5.RISK.INCLINATION_FLATTENING_CORRECTION"
        )
        flattening["unless_support_policy"] = "ALL"
        engine = ExpertConflictEngine(knowledge)

        volcanic_only = engine.evaluate(
            "R5",
            "Alpha Formation",
            pages=[
                page(
                    8,
                    "The Alpha Formation direction uses a flattening correction. "
                    "Intercalated volcanic rocks in the Alpha Formation have a similar "
                    "R-value and corroborate the inclination.",
                )
            ],
        )
        self.assertEqual(volcanic_only["relation"], "CONFLICT")

        both = engine.evaluate(
            "R5",
            "Alpha Formation",
            pages=[
                page(
                    8,
                    "The Alpha Formation direction uses a flattening correction. "
                    "Intercalated volcanic rocks in the Alpha Formation have a similar "
                    "R-value and corroborate the inclination. Other sedimentary rocks "
                    "within the same Alpha Formation sequence do not require an "
                    "inclination-flattening correction and corroborate the inclination.",
                )
            ],
        )
        self.assertEqual(both["relation"], "SUPPORT")
        risk = next(
            item
            for item in both["relations"]
            if item["rule_id"] == "R5.RISK.INCLINATION_FLATTENING_CORRECTION"
        )
        self.assertEqual(risk["defeasible"]["status"], "DEFEATED")
        self.assertEqual(
            set(risk["defeasible"]["matched_support_groups"]),
            {
                "flattening_volcanic_corroboration",
                "flattening_sedimentary_corroboration",
            },
        )

    def test_r5_temporal_review_does_not_conflict_with_primary_intrusion_support(self):
        result = self.engine.evaluate(
            "R5",
            "Epsilon Complex",
            pages=[
                page(
                    5,
                    "A secondary post-folding component occurs in the Epsilon Complex.\n\n"
                    "The post-kinematic Epsilon intrusion also preserves a primary TRM.",
                )
            ],
        )

        self.assertEqual(result["relation"], "SUPPORT")
        self.assertEqual(result["decision"]["recommended_value"], 1)
        self.assertEqual(result["decision"]["outcome"], "POSITIVE")
        review = next(
            item for item in result["relations"] if item["rule_id"] == "R5.TEMP.POST_FOLD_SECONDARY"
        )
        self.assertEqual(review["relation"], "UNKNOWN")
        self.assertEqual(result["evidence_mass"]["contradiction"], 0.0)
        self.assertAlmostEqual(sum(result["evidence_mass"].values()), 1.0, places=5)

    def test_r5_no_structural_correction_is_not_conflict_when_correction_is_unnecessary(self):
        cases = [
            "The undeformed Alpha Formation needed no structural correction.",
            (
                "No structural correction was applied to the Alpha Formation because "
                "it has no tectonic disturbance."
            ),
            (
                "No structural correction was applied to the Alpha Formation because "
                "the bedding tilt is 5 degrees."
            ),
            (
                "No structural correction was applied to the Alpha Formation because "
                "the bedding tilt is <10 degrees."
            ),
            (
                "No structural correction was applied to the Alpha Formation. "
                "Therefore, correction was unnecessary because bedding tilt is 5 degrees."
            ),
        ]
        for text in cases:
            with self.subTest(text=text):
                result = self.engine.evaluate(
                    "R5", "Alpha Formation", pages=[page(4, text)]
                )
                relation = next(
                    item
                    for item in result["relations"]
                    if item["rule_id"] == "R5.RISK.UNCORRECTED_DEFORMATION"
                )

                self.assertNotEqual(result["relation"], "CONFLICT")
                self.assertEqual(relation["relation"], "UNKNOWN")
                self.assertEqual(relation["conflict_type"], "MODALITY")
                self.assertEqual(
                    relation["evidence"]["claim_scope"]["reason"],
                    "structural_correction_unnecessary",
                )

    def test_r5_uncorrected_folded_target_remains_a_real_conflict(self):
        result = self.engine.evaluate(
            "R5",
            "Alpha Formation",
            pages=[
                page(
                    4,
                    "The folded Alpha Formation direction was reported without "
                    "structural correction.",
                )
            ],
        )

        self.assertEqual(result["relation"], "CONFLICT")
        self.assertEqual(result["decision"]["recommended_value"], 0)
        relation = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R5.RISK.UNCORRECTED_DEFORMATION"
        )
        self.assertEqual(relation["relation"], "CONFLICT")
        self.assertTrue(relation["evidence"]["claim_scope"]["passed"])

    def test_r5_horizontal_rescue_requires_geological_attitude(self):
        valid = self.engine.evaluate(
            "R5",
            "Alpha Formation",
            pages=[page(5, "The bedding of the Alpha Formation is nearly horizontal.")],
        )
        self.assertEqual(valid["relation"], "SUPPORT")
        self.assertIn(
            "R5.RESCUE.UNDEFORMED_OR_NEGLIGIBLE_TILT",
            {item["rule_id"] for item in valid["relations"] if item["relation"] == "SUPPORT"},
        )

        invalid_claims = [
            "The Alpha Formation magnetic direction is nearly horizontal.",
            "The Alpha Formation rotation axis is nearly horizontal.",
            "The Alpha Formation inclination is sub-horizontal.",
            "The Alpha Formation palaeopole is nearly horizontal.",
        ]
        for text in invalid_claims:
            with self.subTest(text=text):
                result = self.engine.evaluate(
                    "R5", "Alpha Formation", pages=[page(5, text)]
                )
                self.assertNotIn(
                    "R5.RESCUE.UNDEFORMED_OR_NEGLIGIBLE_TILT",
                    {
                        item["rule_id"]
                        for item in result["relations"]
                        if item["relation"] == "SUPPORT"
                    },
                )

    def test_r7_risk_patterns_cannot_join_claims_across_sentences(self):
        result = self.engine.evaluate(
            "R7",
            "Alpha Formation",
            pages=[
                page(
                    6,
                    "The Alpha Formation pole matches the reference pole. "
                    "The reference pole is younger.",
                )
            ],
        )

        self.assertEqual(result["relation"], "UNKNOWN")
        self.assertIsNone(result["decision"]["recommended_value"])
        relation = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R7.RISK.YOUNGER_POLE_MATCH"
        )
        self.assertEqual(relation["relation"], "UNKNOWN")
        self.assertEqual(relation["conflict_type"], "COMPONENT_SCOPE")
        self.assertEqual(
            relation["evidence"]["claim_scope"]["reason"], "cross_sentence_match"
        )

    def test_r7_younger_pole_mismatch_is_support_not_risk(self):
        claims = [
            "The Alpha Formation pole does not match the younger reference pole.",
            "The Alpha Formation pole does not resemble the younger reference pole.",
            "The Alpha Formation pole fails to coincide with the younger reference pole.",
            "The Alpha Formation pole is different from the younger reference pole.",
        ]
        for text in claims:
            with self.subTest(text=text):
                result = self.engine.evaluate(
                    "R7", "Alpha Formation", pages=[page(6, text)]
                )

                self.assertEqual(result["relation"], "SUPPORT")
                self.assertIsNone(result["decision"]["recommended_value"])
                self.assertIn(
                    "R7.SUPPORT.YOUNGER_POLE_MISMATCH",
                    {
                        item["rule_id"]
                        for item in result["relations"]
                        if item["relation"] == "SUPPORT"
                    },
                )
                self.assertNotIn(
                    "R7.RISK.YOUNGER_POLE_MATCH",
                    {
                        item["rule_id"]
                        for item in result["relations"]
                        if item["relation"] == "CONFLICT"
                    },
                )

    def test_r7_removed_or_absent_secondary_overprint_is_not_conflict(self):
        claims = [
            "The Alpha Formation has no secondary overprint.",
            "The secondary overprint of the Alpha Formation was removed by demagnetization.",
            "The Alpha Formation secondary overprint was isolated away during cleaning.",
        ]
        for text in claims:
            with self.subTest(text=text):
                result = self.engine.evaluate(
                    "R7", "Alpha Formation", pages=[page(6, text)]
                )

                self.assertNotEqual(result["relation"], "CONFLICT")
                self.assertNotIn(
                    "R7.RISK.REMAGNETIZATION",
                    {
                        item["rule_id"]
                        for item in result["relations"]
                        if item["relation"] == "CONFLICT"
                    },
                )

    def test_r7_unqualified_younger_pole_match_is_review_only(self):
        result = self.engine.evaluate(
            "R7",
            "Alpha Formation",
            pages=[page(6, "The Alpha Formation pole matches the younger reference pole.")],
        )

        self.assertEqual(result["relation"], "UNKNOWN")
        self.assertIsNone(result["decision"]["recommended_value"])
        relation = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R7.RISK.YOUNGER_POLE_MATCH"
        )
        self.assertEqual(relation["relation"], "UNKNOWN")

    def test_r7_explicit_younger_remagnetization_remains_conflict(self):
        result = self.engine.evaluate(
            "R7",
            "Alpha Formation",
            pages=[
                page(
                    6,
                    "The Alpha Formation component is interpreted as a younger remagnetization.",
                )
            ],
        )

        self.assertEqual(result["relation"], "CONFLICT")
        self.assertEqual(result["decision"]["recommended_value"], 0)

    def test_r7_uncertain_negated_or_removed_overprints_are_review_only(self):
        claims = (
            "The Alpha Formation component may be a secondary overprint.",
            "Either the Alpha Formation direction is primary or it records a later remagnetization.",
            "The Alpha Formation component cannot be explained as a secondary overprint.",
            "The Alpha Formation primary remanence was preserved from later overprints.",
        )
        for text in claims:
            with self.subTest(text=text):
                result = self.engine.evaluate(
                    "R7", "Alpha Formation", pages=[page(6, text)]
                )

                self.assertNotEqual(result["relation"], "CONFLICT")
                self.assertIsNone(result["decision"]["recommended_value"])

    def test_r7_age_appropriate_apwp_match_is_support(self):
        result = self.engine.evaluate(
            "R7",
            "Alpha Formation",
            pages=[
                page(
                    7,
                    "The Alpha Formation pole agrees with the coeval age-appropriate APWP.",
                )
            ],
        )

        self.assertEqual(result["relation"], "SUPPORT")
        self.assertIsNone(result["decision"]["recommended_value"])
        relation = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R7.SUPPORT.AGE_APPROPRIATE_APWP_MATCH"
        )
        self.assertEqual(relation["claim"]["temporal"]["relation"], "COEVAL")

    def test_r7_formal_complete_search_and_field_age_paths_support_decision(self):
        claims = (
            "A complete search for the Alpha Formation pole across all eligible R>=3 "
            "younger poles in the stable connected Laurentian craton found no A95 overlap.",
            "A positive fold test for the Alpha Formation proves that its magnetization "
            "predates the younger reference pole it resembles.",
        )
        for text in claims:
            with self.subTest(text=text):
                result = self.engine.evaluate(
                    "R7", "Alpha Formation", pages=[page(7, text)]
                )

                self.assertEqual(result["decision"]["recommended_value"], 1)
                self.assertEqual(result["decision"]["outcome"], "POSITIVE")
                self.assertIn(
                    "formal_r7_decision", result["decision"]["support_groups"]
                )

    def test_r7_high_risk_claim_requires_target_in_same_sentence(self):
        result = self.engine.evaluate(
            "R7",
            "Alpha Formation",
            pages=[
                page(
                    8,
                    "The Alpha Formation was sampled. This component is interpreted "
                    "as a younger remagnetization.",
                )
            ],
        )

        self.assertEqual(result["relation"], "UNKNOWN")
        relation = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R7.RISK.REMAGNETIZATION"
        )
        self.assertEqual(relation["relation"], "UNKNOWN")
        self.assertEqual(relation["conflict_type"], "TARGET_SCOPE")
        self.assertEqual(
            relation["evidence"]["claim_scope"]["reason"],
            "target_not_in_same_claim",
        )

    def test_r7_reference_section_evidence_remains_unknown(self):
        result = self.engine.evaluate(
            "R7",
            "Alpha Formation",
            pages=[
                {
                    **page(
                        12,
                        "The Alpha Formation pole matches the younger reference pole.",
                    ),
                    "section_hints": ["References"],
                }
            ],
        )

        self.assertEqual(result["relation"], "UNKNOWN")
        relation = next(
            item
            for item in result["relations"]
            if item["rule_id"] == "R7.RISK.YOUNGER_POLE_MATCH"
        )
        self.assertEqual(relation["relation"], "UNKNOWN")
        self.assertEqual(relation["conflict_type"], "PROVENANCE")
        self.assertEqual(
            relation["evidence"]["claim_scope"]["reason"], "reference_section"
        )

    def test_r7_binding_threshold_rejects_weak_and_retains_boundary_evidence(self):
        def evaluate(binding):
            return self.engine.evaluate(
                "R7",
                "Alpha Formation",
                retrieval={
                    "hits": [
                        {
                            "quote": (
                                "The Alpha Formation component is interpreted as a "
                                "younger remagnetization."
                            ),
                            "page": 8,
                            "section": "Discussion",
                            "source": "page_cache",
                            "binding": binding,
                            "target_anchor": "retrieval_target_score",
                        }
                    ]
                },
            )

        weak = evaluate(0.81)
        weak_relation = next(
            item for item in weak["relations"] if item["rule_id"] == "R7.RISK.REMAGNETIZATION"
        )
        self.assertEqual(weak["relation"], "UNKNOWN")
        self.assertEqual(weak_relation["relation"], "UNKNOWN")
        self.assertEqual(weak_relation["conflict_type"], "TARGET_SCOPE")

        boundary = evaluate(0.82)
        boundary_relation = next(
            item
            for item in boundary["relations"]
            if item["rule_id"] == "R7.RISK.REMAGNETIZATION"
        )
        self.assertEqual(boundary["relation"], "CONFLICT")
        self.assertEqual(boundary["decision"]["recommended_value"], 0)
        self.assertEqual(boundary_relation["relation"], "CONFLICT")

    def test_multi_target_page_does_not_transfer_risk_evidence_between_targets(self):
        cases = [
            (
                "R5",
                "The Beta Formation carries a secondary post-folding magnetization component.",
                "R5.TEMP.POST_FOLD_SECONDARY",
            ),
            (
                "R7",
                "The Beta Formation component is interpreted as a younger remagnetization.",
                "R7.RISK.REMAGNETIZATION",
            ),
        ]
        for indicator, off_target_evidence, rule_id in cases:
            with self.subTest(indicator=indicator):
                result = self.engine.evaluate(
                    indicator,
                    "Alpha Formation",
                    pages=[
                        page(
                            9,
                            "Samples from the Alpha Formation were also studied.\n\n"
                            + off_target_evidence,
                        )
                    ],
                    multi_target=True,
                )
                relation = next(item for item in result["relations"] if item["rule_id"] == rule_id)

                self.assertEqual(result["relation"], "UNKNOWN")
                self.assertIsNone(result["decision"]["recommended_value"])
                self.assertEqual(relation["relation"], "UNKNOWN")
                self.assertEqual(relation["conflict_type"], "TARGET_SCOPE")
                self.assertEqual(relation["evidence"]["target_binding"]["score"], 0.48)

    def test_missing_source_is_unknown(self):
        result = self.engine.evaluate("R7", "Zeta Formation", pages=[])

        self.assertFalse(result["source_available"])
        self.assertEqual(result["relation"], "UNKNOWN")
        self.assertIsNone(result["decision"]["recommended_value"])
        self.assertEqual(result["decision"]["outcome"], "UNKNOWN")
        self.assertEqual(result["evidence_status"], "NEITHER")
        self.assertEqual(result["decision"]["policy_decision"], "ABSTAIN")
        self.assertEqual(
            result["evidence_mass"],
            {"support": 0.0, "conflict": 0.0, "unknown": 1.0, "contradiction": 0.0},
        )
        self.assertEqual(result["relations"], [])


if __name__ == "__main__":
    unittest.main()
