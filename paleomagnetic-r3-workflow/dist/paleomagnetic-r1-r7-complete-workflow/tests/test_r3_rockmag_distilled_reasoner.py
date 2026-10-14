import json
from pathlib import Path
import unittest

from scripts.r3_rockmag_distilled_reasoner import (
    ARTIFACT_CLASS,
    R3RockmagArtifactValidationError,
    R3RockmagDistilledReasoner,
    R3RockmagRuntimeBoundaryError,
    reason_r3_rockmag_distilled,
    validate_r3_rockmag_artifact,
)
from scripts.paper_text_retrieval import retrieve_descriptions


def concept(concept_id, *patterns):
    return {
        "concept_id": concept_id,
        "runtime_original_text_patterns": list(patterns),
        "training_statistics": {
            "paper_deduplicated": {
                "observations": 10,
                "positive": 10,
                "negative": 0,
            }
        },
    }


def artifact():
    return {
        "artifact_class": ARTIFACT_CLASS,
        "schema_version": "r3_rockmag_text_distillation_v1",
        "profile": "rockmag_text_distilled_v1",
        "provenance": {
            "label_supervision": True,
            "phase": "offline_training_only",
            "raw_field_values_retained": False,
            "row_identity_mapping_retained": False,
            "source_paper_quotes_retained": False,
            "supervision_scope": "pre_registered_training_partition_only",
        },
        "runtime_policy": {
            "allowed_scoring_sources": [
                "identity manifest",
                "original paper page text",
                "this frozen aggregate artifact",
            ],
            "contains_record_answer_mapping": False,
            "previous_prediction_access": False,
            "source_metadata_table_access": False,
            "supervised_field_access": False,
        },
        "split": {
            "held_out_supervision_accessed": False,
            "unit": "paper_identity",
        },
        "method_taxonomy": [
            concept("irm_acquisition", r"\b(?:IRM|SIRM)\b", r"IRM\s+acquisition\s+curves?"),
            concept("thermomagnetic_curve", r"thermo[- ]?magnetic", r"\bJ(?:s|rs)[- ]?T\b"),
            concept(
                "temperature_dependent_susceptibility",
                r"susceptibility.{0,80}(?:heating|cooling|temperature)",
            ),
            concept("hysteresis_forc", r"hysteresis\s+(?:loops?|properties)", r"\bFORC\b"),
            concept(
                "petrographic_microscopy",
                r"reflected[- ]light\s+microscop",
                r"polished\s+thin\s+sections?",
            ),
            concept("mineralogical_phase_analysis", r"X[- ]?ray\s+diffraction"),
        ],
        "carrier_taxonomy": [
            concept("magnetite", r"\bmagnetite\b"),
            concept("hematite", r"\bha?ematite\b"),
        ],
        "diagnostic_taxonomy": [
            concept(
                "saturation_behavior",
                r"(?:does\s+not|did\s+not|failed\s+to)\s+saturate",
                r"saturat(?:e|ed|ion).{0,50}(?:at|by)\s*[0-9]",
            ),
            concept("curie_temperature", r"Curie\s+(?:point|temperature)"),
            concept("domain_state", r"(?:single|pseudo[- ]single|multi)[- ]domain"),
        ],
        "decision_policy": {
            "always_required": ["original-paper evidence", "target binding"],
            "missing_or_ambiguous_source": "ABSTAIN",
            "positive_requires_any": [
                "target-bound performed method plus diagnostic result",
                "target-bound performed method plus named carrier identification",
            ],
            "standalone_non_support": [
                "component unblocking-temperature interval",
                "bare susceptibility code",
                "bare anisotropy code",
                "bare carrier name",
            ],
        },
        "standalone_exclusions": [
            {
                "exclusion_id": "demagnetization_unblocking_interval",
                "field_pattern": r"\bT(?:ub|b)\s*(?:=|>|<|:)",
                "policy": "Not independently sufficient.",
            }
        ],
    }


def hit(
    category,
    subtype,
    quote,
    *,
    page=4,
    relevance=0.9,
    local_target=True,
    section="Results",
    reference_noise=False,
    **extra,
):
    return {
        "category": category,
        "subtype": subtype,
        "quote": quote,
        "matched_text": quote,
        "page": page,
        "target_relevance": relevance,
        "local_target": local_target,
        "target_anchor": "target_same_sentence" if local_target else "same_target_page",
        "section": section,
        "reference_noise": reference_noise,
        **extra,
    }


def retrieval(*hits, source_available=True):
    categories = {category: [] for category in ("method", "carrier", "bridge", "diagnostic")}
    for member in hits:
        categories.setdefault(member["category"], []).append(member)
    return {
        "target": "Alpha Formation",
        "indicator": "R3",
        "multi_target": False,
        "source_available": source_available,
        "target_anchor": {"status": "exact", "strength": "strong"},
        "hits": list(hits),
        "categories": categories,
        "best_chain": {"hits": {}},
        "provenance": "original paper page cache only",
    }


class R3RockmagArtifactTests(unittest.TestCase):
    def test_valid_supervised_aggregate_artifact_is_accepted(self):
        validated = validate_r3_rockmag_artifact(artifact())

        self.assertEqual(validated["artifact_class"], ARTIFACT_CLASS)
        self.assertTrue(validated["provenance"]["label_supervision"])
        self.assertFalse(validated["runtime_policy"]["supervised_field_access"])

    def test_generated_workspace_artifact_is_directly_compatible(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "data/calibration/r3_rockmag_text_distillation_v1.json"
        )
        generated = json.loads(path.read_text(encoding="utf-8"))

        validated = validate_r3_rockmag_artifact(generated)

        self.assertEqual(validated["artifact_class"], ARTIFACT_CLASS)
        self.assertGreater(len(validated["method_taxonomy"]), 5)

    def test_wrong_artifact_class_or_label_contract_is_rejected(self):
        wrong_class = artifact()
        wrong_class["artifact_class"] = "ordinary_model"
        with self.assertRaisesRegex(R3RockmagArtifactValidationError, "artifact_class"):
            validate_r3_rockmag_artifact(wrong_class)

        unsupervised = artifact()
        unsupervised["provenance"]["label_supervision"] = False
        with self.assertRaisesRegex(R3RockmagArtifactValidationError, "label_supervision"):
            validate_r3_rockmag_artifact(unsupervised)

    def test_runtime_table_or_label_access_is_rejected(self):
        source_table = artifact()
        source_table["runtime_policy"]["source_metadata_table_access"] = True
        with self.assertRaisesRegex(R3RockmagArtifactValidationError, "metadata_table"):
            validate_r3_rockmag_artifact(source_table)

        labels = artifact()
        labels["runtime_policy"]["supervised_field_access"] = True
        with self.assertRaisesRegex(R3RockmagArtifactValidationError, "supervised_field"):
            validate_r3_rockmag_artifact(labels)

    def test_record_answer_mapping_is_rejected(self):
        declared = artifact()
        declared["runtime_policy"]["contains_record_answer_mapping"] = True
        with self.assertRaisesRegex(R3RockmagArtifactValidationError, "must be false"):
            validate_r3_rockmag_artifact(declared)

        embedded = artifact()
        embedded["RESULTNO"] = {"4": 1}
        with self.assertRaisesRegex(R3RockmagArtifactValidationError, "record identifier"):
            validate_r3_rockmag_artifact(embedded)

    def test_invalid_runtime_taxonomy_regex_is_rejected(self):
        invalid = artifact()
        invalid["method_taxonomy"][0]["runtime_original_text_patterns"] = ["("]
        with self.assertRaisesRegex(R3RockmagArtifactValidationError, "invalid regex"):
            validate_r3_rockmag_artifact(invalid)

    def test_invalid_proxy_policy_or_identity_granularity_is_rejected(self):
        invalid_proxy = artifact()
        invalid_proxy["decision_policy"]["rockmag_proxy_policy"] = "method_name_only"
        with self.assertRaisesRegex(R3RockmagArtifactValidationError, "rockmag_proxy_policy"):
            validate_r3_rockmag_artifact(invalid_proxy)

        invalid_identity = artifact()
        invalid_identity["decision_policy"]["identity_granularity"] = "resultno_lookup"
        with self.assertRaisesRegex(R3RockmagArtifactValidationError, "identity_granularity"):
            validate_r3_rockmag_artifact(invalid_identity)


class R3RockmagDistilledDecisionTests(unittest.TestCase):
    def setUp(self):
        self.reasoner = R3RockmagDistilledReasoner(artifact())

    def test_method_and_same_page_diagnostic_result_trigger_positive(self):
        method = hit(
            "method",
            "rock_magnetic_experiment",
            "IRM acquisition curves for the Alpha Formation show no saturation up to 1 T.",
        )

        result = self.reasoner.evaluate(retrieval(method))

        self.assertEqual(result["policy_decision"], 1)
        self.assertEqual(result["proxy_decision"], 1)
        self.assertEqual(result["decision_path"], "ACCEPTED_METHOD_WITH_DIAGNOSTIC_RESULT")
        self.assertEqual(result["conditions"]["selected_page_span"], 0)
        self.assertEqual(result["rockmag_proxy_r3"]["policy_decision"], 1)
        self.assertEqual(result["normative_r3"]["policy_decision"], "ABSTAIN")
        self.assertIsNone(result["normative_r3"]["value"])
        self.assertEqual(result["normative_r3"]["outcome"], "UNKNOWN")
        self.assertTrue(result["criterion_practice_conflict"])
        self.assertTrue(result["manual_review"])
        self.assertEqual(
            result["criterion_practice_comparison"]["status"],
            "INCOMPLETE_NORMATIVE_CHAIN",
        )

    def test_same_claim_methodology_cannot_serve_as_its_own_diagnostic(self):
        methodology = hit(
            "method",
            "domain_state_experiment",
            "Hysteresis properties were plotted to identify pseudo-single-domain behavior.",
        )

        result = self.reasoner.evaluate(retrieval(methodology))

        self.assertEqual(result["policy_decision"], 0)
        self.assertTrue(result["conditions"]["accepted_method"])
        self.assertTrue(result["conditions"]["diagnostic_result"])
        self.assertFalse(result["conditions"]["connected_positive_path"])

    def test_distilled_proxy_accepts_performed_method_without_formal_chain(self):
        configured = artifact()
        configured["decision_policy"]["rockmag_proxy_policy"] = (
            "distilled_method_or_self_authenticating_result"
        )
        reasoner = R3RockmagDistilledReasoner(configured)
        method = hit(
            "method",
            "thermal_magnetic_experiment",
            "Thermomagnetic curves were measured to estimate the Curie temperature.",
        )

        result = reasoner.evaluate(retrieval(method))

        self.assertEqual(result["policy_decision"], 1)
        self.assertEqual(result["decision_path"], "ACCEPTED_DISTILLED_ROCKMAG_METHOD")
        self.assertEqual(result["selected_evidence"]["proxy"]["category"], "method")
        self.assertEqual(result["normative_r3"]["policy_decision"], "ABSTAIN")
        self.assertTrue(result["criterion_practice_conflict"])

    def test_distilled_proxy_accepts_standalone_result_that_entails_method(self):
        configured = artifact()
        configured["decision_policy"]["rockmag_proxy_policy"] = (
            "distilled_method_or_self_authenticating_result"
        )
        reasoner = R3RockmagDistilledReasoner(configured)
        diagnostic = hit(
            "diagnostic",
            "hysteresis_ratio_result",
            "For Alpha Formation samples, Hcr/Hc=2.79.",
        )

        result = reasoner.evaluate(retrieval(diagnostic))

        self.assertEqual(result["policy_decision"], 1)
        self.assertEqual(
            result["decision_path"],
            "ACCEPTED_SELF_AUTHENTICATING_DIAGNOSTIC_RESULT",
        )
        self.assertEqual(result["selected_evidence"]["proxy"]["category"], "diagnostic")

    def test_canonical_target_identity_policy_does_not_use_opaque_resultno(self):
        configured = artifact()
        configured["decision_policy"]["identity_granularity"] = "canonical_target"
        reasoner = R3RockmagDistilledReasoner(configured)
        strong = hit(
            "method",
            "rock_magnetic_experiment",
            "IRM curves show no saturation up to 1 T.",
        )

        result = reasoner.evaluate(retrieval(strong), identity_ambiguous=True)

        self.assertEqual(result["policy_decision"], 1)
        self.assertTrue(result["conditions"]["identity_clear"])
        self.assertTrue(result["conditions"]["result_component_identity_ambiguous"])
        self.assertEqual(result["conditions"]["identity_granularity"], "canonical_target")
        self.assertTrue(result["manual_review"])

    def test_distinct_method_and_diagnostic_claims_can_pair(self):
        method = hit(
            "method",
            "domain_state_experiment",
            "Hysteresis properties were measured for the Alpha Formation.",
        )
        diagnostic = hit(
            "diagnostic",
            "domain_state",
            "The grains have pseudo-single-domain behavior.",
        )

        result = self.reasoner.evaluate(retrieval(method, diagnostic))

        self.assertEqual(result["policy_decision"], 1)
        self.assertEqual(
            result["selected_evidence"]["companion"]["category"],
            "diagnostic",
        )

    def test_quantified_curie_claim_can_self_authenticate(self):
        quantified = hit(
            "method",
            "thermal_magnetic_experiment",
            "Thermomagnetic curves show a Curie temperature of about 580 C.",
        )
        method_only = hit(
            "method",
            "thermal_magnetic_experiment",
            "Thermomagnetic curves were measured to estimate the Curie temperature.",
        )
        numeric_method_only = hit(
            "method",
            "thermal_magnetic_experiment",
            "Thermomagnetic curves were heated to 700 C to estimate the Curie temperature.",
        )

        positive = self.reasoner.evaluate(retrieval(quantified))
        negative = self.reasoner.evaluate(retrieval(method_only))
        numeric_negative = self.reasoner.evaluate(retrieval(numeric_method_only))

        self.assertEqual(positive["policy_decision"], 1)
        self.assertEqual(negative["policy_decision"], 0)
        self.assertFalse(negative["conditions"]["connected_positive_path"])
        self.assertEqual(numeric_negative["policy_decision"], 0)
        self.assertFalse(numeric_negative["conditions"]["connected_positive_path"])

    def test_saturation_without_field_endpoint_cannot_self_authenticate(self):
        unquantified = hit(
            "method",
            "rock_magnetic_experiment",
            "IRM acquisition curves for the Alpha Formation show no saturation.",
        )
        applied_field_only = hit(
            "method",
            "rock_magnetic_experiment",
            "IRM curves show no saturation; a 1 T field was used for a separate calibration.",
        )

        result = self.reasoner.evaluate(retrieval(unquantified))
        applied_field_result = self.reasoner.evaluate(retrieval(applied_field_only))

        self.assertEqual(result["policy_decision"], 0)
        self.assertTrue(result["conditions"]["diagnostic_result"])
        self.assertFalse(result["conditions"]["connected_positive_path"])
        self.assertEqual(applied_field_result["policy_decision"], 0)
        self.assertFalse(
            applied_field_result["conditions"]["connected_positive_path"]
        )

    def test_do_not_saturate_until_field_endpoint_can_self_authenticate(self):
        endpoint = hit(
            "method",
            "rock_magnetic_experiment",
            "Alpha Formation IRM acquisition curves do not saturate until 2 T.",
        )

        result = self.reasoner.evaluate(retrieval(endpoint))

        self.assertEqual(result["policy_decision"], 1)
        self.assertEqual(
            result["decision_path"],
            "ACCEPTED_METHOD_WITH_DIAGNOSTIC_RESULT",
        )

    def test_quantified_hysteresis_and_domain_claims_can_self_authenticate(self):
        claims = (
            "Hysteresis properties give Hcr/Hc values between 2.5 and 4.",
            "Hysteresis properties show that most samples fall in the "
            "pseudo-single-domain region.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                result = self.reasoner.evaluate(
                    retrieval(hit("method", "domain_state_experiment", claim))
                )
                self.assertEqual(result["policy_decision"], 1)

    def test_independent_diagnostic_hit_is_preferred_to_method_copy(self):
        claim = "Hysteresis properties give Hcr/Hc=2.79 for the Alpha Formation."
        method = hit(
            "method",
            "domain_state_experiment",
            claim,
            relevance=0.99,
            claim_text=claim,
        )
        diagnostic = hit(
            "diagnostic",
            "hysteresis_ratio_result",
            "Hcr/Hc=2.79",
            relevance=0.61,
            claim_text=claim,
        )

        result = self.reasoner.evaluate(retrieval(method, diagnostic))

        self.assertEqual(result["policy_decision"], 1)
        self.assertEqual(result["selected_evidence"]["method"]["category"], "method")
        self.assertEqual(
            result["selected_evidence"]["companion"]["category"],
            "diagnostic",
        )

    def test_retrieved_adjacent_sentence_diagnostic_forms_explicit_pair(self):
        cache_row = {
            "source": {"source_quality_flags": []},
            "pages": [
                {
                    "page": 2,
                    "source": "synthetic",
                    "section_hints": ["Rock magnetism"],
                    "text": (
                        "Alpha Formation IRM acquisition curves were measured. "
                        "The curves did not saturate up to 1 T."
                    ),
                }
            ],
        }
        source = retrieve_descriptions(cache_row, "Alpha Formation", False, "R3")

        result = self.reasoner.evaluate(source)

        self.assertEqual(result["policy_decision"], 1)
        self.assertEqual(result["decision_path"], "ACCEPTED_METHOD_WITH_DIAGNOSTIC_RESULT")
        self.assertEqual(result["selected_evidence"]["companion"]["category"], "diagnostic")
        self.assertEqual(
            result["selected_evidence"]["companion"]["subtype"],
            "saturation_behavior",
        )

    def test_unperformed_thermomagnetic_method_and_bare_curie_property_do_not_trigger(self):
        cache_row = {
            "source": {"source_quality_flags": []},
            "pages": [
                {
                    "page": 3,
                    "source": "synthetic",
                    "section_hints": ["Rock magnetism"],
                    "text": (
                        "It was impossible to perform thermomagnetic analysis on the Alpha "
                        "Formation samples. The NRM was acquired while cooling from a "
                        "temperature exceeding the magnetite Curie point."
                    ),
                }
            ],
        }
        source = retrieve_descriptions(cache_row, "Alpha Formation", False, "R3")

        result = self.reasoner.evaluate(source)

        self.assertEqual(source["categories"]["diagnostic"], [])
        self.assertEqual(result["policy_decision"], 0)
        self.assertEqual(result["evidence_counts"]["accepted_methods"], 0)

    def test_method_and_adjacent_page_carrier_trigger_positive(self):
        method = hit(
            "method",
            "thermal_magnetic_experiment",
            "Thermomagnetic Js-T curves were measured for the Alpha Formation.",
            page=6,
        )
        carrier = hit(
            "carrier",
            "named_carrier",
            "The curves identify hematite with minor magnetite.",
            page=7,
        )

        result = reason_r3_rockmag_distilled(
            retrieval(method, carrier),
            source_flags=[],
            multi_target=False,
            identity_ambiguous=False,
            artifact=artifact(),
        )

        self.assertEqual(result["policy_decision"], 1)
        self.assertEqual(result["decision_path"], "ACCEPTED_METHOD_WITH_CARRIER_IDENTIFICATION")
        self.assertEqual(result["conditions"]["selected_page_span"], 1)

    def test_connected_method_carrier_significance_chain_sets_both_tracks_positive(self):
        method = hit(
            "method",
            "rock_magnetic_experiment",
            "IRM acquisition curves were measured for the Alpha Formation.",
            page=3,
        )
        carrier = hit(
            "carrier",
            "named_carrier",
            "The measurements identify magnetite.",
            page=3,
        )
        bridge = hit(
            "bridge",
            "mineral_remanence_bridge",
            "The characteristic remanence is carried by magnetite.",
            page=3,
        )

        result = self.reasoner.evaluate(retrieval(method, carrier, bridge))

        self.assertEqual(result["normative_r3"]["policy_decision"], 1)
        self.assertEqual(result["normative_r3"]["value"], 1)
        self.assertEqual(result["normative_r3"]["binary_value"], 1)
        self.assertEqual(result["normative_r3"]["outcome"], "POSITIVE")
        self.assertEqual(result["rockmag_proxy_r3"]["policy_decision"], 1)
        self.assertFalse(result["criterion_practice_conflict"])
        self.assertEqual(result["criterion_practice_comparison"]["status"], "AGREE")
        self.assertFalse(result["manual_review"])

    def test_disconnected_method_and_carrier_do_not_trigger(self):
        method = hit(
            "method",
            "thermal_magnetic_experiment",
            "Thermomagnetic curves were measured.",
            page=2,
        )
        carrier = hit(
            "carrier",
            "named_carrier",
            "Magnetite is the carrier.",
            page=4,
        )

        result = self.reasoner.evaluate(retrieval(method, carrier))

        self.assertEqual(result["policy_decision"], 0)
        self.assertFalse(result["conditions"]["connected_positive_path"])

    def test_microscopy_and_alteration_trigger_positive(self):
        microscopy = hit(
            "method",
            "mineralogical_observation",
            "Reflected-light microscopy of Alpha Formation polished sections was performed.",
            page=8,
        )
        alteration = hit(
            "diagnostic",
            "mineral_alteration",
            "The opaque grains show oxidation and exsolution textures.",
            page=8,
        )

        result = self.reasoner.evaluate(retrieval(microscopy, alteration))

        self.assertEqual(result["policy_decision"], 1)
        self.assertEqual(result["decision_path"], "MICROSCOPY_WITH_CARRIER_OR_ALTERATION")

    def test_demagnetization_or_tub_only_cannot_trigger(self):
        demag = hit(
            "method",
            "demagnetization",
            "Thermal demagnetization gives Tub=150-450 C for N and above 600 C for R.",
        )
        carrier = hit("carrier", "named_carrier", "Hematite is present.")

        result = self.reasoner.evaluate(retrieval(demag, carrier))

        self.assertEqual(result["policy_decision"], 0)
        self.assertEqual(result["evidence_counts"]["accepted_methods"], 0)
        self.assertEqual(result["evidence_counts"]["weak_only"], 1)

    def test_bare_susceptibility_anisotropy_and_mineral_are_insufficient(self):
        for subtype, quote in (
            ("bare_susceptibility", "Magnetic susceptibility was 0.03 SI."),
            ("bare_anisotropy", "AN was small."),
        ):
            with self.subTest(subtype=subtype):
                weak = hit("method", subtype, quote)
                mineral = hit("carrier", "named_carrier", "Magnetite.")
                result = self.reasoner.evaluate(retrieval(weak, mineral))
                self.assertEqual(result["policy_decision"], 0)
                self.assertEqual(result["evidence_counts"]["accepted_methods"], 0)

        mineral_only = self.reasoner.evaluate(
            retrieval(hit("carrier", "named_carrier", "Magnetite and hematite."))
        )
        self.assertEqual(mineral_only["policy_decision"], 0)

    def test_reference_not_executed_and_negated_claims_do_not_trigger(self):
        reference = hit(
            "method",
            "rock_magnetic_experiment",
            "Smith et al. used IRM acquisition curves with no saturation.",
            section="References",
            reference_noise=True,
        )
        not_executed = hit(
            "method",
            "rock_magnetic_experiment",
            "IRM experiments were not performed for the Alpha Formation.",
            page=5,
        )
        negated_carrier = hit(
            "carrier",
            "named_carrier",
            "No magnetite was identified in the Alpha Formation.",
            page=5,
        )

        result = self.reasoner.evaluate(retrieval(reference, not_executed, negated_carrier))

        self.assertEqual(result["policy_decision"], 0)
        self.assertEqual(result["evidence_counts"]["accepted_methods"], 0)
        self.assertEqual(result["evidence_counts"]["carrier_identifications"], 0)
        self.assertGreaterEqual(
            result["rejected_evidence_counts"]["negated_or_not_executed"], 1
        )

    def test_unresolved_or_excluded_carrier_bridges_do_not_trigger(self):
        method = hit(
            "method",
            "rock_magnetic_experiment",
            "IRM acquisition curves were measured for the Alpha Formation.",
        )
        claims = (
            "The magnetic carrier of remanence remains unknown.",
            "The carrier of magnetization could not be identified.",
            "It is unclear which mineral carries the remanence.",
            "Magnetite is not the carrier of remanence.",
            "The results exclude magnetite as the carrier of remanence.",
            "There is no evidence that hematite carries the remanence.",
            "The experiment was intended to identify the remanence carrier, but results were inconclusive.",
            "The remanence carrier has yet to be identified.",
            "The carrier was not established.",
            "The identity of the remanence carrier is ambiguous.",
            "No specific remanence carrier was identified.",
            "The data do not constrain the remanence carrier.",
            "We cannot determine which mineral carries the remanence.",
            "Magnetite was not established as the carrier.",
            "The data do not support magnetite as the remanence carrier.",
            "Magnetite cannot be confirmed as the carrier.",
            "Whether magnetite carries the remanence is ambiguous.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                bridge = hit("bridge", "explicit_carrier_bridge", claim)
                result = self.reasoner.evaluate(retrieval(method, bridge))
                self.assertEqual(result["policy_decision"], 0)
                self.assertEqual(result["evidence_counts"]["carrier_identifications"], 0)
                self.assertEqual(result["evidence_counts"]["carrier_significance"], 0)

        positive = hit(
            "bridge",
            "mineral_remanence_bridge",
            "Magnetite carries the remanence, although its acquisition age remains uncertain.",
        )
        positive_result = self.reasoner.evaluate(retrieval(method, positive))
        self.assertEqual(positive_result["policy_decision"], 1)
        self.assertEqual(positive_result["evidence_counts"]["carrier_identifications"], 1)
        self.assertEqual(positive_result["evidence_counts"]["carrier_significance"], 1)

    def test_cross_target_hits_do_not_trigger_in_multi_target_paper(self):
        method = hit(
            "method",
            "rock_magnetic_experiment",
            "IRM curves show no saturation for the Beta Formation.",
            local_target=False,
        )

        result = self.reasoner.evaluate(retrieval(method), multi_target=True)

        self.assertEqual(result["policy_decision"], 0)
        self.assertEqual(
            result["rejected_evidence_counts"]["cross_target_or_unbound"], 1
        )

    def test_local_weak_hit_cannot_absorb_sibling_rockmag_from_wide_quote(self):
        cache_row = {
            "source": {"source_quality_flags": []},
            "pages": [
                {
                    "page": 1,
                    "source": "synthetic",
                    "section_hints": ["Rock magnetism"],
                    "text": (
                        "The Alpha Formation was thermally demagnetized.\n\n"
                        "The Beta Formation was tested using IRM acquisition curves, "
                        "which identify magnetite as the remanence carrier."
                    ),
                }
            ],
        }
        source = retrieve_descriptions(
            cache_row,
            "Alpha Formation",
            True,
            "R3",
            competing_targets=["Beta Formation"],
        )

        result = self.reasoner.evaluate(source, multi_target=True)

        self.assertEqual(result["policy_decision"], 0)
        self.assertEqual(result["evidence_counts"]["accepted_methods"], 0)
        self.assertEqual(result["evidence_counts"]["carrier_identifications"], 0)
        self.assertGreater(
            result["rejected_evidence_counts"]["cross_target_or_unbound"],
            0,
        )

    def test_promoted_multi_target_scope_must_connect_the_evidence_pair(self):
        method = hit(
            "method",
            "rock_magnetic_experiment",
            "IRM acquisition was performed.",
            target_anchor="target_same_paragraph",
            scope_id="page:4:paragraph:2",
            scope_kind="paragraph",
        )
        same_scope_carrier = hit(
            "carrier",
            "named_carrier",
            "Magnetite was identified.",
            target_anchor="target_same_paragraph",
            scope_id="page:4:paragraph:2",
            scope_kind="paragraph",
        )
        other_scope_carrier = hit(
            "carrier",
            "named_carrier",
            "Magnetite was identified.",
            target_anchor="target_same_paragraph",
            scope_id="page:4:paragraph:3",
            scope_kind="paragraph",
        )

        connected = self.reasoner.evaluate(
            retrieval(method, same_scope_carrier),
            multi_target=True,
        )
        disconnected = self.reasoner.evaluate(
            retrieval(method, other_scope_carrier),
            multi_target=True,
        )

        self.assertEqual(connected["policy_decision"], 1)
        self.assertEqual(connected["selected_evidence"]["method"]["scope_id"], "page:4:paragraph:2")
        self.assertEqual(disconnected["policy_decision"], 0)
        self.assertFalse(disconnected["conditions"]["connected_positive_path"])

    def test_source_unavailable_ocr_risk_and_identity_ambiguity_abstain(self):
        strong = hit(
            "method",
            "rock_magnetic_experiment",
            "IRM curves show no saturation up to 1 T.",
        )
        cases = (
            (retrieval(source_available=False), [], False, "SOURCE_UNAVAILABLE"),
            (retrieval(strong), ["glyph_or_cid_text"], False, "SOURCE_QUALITY_RISK"),
            (retrieval(strong), [], True, "IDENTITY_AMBIGUOUS"),
        )
        for source, flags, ambiguous, reason_code in cases:
            with self.subTest(reason_code=reason_code):
                result = self.reasoner.evaluate(
                    source,
                    source_flags=flags,
                    identity_ambiguous=ambiguous,
                )
                self.assertEqual(result["policy_decision"], "ABSTAIN")
                self.assertEqual(result["proxy_decision"], 0)
                self.assertEqual(result["outcome"], "UNKNOWN")
                self.assertTrue(result["manual_review"])
                self.assertEqual(result["reason_code"], reason_code)
                self.assertEqual(result["normative_r3"]["policy_decision"], "ABSTAIN")
                self.assertEqual(result["rockmag_proxy_r3"]["value"], None)
                self.assertIsNone(result["criterion_practice_conflict"])

    def test_complete_high_quality_source_without_strong_evidence_is_zero(self):
        result = self.reasoner.evaluate(retrieval())

        self.assertEqual(result["policy_decision"], 0)
        self.assertEqual(result["proxy_decision"], 0)
        self.assertEqual(result["outcome"], "NEGATIVE")
        self.assertFalse(result["manual_review"])
        self.assertEqual(result["evidence_state"], "NEITHER")
        self.assertEqual(result["normative_r3"]["policy_decision"], "ABSTAIN")
        self.assertEqual(result["normative_r3"]["outcome"], "UNKNOWN")
        self.assertFalse(result["criterion_practice_conflict"])

    def test_runtime_retrieval_rejects_forbidden_source_fields(self):
        leaky = retrieval()
        leaky["ROCKMAG"] = "IRM(MAGN)"

        with self.assertRaisesRegex(R3RockmagRuntimeBoundaryError, "forbidden runtime"):
            self.reasoner.evaluate(leaky)

    def test_output_declares_dual_track_and_runtime_boundary(self):
        method = hit(
            "method",
            "rock_magnetic_experiment",
            "IRM curves show no saturation up to 1 T.",
        )

        result = self.reasoner.evaluate(retrieval(method))

        self.assertEqual(result["policy_decision"], 1)
        self.assertEqual(result["binary_compatibility_value"], result["proxy_decision"])
        self.assertEqual(
            result["rockmag_proxy_r3"]["binary_value"],
            result["binary_compatibility_value"],
        )
        self.assertEqual(
            set(result["rockmag_proxy_r3"]),
            {"policy_decision", "value", "binary_value", "outcome", "decision_path", "evidence"},
        )
        self.assertIn("original paper page text", result["runtime_scoring_sources"])
        self.assertIn("PMAGRESULT", result["forbidden_runtime_scoring_sources"])
        self.assertFalse(result["provenance"]["runtime_label_access"])
        self.assertFalse(result["provenance"]["contains_record_answer_mapping"])


if __name__ == "__main__":
    unittest.main()
