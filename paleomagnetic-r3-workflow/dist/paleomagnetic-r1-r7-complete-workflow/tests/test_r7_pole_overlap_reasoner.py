import json
import math
import unittest

from scripts.r7_pole_overlap_reasoner import (
    CRITERION_SOURCE_SHA256,
    R7InputBoundaryError,
    a95_envelopes_overlap,
    a95_overlap_details,
    angular_distance_degrees,
    evaluate_r7_pole_overlap,
)


def target_pole(**overrides):
    pole = {
        "pole_id": "target-pole",
        "lat": 0.0,
        "lon": 0.0,
        "A95": 5.0,
        "age": 1700.0,
        "period": "Statherian",
        "craton": "Kalahari Craton",
        "stable_connected_region": "Kalahari stable connected region",
        "reliability_R": 4,
        "source": {
            "citation": "Synthetic target source",
            "quote": "target source passage",
        },
    }
    pole.update(overrides)
    return pole


def younger_pole(**overrides):
    pole = {
        "pole_id": "younger-pole-1",
        "lat": 0.0,
        "lon": 5.0,
        "A95": 5.0,
        "age": 520.0,
        "period": "Cambrian",
        "craton": "Kalahari",
        "stable_connected_region": "Kalahari",
        "reliability_R": 3,
        "source": {
            "citation": "Synthetic younger-pole source",
            "quote": "younger pole source passage",
        },
    }
    pole.update(overrides)
    return pole


def complete_search(**overrides):
    search = {
        "qualified_search_complete": True,
        "source": {
            "knowledge_base_version": "synthetic-younger-poles-v1",
            "query_scope": "R>=3, Kalahari stable connected region",
        },
    }
    search.update(overrides)
    return search


class SphericalGeometryTests(unittest.TestCase):
    def test_angular_distance_boundaries(self):
        self.assertAlmostEqual(angular_distance_degrees(0, 0, 0, 0), 0.0, places=12)
        self.assertAlmostEqual(angular_distance_degrees(0, 0, 0, 90), 90.0, places=12)
        self.assertAlmostEqual(angular_distance_degrees(90, 0, -90, 0), 180.0, places=12)

        with self.assertRaisesRegex(ValueError, "latitude"):
            angular_distance_degrees(91, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            angular_distance_degrees(0, math.inf, 0, 0)

    def test_crosses_antimeridian_using_short_arc(self):
        distance = angular_distance_degrees(0, 179, 0, -179)

        self.assertAlmostEqual(distance, 2.0, places=10)
        self.assertTrue(a95_envelopes_overlap(0, 179, 1, 0, -179, 1))

    def test_a95_envelopes_overlap_at_inclusive_boundary(self):
        details = a95_overlap_details(0, 0, 10, 0, 20, 10)

        self.assertTrue(details["overlaps"])
        self.assertTrue(details["boundary_is_inclusive"])
        self.assertAlmostEqual(details["angular_distance_deg"], 20.0, places=10)
        self.assertAlmostEqual(details["effective_overlap_threshold_deg"], 20.0)
        self.assertFalse(a95_envelopes_overlap(0, 0, 10, 0, 20.001, 10))

    def test_large_caps_use_a_spherical_180_degree_threshold(self):
        details = a95_overlap_details(0, 0, 100, 0, 180, 100)

        self.assertTrue(details["overlaps"])
        self.assertEqual(details["raw_a95_sum_deg"], 200)
        self.assertEqual(details["effective_overlap_threshold_deg"], 180)


class R7PoleOverlapReasonerTests(unittest.TestCase):
    def test_target_does_not_require_reliability_r(self):
        target = target_pole()
        target.pop("reliability_R")
        candidate = younger_pole()

        result = evaluate_r7_pole_overlap(target, [candidate])

        self.assertEqual(result["relation"], "CONFLICT")
        self.assertNotIn("INVALID_RELIABILITY_R", result["target"]["validation_errors"])

    def test_eligible_younger_overlap_is_conflict(self):
        result = evaluate_r7_pole_overlap(target_pole(), [younger_pole()])

        self.assertEqual(result["relation"], "CONFLICT")
        self.assertEqual(result["decision"]["recommended_value"], 0)
        relation = result["relations"][0]
        self.assertTrue(relation["accepted_for_comparison"])
        self.assertEqual(relation["reason_code"], "A95_ENVELOPES_OVERLAP")
        self.assertTrue(relation["eligibility"]["reliability"]["passed"])
        self.assertTrue(relation["eligibility"]["geography"]["passed"])
        self.assertTrue(relation["eligibility"]["time"]["passed"])
        self.assertGreaterEqual(relation["eligibility"]["time"]["period_rank_gap"], 2)

    def test_disjoint_envelopes_require_an_auditable_complete_search(self):
        incomplete = evaluate_r7_pole_overlap(
            target_pole(),
            [younger_pole(lon=60.0, A95=4.0)],
        )
        complete = evaluate_r7_pole_overlap(
            target_pole(),
            [younger_pole(lon=60.0, A95=4.0)],
            qualified_search=complete_search(),
        )

        self.assertEqual(incomplete["relations"][0]["relation"], "SUPPORT")
        self.assertEqual(incomplete["relation"], "UNKNOWN")
        self.assertIsNone(incomplete["decision"]["recommended_value"])
        self.assertEqual(
            incomplete["decision"]["reason_code"],
            "QUALIFIED_SEARCH_NOT_AUDITABLY_COMPLETE",
        )
        self.assertFalse(incomplete["qualified_search"]["accepted_as_complete"])

        self.assertEqual(complete["relation"], "SUPPORT")
        self.assertEqual(complete["decision"]["recommended_value"], 1)
        self.assertTrue(complete["qualified_search"]["accepted_as_complete"])
        self.assertEqual(complete["relations"][0]["reason_code"], "A95_ENVELOPES_DISJOINT")

    def test_complete_search_flag_without_source_is_not_auditable(self):
        for source in (None, "", 7):
            with self.subTest(source=source):
                search = {"qualified_search_complete": True}
                if source is not None:
                    search["source"] = source
                result = evaluate_r7_pole_overlap(
                    target_pole(),
                    [younger_pole(lon=60.0)],
                    qualified_search=search,
                )

                self.assertEqual(result["relation"], "UNKNOWN")
                self.assertFalse(result["qualified_search"]["accepted_as_complete"])
                self.assertIn(
                    "COMPLETE_SEARCH_REQUIRES_AUDITABLE_SOURCE",
                    result["qualified_search"]["validation_errors"],
                )

    def test_candidate_reliability_boundary_is_three(self):
        accepted = evaluate_r7_pole_overlap(
            target_pole(), [younger_pole(reliability_R=3)]
        )
        rejected = evaluate_r7_pole_overlap(
            target_pole(), [younger_pole(reliability_R=2.999)]
        )

        self.assertEqual(accepted["relation"], "CONFLICT")
        self.assertEqual(rejected["relation"], "UNKNOWN")
        self.assertEqual(
            rejected["relations"][0]["reason_code"],
            "CANDIDATE_RELIABILITY_BELOW_3",
        )
        self.assertIsNone(rejected["decision"]["recommended_value"])

    def test_candidate_must_be_younger_by_more_than_one_period(self):
        adjacent = younger_pole(
            age=180.0,
            period="Jurassic",
            reliability_R=5,
        )
        triassic_target = target_pole(age=230.0, period="Triassic")

        result = evaluate_r7_pole_overlap(triassic_target, [adjacent])

        self.assertEqual(result["relation"], "UNKNOWN")
        self.assertEqual(
            result["relations"][0]["reason_code"],
            "NOT_YOUNGER_BY_MORE_THAN_ONE_PERIOD",
        )
        self.assertEqual(
            result["relations"][0]["eligibility"]["time"]["period_rank_gap"],
            1,
        )

    def test_broad_time_unit_is_resolved_from_age_without_labels(self):
        result = evaluate_r7_pole_overlap(
            target_pole(period="Paleoproterozoic"),
            [younger_pole(lon=60)],
            qualified_search=complete_search(),
        )

        self.assertEqual(result["relation"], "SUPPORT")
        self.assertEqual(result["target"]["period"]["resolved_period"], "Statherian")
        self.assertEqual(
            result["target"]["period"]["resolution_method"],
            "formal_period_inferred_within_named_broad_unit",
        )

    def test_different_craton_or_connected_region_is_not_compared(self):
        cases = (
            (
                younger_pole(craton="Laurentia", stable_connected_region="Laurentia"),
                "DIFFERENT_CRATON",
            ),
            (
                younger_pole(stable_connected_region="Zimbabwe stable block"),
                "DIFFERENT_STABLE_CONNECTED_REGION",
            ),
        )
        for candidate, reason in cases:
            with self.subTest(reason=reason):
                result = evaluate_r7_pole_overlap(target_pole(), [candidate])
                self.assertEqual(result["relation"], "UNKNOWN")
                self.assertEqual(result["relations"][0]["reason_code"], reason)
                self.assertFalse(result["relations"][0]["accepted_for_comparison"])

    def test_orogenic_and_unknown_scope_are_downgraded_to_unknown(self):
        cases = (
            younger_pole(stable_connected_region="Damara Orogenic Belt"),
            younger_pole(stable_connected_region="unknown"),
            younger_pole(craton="unknown"),
        )
        for candidate in cases:
            with self.subTest(candidate=candidate):
                result = evaluate_r7_pole_overlap(target_pole(), [candidate])
                self.assertEqual(result["relation"], "UNKNOWN")
                self.assertIn(
                    result["relations"][0]["reason_code"],
                    {"OROGENIC_SCOPE", "UNKNOWN_STABLE_SCOPE"},
                )

    def test_field_age_constraint_rescues_an_overlapping_younger_pole(self):
        field_constraint = {
            "constraint_id": "positive-fold-test",
            "kind": "magnetization_older_than",
            "bound_age_ma": 800.0,
            "field_test": "positive_fold_test",
            "established": True,
            "candidate_id": "younger-pole-1",
            "source": {
                "citation": "Synthetic field-test source",
                "quote": "A field test constrains magnetization to predate 800 Ma.",
            },
        }

        result = evaluate_r7_pole_overlap(
            target_pole(), [younger_pole()], [field_constraint]
        )

        self.assertEqual(result["relation"], "SUPPORT")
        self.assertEqual(result["decision"]["recommended_value"], 1)
        relation = result["relations"][0]
        self.assertEqual(relation["reason_code"], "FIELD_EVIDENCE_RESCUE")
        self.assertEqual(
            relation["geometric_relation_before_field_rescue"], "CONFLICT"
        )
        self.assertEqual(len(relation["field_rescues"]), 1)
        self.assertGreater(
            relation["field_rescues"][0]["formula_parameters"]["older_age_margin_ma"],
            0,
        )
        self.assertEqual(result["field_relations"][0]["relation"], "SUPPORT")

    def test_equal_field_age_bound_does_not_prove_older_magnetization(self):
        field_constraint = {
            "constraint_id": "equal-age-test",
            "kind": "magnetization_older_than",
            "bound_age_ma": 520.0,
            "field_test": "positive_contact_test",
            "established": True,
            "source": "synthetic field evidence",
        }

        result = evaluate_r7_pole_overlap(
            target_pole(), [younger_pole()], [field_constraint]
        )

        self.assertEqual(result["relation"], "CONFLICT")
        self.assertEqual(result["relations"][0]["field_rescues"], [])
        self.assertEqual(result["field_relations"][0]["relation"], "UNKNOWN")

    def test_any_unrescued_overlap_has_precedence_over_disjoint_support(self):
        candidates = [
            younger_pole(pole_id="overlap", lon=3.0),
            younger_pole(pole_id="disjoint", lon=70.0),
        ]

        result = evaluate_r7_pole_overlap(
            target_pole(),
            candidates,
            qualified_search=complete_search(),
        )

        self.assertEqual(result["relation"], "CONFLICT")
        self.assertEqual(result["decision"]["recommended_value"], 0)
        self.assertEqual(result["decision"]["support_count"], 1)
        self.assertEqual(result["decision"]["conflict_count"], 1)

    def test_complete_search_with_unresolved_scope_is_mixed_for_review(self):
        candidates = [
            younger_pole(pole_id="disjoint", lon=70.0),
            younger_pole(
                pole_id="unresolved-scope",
                lon=80.0,
                stable_connected_region="unknown",
            ),
        ]

        result = evaluate_r7_pole_overlap(
            target_pole(),
            candidates,
            qualified_search=complete_search(),
        )

        self.assertEqual(result["relation"], "MIXED")
        self.assertIsNone(result["decision"]["recommended_value"])
        self.assertTrue(result["decision"]["manual_review_required"])
        self.assertEqual(
            result["decision"]["reason_code"],
            "COMPLETE_SEARCH_CONTAINS_UNRESOLVED_CANDIDATES",
        )

    def test_no_candidates_is_unknown_instead_of_a_default_positive(self):
        result = evaluate_r7_pole_overlap(target_pole(), [])

        self.assertEqual(result["relation"], "UNKNOWN")
        self.assertIsNone(result["decision"]["recommended_value"])
        self.assertEqual(
            result["decision"]["reason_code"],
            "NO_ELIGIBLE_YOUNGER_POLE_COMPARISON",
        )

    def test_forbidden_identifier_and_label_fields_are_rejected_recursively(self):
        bad_inputs = (
            target_pole(REFNO=123),
            target_pole(source={"RESULTNO": 9}),
            target_pole(metadata={"gold_label": 0}),
            target_pole(metadata={"label": 1}),
            target_pole(metadata={"label_supervision": False}),
            target_pole(R7=0),
        )
        for target in bad_inputs:
            with self.subTest(target=target):
                with self.assertRaisesRegex(
                    R7InputBoundaryError,
                    "forbidden identifier or supervised-label field",
                ):
                    evaluate_r7_pole_overlap(target, [younger_pole()])

    def test_output_is_auditable_and_contains_only_source_hashes(self):
        target = target_pole(
            source={
                "citation": "Synthetic target paper",
                "quote": "private target evidence passage",
            }
        )
        candidate = younger_pole(
            source={
                "citation": "Synthetic comparison paper",
                "quote": "private candidate evidence passage",
            }
        )

        first = evaluate_r7_pole_overlap(target, [candidate])
        second = evaluate_r7_pole_overlap(target, [candidate])
        serialized = json.dumps(first, sort_keys=True)

        self.assertEqual(first, second)
        self.assertEqual(
            first["provenance"]["criterion_source_sha256"],
            CRITERION_SOURCE_SHA256,
        )
        self.assertEqual(len(first["provenance"]["input_sha256"]), 64)
        self.assertTrue(
            all(
                len(source_hash) == 64
                for source_hash in first["provenance"]["evidence_source_sha256"]
            )
        )
        self.assertIn("angular_distance", first["formula"])
        self.assertIn("period_scale_sha256", first["parameters"])
        self.assertIn("formula_parameters", first["relations"][0])
        self.assertNotIn("private target evidence passage", serialized)
        self.assertNotIn("private candidate evidence passage", serialized)
        self.assertEqual(first["provenance"]["supervision"], "none")
        self.assertFalse(first["provenance"]["uses_answer_lookup"])


if __name__ == "__main__":
    unittest.main()
