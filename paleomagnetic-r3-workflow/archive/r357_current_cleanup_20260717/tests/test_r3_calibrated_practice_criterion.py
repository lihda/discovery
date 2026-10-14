from __future__ import annotations

import copy
import unittest

from scripts.r3_calibrated_practice_criterion import (
    DEFAULT_CRITERION,
    R3CalibratedCriterionError,
    decide_probability,
    load_criterion,
    validate_criterion,
    verify_dependencies,
)


class R3CalibratedPracticeCriterionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.criterion = load_criterion(DEFAULT_CRITERION)

    def test_repository_criterion_and_dependencies_validate(self) -> None:
        audit = verify_dependencies(self.criterion)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(len(audit["dependencies"]), 5)

    def test_three_semantic_routes_replace_complete_chain_for_practice_head(self) -> None:
        routes = {
            route["route_id"] for route in self.criterion["evidence_routes"]
        }
        self.assertEqual(
            routes,
            {
                "R3.PRACTICE.ROUTE.PERFORMED_METHOD",
                "R3.PRACTICE.ROUTE.SELF_AUTHENTICATING_RESULT",
                "R3.PRACTICE.ROUTE.CARRIER_ATTRIBUTION",
            },
        )
        self.assertEqual(
            self.criterion["semantic_revision"]["revised_requirement"]["operator"],
            "any",
        )

    def test_selective_threshold_change_is_rejected(self) -> None:
        changed = copy.deepcopy(self.criterion)
        changed["decision_policy"]["selective_policy"][
            "MEETS_PRACTICE_PROXY"
        ]["minimum_probability"] = 0.6
        with self.assertRaises(R3CalibratedCriterionError):
            validate_criterion(changed)

    def test_runtime_pmagresult_access_is_rejected(self) -> None:
        changed = copy.deepcopy(self.criterion)
        changed["runtime_policy"]["forbidden_sources"].remove(
            "PMAGRESULT_at_runtime"
        )
        with self.assertRaises(R3CalibratedCriterionError):
            validate_criterion(changed)

    def test_supervision_must_be_disclosed(self) -> None:
        changed = copy.deepcopy(self.criterion)
        changed["provenance"]["rockmag_field_supervision"] = False
        with self.assertRaises(R3CalibratedCriterionError):
            validate_criterion(changed)

    def test_probability_policy_preserves_boundaries_and_abstention(self) -> None:
        self.assertEqual(
            decide_probability(self.criterion, 0.30),
            "DOES_NOT_MEET_PRACTICE_PROXY",
        )
        self.assertEqual(decide_probability(self.criterion, 0.50), "ABSTAIN")
        self.assertEqual(
            decide_probability(self.criterion, 0.70),
            "MEETS_PRACTICE_PROXY",
        )
        self.assertEqual(
            decide_probability(
                self.criterion, 0.99, original_pdf_available=False
            ),
            "SKIP_NO_SOURCE",
        )
        with self.assertRaises(R3CalibratedCriterionError):
            decide_probability(self.criterion, 1.01)


if __name__ == "__main__":
    unittest.main()
