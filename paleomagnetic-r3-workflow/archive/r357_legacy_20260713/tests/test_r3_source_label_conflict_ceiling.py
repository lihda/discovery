from __future__ import annotations

import inspect
import unittest

from scripts import audit_r3_source_label_conflict_ceiling as audit


def candidate_template(fingerprint: str = "same") -> dict:
    return {
        "representation_fingerprint": fingerprint,
        "semantic_signature": "semantic",
        "refno": "10",
        "canonical_target": "alpha formation",
        "rocknames": ["Alpha Formation"],
        "proxy_support_status": "HIGH_CONFIDENCE_SAME_SCOPE",
        "criterion_support_status": "HIGH_CONFIDENCE_COMPLETE_CHAIN",
        "review_quotes": [],
    }


class R3SourceLabelConflictCeilingTests(unittest.TestCase):
    def test_same_scope_own_study_method_carrier_significance_is_high_confidence(self) -> None:
        text = (
            "We measured IRM acquisition curves for Alpha Formation samples. "
            "Magnetite carries the characteristic remanence component."
        )
        scope_id = audit.text_sha256(audit.component.normalized_text(text))[:24]
        paper = {
            "refno": "10",
            "target_count": 1,
            "source_scopes": [
                {
                    "scope_id": scope_id,
                    "page": "3",
                    "kind": "paragraph",
                    "target_hits": {"alpha formation": ["Alpha Formation"]},
                    "concepts": [
                        "method:irm_acquisition",
                        "carrier:magnetite",
                    ],
                }
            ],
        }
        target = {
            "target_id": "target",
            "canonical_target": "alpha formation",
            "rocknames": ["Alpha Formation"],
            "resultnos": ["100"],
            "scope_links": {"direct_scope_ids": [scope_id], "bridged_scope_ids": []},
            "features": {
                "direct_scope_count": 1,
                "linked_scope_count": 1,
                "sample_id_count": 0,
                "component_count": 1,
            },
            "sample_ids": [],
            "components": ["chrm"],
        }

        result = audit.classify_target_candidate(paper, target, {scope_id: text})

        self.assertEqual(result["proxy_support_status"], "HIGH_CONFIDENCE_SAME_SCOPE")
        self.assertEqual(
            result["criterion_support_status"],
            "HIGH_CONFIDENCE_COMPLETE_CHAIN",
        )
        self.assertTrue(result["own_study_flags"]["remanence_significance"])

    def test_method_carrier_without_own_study_language_is_not_high_confidence(self) -> None:
        text = "Smith et al. reported IRM and magnetite in Alpha Formation."
        scope_id = audit.text_sha256(audit.component.normalized_text(text))[:24]
        paper = {
            "refno": "10",
            "target_count": 1,
            "source_scopes": [
                {
                    "scope_id": scope_id,
                    "page": "2",
                    "kind": "paragraph",
                    "target_hits": {"alpha formation": ["Alpha Formation"]},
                    "concepts": ["method:irm_acquisition", "carrier:magnetite"],
                }
            ],
        }
        target = {
            "target_id": "target",
            "canonical_target": "alpha formation",
            "rocknames": ["Alpha Formation"],
            "resultnos": ["100"],
            "scope_links": {"direct_scope_ids": [scope_id], "bridged_scope_ids": []},
            "features": {
                "direct_scope_count": 1,
                "linked_scope_count": 1,
                "sample_id_count": 0,
                "component_count": 0,
            },
            "sample_ids": [],
            "components": [],
        }

        result = audit.classify_target_candidate(paper, target, {scope_id: text})

        self.assertEqual(result["proxy_support_status"], "NO_FROZEN_CANDIDATE")
        self.assertEqual(
            result["criterion_support_status"], "INCOMPLETE_OR_ABSENT_CHAIN"
        )

    def test_exact_representation_error_lower_bound_uses_minority_count(self) -> None:
        same = candidate_template("same")
        other = candidate_template("other")
        rows = [
            {"candidate": same, "resultno": "1", "label": 0},
            {"candidate": same, "resultno": "2", "label": 1},
            {"candidate": same, "resultno": "3", "label": 1},
            {"candidate": other, "resultno": "4", "label": 0},
        ]

        result = audit.exact_representation_bound(rows)

        self.assertEqual(result["mixed_representation_groups"], 1)
        self.assertEqual(result["minimum_unavoidable_errors"], 1)
        self.assertEqual(result["error_rate_lower_bound"], 0.25)
        self.assertEqual(result["accuracy_upper_bound"], 0.75)
        self.assertIn("not a mathematical ceiling for all source-only", result["limitation"])

    def test_freeze_api_has_no_label_or_target_workbook_argument(self) -> None:
        parameters = inspect.signature(audit.freeze_candidates).parameters
        self.assertNotIn("targets_path", parameters)
        self.assertNotIn("labels_path", parameters)
        self.assertNotIn("labels", parameters)

    def test_bottom_k_overlap_is_bounded_and_identical_for_same_text(self) -> None:
        text = "IRM acquisition curves identify magnetite as the remanence carrier"
        sketch = audit.bottom_k_shingle_sketch(text)
        self.assertEqual(audit.sketch_overlap(sketch, sketch), 1.0)
        self.assertEqual(audit.sketch_overlap(sketch, []), 0.0)


if __name__ == "__main__":
    unittest.main()
