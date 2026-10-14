import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.criterion_spec import DEFAULT_CRITERION_SPEC
from scripts.r3_practice_criterion import (
    DEFAULT_R3_PRACTICE_CRITERION,
    R3PracticeCriterionValidationError,
    load_r3_practice_criterion,
    validate_r3_practice_criterion,
    verify_r3_practice_dependencies,
)


ROOT = Path(__file__).resolve().parents[1]


def by_id(items, key, identifier):
    return next(item for item in items if item[key] == identifier)


class R3PracticeCriterionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.criterion = load_r3_practice_criterion()

    def test_practice_criterion_is_separate_opt_in_and_record_independent(self):
        self.assertNotEqual(DEFAULT_R3_PRACTICE_CRITERION, DEFAULT_CRITERION_SPEC)
        self.assertEqual(DEFAULT_CRITERION_SPEC.name, "r357_criterion_spec_v2.json")
        normative_digest = hashlib.sha256(DEFAULT_CRITERION_SPEC.read_bytes()).hexdigest()
        relationship = self.criterion["normative_relationship"]

        self.assertEqual(
            relationship["relationship"], "non_normative_practice_proxy"
        )
        self.assertEqual(relationship["normative_sha256"], normative_digest)
        self.assertFalse(relationship["replaces_normative"])
        self.assertFalse(relationship["default_scoring_enabled"])
        self.assertEqual(
            self.criterion["runtime_policy"]["integration_mode"],
            "explicit_opt_in_only",
        )
        self.assertFalse(
            self.criterion["runtime_policy"]["overwrite_normative_r3"]
        )
        serialized = json.dumps(self.criterion, ensure_ascii=False).casefold()
        self.assertNotIn("refno", serialized)
        self.assertNotIn("resultno", serialized)

    def test_episode_is_the_documented_rockmag_practice_downgrade(self):
        episode = self.criterion["evidence_episode"]
        facets = {item["facet_id"] for item in episode["required_facets"]}

        self.assertEqual(
            episode["unit"], "target_bound_implemented_characterization_episode"
        )
        self.assertEqual(
            facets,
            {
                "R3.F.SOURCE.ORIGINAL_PAPER",
                "R3.F.CONTEXT.ELIGIBLE",
                "R3.F.EXECUTION.AFFIRMED",
                "R3.F.CHARACTERIZATION.ELIGIBLE",
                "R3.F.SCOPE.ATTRIBUTABLE",
            },
        )
        self.assertEqual(
            set(episode["not_required_by_practice_proxy"]),
            {
                "explicit_carrier_identification",
                "explicit_remanence_significance_interpretation",
            },
        )
        self.assertEqual(
            set(self.criterion["decision_logic"]["positive_requires_all"]),
            facets,
        )
        self.assertEqual(
            self.criterion["decision_logic"]["skip_if"],
            "original_paper_unavailable",
        )

    def test_all_required_standalone_and_context_exclusions_are_frozen(self):
        exclusions = {
            item["exclusion_id"]: item for item in self.criterion["exclusions"]
        }

        self.assertEqual(
            set(exclusions),
            {
                "R3.X.STANDALONE.TUB",
                "R3.X.STANDALONE.BARE_SUSC",
                "R3.X.STANDALONE.BARE_AN",
                "R3.X.STANDALONE.BARE_MINERAL_NAME",
                "R3.X.CONTEXT.REFERENCE_SECTION",
                "R3.X.CONTEXT.NOT_IMPLEMENTED",
            },
        )
        for exclusion_id in (
            "R3.X.STANDALONE.TUB",
            "R3.X.STANDALONE.BARE_SUSC",
            "R3.X.STANDALONE.BARE_AN",
            "R3.X.STANDALONE.BARE_MINERAL_NAME",
        ):
            self.assertEqual(
                exclusions[exclusion_id]["effect"],
                "does_not_satisfy_episode_alone",
            )
        for exclusion_id in (
            "R3.X.CONTEXT.REFERENCE_SECTION",
            "R3.X.CONTEXT.NOT_IMPLEMENTED",
        ):
            self.assertEqual(
                exclusions[exclusion_id]["effect"], "reject_evidence_span"
            )

    def test_target_evidence_precedes_only_two_permitted_propagation_paths(self):
        scope = self.criterion["scope_policy"]
        propagation = scope["paper_level_propagation"]
        conditions = {
            item["condition_id"]: set(item["requires_all"])
            for item in propagation["allow_if_any"]
        }

        self.assertEqual(
            scope["evidence_resolution_order"],
            [
                "target_level",
                "explicit_shared_sample_scope",
                "single_target_paper_scope",
            ],
        )
        self.assertEqual(propagation["default"], "forbidden")
        self.assertEqual(
            conditions,
            {
                "R3.SCOPE.SINGLE_TARGET_PAPER": {"paper_has_single_target"},
                "R3.SCOPE.EXPLICIT_SHARED_SAMPLE": {
                    "explicit_shared_sample_scope",
                    "scope_includes_target",
                    "episode_applies_to_shared_samples",
                },
            },
        )
        self.assertEqual(propagation["unresolved_scope_outcome"], "ABSTAIN")

    def test_repository_dependencies_and_all_concept_refs_verify(self):
        result = verify_r3_practice_dependencies(self.criterion, ROOT)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["concept_refs_verified"], 14)
        self.assertEqual(
            {item["basis_id"] for item in result["dependencies"]},
            {"R3.BASIS.NORMATIVE.V2", "R3.BASIS.ROCKMAG.LEXICON.V1"},
        )

    def test_validation_rejects_removing_or_weakening_an_exclusion(self):
        missing_tub = copy.deepcopy(self.criterion)
        missing_tub["exclusions"] = [
            item
            for item in missing_tub["exclusions"]
            if item["exclusion_id"] != "R3.X.STANDALONE.TUB"
        ]
        with self.assertRaisesRegex(
            R3PracticeCriterionValidationError, "must cover Tub"
        ):
            validate_r3_practice_criterion(missing_tub)

        accepted_reference = copy.deepcopy(self.criterion)
        reference = by_id(
            accepted_reference["exclusions"],
            "exclusion_id",
            "R3.X.CONTEXT.REFERENCE_SECTION",
        )
        reference["effect"] = "does_not_satisfy_episode_alone"
        with self.assertRaisesRegex(
            R3PracticeCriterionValidationError, "matcher or effect"
        ):
            validate_r3_practice_criterion(accepted_reference)

        overbroad_an = copy.deepcopy(self.criterion)
        bare_an = by_id(
            overbroad_an["exclusions"],
            "exclusion_id",
            "R3.X.STANDALONE.BARE_AN",
        )
        bare_an["matcher"]["values"] = [".*AN.*"]
        with self.assertRaisesRegex(
            R3PracticeCriterionValidationError, "incorrectly rejects contextual case"
        ):
            validate_r3_practice_criterion(overbroad_an)

    def test_validation_rejects_cross_target_paper_propagation(self):
        default_allowed = copy.deepcopy(self.criterion)
        default_allowed["scope_policy"]["paper_level_propagation"][
            "default"
        ] = "allowed"
        with self.assertRaisesRegex(
            R3PracticeCriterionValidationError, "forbidden by default"
        ):
            validate_r3_practice_criterion(default_allowed)

        any_multi_target = copy.deepcopy(self.criterion)
        condition = by_id(
            any_multi_target["scope_policy"]["paper_level_propagation"][
                "allow_if_any"
            ],
            "condition_id",
            "R3.SCOPE.EXPLICIT_SHARED_SAMPLE",
        )
        condition["requires_all"] = ["paper_has_multiple_targets"]
        with self.assertRaisesRegex(
            R3PracticeCriterionValidationError, "allowed only"
        ):
            validate_r3_practice_criterion(any_multi_target)

    def test_validation_rejects_identifiers_answers_and_supervision(self):
        identifier = copy.deepcopy(self.criterion)
        identifier["RESULTNO_lookup"] = {}
        with self.assertRaisesRegex(
            R3PracticeCriterionValidationError, "record identifier key"
        ):
            validate_r3_practice_criterion(identifier)

        answer = copy.deepcopy(self.criterion)
        answer["evidence_episode"]["eligible_families"][0]["gold_label"] = 1
        with self.assertRaisesRegex(
            R3PracticeCriterionValidationError, "answer or label key"
        ):
            validate_r3_practice_criterion(answer)

        supervised = copy.deepcopy(self.criterion)
        supervised["provenance"]["label_supervision"] = True
        with self.assertRaisesRegex(
            R3PracticeCriterionValidationError, "label_supervision must be false"
        ):
            validate_r3_practice_criterion(supervised)

    def test_dependency_verification_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "data/knowledge/r357_criterion_spec_v2.json",
                "data/knowledge/r3_rockmag_lexicon_v1.json",
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT / relative).read_bytes())
            lexicon = root / "data/knowledge/r3_rockmag_lexicon_v1.json"
            lexicon.write_bytes(lexicon.read_bytes() + b"\n")

            with self.assertRaisesRegex(
                R3PracticeCriterionValidationError, "dependency SHA-256 mismatch"
            ):
                verify_r3_practice_dependencies(self.criterion, root)

    def test_loading_invalid_artifact_fails_closed(self):
        invalid = copy.deepcopy(self.criterion)
        invalid["runtime_policy"]["overwrite_normative_r3"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(
                R3PracticeCriterionValidationError, "opt-in and separate"
            ):
                load_r3_practice_criterion(path)


if __name__ == "__main__":
    unittest.main()
