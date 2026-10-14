import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.criterion_spec import (
    DEFAULT_CRITERION_SPEC,
    CriterionSpecValidationError,
    load_criterion_spec,
    validate_criterion_spec,
    verify_source_artifact,
)


def by_id(items, key, identifier):
    return next(item for item in items if item[key] == identifier)


class CriterionSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_criterion_spec(DEFAULT_CRITERION_SPEC)

    def test_default_spec_is_frozen_bilingual_sourced_and_record_independent(self):
        self.assertEqual(self.spec["freeze_status"], "frozen")
        self.assertFalse(self.spec["provenance"]["label_supervision"])
        self.assertFalse(self.spec["provenance"]["contains_record_identifiers"])
        self.assertFalse(self.spec["provenance"]["contains_answer_mapping"])
        self.assertEqual(set(self.spec["criteria"]), {"R3", "R5", "R7"})
        self.assertEqual(len(self.spec["source_propositions"]), 19)
        self.assertTrue(
            all(
                source["statement"]["en"] and source["statement"]["zh"]
                for source in self.spec["source_propositions"]
            )
        )
        for indicator, criterion in self.spec["criteria"].items():
            self.assertTrue(criterion["source_refs"], indicator)
            for proposition in criterion["propositions"]:
                self.assertTrue(proposition["statement"]["en"])
                self.assertTrue(proposition["statement"]["zh"])
                self.assertTrue(proposition["source_refs"])
        serialized = json.dumps(self.spec, ensure_ascii=False).casefold()
        self.assertNotIn("refno", serialized)
        self.assertNotIn("resultno", serialized)

    def test_frozen_attachment_digest_is_exact(self):
        artifact = self.spec["source_artifact"]
        self.assertEqual(
            artifact["sha256"],
            "8433f968a4658118dc43a060114b8d2a6a0ca0cbf6ed82ee006d924d970b18e5",
        )
        self.assertEqual(artifact["byte_size"], 11793)

    def test_source_artifact_verification_accepts_exact_bytes_and_rejects_change(self):
        payload = b"normative source bytes\n"
        local_spec = copy.deepcopy(self.spec)
        local_spec["source_artifact"]["sha256"] = hashlib.sha256(payload).hexdigest()
        local_spec["source_artifact"]["byte_size"] = len(payload)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_bytes(payload)
            result = verify_source_artifact(local_spec, path)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())

            path.write_bytes(payload + b"changed")
            with self.assertRaisesRegex(CriterionSpecValidationError, "SHA-256 mismatch"):
                verify_source_artifact(local_spec, path)

    def test_r3_requires_eligible_method_identification_and_significance(self):
        criterion = self.spec["criteria"]["R3"]
        groups = criterion["required_groups"]
        method = by_id(groups, "group_id", "R3.G.ELIGIBLE_METHOD")
        identify = by_id(groups, "group_id", "R3.G.IDENTIFY")
        significance = by_id(groups, "group_id", "R3.G.SIGNIFICANCE")

        self.assertEqual(method["expression"]["op"], "any")
        self.assertEqual(set(method["expression"]["args"]), set(criterion["eligible_methods"]))
        self.assertEqual(
            set(identify["expression"]["args"]),
            {
                "R3.P.CARRIER.IDENTIFIED",
                "R3.P.CARRIER.REASONABLE_INCONCLUSIVE_ATTEMPT",
            },
        )
        self.assertEqual(significance["expression"], "R3.P.CARRIER.SIGNIFICANCE_COMMENTED")
        self.assertEqual(
            criterion["decision_logic"]["meets"],
            {
                "op": "all",
                "args": [
                    "R3.G.ELIGIBLE_METHOD",
                    "R3.G.IDENTIFY",
                    "R3.G.SIGNIFICANCE",
                ],
            },
        )

    def test_r5_encodes_rigid_frame_disqualifiers_intrusion_exception_and_flattening_guard(self):
        criterion = self.spec["criteria"]["R5"]
        groups = criterion["required_groups"]
        disqualifying = by_id(groups, "group_id", "R5.G.DISQUALIFYING_SETTING")
        self.assertEqual(
            set(disqualifying["expression"]["args"]),
            {
                "R5.P.SETTING.ALLOCHTHONOUS",
                "R5.P.SETTING.PARAUTOCHTHONOUS",
                "R5.P.SETTING.NON_STRATIFIED",
                "R5.P.SETTING.INTERNAL_VERTICAL_AXIS_ROTATION",
            },
        )
        rigid = by_id(groups, "group_id", "R5.G.RIGID_FRAME")
        self.assertEqual(rigid["expression"]["op"], "any")

        flattening = by_id(groups, "group_id", "R5.G.FLATTENING_GUARD")
        self.assertEqual(
            flattening["expression"],
            {
                "op": "implies",
                "args": [
                    "R5.P.FLATTENING_CORRECTION_USED",
                    "R5.G.INDEPENDENT_FLATTENING_CORROBORATION",
                ],
            },
        )
        corroboration = by_id(
            groups, "group_id", "R5.G.INDEPENDENT_FLATTENING_CORROBORATION"
        )
        self.assertEqual(len(corroboration["expression"]["args"]), 2)

        exception = criterion["exceptions"][0]
        self.assertEqual(exception["trigger"], "R5.P.EXCEPTION.POST_DEFORMATION_INTRUSION")
        self.assertEqual(exception["effect"]["outcome"], "PERMITS_MEETING")
        self.assertEqual(exception["effect"]["overrides"], ["R5.G.DISQUALIFYING_SETTING"])
        self.assertNotIn("R5.G.FLATTENING_GUARD", exception["effect"]["overrides"])
        self.assertIn("R5.G.FLATTENING_GUARD", criterion["decision_logic"]["meets"]["args"])

    def test_r7_conflict_requires_all_four_gates_and_older_field_evidence_rescues(self):
        criterion = self.spec["criteria"]["R7"]
        groups = criterion["required_groups"]
        resemblance = by_id(groups, "group_id", "R7.G.QUALIFIED_YOUNGER_RESEMBLANCE")
        self.assertEqual(
            set(resemblance["expression"]["args"]),
            {
                "R7.P.COMPARISON.A95_OVERLAP",
                "R7.P.COMPARISON.YOUNGER_R_GE_3",
                "R7.P.COMPARISON.MORE_THAN_ONE_PERIOD_YOUNGER",
                "R7.G.GEOGRAPHICALLY_VALID_COMPARATOR",
            },
        )
        geography = by_id(groups, "group_id", "R7.G.GEOGRAPHICALLY_VALID_COMPARATOR")
        self.assertEqual(geography["expression"]["op"], "all")
        self.assertIn("R7.P.COMPARISON.STABLE_CONNECTED_CRATON", geography["expression"]["args"])

        exception = criterion["exceptions"][0]
        self.assertEqual(exception["effect"]["outcome"], "MEETS")
        self.assertEqual(
            exception["effect"]["overrides"], ["R7.G.QUALIFIED_YOUNGER_RESEMBLANCE"]
        )
        self.assertIn(exception["exception_id"], criterion["decision_logic"]["meets"]["args"])

    def test_validation_rejects_record_identifiers_answer_keys_and_label_supervision(self):
        labelled = copy.deepcopy(self.spec)
        labelled["provenance"]["label_supervision"] = True
        with self.assertRaisesRegex(CriterionSpecValidationError, "label_supervision"):
            validate_criterion_spec(labelled)

        identifier_key = copy.deepcopy(self.spec)
        identifier_key["REFNO_lookup"] = {}
        with self.assertRaisesRegex(CriterionSpecValidationError, "record identifier key"):
            validate_criterion_spec(identifier_key)

        identifier_text = copy.deepcopy(self.spec)
        identifier_text["title"]["en"] = "Lookup using RESULTNO"
        with self.assertRaisesRegex(CriterionSpecValidationError, "identifier reference"):
            validate_criterion_spec(identifier_text)

        answer_key = copy.deepcopy(self.spec)
        answer_key["criteria"]["R3"]["propositions"][0]["gold_label"] = 1
        with self.assertRaisesRegex(CriterionSpecValidationError, "forbidden answer key"):
            validate_criterion_spec(answer_key)

        answer_mapping = copy.deepcopy(self.spec)
        answer_mapping["provenance"]["contains_answer_mapping"] = True
        with self.assertRaisesRegex(CriterionSpecValidationError, "contains_answer_mapping"):
            validate_criterion_spec(answer_mapping)

        undeclared_payload = copy.deepcopy(self.spec)
        undeclared_payload["lookup_payload"] = {"opaque": 1}
        with self.assertRaisesRegex(CriterionSpecValidationError, "unexpected keys"):
            validate_criterion_spec(undeclared_payload)

    def test_validation_rejects_unknown_logic_reference_bad_operator_and_cycle(self):
        unknown = copy.deepcopy(self.spec)
        unknown["criteria"]["R3"]["decision_logic"]["meets"]["args"].append(
            "R3.P.UNDECLARED"
        )
        with self.assertRaisesRegex(CriterionSpecValidationError, "unknown logic identifier"):
            validate_criterion_spec(unknown)

        bad_operator = copy.deepcopy(self.spec)
        bad_operator["criteria"]["R5"]["required_groups"][0]["expression"] = {
            "op": "xor",
            "args": ["R5.P.RIGID_CRATON_SINCE_ACQUISITION"],
        }
        with self.assertRaisesRegex(CriterionSpecValidationError, "op is invalid"):
            validate_criterion_spec(bad_operator)

        cycle = copy.deepcopy(self.spec)
        groups = cycle["criteria"]["R5"]["required_groups"]
        by_id(groups, "group_id", "R5.G.RIGID_FRAME")["expression"] = "R5.G.STRUCTURAL_BASIS"
        with self.assertRaisesRegex(CriterionSpecValidationError, "cyclic required-group logic"):
            validate_criterion_spec(cycle)

    def test_validation_rejects_unsourced_or_wrongly_sourced_specification(self):
        no_artifact = copy.deepcopy(self.spec)
        del no_artifact["source_artifact"]
        with self.assertRaisesRegex(CriterionSpecValidationError, "source_artifact"):
            validate_criterion_spec(no_artifact)

        no_source_refs = copy.deepcopy(self.spec)
        no_source_refs["criteria"]["R7"]["propositions"][0]["source_refs"] = []
        with self.assertRaisesRegex(CriterionSpecValidationError, "non-empty string list"):
            validate_criterion_spec(no_source_refs)

        wrong_criterion_source = copy.deepcopy(self.spec)
        wrong_criterion_source["criteria"]["R7"]["propositions"][0]["source_refs"] = [
            "SRC.R5.P1"
        ]
        with self.assertRaisesRegex(CriterionSpecValidationError, "another criterion"):
            validate_criterion_spec(wrong_criterion_source)

    def test_loading_rejects_invalid_artifact_on_disk(self):
        invalid = copy.deepcopy(self.spec)
        invalid["source_artifact"]["sha256"] = "not-a-digest"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(CriterionSpecValidationError, "SHA-256"):
                load_criterion_spec(path)


if __name__ == "__main__":
    unittest.main()
