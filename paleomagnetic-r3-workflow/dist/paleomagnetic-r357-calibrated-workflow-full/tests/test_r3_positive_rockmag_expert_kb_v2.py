import json
from pathlib import Path
import re
import tempfile
import unittest

from openpyxl import Workbook

from scripts.train_r3_positive_rockmag_expert_kb_v2 import (
    HOLDOUT_FRACTION,
    SPLIT_SALT,
    build_artifact,
    classify_rockmag_value,
    collect_training_rockmag_rows,
    holdout_bucket,
    induce_statistics,
    partition_target_rows,
    validate_artifact_boundary,
)


class Cell:
    def __init__(self, value):
        self.value = value


class ForbiddenCell:
    @property
    def value(self):
        raise AssertionError("forbidden held-out cell was opened")


def identity_for_partition(holdout: bool, offset: int = 0) -> str:
    found = 0
    for index in range(100000):
        value = f"paper-{index}"
        if (holdout_bucket(value) < HOLDOUT_FRACTION) is holdout:
            if found == offset:
                return value
            found += 1
    raise AssertionError("could not construct a deterministic partition identity")


def write_workbook(path: Path, header, rows) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(list(header))
    for row in rows:
        sheet.append(list(row))
    workbook.save(path)


class R3PositiveRockmagExpertKbV2Tests(unittest.TestCase):
    def test_split_opens_only_paper_identity_for_holdout_rows(self):
        self.assertEqual(SPLIT_SALT, "r3-rockmag-distillation-v1")
        self.assertEqual(HOLDOUT_FRACTION, 0.20)
        training = identity_for_partition(False)
        holdout = identity_for_partition(True)

        records, audit = partition_target_rows(
            ["REFNO", "RESULTNO", "R3"],
            [
                [Cell(training), Cell("train-record"), Cell(1)],
                [Cell(holdout), ForbiddenCell(), ForbiddenCell()],
            ],
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["label"], 1)
        self.assertEqual(audit["training"], {"rows": 1, "paper_groups": 1})
        self.assertEqual(audit["holdout"], {"rows": 1, "paper_groups": 1})

    def test_source_table_never_opens_nontraining_field_cells(self):
        values, audit = collect_training_rockmag_rows(
            ["RESULTNO", "ROCKMAG"],
            [
                [Cell("train-record"), Cell("IRM, no saturation, magnetite")],
                [Cell("held-record"), ForbiddenCell()],
                [Cell("unrelated-record"), ForbiddenCell()],
            ],
            {"train-record"},
        )

        self.assertEqual(
            values, {"train-record": ["IRM, no saturation, magnetite"]}
        )
        self.assertEqual(audit["training_field_cells_opened"], 1)
        self.assertEqual(audit["nontraining_field_cells_opened"], 0)

    def test_user_example_normalizes_to_method_diagnostic_and_carriers(self):
        result = classify_rockmag_value(
            "IRM (no sat. 1T), Js-T (HEM with minor MAGN)"
        )

        self.assertTrue(result["strong_chain"])
        self.assertEqual(result["evidence_class"], "method_diagnostic_carrier")
        self.assertIn("irm_acquisition", result["method_concepts"])
        self.assertIn("thermomagnetic_curve", result["method_concepts"])
        self.assertIn("saturation_behavior", result["diagnostic_concepts"])
        self.assertEqual(set(result["carrier_concepts"]), {"hematite", "magnetite"})

    def test_training_code_variants_map_to_safe_concepts(self):
        temperature = classify_rockmag_value(
            "SUCS-T, Tc=580, TMAGN, PYRH, GOEH"
        )
        coercivity = classify_rockmag_value("Hcr/Hc, Bcr/Bc, HYST, PSD")
        remanence_temperature = classify_rockmag_value("SIRM-T, LOW-T IRM")
        susceptibility_alias = classify_rockmag_value("SUSC vs. TEMP")

        self.assertIn(
            "temperature_dependent_susceptibility",
            temperature["method_concepts"],
        )
        self.assertIn("curie_temperature", temperature["diagnostic_concepts"])
        self.assertEqual(
            set(temperature["carrier_concepts"]),
            {"titanomagnetite", "pyrrhotite", "goethite"},
        )
        self.assertIn("coercivity_backfield", coercivity["method_concepts"])
        self.assertIn("coercivity_parameter", coercivity["diagnostic_concepts"])
        self.assertIn("domain_state", coercivity["diagnostic_concepts"])
        self.assertIn(
            "remanence_temperature_experiment",
            remanence_temperature["method_concepts"],
        )
        self.assertIn(
            "temperature_dependent_susceptibility",
            susceptibility_alias["method_concepts"],
        )

    def test_tub_susc_an_and_bare_carrier_are_nonqualifying(self):
        tub = classify_rockmag_value("Tub=150-450C for N, Tub>600C for R")
        susc = classify_rockmag_value("SUSC")
        anisotropy = classify_rockmag_value("AN")
        bundle = classify_rockmag_value("SUSC / AN")
        carrier = classify_rockmag_value("MAGN")

        self.assertIn(
            "tub_unblocking_interval_without_independent_method",
            tub["nonqualifying_classes"],
        )
        self.assertIn("bare_susceptibility", susc["nonqualifying_classes"])
        self.assertIn("bare_anisotropy", anisotropy["nonqualifying_classes"])
        self.assertIn("bare_susc_an_bundle", bundle["nonqualifying_classes"])
        self.assertIn("carrier_without_method", carrier["nonqualifying_classes"])
        for result in (tub, susc, anisotropy, bundle, carrier):
            self.assertFalse(result["strong_chain"])

        qualified = classify_rockmag_value("IRM, Tub=580C, MAGN")
        self.assertTrue(qualified["strong_chain"])
        self.assertNotIn(
            "tub_unblocking_interval_without_independent_method",
            qualified["nonqualifying_classes"],
        )

    def test_statistics_emit_only_aggregate_frequency_and_cooccurrence(self):
        records = [
            {"paper_identity": "paper-a", "record_identity": "a", "label": 1},
            {"paper_identity": "paper-a", "record_identity": "b", "label": 1},
            {"paper_identity": "paper-b", "record_identity": "c", "label": 0},
            {"paper_identity": "paper-c", "record_identity": "d", "label": 1},
        ]
        values = {
            "a": ["IRM, no saturation, MAGN"],
            "b": ["IRM, no saturation, MAGN"],
            "c": ["IRM, no saturation, MAGN"],
            "d": ["SUSC"],
        }

        result = induce_statistics(records, values, minimum_cooccurrence=2)
        irm = result["concept_statistics"]["method:irm_acquisition"]
        bare_susc = result["nonqualifying_statistics"]["bare_susceptibility"]
        edge = next(
            item
            for item in result["positive_concept_cooccurrences"]
            if {
                item["source_concept"],
                item["target_concept"],
            }
            == {"method:irm_acquisition", "carrier:magnetite"}
        )

        self.assertEqual(irm["row_level"]["positive_rows"], 2)
        self.assertEqual(irm["row_level"]["negative_rows"], 1)
        self.assertAlmostEqual(irm["row_level"]["positive_precision"], 2 / 3)
        self.assertEqual(
            irm["paper_deduplicated"]["label_conflict_papers"], 0
        )
        self.assertEqual(bare_susc["row_level"]["positive_rows"], 1)
        self.assertEqual(edge["positive_row_count"], 2)
        self.assertEqual(edge["positive_paper_count"], 1)

        serialized = json.dumps(result, sort_keys=True)
        for secret in ("paper-a", "paper-b", "paper-c", '"a"', '"b"', '"c"', '"d"'):
            self.assertNotIn(secret, serialized)

    def test_holdout_supervision_changes_do_not_change_artifact(self):
        training = identity_for_partition(False)
        holdout = identity_for_partition(True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets_a = root / "targets-a.xlsx"
            targets_b = root / "targets-b.xlsx"
            source_a = root / "source-a.xlsx"
            source_b = root / "source-b.xlsx"
            target_header = ["REFNO", "RESULTNO", "ROCKNAME", "R3"]
            write_workbook(
                targets_a,
                target_header,
                [
                    [training, "train-secret", "train-target", 1],
                    [holdout, "holdout-secret", "holdout-target-a", 0],
                ],
            )
            write_workbook(
                targets_b,
                target_header,
                [
                    [training, "train-secret", "train-target", 1],
                    [holdout, "changed-held-secret", "holdout-target-b", 1],
                ],
            )
            source_header = ["RESULTNO", "ROCKMAG"]
            write_workbook(
                source_a,
                source_header,
                [
                    ["train-secret", "IRM, no saturation, magnetite"],
                    ["holdout-secret", "held-field-secret-a"],
                ],
            )
            write_workbook(
                source_b,
                source_header,
                [
                    ["train-secret", "IRM, no saturation, magnetite"],
                    ["changed-held-secret", "held-field-secret-b"],
                ],
            )

            artifact_a = build_artifact(targets_a, source_a)
            artifact_b = build_artifact(targets_b, source_b)

        self.assertEqual(artifact_a, artifact_b)

    def test_artifact_contract_excludes_identities_raw_values_and_answer_maps(self):
        training = identity_for_partition(False)
        holdout = identity_for_partition(True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = root / "targets.xlsx"
            source = root / "source.xlsx"
            write_workbook(
                targets,
                ["REFNO", "RESULTNO", "ROCKNAME", "R3"],
                [
                    [training, "train-record-secret", "train-target-secret", 1],
                    [holdout, "held-record-secret", "held-target-secret", 0],
                ],
            )
            write_workbook(
                source,
                ["RESULTNO", "ROCKMAG"],
                [
                    ["train-record-secret", "IRM, no saturation, magnetite"],
                    ["held-record-secret", "held-field-secret"],
                ],
            )

            artifact = build_artifact(targets, source)

        validate_artifact_boundary(artifact)
        serialized = json.dumps(artifact, ensure_ascii=False, sort_keys=True)
        self.assertTrue(artifact["provenance"]["label_supervision"])
        self.assertFalse(artifact["provenance"]["held_out_labels_opened"])
        self.assertFalse(artifact["provenance"]["held_out_field_cells_opened"])
        self.assertFalse(artifact["provenance"]["raw_field_values_retained"])
        self.assertFalse(artifact["runtime_policy"]["contains_record_answer_mapping"])
        for forbidden in (
            "REFNO",
            "RESULTNO",
            "ROCKNAME",
            training,
            holdout,
            "train-record-secret",
            "held-record-secret",
            "train-target-secret",
            "held-target-secret",
            "IRM, no saturation, magnetite",
            "held-field-secret",
        ):
            self.assertNotIn(forbidden, serialized)

        for taxonomy_name in (
            "method_taxonomy",
            "diagnostic_taxonomy",
            "carrier_taxonomy",
        ):
            for concept in artifact[taxonomy_name]:
                self.assertEqual(
                    concept["runtime_original_text_patterns"],
                    concept["original_text_patterns"],
                )
                for pattern in concept["field_patterns"]:
                    re.compile(pattern, re.I)
                for pattern in concept["original_text_patterns"]:
                    re.compile(pattern, re.I | re.S)
                for pattern in concept["runtime_original_text_patterns"]:
                    re.compile(pattern, re.I | re.S)


if __name__ == "__main__":
    unittest.main()
