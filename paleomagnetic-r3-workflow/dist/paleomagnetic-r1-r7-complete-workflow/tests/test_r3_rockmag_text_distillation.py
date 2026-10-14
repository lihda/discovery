import json
from pathlib import Path
import re
import tempfile
import unittest

from openpyxl import Workbook

from scripts.train_r3_rockmag_text_distillation import (
    HOLDOUT_FRACTION,
    SPLIT_SALT,
    build_artifact,
    classify_rockmag_value,
    collect_training_rockmag_rows,
    holdout_bucket,
    induce_training_statistics,
    partition_target_rows,
    wilson_lower_bound,
)


class Cell:
    def __init__(self, value):
        self.value = value


class ForbiddenCell:
    @property
    def value(self):
        raise AssertionError("forbidden held-out cell was dereferenced")


def identity_for_partition(holdout: bool) -> str:
    for index in range(10000):
        value = f"paper-{index}"
        if (holdout_bucket(value) < HOLDOUT_FRACTION) is holdout:
            return value
    raise AssertionError("could not construct deterministic test identity")


def write_workbook(path: Path, header, rows) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(list(header))
    for row in rows:
        sheet.append(list(row))
    workbook.save(path)


class R3RockmagTextDistillationTests(unittest.TestCase):
    def test_split_is_fixed_deterministic_and_grouped_by_paper(self):
        self.assertEqual(SPLIT_SALT, "r3-rockmag-distillation-v1")
        self.assertEqual(HOLDOUT_FRACTION, 0.20)
        paper = "same-paper"
        self.assertEqual(holdout_bucket(paper), holdout_bucket(paper))

        train = identity_for_partition(False)
        held_out = identity_for_partition(True)
        rows = [
            [Cell(train), Cell("train-a"), Cell("Alpha"), Cell(1)],
            [Cell(train), Cell("train-b"), Cell("Beta"), Cell(0)],
            [Cell(held_out), ForbiddenCell(), ForbiddenCell(), ForbiddenCell()],
        ]

        records, audit = partition_target_rows(
            ["REFNO", "RESULTNO", "ROCKNAME", "R3"],
            rows,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual({row["paper_identity"] for row in records}, {train})
        self.assertEqual(audit["training"], {"rows": 2, "paper_groups": 1})
        self.assertEqual(audit["holdout"], {"rows": 1, "paper_groups": 1})

    def test_source_table_never_dereferences_nontraining_rockmag(self):
        values, audit = collect_training_rockmag_rows(
            ["RESULTNO", "ROCKMAG"],
            [
                [Cell("train-a"), Cell("IRM (no sat. 1T), Js-T (HEM with minor MAGN)")],
                [Cell("held-out-a"), ForbiddenCell()],
                [Cell("unrelated-a"), ForbiddenCell()],
            ],
            {"train-a"},
        )

        self.assertEqual(
            values,
            {"train-a": ["IRM (no sat. 1T), Js-T (HEM with minor MAGN)"]},
        )
        self.assertEqual(audit["training_rows_with_field_cell_access"], 1)
        self.assertEqual(audit["nonempty_training_field_values"], 1)

    def test_user_positive_example_maps_to_method_diagnostic_and_carrier(self):
        result = classify_rockmag_value(
            "IRM (no sat. 1T), Js-T (HEM with minor MAGN)"
        )

        self.assertTrue(result["reportable_template"])
        self.assertIn("irm_acquisition", result["method_concepts"])
        self.assertIn("thermomagnetic_curve", result["method_concepts"])
        self.assertIn("saturation_behavior", result["diagnostic_concepts"])
        self.assertEqual(set(result["carrier_concepts"]), {"hematite", "magnetite"})

    def test_tub_and_other_standalone_values_do_not_trigger_positive(self):
        tub = classify_rockmag_value("Tub=150-450C for N, Tub>600C for R")
        bare_susc = classify_rockmag_value("SUSC")
        bare_an = classify_rockmag_value("AN")
        bare_carrier = classify_rockmag_value("MAGN")

        for result in (tub, bare_susc, bare_an, bare_carrier):
            self.assertFalse(result["reportable_template"])
        self.assertIn("demagnetization_unblocking_interval", tub["standalone_exclusions"])
        self.assertIn("bare_susceptibility", bare_susc["standalone_exclusions"])
        self.assertIn("bare_anisotropy_code", bare_an["standalone_exclusions"])
        self.assertIn("bare_carrier_name", bare_carrier["standalone_exclusions"])

    def test_temperature_susceptibility_and_ams_require_nonbare_forms(self):
        susceptibility = classify_rockmag_value("SUSC-T, CT 580, MAGN")
        fabric = classify_rockmag_value("AMS, domain state PSD")

        self.assertTrue(susceptibility["reportable_template"])
        self.assertIn(
            "temperature_dependent_susceptibility",
            susceptibility["method_concepts"],
        )
        self.assertTrue(fabric["reportable_template"])
        self.assertIn("magnetic_fabric", fabric["method_concepts"])

    def test_statistics_include_row_and_paper_deduplicated_precision(self):
        records = [
            {"paper_identity": "p1", "record_identity": "a", "label": 1},
            {"paper_identity": "p1", "record_identity": "b", "label": 1},
            {"paper_identity": "p2", "record_identity": "c", "label": 0},
        ]
        values = {
            "a": ["IRM (no sat. 1T)"],
            "b": ["IRM (no sat. 1T)"],
            "c": ["IRM (no sat. 1T)"],
        }

        result = induce_training_statistics(records, values)
        row = result["row_level_statistics"]["method:irm_acquisition"]
        paper = result["paper_deduplicated_statistics"]["method:irm_acquisition"]

        self.assertEqual(row["observations"], 3)
        self.assertEqual(row["positive"], 2)
        self.assertAlmostEqual(row["precision"], 2 / 3, places=7)
        self.assertEqual(paper["observations"], 2)
        self.assertEqual(paper["positive"], 1)
        self.assertEqual(paper["negative"], 1)
        self.assertEqual(paper["precision"], 0.5)
        self.assertGreater(row["wilson_lower_bound_95"], 0.0)
        evaluation = result["training_rule_evaluation"]
        self.assertEqual(
            evaluation["rockmag_nonempty_baseline"]["confusion_matrix"],
            {
                "true_negative": 0,
                "false_positive": 1,
                "false_negative": 0,
                "true_positive": 2,
            },
        )
        self.assertFalse(evaluation["independent_generalization_claim"])

    def test_conflicting_rows_in_one_paper_are_excluded_from_paper_precision(self):
        records = [
            {"paper_identity": "p1", "record_identity": "a", "label": 1},
            {"paper_identity": "p1", "record_identity": "b", "label": 0},
        ]
        values = {"a": ["IRM (no sat. 1T)"], "b": ["IRM (no sat. 1T)"]}

        result = induce_training_statistics(records, values)
        paper = result["paper_deduplicated_statistics"]["method:irm_acquisition"]

        self.assertEqual(paper["observations"], 0)
        self.assertEqual(paper["conflict_excluded"], 1)
        self.assertIsNone(paper["precision"])

    def test_wilson_lower_bound_is_conservative(self):
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)
        self.assertLess(wilson_lower_bound(10, 10), 1.0)
        self.assertGreater(wilson_lower_bound(10, 10), wilson_lower_bound(2, 2))
        self.assertLess(wilson_lower_bound(5, 10), 0.5)

    def test_artifact_is_aggregate_honest_and_contains_safe_runtime_patterns(self):
        train = identity_for_partition(False)
        held_out = identity_for_partition(True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = root / "targets.xlsx"
            source = root / "source.xlsx"
            write_workbook(
                targets,
                ["REFNO", "RESULTNO", "ROCKNAME", "R3"],
                [
                    [train, "train-result-secret", "training-target-secret", 1],
                    [held_out, "holdout-result-secret", "holdout-target-secret", 0],
                ],
            )
            write_workbook(
                source,
                ["RESULTNO", "ROCKMAG"],
                [
                    ["train-result-secret", "IRM (no sat. 1T), Js-T (HEM with minor MAGN)"],
                    ["holdout-result-secret", "holdout-rockmag-secret"],
                ],
            )

            artifact = build_artifact(targets, source)

            self.assertEqual(
                artifact["decision_policy"]["rockmag_proxy_policy"],
                "distilled_method_or_self_authenticating_result",
            )
            self.assertEqual(
                artifact["decision_policy"]["identity_granularity"],
                "canonical_target",
            )

        serialized = json.dumps(artifact, ensure_ascii=False, sort_keys=True)
        self.assertEqual(artifact["artifact_class"], "historically_supervised_distillation")
        self.assertTrue(artifact["provenance"]["label_supervision"])
        self.assertEqual(
            artifact["provenance"]["identity_use_offline"]["paper_identity"],
            "paper-group split only",
        )
        self.assertFalse(artifact["split"]["held_out_supervision_accessed"])
        self.assertEqual(artifact["split"]["salt"], SPLIT_SALT)
        self.assertEqual(artifact["split"]["holdout_fraction"], 0.20)
        self.assertEqual(len(artifact["provenance"]["input_sha256"]["target_manifest_source"]), 64)
        self.assertEqual(len(artifact["provenance"]["input_sha256"]["source_metadata_table"]), 64)
        self.assertEqual(artifact["split"]["identity_counts"]["training"]["rows"], 1)
        self.assertEqual(artifact["split"]["identity_counts"]["holdout"]["rows"], 1)
        self.assertEqual(
            artifact["training_statistics"]["aggregate_counts"][
                "rows_matching_reportable_template"
            ],
            1,
        )

        for forbidden in (
            "REFNO",
            "RESULTNO",
            "ROCKNAME",
            train,
            held_out,
            "train-result-secret",
            "holdout-result-secret",
            "training-target-secret",
            "holdout-target-secret",
            "IRM (no sat. 1T), Js-T (HEM with minor MAGN)",
            "holdout-rockmag-secret",
        ):
            self.assertNotIn(forbidden, serialized)

        runtime_patterns = [
            pattern
            for taxonomy_name in (
                "method_taxonomy",
                "diagnostic_taxonomy",
                "carrier_taxonomy",
            )
            for concept in artifact[taxonomy_name]
            for pattern in concept["runtime_original_text_patterns"]
        ]
        for pattern in runtime_patterns:
            re.compile(pattern, re.I | re.S)
        self.assertFalse(any(pattern in {r"\bOP\b", r"\bTS\b", r"\bTM\b", r"\bAN\b"} for pattern in runtime_patterns))
        irm = next(
            concept
            for concept in artifact["method_taxonomy"]
            if concept["concept_id"] == "irm_acquisition"
        )
        self.assertEqual(
            irm["training_statistics"]["row_level"]["precision"],
            1.0,
        )
        self.assertGreater(
            irm["training_statistics"]["row_level"]["wilson_lower_bound_95"],
            0.0,
        )
        self.assertFalse(artifact["runtime_policy"]["source_metadata_table_access"])
        self.assertFalse(artifact["runtime_policy"]["supervised_field_access"])


if __name__ == "__main__":
    unittest.main()
