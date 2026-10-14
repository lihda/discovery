import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.audit_r357_scoring_boundary import (
    audit_scoring_boundary,
    main as audit_main,
)
from scripts.build_strict_identity_manifest import (
    ManifestValidationError,
    build_manifest,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StrictIdentityManifestTests(unittest.TestCase):
    def test_manifest_reads_only_identity_allowlist_and_audits_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.xlsx"
            output = root / "identity.xlsx"
            audit_path = root / "identity.audit.json"
            pd.DataFrame(
                [
                    {
                        "REFNO": "100",
                        "RESULTNO": "1",
                        "ROCKNAME": "Alpha Formation",
                        "ROCKUNITNO": "RU-1",
                        "R1": 0,
                        "R3": 1,
                        "R5": 0,
                        "R7": 1,
                    },
                    {
                        "REFNO": "100",
                        "RESULTNO": "2",
                        "ROCKNAME": "Beta Formation",
                        "ROCKUNITNO": "RU-2",
                        "R1": 1,
                        "R3": 0,
                        "R5": 1,
                        "R7": 0,
                    },
                ]
            ).to_excel(source, index=False)

            report = build_manifest(
                source,
                output,
                audit_path,
                include_rockunitno=True,
            )

            identity = pd.read_excel(output, dtype=str, keep_default_na=False)
            self.assertEqual(
                identity.columns.tolist(),
                ["REFNO", "RESULTNO", "ROCKNAME", "ROCKUNITNO"],
            )
            self.assertEqual(report["input"]["sha256"], sha256(source))
            self.assertEqual(report["output"]["sha256"], sha256(output))
            self.assertEqual(
                report["input"]["columns_read"],
                ["REFNO", "RESULTNO", "ROCKNAME", "ROCKUNITNO"],
            )
            self.assertEqual(report["criterion_value_columns_read"], [])
            self.assertFalse(report["guarantees"]["r1_r7_values_read"])
            self.assertFalse(report["guarantees"]["rockunitno_used_for_scoring"])
            self.assertEqual(
                json.loads(audit_path.read_text(encoding="utf-8")), report
            )

    def test_manifest_rejects_empty_and_duplicate_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty_source = root / "empty.xlsx"
            duplicate_source = root / "duplicate.xlsx"
            pd.DataFrame(
                [{"REFNO": "100", "RESULTNO": "", "ROCKNAME": "Alpha"}]
            ).to_excel(empty_source, index=False)
            pd.DataFrame(
                [
                    {"REFNO": "100", "RESULTNO": "1", "ROCKNAME": "Alpha"},
                    {"REFNO": "101", "RESULTNO": "1", "ROCKNAME": "Beta"},
                ]
            ).to_excel(duplicate_source, index=False)

            for source in (empty_source, duplicate_source):
                with self.subTest(source=source.name):
                    with self.assertRaises(ManifestValidationError):
                        build_manifest(
                            source,
                            root / f"{source.stem}.out.xlsx",
                            root / f"{source.stem}.audit.json",
                        )


class ScoringBoundaryAuditTests(unittest.TestCase):
    def _write_prediction_xlsx(self, path: Path, indicator: str, payload: dict):
        pd.DataFrame(
            [
                {
                    "REFNO": "100",
                    "RESULTNO": "1",
                    "TARGET": "Alpha Formation",
                    indicator: payload[indicator],
                    f"{indicator}_RESULT_JSON": json.dumps(payload),
                }
            ]
        ).to_excel(path, index=False)

    def test_legal_predictions_and_provenanced_record_independent_kb_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r3_path = root / "r3.xlsx"
            r7_path = root / "r7.jsonl"
            kb_path = root / "expert_knowledge.json"
            self._write_prediction_xlsx(
                r3_path,
                "R3",
                {
                    "R3": 1,
                    "source_metadata": {},
                    "runtime_scoring_sources": [
                        "strict identity manifest",
                        "target-bound original paper text",
                        "expert knowledge artifact with source provenance",
                    ],
                },
            )
            r7_path.write_text(
                json.dumps(
                    {
                        "REFNO": "100",
                        "RESULTNO": "1",
                        "R7": 1,
                        "R7_RESULT_JSON": {
                            "R7": 1,
                            "runtime_scoring_sources": [
                                "strict identity manifest",
                                "original paper page text",
                            ],
                            "source_metadata": None,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            kb_path.write_text(
                json.dumps(
                    {
                        "version": "r357_expert_knowledge_v1",
                        "provenance": {
                            "source": "published criterion definitions",
                            "citation": "Example et al. (2026)",
                        },
                        "label_supervision": False,
                        "rules": [
                            {
                                "criterion": "R3",
                                "pattern": "hysteresis|thermomagnetic",
                                "effect": "carrier investigation evidence",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = audit_scoring_boundary([r3_path, r7_path], [kb_path])

            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["violations"], [])

    def test_forbidden_prediction_sources_metadata_and_r5_model_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "r5.xlsx"
            self._write_prediction_xlsx(
                prediction,
                "R5",
                {
                    "R5": 0,
                    "runtime_scoring_sources": [
                        "PMAGRESULT and ROCKUNIT metadata",
                        "evaluation labels from a previous prediction workbook",
                    ],
                    "source_metadata": {"PMAGRESULT.TILT": "100"},
                    "model_version": "r5_paper_text_model_v1",
                },
            )

            report = audit_scoring_boundary([prediction], [])
            codes = {violation["code"] for violation in report["violations"]}

            self.assertFalse(report["passed"])
            self.assertIn("forbidden_runtime_scoring_source", codes)
            self.assertIn("nonempty_source_metadata", codes)
            self.assertIn("forbidden_r5_calibration_model", codes)

    def test_kb_identity_mapping_supervision_and_missing_provenance_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "leaky_knowledge.json"
            artifact.write_text(
                json.dumps(
                    {
                        "label_supervision": True,
                        "REFNO": "100",
                        "answers_by_resultno": {
                            "1": {"R5": 0},
                            "2": {"R5": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = audit_scoring_boundary([], [artifact])
            codes = {violation["code"] for violation in report["violations"]}

            self.assertFalse(report["passed"])
            self.assertIn("missing_knowledge_provenance", codes)
            self.assertIn("forbidden_identity_key_in_knowledge", codes)
            self.assertIn("record_level_mapping_in_knowledge", codes)
            self.assertIn("label_supervision_enabled", codes)

    def test_cli_writes_machine_report_and_returns_nonzero_for_violation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "r3.json"
            report_path = root / "boundary-report.json"
            prediction.write_text(json.dumps({"R3": 1}), encoding="utf-8")

            exit_code = audit_main(
                [
                    "--prediction",
                    str(prediction),
                    "--report",
                    str(report_path),
                ]
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(
                report["violations"][0]["code"],
                "missing_runtime_scoring_sources",
            )


if __name__ == "__main__":
    unittest.main()
