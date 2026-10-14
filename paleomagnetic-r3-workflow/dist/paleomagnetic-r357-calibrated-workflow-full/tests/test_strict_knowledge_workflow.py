import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

import pandas as pd

from scripts.evaluate_r357_strict_predictions import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_HOLDOUT_SALT,
    holdout_bucket,
    main as evaluate_main,
)
from scripts.run_r357_strict_knowledge_workflow import (
    DEFAULT_CRITERION_SPEC,
    ROOT,
    StrictInputError,
    main as strict_main,
    validate_identity_manifest,
    validate_page_cache,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_cache(path: Path, refno: str = "100") -> None:
    record = {
        "REFNO": refno,
        "source": {
            "refno": refno,
            "source_kind": "synthetic_original_paper",
            "parsed_path": None,
            "pdf_path": None,
            "page_count": 1,
            "text_length": 193,
            "cache_sha256": "fixture",
            "source_quality_flags": [],
        },
        "pages": [
            {
                "page": 1,
                "source": "synthetic_original_paper",
                "text": (
                    "Alpha Formation samples were examined using hysteresis loops, "
                    "IRM acquisition curves, and thermomagnetic experiments. "
                    "Magnetite is interpreted as the principal remanence carrier."
                ),
                "text_sha256": "fixture",
                "section_hints": ["Rock magnetic results"],
            }
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


class StrictScoringWorkerTests(unittest.TestCase):
    def test_worker_physically_blocks_an_unlisted_spreadsheet_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            allowed = root / "allowed.json"
            forbidden = root / "historical_evaluation.xlsx"
            attacker = root / "reader.py"
            allowed.write_text("{}\n", encoding="utf-8")
            forbidden.write_bytes(b"not actually an xlsx")
            attacker.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "Path(sys.argv[1]).read_bytes()\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/strict_scoring_worker.py"),
                    "--script",
                    str(attacker),
                    "--allow-read",
                    str(allowed),
                    "--write-root",
                    str(output),
                    "--",
                    str(forbidden),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("blocked undeclared data read", result.stderr)


class StrictKnowledgeWorkflowTests(unittest.TestCase):
    def test_manifest_and_cache_contract_reject_labels_and_missing_refnos(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labelled = root / "labelled.xlsx"
            cache = root / "cache.jsonl"
            pd.DataFrame(
                [
                    {
                        "REFNO": "100",
                        "RESULTNO": "1",
                        "ROCKNAME": "Alpha Formation",
                        "R3": 1,
                    }
                ]
            ).to_excel(labelled, index=False)
            write_cache(cache, refno="101")

            with self.assertRaises(StrictInputError):
                validate_identity_manifest(labelled)
            with self.assertRaises(StrictInputError):
                validate_page_cache(cache, ["100"])

    def test_end_to_end_scoring_then_separate_evaluation_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = root / "identity.xlsx"
            cache = root / "page_cache.jsonl"
            predictions = root / "predictions"
            labels = root / "labels.xlsx"
            evaluation = root / "evaluation.json"
            knowledge = ROOT / "data/knowledge/r357_expert_kb_v1.json"
            pd.DataFrame(
                [
                    {
                        "REFNO": "100",
                        "RESULTNO": "1",
                        "ROCKNAME": "Alpha Formation",
                    }
                ]
            ).to_excel(identity, index=False)
            write_cache(cache)

            exit_code = strict_main(
                [
                    "--identity-manifest",
                    str(identity),
                    "--page-cache",
                    str(cache),
                    "--knowledge-base",
                    str(knowledge),
                    "--output-dir",
                    str(predictions),
                ]
            )

            self.assertEqual(exit_code, 0)
            output_paths = {
                "R3": predictions / "r3_predictions.jsonl",
                "R5": predictions / "r5_predictions.xlsx",
                "R7": predictions / "r7_predictions.xlsx",
            }
            before = {indicator: sha256(path) for indicator, path in output_paths.items()}
            run_manifest = json.loads(
                (predictions / "strict_run_manifest.json").read_text(encoding="utf-8")
            )
            boundary = json.loads(
                (predictions / "scoring_boundary_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_manifest["evaluation"]["status"], "NOT_RUN")
            self.assertFalse(
                run_manifest["scoring_contract"]["labels_accepted_by_this_command"]
            )
            self.assertEqual(boundary["status"], "PASS")
            spec_input = run_manifest["inputs"]["criterion_spec"]
            self.assertEqual(spec_input["path"], str(DEFAULT_CRITERION_SPEC.resolve()))
            self.assertEqual(spec_input["sha256"], sha256(DEFAULT_CRITERION_SPEC))
            self.assertEqual(spec_input["spec_version"], "r357_criterion_spec_v2")
            self.assertEqual(
                spec_input["source_sha256"],
                "8433f968a4658118dc43a060114b8d2a6a0ca0cbf6ed82ee006d924d970b18e5",
            )
            self.assertTrue(
                run_manifest["scoring_contract"]["criterion_spec_required"]
            )
            self.assertIn(
                str(DEFAULT_CRITERION_SPEC.resolve()),
                boundary["inputs"]["knowledge_artifacts"],
            )

            pd.DataFrame(
                [
                    {
                        "REFNO": "100",
                        "RESULTNO": "1",
                        "R3": 1,
                        "R5": 1,
                        "R7": 1,
                    }
                ]
            ).to_excel(labels, index=False)
            eval_exit = evaluate_main(
                [
                    "--r3",
                    str(output_paths["R3"]),
                    "--r5",
                    str(output_paths["R5"]),
                    "--r7",
                    str(output_paths["R7"]),
                    "--labels",
                    str(labels),
                    "--output",
                    str(evaluation),
                ]
            )

            self.assertEqual(eval_exit, 0)
            after = {indicator: sha256(path) for indicator, path in output_paths.items()}
            self.assertEqual(after, before)
            report = json.loads(evaluation.read_text(encoding="utf-8"))
            contract = report["evaluation_contract"]
            self.assertTrue(contract["prediction_hashes_recorded_before_labels_opened"])
            self.assertTrue(
                contract["prediction_hash_receipt_written_before_labels_opened"]
            )
            self.assertFalse(contract["scoring_code_called"])
            self.assertFalse(contract["prediction_files_modified"])
            receipt = Path(report["prediction_hash_receipt"]["path"])
            self.assertTrue(receipt.is_file())
            self.assertEqual(report["prediction_hash_receipt"]["sha256"], sha256(receipt))
            holdout = report["refno_holdout"]
            self.assertEqual(holdout["salt"], DEFAULT_HOLDOUT_SALT)
            self.assertEqual(holdout["fraction"], DEFAULT_HOLDOUT_FRACTION)
            self.assertEqual(
                holdout["holdout_rows"],
                int(holdout_bucket("100", DEFAULT_HOLDOUT_SALT) < DEFAULT_HOLDOUT_FRACTION),
            )
            for indicator in ("R3", "R5", "R7"):
                self.assertIn("confusion_matrix", holdout["indicators"][indicator])
            for indicator in ("R3", "R5", "R7"):
                metrics = report["indicators"][indicator]
                for key in (
                    "accuracy",
                    "coverage",
                    "known_accuracy",
                    "macro_f1",
                    "f1_0",
                    "f1_1",
                    "confusion_matrix",
                ):
                    self.assertIn(key, metrics)

    def test_tampered_or_illegal_criterion_spec_fails_before_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = root / "identity.xlsx"
            cache = root / "page_cache.jsonl"
            knowledge = ROOT / "data/knowledge/r357_expert_kb_v1.json"
            pd.DataFrame(
                [
                    {
                        "REFNO": "100",
                        "RESULTNO": "1",
                        "ROCKNAME": "Alpha Formation",
                    }
                ]
            ).to_excel(identity, index=False)
            write_cache(cache)
            original = json.loads(DEFAULT_CRITERION_SPEC.read_text(encoding="utf-8"))

            invalid_specs = []
            tampered = json.loads(json.dumps(original))
            tampered["source_artifact"]["sha256"] = "tampered"
            invalid_specs.append(tampered)
            illegal = json.loads(json.dumps(original))
            illegal["provenance"]["label_supervision"] = True
            invalid_specs.append(illegal)

            for index, payload in enumerate(invalid_specs):
                with self.subTest(index=index):
                    spec = root / f"invalid_spec_{index}.json"
                    output = root / f"predictions_{index}"
                    spec.write_text(json.dumps(payload), encoding="utf-8")
                    errors = io.StringIO()
                    with redirect_stderr(errors):
                        exit_code = strict_main(
                            [
                                "--identity-manifest",
                                str(identity),
                                "--page-cache",
                                str(cache),
                                "--knowledge-base",
                                str(knowledge),
                                "--criterion-spec",
                                str(spec),
                                "--output-dir",
                                str(output),
                            ]
                        )

                    self.assertEqual(exit_code, 2)
                    self.assertIn("Criterion spec failed validation", errors.getvalue())
                    self.assertFalse(output.exists())

    def test_knowledge_base_must_declare_the_validated_criterion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = root / "identity.xlsx"
            cache = root / "page_cache.jsonl"
            knowledge = root / "mismatched_knowledge.json"
            predictions = root / "predictions"
            pd.DataFrame(
                [{"REFNO": "100", "RESULTNO": "1", "ROCKNAME": "Alpha Formation"}]
            ).to_excel(identity, index=False)
            write_cache(cache)
            document = json.loads(
                (ROOT / "data/knowledge/r357_expert_kb_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            document["provenance"]["criterion_spec"]["source_sha256"] = "0" * 64
            knowledge.write_text(json.dumps(document), encoding="utf-8")

            exit_code = strict_main(
                [
                    "--identity-manifest",
                    str(identity),
                    "--page-cache",
                    str(cache),
                    "--knowledge-base",
                    str(knowledge),
                    "--output-dir",
                    str(predictions),
                ]
            )

            self.assertEqual(exit_code, 2)
            self.assertFalse(predictions.exists())


if __name__ == "__main__":
    unittest.main()
