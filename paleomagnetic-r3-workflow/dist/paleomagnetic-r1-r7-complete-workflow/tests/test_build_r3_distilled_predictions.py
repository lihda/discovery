import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from openpyxl import Workbook

from scripts.build_r3_distilled_predictions import (
    DEFAULT_ARTIFACT,
    R3DistilledPredictionError,
    build_predictions,
    canonical_json_sha256,
)
from scripts.r3_rockmag_distilled_reasoner import R3RockmagDistilledReasoner


def write_manifest(path: Path, rows: list[tuple[str, str, str]], *, extra_header=None) -> None:
    workbook = Workbook()
    sheet = workbook.active
    header = ["REFNO", "RESULTNO", "ROCKNAME"]
    if extra_header is not None:
        header.append(extra_header)
    sheet.append(header)
    for row in rows:
        values = list(row)
        if extra_header is not None:
            values.append(1)
        sheet.append(values)
    workbook.save(path)


def text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cache_record(refno: str, text: str = "") -> dict:
    pages = []
    page_hashes = []
    if text:
        digest = text_hash(text)
        page_hashes.append(digest)
        pages.append(
            {
                "page": 1,
                "source": "synthetic",
                "text": text,
                "text_sha256": digest,
                "section_hints": ["Rock magnetism"],
            }
        )
    return {
        "REFNO": refno,
        "source": {
            "refno": refno,
            "source_kind": "synthetic" if pages else "missing",
            "parsed_path": None,
            "pdf_path": None,
            "page_count": len(pages),
            "text_length": len("\n".join(page["text"] for page in pages)),
            "cache_sha256": hashlib.sha256("".join(page_hashes).encode("ascii")).hexdigest(),
            "source_quality_flags": [] if pages else ["no_extractable_text"],
        },
        "pages": pages,
    }


def write_cache(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


class FakeReasoner:
    conflict = True

    def __init__(self, artifact):
        self.artifact_sha256 = canonical_json_sha256(artifact)

    @staticmethod
    def _track(decision: int, path: str) -> dict:
        return {
            "policy_decision": decision,
            "value": decision,
            "binary_value": decision,
            "outcome": "POSITIVE" if decision else "NEGATIVE",
            "decision_path": path,
            "evidence": {"selected": [{"quote": "Synthetic evidence."}]},
        }

    def evaluate(self, retrieval, **kwargs):
        normative = self._track(1, "FORMAL_CHAIN")
        proxy = self._track(0, "NO_ROCKMAG_TEMPLATE")
        conflict = self.conflict
        return {
            "schema_version": "fake_decision_v1",
            "engine_version": "fake_reasoner_v1",
            "artifact_sha256": self.artifact_sha256,
            "policy_decision": proxy["policy_decision"],
            "binary_compatibility_value": proxy["binary_value"],
            "normative_r3": normative,
            "rockmag_proxy_r3": proxy,
            "criterion_practice_conflict": conflict,
            "criterion_practice_comparison": {
                "status": "TRACK_INCONSISTENCY" if conflict else "AGREE",
                "normative_policy_decision": normative["policy_decision"],
                "rockmag_proxy_policy_decision": proxy["policy_decision"],
            },
            "reason_code": "NO_ROCKMAG_TEMPLATE",
            "manual_review": False,
            "rejected_evidence_counts": {"weak_only": 1},
        }


class InconsistentReasoner(FakeReasoner):
    conflict = False


class RecordingReasoner(R3RockmagDistilledReasoner):
    evaluations = []

    def evaluate(self, retrieval, **kwargs):
        type(self).evaluations.append(
            {
                "target": retrieval.get("target"),
                "multi_target": kwargs.get("multi_target"),
                "identity_ambiguous": kwargs.get("identity_ambiguous"),
                "scope_binding": retrieval.get("scope_binding"),
                "scope_binding": retrieval.get("scope_binding"),
            }
        )
        return super().evaluate(retrieval, **kwargs)


class R3DistilledPredictionBuilderTests(unittest.TestCase):
    def test_builds_hashed_dual_track_jsonl_from_strict_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            cache = root / "pages.jsonl"
            output = root / "predictions.jsonl"
            summary_path = root / "summary.json"
            write_manifest(manifest, [("10", "100", "Alpha Formation")])
            write_cache(
                cache,
                [
                    cache_record(
                        "10",
                        "IRM acquisition was performed on the Alpha Formation and "
                        "magnetite carries the characteristic remanence.",
                    )
                ],
            )

            summary = build_predictions(
                manifest,
                cache,
                DEFAULT_ARTIFACT,
                output,
                summary_json=summary_path,
                reasoner_factory=FakeReasoner,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual((row["REFNO"], row["RESULTNO"]), ("10", "100"))
            self.assertEqual(row["R3"], 0)
            self.assertEqual(row["OUTCOME"], "NEGATIVE")
            self.assertEqual(row["normative_r3"]["policy_decision"], 1)
            self.assertEqual(row["rockmag_proxy_r3"]["policy_decision"], 0)
            self.assertIs(row["criterion_practice_conflict"], True)
            self.assertEqual(
                row["criterion_practice_comparison"]["status"],
                "TRACK_INCONSISTENCY",
            )
            self.assertTrue(row["runtime_scoring_sources"])
            self.assertFalse(row["runtime_boundary"]["labels_read"])
            self.assertRegex(row["input_hashes"]["artifact_file_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(row["input_hashes"]["page_cache_record_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(summary["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(summary["criterion_practice_conflict_rows"], 1)
            self.assertEqual(
                summary["retrieval_policy"],
                {
                    "total_limit": 30,
                    "per_category_limit": 12,
                    "target_scope_policy": "exclusive_scope",
                },
            )
            self.assertEqual(json.loads(summary_path.read_text())["status"], "PASS")

    def test_rejects_identity_manifest_with_label_column(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            cache = root / "pages.jsonl"
            write_manifest(
                manifest,
                [("10", "100", "Alpha Formation")],
                extra_header="R3",
            )
            write_cache(cache, [cache_record("10")])

            with self.assertRaisesRegex(
                R3DistilledPredictionError,
                "exactly REFNO, RESULTNO, ROCKNAME",
            ):
                build_predictions(
                    manifest,
                    cache,
                    DEFAULT_ARTIFACT,
                    root / "predictions.jsonl",
                    reasoner_factory=FakeReasoner,
                )

    def test_rejects_missing_and_extra_cache_refnos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            cache = root / "pages.jsonl"
            write_manifest(
                manifest,
                [
                    ("10", "100", "Alpha Formation"),
                    ("20", "200", "Beta Formation"),
                ],
            )
            write_cache(cache, [cache_record("10"), cache_record("30")])

            with self.assertRaisesRegex(
                R3DistilledPredictionError,
                "missing REFNO=20; extra REFNO=30",
            ):
                build_predictions(
                    manifest,
                    cache,
                    DEFAULT_ARTIFACT,
                    root / "predictions.jsonl",
                    reasoner_factory=FakeReasoner,
                )

    def test_rejects_page_text_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            cache = root / "pages.jsonl"
            write_manifest(manifest, [("10", "100", "Alpha Formation")])
            record = cache_record("10", "Original text")
            record["pages"][0]["text"] = "Tampered text"
            write_cache(cache, [record])

            with self.assertRaisesRegex(
                R3DistilledPredictionError,
                "text_sha256 does not match",
            ):
                build_predictions(
                    manifest,
                    cache,
                    DEFAULT_ARTIFACT,
                    root / "predictions.jsonl",
                    reasoner_factory=FakeReasoner,
                )

    def test_rejects_inconsistent_reasoner_dual_track_result_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            cache = root / "pages.jsonl"
            output = root / "predictions.jsonl"
            write_manifest(manifest, [("10", "100", "Alpha Formation")])
            write_cache(cache, [cache_record("10", "IRM and magnetite were reported.")])

            with self.assertRaisesRegex(
                R3DistilledPredictionError,
                "criterion_practice_conflict is inconsistent",
            ):
                build_predictions(
                    manifest,
                    cache,
                    DEFAULT_ARTIFACT,
                    output,
                    reasoner_factory=InconsistentReasoner,
                )
            self.assertFalse(output.exists())

    def test_rejects_invalid_retrieval_limit_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            cache = root / "pages.jsonl"
            output = root / "predictions.jsonl"
            write_manifest(manifest, [("10", "100", "Alpha Formation")])
            write_cache(cache, [cache_record("10", "IRM was measured.")])

            with self.assertRaisesRegex(
                R3DistilledPredictionError,
                "per-category retrieval limit cannot exceed",
            ):
                build_predictions(
                    manifest,
                    cache,
                    DEFAULT_ARTIFACT,
                    output,
                    retrieval_limit=4,
                    per_category_limit=5,
                    reasoner_factory=FakeReasoner,
                )
            self.assertFalse(output.exists())

    def test_rejects_unknown_target_scope_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            cache = root / "pages.jsonl"
            output = root / "predictions.jsonl"
            write_manifest(manifest, [("10", "100", "Alpha Formation")])
            write_cache(cache, [cache_record("10", "IRM was measured.")])

            with self.assertRaisesRegex(
                R3DistilledPredictionError,
                "target_scope_policy",
            ):
                build_predictions(
                    manifest,
                    cache,
                    DEFAULT_ARTIFACT,
                    output,
                    target_scope_policy="same_page",
                    reasoner_factory=FakeReasoner,
                )
            self.assertFalse(output.exists())

    def test_duplicate_same_target_rows_share_target_level_r3_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            cache = root / "pages.jsonl"
            output = root / "predictions.jsonl"
            write_manifest(
                manifest,
                [
                    ("10", "100", "Alpha Formation"),
                    ("10", "101", "Alpha Formation"),
                ],
            )
            write_cache(
                cache,
                [cache_record("10", "Alpha Formation IRM results identify magnetite.")],
            )
            RecordingReasoner.evaluations = []

            build_predictions(
                manifest,
                cache,
                DEFAULT_ARTIFACT,
                output,
                reasoner_factory=RecordingReasoner,
            )

            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(RecordingReasoner.evaluations), 1)
            self.assertEqual(
                RecordingReasoner.evaluations[0],
                {
                    "target": "Alpha Formation",
                    "multi_target": False,
                    "identity_ambiguous": True,
                    "scope_binding": {
                        "status": "not_required_single_target",
                        "competing_targets": [],
                        "policy": (
                            "same sentence, exclusive bounded paragraph/table row, or an "
                            "exact-target forward discourse window with sibling exclusion"
                        ),
                        "structured_scope_enabled": True,
                        "discourse_scope_enabled": False,
                    },
                },
            )
            self.assertEqual({row["policy_decision"] for row in rows}, {1})
            self.assertEqual({row["OUTCOME"] for row in rows}, {"POSITIVE"})
            self.assertTrue(
                all(
                    row["reasoning_details"]["conditions"]["identity_granularity"]
                    == "canonical_target"
                    for row in rows
                )
            )

    def test_normalized_same_target_rows_share_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            cache = root / "pages.jsonl"
            output = root / "predictions.jsonl"
            write_manifest(
                manifest,
                [
                    ("10", "100", "Sao-Tome Formation"),
                    ("10", "101", "Sao Tome Formation"),
                ],
            )
            write_cache(
                cache,
                [cache_record("10", "Sao Tome Formation IRM results identify magnetite.")],
            )
            RecordingReasoner.evaluations = []

            build_predictions(
                manifest,
                cache,
                DEFAULT_ARTIFACT,
                output,
                reasoner_factory=RecordingReasoner,
            )

            self.assertEqual(len(RecordingReasoner.evaluations), 1)
            self.assertFalse(RecordingReasoner.evaluations[0]["multi_target"])
            self.assertTrue(RecordingReasoner.evaluations[0]["identity_ambiguous"])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual({row["policy_decision"] for row in rows}, {1})
            self.assertEqual({row["OUTCOME"] for row in rows}, {"POSITIVE"})

    def test_distinct_targets_remain_multi_target_and_are_scored_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            cache = root / "pages.jsonl"
            output = root / "predictions.jsonl"
            write_manifest(
                manifest,
                [
                    ("10", "100", "Alpha Formation"),
                    ("10", "101", "Beta Formation"),
                ],
            )
            write_cache(
                cache,
                [
                    cache_record(
                        "10",
                        "Alpha Formation IRM results identify magnetite. "
                        "Beta Formation IRM results identify hematite.",
                    )
                ],
            )
            RecordingReasoner.evaluations = []

            build_predictions(
                manifest,
                cache,
                DEFAULT_ARTIFACT,
                output,
                reasoner_factory=RecordingReasoner,
            )

            self.assertEqual(len(RecordingReasoner.evaluations), 2)
            self.assertEqual(
                {item["target"] for item in RecordingReasoner.evaluations},
                {"Alpha Formation", "Beta Formation"},
            )
            self.assertTrue(
                all(item["multi_target"] for item in RecordingReasoner.evaluations)
            )
            self.assertTrue(
                all(
                    not item["identity_ambiguous"]
                    for item in RecordingReasoner.evaluations
                )
            )
            self.assertEqual(
                {
                    tuple(item["scope_binding"]["competing_targets"])
                    for item in RecordingReasoner.evaluations
                },
                {("alpha formation",), ("beta formation",)},
            )
            self.assertTrue(
                all(
                    item["scope_binding"]["status"]
                    == "enabled_with_competing_target_exclusion"
                    for item in RecordingReasoner.evaluations
                )
            )
            self.assertEqual(
                {
                    tuple(item["scope_binding"]["competing_targets"])
                    for item in RecordingReasoner.evaluations
                },
                {("alpha formation",), ("beta formation",)},
            )
            self.assertTrue(
                all(
                    item["scope_binding"]["status"]
                    == "enabled_with_competing_target_exclusion"
                    for item in RecordingReasoner.evaluations
                )
            )


if __name__ == "__main__":
    unittest.main()
