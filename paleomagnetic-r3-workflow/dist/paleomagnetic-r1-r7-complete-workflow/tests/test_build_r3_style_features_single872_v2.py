from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from openpyxl import Workbook

from scripts import build_r3_style_features_single872_v2 as builder
from scripts import build_r3_trace_page_graph_features as trace_freeze
from scripts import experiment_r3_pdf_only_strategies as knowledge_source


ROOT = Path(__file__).resolve().parents[1]


def write_identity(path: Path, records: list[dict[str, str]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["REFNO", "RESULTNO", "ROCKNAME"])
    for record in records:
        sheet.append([record["REFNO"], record["RESULTNO"], record["ROCKNAME"]])
    workbook.save(path)
    workbook.close()


def write_fixture(root: Path) -> dict[str, Path]:
    records = [
        {"REFNO": "100", "RESULTNO": "1000", "ROCKNAME": "Alpha Formation"},
        {"REFNO": "200", "RESULTNO": "2000", "ROCKNAME": "Beta Formation"},
    ]
    identity = root / "identity.xlsx"
    page_cache = root / "page_cache.jsonl"
    pdf_dir = root / "pdfs"
    pdf_dir.mkdir()
    write_identity(identity, records)
    pages_by_refno = {
        "100": [
            {
                "page": 1,
                "source": "mineru_text",
                "section_hints": ["ABSTRACT", "METHODS"],
                "text": (
                    "Geophys. J. R. astr. Soc. (1979) 58, 1-20\n"
                    "Paleomagnetism of the Alpha Formation\nAbstract\n"
                    "We measured IRM acquisition curves for Alpha Formation samples "
                    "and identified magnetite as the remanence carrier. " * 6
                ),
            }
        ],
        "200": [
            {
                "page": 1,
                "source": "pypdf",
                "section_hints": ["RESULTS"],
                "text": (
                    "Journal of Geophysical Research (1991) 96, 1-12\n"
                    "Paleomagnetism of the Beta Formation\nResults\n"
                    "Beta Formation specimens show hematite and hysteresis behavior. " * 7
                ),
            }
        ],
    }
    for refno in pages_by_refno:
        (pdf_dir / f"{refno}.pdf").write_bytes((f"synthetic PDF {refno}\n" * 20).encode())
    with page_cache.open("w", encoding="utf-8") as handle:
        for refno, pages in pages_by_refno.items():
            handle.write(json.dumps({"REFNO": refno, "pages": pages, "source": "test"}))
            handle.write("\n")

    pdf_hashes = {
        refno: trace_freeze.file_sha256(pdf_dir / f"{refno}.pdf")
        for refno in pages_by_refno
    }
    groups, text_hashes, characters = trace_freeze.normalized_text_document_groups(
        {
            refno: {"REFNO": refno, "pages": pages}
            for refno, pages in pages_by_refno.items()
        },
        pdf_hashes,
    )
    fold_ids = np.asarray([1, 2], dtype=np.int8)
    alignment_sha = trace_freeze._record_alignment_sha256(records)
    fold = {
        "schema_version": trace_freeze.FOLD_SCHEMA_VERSION,
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "alignment_sha256": alignment_sha,
        "fold_id_sha256": hashlib.sha256(fold_ids.tobytes()).hexdigest(),
        "labels_read": False,
        "rows": [
            {
                "row_index": index,
                "REFNO": record["REFNO"],
                "RESULTNO": record["RESULTNO"],
                "document_sha256": groups[record["REFNO"]],
                "pdf_sha256": pdf_hashes[record["REFNO"]],
                "normalized_text_sha256": text_hashes[record["REFNO"]],
                "normalized_text_characters": characters[record["REFNO"]],
                "fold": int(fold_ids[index]),
            }
            for index, record in enumerate(records)
        ],
    }
    fold_path = root / "folds.json"
    trace_freeze.atomic_json(fold_path, fold)
    v4 = {
        "schema_version": trace_freeze.FEATURE_SCHEMA_VERSION,
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "rows": 2,
        "contains_labels": False,
        "feature_archive_contains_identifiers": False,
        "alignment_sha256": alignment_sha,
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_read": False,
            "evaluation_results_read": False,
        },
        "inputs": {
            "identity": {
                "path": str(identity.resolve()),
                "sha256": trace_freeze.file_sha256(identity),
            },
            "page_cache": {
                "path": str(page_cache.resolve()),
                "sha256": trace_freeze.file_sha256(page_cache),
            },
            "pdf_membership": {
                "path": str(pdf_dir.resolve()),
                "sha256": builder.pdf_membership_sha256(pdf_dir),
            },
        },
        "outputs": {
            "fold_manifest": {
                "path": str(fold_path.resolve()),
                "sha256": trace_freeze.file_sha256(fold_path),
            }
        },
    }
    v4_path = root / "v4.manifest.json"
    trace_freeze.atomic_json(v4_path, v4)
    return {
        "v4": v4_path,
        "fold": fold_path,
        "page_cache": page_cache,
        "pdf_dir": pdf_dir,
    }


class StyleFeatureSingle872V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.concepts, cls.knowledge_audit = knowledge_source.load_knowledge_patterns(
            builder.DEFAULT_TAXONOMY, builder.DEFAULT_POSITIVE_KB
        )

    def test_fixed_knowledge_contract_has_38_concepts(self) -> None:
        self.assertEqual(len(self.concepts), 38)
        self.assertEqual(self.knowledge_audit["runtime_pattern_count"], 80)
        self.assertEqual(
            trace_freeze.file_sha256(builder.DEFAULT_TAXONOMY),
            builder.EXPECTED_TAXONOMY_SHA256,
        )
        self.assertEqual(
            trace_freeze.file_sha256(builder.DEFAULT_POSITIVE_KB),
            builder.EXPECTED_POSITIVE_KB_SHA256,
        )

    def test_v4_alignment_controls_identity_order_and_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_fixture(Path(temporary))
            value = builder.load_v4_single_alignment(
                paths["v4"],
                paths["fold"],
                expected_v4_rows=2,
                expected_single_rows=2,
            )
            self.assertEqual(
                [record["RESULTNO"] for record in value["records"]],
                ["1000", "2000"],
            )
            self.assertEqual(value["source_row_indices"].tolist(), [0, 1])
            self.assertEqual(value["fold_ids"].tolist(), [1, 2])

    def test_style_knowledge_and_all_matrices_reuse_legacy_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_fixture(Path(temporary))
            records = [
                {"REFNO": "100", "RESULTNO": "1000", "ROCKNAME": "Alpha Formation"},
                {"REFNO": "200", "RESULTNO": "2000", "ROCKNAME": "Beta Formation"},
            ]
            pages, present = builder.load_style_pages(
                paths["page_cache"], {"100", "200"}
            )
            result = builder.build_matrices(
                records, pages, present, self.concepts, paths["pdf_dir"]
            )
            self.assertEqual(result["style"].shape, (2, 105))
            self.assertEqual(result["knowledge"].shape, (2, 84))
            self.assertEqual(result["all"].shape, (2, 189))
            np.testing.assert_array_equal(
                result["all"], np.hstack((result["style"], result["knowledge"]))
            )
            self.assertEqual(len(set(result["style_feature_names"])), 105)
            self.assertEqual(len(set(result["knowledge_feature_names"])), 84)
            self.assertEqual(len(set(result["all_feature_names"])), 189)
            self.assertEqual(len(result["legacy_knowledge_feature_names"]), 84)
            self.assertGreater(
                result["style"][0, result["style_feature_names"].index("title_target_mention")],
                0.0,
            )
            local_irm = [
                index
                for index, name in enumerate(result["knowledge_feature_names"])
                if name.startswith("target_concept_") and "irm_acquisition" in name
            ]
            self.assertTrue(local_irm)
            self.assertTrue(any(result["knowledge"][0, index] > 0 for index in local_irm))

    def test_missing_sources_keep_rows_with_explicit_zero_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_fixture(root)
            (paths["pdf_dir"] / "200.pdf").unlink()
            records = [
                {"REFNO": "100", "RESULTNO": "1000", "ROCKNAME": "Alpha Formation"},
                {"REFNO": "200", "RESULTNO": "2000", "ROCKNAME": "Beta Formation"},
            ]
            pages, present = builder.load_style_pages(
                paths["page_cache"], {"100", "200"}
            )
            result = builder.build_matrices(
                records, pages, present, self.concepts, paths["pdf_dir"]
            )
            self.assertEqual(result["available"].tolist(), [True, False])
            self.assertEqual(result["pdf_missing"].tolist(), [False, True])
            self.assertTrue(np.all(result["all"][1] == 0.0))

    def test_page_loader_rejects_forbidden_outcome_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pages.jsonl"
            path.write_text(
                json.dumps({"REFNO": "1", "R3": 1, "pages": [{"text": "x"}]})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(builder.StyleFeatureError, "forbidden fields"):
                builder.load_style_pages(path, {"1"})

    def test_end_to_end_freeze_is_hashed_and_self_describing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_fixture(root)
            output = root / "output"
            built = builder.build_style_features(
                v4_manifest_path=paths["v4"],
                fold_path=paths["fold"],
                page_cache_path=paths["page_cache"],
                pdf_dir=paths["pdf_dir"],
                taxonomy_path=builder.DEFAULT_TAXONOMY,
                positive_kb_path=builder.DEFAULT_POSITIVE_KB,
                output_dir=output,
                expected_v4_rows=2,
                expected_single_rows=2,
            )
            manifest = json.loads(built["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["status"], "COMPLETE_FROZEN_WITHOUT_LABEL_ACCESS"
            )
            self.assertEqual(manifest["available_rows"], 2)
            with np.load(built["features"], allow_pickle=False) as archive:
                self.assertEqual(archive["style"].shape, (2, 105))
                self.assertEqual(archive["knowledge"].shape, (2, 84))
                self.assertEqual(archive["all"].shape, (2, 189))
                self.assertEqual(archive["fold_ids"].tolist(), [1, 2])
                for name in archive.files:
                    self.assertEqual(
                        manifest["array_sha256"][name],
                        trace_freeze.array_sha256(archive[name]),
                    )
            self.assertEqual(
                manifest["outputs"]["features"]["sha256"],
                trace_freeze.file_sha256(built["features"]),
            )

    def test_cli_has_no_label_prediction_or_evaluation_input(self) -> None:
        options = {
            option
            for action in builder.build_parser()._actions
            for option in action.option_strings
        }
        self.assertEqual(
            options,
            {
                "-h",
                "--help",
                "--v4-manifest",
                "--fold-manifest",
                "--page-cache",
                "--pdf-dir",
                "--taxonomy",
                "--positive-kb",
                "--output-dir",
            },
        )


if __name__ == "__main__":
    unittest.main()
