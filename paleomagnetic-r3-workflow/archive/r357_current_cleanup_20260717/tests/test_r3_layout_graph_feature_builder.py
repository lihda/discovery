from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.build_r3_layout_graph_features import (
    CANDIDATE_NAME,
    FEATURE_NAME,
    MANIFEST_NAME,
    LayoutGraphFeatureBuilderError,
    aggregate_target_page_features,
    build_layout_graph_feature_artifacts,
    file_sha256,
    json_sha256,
)
from scripts.r3_layout_graph_features import FEATURE_NAMES as PAGE_FEATURE_NAMES


def atomic_json_fixture(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


class LayoutGraphFeatureBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output_dir = self.root / "output"
        self.pdf = self.root / "10.pdf"
        self.pdf.write_bytes(b"%PDF layout pilot fixture")
        self.pdf_hash = file_sha256(self.pdf)
        self.selection = self.root / "selection.json"
        self.sidecar = self.root / "sidecar.jsonl"
        self.fold = self.root / "fold.json"
        self.concepts = self.root / "concepts.json"
        self.criterion_source = self.root / "criterion.json"
        self.normative_source = self.root / "normative.json"
        self.criterion_source.write_text("{}\n", encoding="utf-8")
        self.normative_source.write_text("{}\n", encoding="utf-8")
        self.selected_rows = [
            {"REFNO": "10", "RESULTNO": str(index), "ROCKNAME": rockname}
            for index, rockname in enumerate(
                (
                    "Alpha Formation",
                    "Alpha Formation",
                    "Beta Formation",
                    "Gamma Formation",
                    "Delta Formation",
                    "Epsilon Formation",
                    "Zeta Formation",
                    "Eta Formation",
                ),
                start=1,
            )
        ]
        self._write_selection()
        self._write_fold()
        self._write_concepts()
        self._write_sidecar()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_selection(self):
        rule = {
            "id": "identity_row_count_gte_8_and_unique_local_pdf_v1",
            "minimum_identity_rows_inclusive": 8,
            "label_free": True,
        }
        papers = [
            {
                "REFNO": "10",
                "identity_row_count": 8,
                "pdf": {
                    "path": str(self.pdf),
                    "name": self.pdf.name,
                    "size_bytes": self.pdf.stat().st_size,
                    "sha256": self.pdf_hash,
                },
            }
        ]
        frozen_pdfs = [
            {"REFNO": "10", "name": self.pdf.name, "sha256": self.pdf_hash}
        ]
        payload = {
            "schema_version": "r3_layout_pilot_manifest_v1",
            "status": "FROZEN_WITHOUT_LABEL_ACCESS",
            "identity_fields": ["REFNO", "RESULTNO", "ROCKNAME"],
            "selection_rule": rule,
            "selection_rule_sha256": json_sha256(rule),
            "selection_sha256": json_sha256(
                {
                    "selection_rule": rule,
                    "rows": self.selected_rows,
                    "pdfs": frozen_pdfs,
                }
            ),
            "selected_rows_sha256": json_sha256(self.selected_rows),
            "selected_pdf_set_sha256": json_sha256(frozen_pdfs),
            "runtime_boundary": {
                "labels_read": False,
                "pmagresult_or_rockmag_read": False,
                "prior_predictions_read": False,
                "evaluation_results_read": False,
            },
            "papers": papers,
            "rows": self.selected_rows,
        }
        atomic_json_fixture(self.selection, payload)

    def _write_fold(self, *, corrupt_hash=False):
        rows = []
        for index, selected in enumerate(self.selected_rows):
            rows.append(
                {
                    "row_index": index,
                    "REFNO": "10",
                    "RESULTNO": selected["RESULTNO"],
                    "document_sha256": "a" * 64,
                    "pdf_sha256": self.pdf_hash,
                    "normalized_text_sha256": "b" * 64,
                    "normalized_text_characters": 100,
                    "fold": 1,
                }
            )
        rows.append(
            {
                "row_index": 8,
                "REFNO": "20",
                "RESULTNO": "9",
                "document_sha256": "c" * 64,
                "pdf_sha256": "d" * 64,
                "normalized_text_sha256": "e" * 64,
                "normalized_text_characters": 100,
                "fold": 2,
            }
        )
        folds = np.asarray([row["fold"] for row in rows], dtype=np.int8)
        payload = {
            "schema_version": "r3_trace_refno_fold_manifest_v4",
            "status": "FROZEN_WITHOUT_LABEL_ACCESS",
            "alignment_sha256": "f" * 64,
            "fold_id_sha256": (
                "0" * 64
                if corrupt_hash
                else hashlib.sha256(folds.tobytes()).hexdigest()
            ),
            "labels_read": False,
            "fold_counts": {"1": 8, "2": 1},
            "rows": rows,
        }
        atomic_json_fixture(self.fold, payload)

    def _write_concepts(self):
        payload = {
            "schema_version": "r3_trace_page_graph_concepts_v4",
            "status": "FROZEN_WITHOUT_LABEL_ACCESS",
            "criterion": {
                "path": str(self.criterion_source),
                "sha256": file_sha256(self.criterion_source),
            },
            "normative_concepts": {
                "path": str(self.normative_source),
                "sha256": file_sha256(self.normative_source),
            },
            "concepts": [
                {
                    "concept_id": "irm",
                    "category": "method",
                    "search_patterns": [r"\bIRM\b"],
                },
                {
                    "concept_id": "saturation",
                    "category": "diagnostic",
                    "search_patterns": [r"saturat(?:e|ed|ion)"],
                },
                {
                    "concept_id": "magnetite",
                    "category": "carrier",
                    "search_patterns": [r"\bmagnetite\b"],
                },
                {
                    "concept_id": "carrier_relation",
                    "category": "bridge",
                    "search_patterns": [r"carrier of (?:the )?ChRM"],
                },
            ],
        }
        atomic_json_fixture(self.concepts, payload)

    def _page(self, number, alpha_text, beta_text):
        return {
            "page": number,
            "width": 600.0,
            "height": 800.0,
            "rotation": 0,
            "blocks": [
                {
                    "block_id": f"p{number}-b1",
                    "text": alpha_text,
                    "layout_confidence": 1.0,
                    "lines": [
                        {
                            "line_id": f"p{number}-l1",
                            "text": alpha_text,
                            "layout_confidence": 1.0,
                        }
                    ],
                },
                {
                    "block_id": f"p{number}-b2",
                    "text": beta_text,
                    "layout_confidence": 0.9,
                    "lines": [
                        {
                            "line_id": f"p{number}-l2",
                            "text": beta_text,
                            "layout_confidence": 0.9,
                        }
                    ],
                },
            ],
            "tables": [],
            "captions": [],
        }

    def _write_sidecar(self, *, supervised=False):
        record = {
            "schema_version": "r3_layout_sidecar_v1",
            "REFNO": "10",
            "pdf": {
                "path": str(self.pdf),
                "sha256": self.pdf_hash,
                "page_count": 2,
            },
            "pages": [
                self._page(
                    1,
                    "Alpha Formation IRM saturated; magnetite is carrier of the ChRM.",
                    "Beta Formation IRM was measured.",
                ),
                self._page(
                    2,
                    "Alpha Formation magnetite is carrier of the ChRM.",
                    "Beta Formation IRM saturated; magnetite is carrier of the ChRM.",
                ),
            ],
        }
        record["pages"][0]["captions"] = [
            {
                "caption_id": "p1-cap1",
                "kind": "table",
                "label": "1",
                "text": "Table 1 Alpha Formation carrier summary",
            }
        ]
        if supervised:
            record["R3"] = 1
        self.sidecar.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def build(self):
        return build_layout_graph_feature_artifacts(
            selection_path=self.selection,
            sidecar_path=self.sidecar,
            fold_manifest_path=self.fold,
            concept_snapshot_path=self.concepts,
            output_dir=self.output_dir,
        )

    def test_builds_complete_alignment_and_keeps_duplicate_identity_equal(self):
        outputs = self.build()
        with np.load(outputs["features"]) as archive:
            self.assertEqual(archive.files, ["layout"])
            matrix = archive["layout"]
        manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
        candidate = json.loads(
            outputs["candidate_freeze"].read_text(encoding="utf-8")
        )

        self.assertEqual(matrix.shape, (9, len(manifest["feature_names"])))
        self.assertTrue(np.all(matrix[8] == 0.0))
        self.assertTrue(np.all(matrix[:8, 0] == 1.0))
        self.assertTrue(np.all(matrix[:8, 1] == 1.0))
        self.assertTrue(np.array_equal(matrix[0], matrix[1]))
        self.assertEqual(manifest["matrix_sha256"], array_hash(matrix))
        self.assertEqual(
            manifest["outputs"]["features"]["sha256"],
            file_sha256(outputs["features"]),
        )
        self.assertEqual(manifest["outputs"]["candidate_freeze"]["path"], str(outputs["candidate_freeze"]))
        self.assertEqual(audit["nonpilot_nonzero_values"], 0)
        self.assertEqual(audit["duplicate_canonical_groups_checked"], 1)
        self.assertFalse(manifest["runtime_boundary"]["labels_read"])
        self.assertEqual(candidate["fixed_decision_threshold"], 0.5)
        self.assertIn("random_forest", candidate["model_candidates"])
        self.assertIn("hgb", candidate["model_candidates"])
        self.assertEqual(
            candidate["model_candidates"]["random_forest"]["random_state"],
            35_720_260,
        )
        self.assertEqual(
            set(candidate["feature_ablation_candidates"]),
            {"base_v4", "pilot_flags_only", "full_layout"},
        )
        self.assertEqual(
            candidate["feature_ablation_candidates"]["pilot_flags_only"][
                "layout_columns"
            ],
            [0, 1],
        )
        self.assertEqual(
            candidate["deployment_route"]["nonpilot_rows"],
            "base_v4_unchanged",
        )
        self.assertEqual(outputs["features"].name, FEATURE_NAME)
        self.assertEqual(outputs["manifest"].name, MANIFEST_NAME)
        self.assertEqual(outputs["candidate_freeze"].name, CANDIDATE_NAME)

    def test_page_aggregation_is_stable_and_interpretable(self):
        first = {name: 0.0 for name in PAGE_FEATURE_NAMES}
        second = dict(first)
        first["layout_method_same_line_available"] = 1.0
        first["layout_method_same_line_mass"] = 2.0
        second["layout_method_same_line_mass"] = 3.0
        first["layout_unique_assignment_count_log"] = math.log1p(2.0)
        second["layout_unique_assignment_count_log"] = math.log1p(3.0)
        first["layout_evidence_mass"] = 2.0
        second["layout_evidence_mass"] = 3.0
        first["layout_evidence_share"] = 0.25
        second["layout_evidence_share"] = 0.75

        result = aggregate_target_page_features([first, second])

        self.assertEqual(result["layout_method_same_line_available"], 1.0)
        self.assertAlmostEqual(result["layout_method_same_line_mass"], math.log1p(5.0))
        self.assertAlmostEqual(
            result["layout_unique_assignment_count_log"], math.log1p(5.0)
        )
        self.assertAlmostEqual(result["layout_evidence_share"], 0.55)

    def test_rejects_supervised_sidecar_field_before_output(self):
        self._write_sidecar(supervised=True)

        with self.assertRaisesRegex(
            LayoutGraphFeatureBuilderError, "forbidden supervised field"
        ):
            self.build()
        self.assertFalse((self.output_dir / FEATURE_NAME).exists())

    def test_rejects_fold_hash_mismatch(self):
        self._write_fold(corrupt_hash=True)

        with self.assertRaisesRegex(LayoutGraphFeatureBuilderError, "fold-id hash"):
            self.build()

    def test_rejects_changed_pilot_pdf(self):
        self.pdf.write_bytes(b"changed after selection freeze")

        with self.assertRaisesRegex(LayoutGraphFeatureBuilderError, "missing or changed"):
            self.build()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
