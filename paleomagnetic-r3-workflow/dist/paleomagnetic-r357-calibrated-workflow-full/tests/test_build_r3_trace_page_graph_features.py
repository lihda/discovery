import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from openpyxl import Workbook

from scripts.build_r3_trace_page_graph_features import (
    build_frozen_trace_features,
    build_trace_concept_snapshot,
    load_sanitized_page_cache,
    load_three_column_identity,
)
from scripts.experiment_r3_trace_clean_nested_oof import load_frozen_inputs


ROOT = Path(__file__).resolve().parents[1]
CRITERION = ROOT / "data/knowledge/r357_criterion_spec_v2.json"
CONCEPTS = ROOT / "data/knowledge/r3_trace_normative_concepts_v1.json"


def write_identity(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    workbook.close()


class BuildTracePageGraphFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_end_to_end_freeze_uses_only_pdf_backed_identity(self):
        identity = self.root / "identity.xlsx"
        write_identity(
            identity,
            ("REFNO", "RESULTNO", "ROCKNAME"),
            [
                ("1", "10", "Alpha Formation"),
                ("1", "11", "Beta Formation"),
                ("2", "12", "Gamma Formation"),
            ],
        )
        pdf_dir = self.root / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "1.pdf").write_bytes(b"%PDF-1.4\n")
        page_cache = self.root / "pages.jsonl"
        cache_rows = [
            {
                "REFNO": "1",
                "pages": [
                    {
                        "page": 1,
                        "text": "Alpha Formation specimens underwent IRM acquisition.",
                    },
                    {
                        "page": 2,
                        "text": (
                            "Thermomagnetic curves indicate magnetite in Alpha Formation. "
                            "Magnetite carries the primary remanence."
                        ),
                    },
                    {
                        "page": 3,
                        "text": "Beta Formation was thermally demagnetized.",
                    },
                ],
            },
            {"REFNO": "2", "pages": [{"page": 1, "text": "Gamma Formation."}]},
        ]
        page_cache.write_text(
            "\n".join(json.dumps(row) for row in cache_rows) + "\n", encoding="utf-8"
        )
        output_dir = self.root / "output"
        outputs = build_frozen_trace_features(
            identity_path=identity,
            page_cache_path=page_cache,
            pdf_dir=pdf_dir,
            criterion_path=CRITERION,
            concepts_path=CONCEPTS,
            output_dir=output_dir,
        )

        matrix = np.load(outputs["features"])["trace"]
        manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        candidate = json.loads(outputs["candidate_freeze"].read_text(encoding="utf-8"))
        folds = json.loads(outputs["fold_manifest"].read_text(encoding="utf-8"))
        snapshot = json.loads(outputs["concept_snapshot"].read_text(encoding="utf-8"))
        context_documents = [
            json.loads(line)
            for line in outputs["target_context"].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(matrix.shape[0], 2)
        self.assertGreater(matrix.shape[1], 0)
        self.assertIn("coref_assigned_event_count_log", manifest["feature_names"])
        self.assertEqual(manifest["coverage"]["skipped_without_pdf_rows"], 1)
        self.assertFalse(manifest["contains_labels"])
        self.assertFalse(candidate["runtime_boundary"]["pmagresult_or_rockmag_read"])
        self.assertEqual(
            candidate["schema_version"], "r3_trace_page_graph_candidate_freeze_v4"
        )
        self.assertIn("within_paper_pairwise", candidate["model_candidates"])
        self.assertTrue(outputs["features"].name.endswith("_v4.npz"))
        self.assertTrue(all(row["document_sha256"] for row in folds["rows"]))
        self.assertTrue(all(row["normalized_text_sha256"] for row in folds["rows"]))
        self.assertEqual(len(context_documents), 2)
        self.assertIn("[TARGET] Alpha Formation", context_documents[0])
        self.assertIn("[METHOD_", context_documents[0])
        self.assertNotIn("global_pair", json.dumps(candidate))
        self.assertEqual(
            snapshot["normative_concepts"]["schema_version"],
            "r3_trace_normative_concepts_v1",
        )
        loaded = load_frozen_inputs(
            identity_path=identity,
            pdf_dir=pdf_dir,
            feature_path=outputs["features"],
            manifest_path=outputs["manifest"],
            context_path=outputs["target_context"],
            candidate_path=outputs["candidate_freeze"],
            fold_path=outputs["fold_manifest"],
        )
        self.assertEqual(loaded[1].shape, matrix.shape)
        self.assertEqual(len(loaded[3]), 2)

    def test_identity_rejects_any_fourth_column(self):
        identity = self.root / "contaminated.xlsx"
        write_identity(
            identity,
            ("REFNO", "RESULTNO", "ROCKNAME", "R3"),
            [("1", "10", "Alpha Formation", "1")],
        )
        with self.assertRaisesRegex(ValueError, "exactly REFNO"):
            load_three_column_identity(identity)

    def test_page_cache_rejects_forbidden_fields(self):
        path = self.root / "cache.jsonl"
        path.write_text(
            json.dumps(
                {
                    "REFNO": "1",
                    "R3": 1,
                    "pages": [
                        {
                            "page": 1,
                            "text": "original text",
                            "ROCKMAG": "IRM",
                            "old_prediction": 0.9,
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "forbidden fields"):
            load_sanitized_page_cache(path, {"1"})

    def test_sanitized_page_cache_keeps_only_source_whitelist(self):
        path = self.root / "cache.jsonl"
        path.write_text(
            json.dumps(
                {
                    "REFNO": "1",
                    "source": {"parser": "synthetic"},
                    "pages": [
                        {
                            "page": 1,
                            "text": "original text",
                            "text_sha256": "abc",
                            "section_hints": ["Results"],
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        rows = load_sanitized_page_cache(path, {"1"})
        self.assertEqual(set(rows["1"]), {"REFNO", "pages"})
        self.assertEqual(set(rows["1"]["pages"][0]), {"page", "text", "section_hints"})

    def test_rejects_historically_rockmag_induced_knowledge(self):
        contaminated = ROOT / "data/knowledge/r357_expert_kb_v1.json"
        with self.assertRaisesRegex(ValueError, "no-leakage|forbidden historical"):
            build_trace_concept_snapshot(CRITERION, contaminated)

    def test_normalized_text_groups_pdf_byte_variants(self):
        from scripts.build_r3_trace_page_graph_features import (
            normalized_text_document_groups,
        )

        text = "Alpha Formation specimens underwent IRM acquisition. " * 8
        rows = {
            "1": {"REFNO": "1", "pages": [{"page": 1, "text": text}]},
            "2": {
                "REFNO": "2",
                "pages": [{"page": 1, "text": text.upper()}],
            },
        }
        groups, text_hashes, counts = normalized_text_document_groups(
            rows, {"1": "a" * 64, "2": "b" * 64}
        )
        self.assertEqual(groups["1"], groups["2"])
        self.assertEqual(text_hashes["1"], text_hashes["2"])
        self.assertGreaterEqual(counts["1"], 256)


if __name__ == "__main__":
    unittest.main()
