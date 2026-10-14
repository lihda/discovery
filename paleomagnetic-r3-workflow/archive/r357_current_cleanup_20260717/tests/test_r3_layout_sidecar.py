from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.r3_layout_sidecar import (
    LayoutSidecarError,
    _valid_bbox,
    build_layout_sidecar,
    load_selection_manifest,
)
from scripts.build_r3_layout_pilot_manifest import json_sha256
from scripts.r3_layout_graph_features import build_layout_graph_target_features


class FakeRow:
    def __init__(self, cells):
        self.cells = cells


class FakeTable:
    bbox = (30, 180, 570, 300)

    def __init__(self):
        self.rows = [
            FakeRow([(30, 180, 260, 220), (260, 180, 570, 220)]),
            FakeRow([(30, 220, 260, 300), (260, 220, 570, 300)]),
        ]

    def extract(self):
        return [["Target", "Carrier"], ["Alpha", "Magnetite"]]


def words_for_line(text, top, start_x):
    words = []
    x = start_x
    for token in text.split():
        width = max(14, len(token) * 5)
        words.append(
            {
                "text": token,
                "x0": x,
                "x1": x + width,
                "top": top,
                "bottom": top + 10,
                "fontname": "Times-Roman",
                "size": 10,
            }
        )
        x += width + 5
    return words


def representative_words():
    return [
        *words_for_line("Table 1 Carrier summary", 20, 40),
        *words_for_line("Alpha samples", 70, 40),
        *words_for_line("Beta samples", 90, 40),
        *words_for_line("Magnetite carrier", 70, 360),
        *words_for_line("Hematite carrier", 90, 360),
        *words_for_line("Results are reported in Table 1", 140, 40),
    ]


class FakePage:
    width = 600
    height = 800
    rotation = 0

    def __init__(self, *, table_error=False):
        self.table_error = table_error

    def extract_words(self, **kwargs):
        return representative_words()

    def find_tables(self, table_settings=None):
        if self.table_error:
            raise RuntimeError("synthetic table detector failure")
        return [FakeTable()]


class FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class LayoutSidecarTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pdf_dir = self.root / "pdfs"
        self.pdf_dir.mkdir()
        self.pdf = self.pdf_dir / "10.pdf"
        self.pdf.write_bytes(b"synthetic pdf bytes")
        self.selection = self.root / "selection.json"
        self.selection.write_text(
            json.dumps(
                {
                    "schema_version": "r3_layout_pilot_selection_v1",
                    "status": "FROZEN_WITHOUT_LABEL_ACCESS",
                    "selected_refnos": ["10"],
                    "runtime_boundary": {
                        "labels_read": False,
                        "pmagresult_or_rockmag_read": False,
                        "prior_predictions_read": False,
                        "evaluation_results_read": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        cache_text = "\n".join(
            [
                "Table 1 Carrier summary",
                "Alpha samples Magnetite carrier",
                "Beta samples Hematite carrier",
                "Results are reported in Table 1",
            ]
        )
        self.cache = self.root / "page_cache.jsonl"
        self.cache.write_text(
            json.dumps(
                {
                    "REFNO": "10",
                    "pages": [{"page": 1, "text": cache_text}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.output = self.root / "sidecar.jsonl"
        self.audit = self.root / "sidecar.audit.json"

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, page):
        return build_layout_sidecar(
            self.selection,
            self.pdf_dir,
            self.cache,
            self.output,
            self.audit,
            pdf_opener=lambda _: FakePDF([page]),
        )

    def record(self):
        return json.loads(self.output.read_text(encoding="utf-8"))

    def test_builds_words_columns_tables_captions_crossrefs_and_alignment(self):
        audit = self.build(FakePage())
        record = self.record()
        page = record["pages"][0]

        self.assertEqual(record["schema_version"], "r3_layout_sidecar_v1")
        self.assertEqual(record["REFNO"], "10")
        self.assertEqual([page["width"], page["height"]], [600.0, 800.0])
        self.assertTrue(page["words"])
        self.assertTrue(all(len(word["bbox"]) == 4 for word in page["words"]))
        self.assertEqual(
            [word["reading_order"] for word in page["words"]],
            list(range(1, len(page["words"]) + 1)),
        )
        self.assertGreaterEqual(len(page["columns"]), 2)
        self.assertTrue(page["blocks"])
        self.assertEqual(page["tables"][0]["row_count"], 2)
        self.assertEqual(page["tables"][0]["column_count"], 2)
        self.assertEqual(page["tables"][0]["cells"][3]["text"], "Magnetite")
        self.assertEqual(page["tables"][0]["rows"][1]["cells"][1]["text"], "Magnetite")
        self.assertEqual(page["captions"][0]["label"], "1")
        self.assertEqual(
            page["captions"][0]["linked_table_id"], page["tables"][0]["table_id"]
        )
        self.assertEqual(
            page["crossrefs"][0]["linked_caption_ids"],
            [page["captions"][0]["caption_id"]],
        )
        self.assertEqual(
            page["crossrefs"][0]["linked_table_ids"],
            [page["tables"][0]["table_id"]],
        )
        self.assertEqual(page["cache_alignment"]["status"], "aligned")
        self.assertGreater(page["cache_alignment"]["token_f1"], 0.8)
        self.assertEqual(audit["status"], "PASS")
        self.assertFalse(audit["runtime_boundary"]["labels_read"])
        self.assertFalse(audit["runtime_boundary"]["pmagresult_or_rockmag_read"])
        self.assertEqual(audit["output"]["records"], 1)
        self.assertFalse(self.output.with_name(self.output.name + ".tmp").exists())
        self.assertFalse(self.audit.with_name(self.audit.name + ".tmp").exists())

        graph = build_layout_graph_target_features(
            page,
            {"alpha": "Alpha"},
            [
                {"concept_id": "method", "category": "method", "search_patterns": ["samples"]},
                {"concept_id": "diagnostic", "category": "diagnostic", "search_patterns": ["summary"]},
                {"concept_id": "carrier", "category": "carrier", "search_patterns": ["magnetite"]},
                {"concept_id": "bridge", "category": "bridge", "search_patterns": ["carrier"]},
            ],
        )
        carrier_row = graph["feature_names"].index(
            "layout_carrier_same_table_row_available"
        )
        self.assertEqual(graph["matrix"][0, carrier_row], 1.0)

    def test_table_detection_failure_degrades_page_without_losing_text(self):
        audit = self.build(FakePage(table_error=True))
        page = self.record()["pages"][0]

        self.assertTrue(page["words"])
        self.assertTrue(page["lines"])
        self.assertTrue(page["captions"])
        self.assertEqual(page["tables"], [])
        self.assertEqual(page["quality"]["table_detection_status"], "degraded")
        self.assertGreaterEqual(page["quality"]["table_detection_error_count"], 1)
        self.assertEqual(audit["status"], "PASS_WITH_DEGRADATION")
        self.assertGreaterEqual(audit["counts"]["table_detection_errors"], 1)
        self.assertTrue(audit["table_detection_degrades_per_page"])

    def test_selection_manifest_rejects_supervised_fields(self):
        self.selection.write_text(
            json.dumps({"selected_refnos": ["10"], "labels": {"10": 1}}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(LayoutSidecarError, "forbidden supervised field"):
            load_selection_manifest(self.selection)

    def test_accepts_and_verifies_frozen_pilot_manifest(self):
        rule = {"id": "pilot", "label_free": True}
        rows = [
            {"REFNO": "10", "RESULTNO": str(index), "ROCKNAME": f"T{index}"}
            for index in range(8)
        ]
        pdfs = [
            {
                "REFNO": "10",
                "name": "10.pdf",
                "sha256": "a" * 64,
            }
        ]
        payload = {
            "schema_version": "r3_layout_pilot_manifest_v1",
            "status": "FROZEN_WITHOUT_LABEL_ACCESS",
            "selection_rule": rule,
            "selection_rule_sha256": json_sha256(rule),
            "rows": rows,
            "selected_rows_sha256": json_sha256(rows),
            "papers": [
                {
                    "REFNO": "10",
                    "pdf": {
                        "name": "10.pdf",
                        "sha256": "a" * 64,
                    },
                }
            ],
            "selected_pdf_set_sha256": json_sha256(pdfs),
            "selection_sha256": json_sha256(
                {"selection_rule": rule, "rows": rows, "pdfs": pdfs}
            ),
            "runtime_boundary": {"labels_read": False},
        }
        self.selection.write_text(json.dumps(payload), encoding="utf-8")

        observed = load_selection_manifest(self.selection)

        self.assertEqual(observed["refnos"], ["10"])
        self.assertEqual(observed["expected_pdf_hashes"], {"10": "a" * 64})
        payload["rows"][0]["ROCKNAME"] = "changed"
        self.selection.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(LayoutSidecarError, "selected_rows_sha256"):
            load_selection_manifest(self.selection)

    def test_bbox_rounding_is_not_reported_as_clamping(self):
        bbox, adjusted = _valid_bbox(
            [1.123456, 2.234567, 50.345678, 60.456789], 100.0, 100.0
        )
        self.assertEqual(bbox, [1.1235, 2.2346, 50.3457, 60.4568])
        self.assertFalse(adjusted)
        bbox, adjusted = _valid_bbox([-0.5, 2.0, 50.0, 60.0], 100.0, 100.0)
        self.assertEqual(bbox, [0.0, 2.0, 50.0, 60.0])
        self.assertTrue(adjusted)

    def test_missing_selected_cache_record_fails_closed(self):
        self.cache.write_text(
            json.dumps({"REFNO": "20", "pages": []}) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(LayoutSidecarError, "missing selected REFNO"):
            self.build(FakePage())


if __name__ == "__main__":
    unittest.main()
