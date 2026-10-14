from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from openpyxl import Workbook

from scripts.build_r3_layout_pilot_manifest import (
    IDENTITY_FIELDS,
    LayoutPilotManifestError,
    build_layout_pilot_manifest,
    file_sha256,
    freeze_layout_pilot_manifest,
    load_identity_rows,
    main,
)


def write_identity(
    path: Path,
    rows: list[tuple[object, ...]],
    header: tuple[str, ...] = IDENTITY_FIELDS,
    *,
    second_sheet: bool = False,
) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(header)
    for row in rows:
        worksheet.append(row)
    if second_sheet:
        workbook.create_sheet("unexpected")
    workbook.save(path)
    workbook.close()


def paper_rows(refno: str, first_resultno: int, count: int) -> list[tuple[str, str, str]]:
    return [
        (refno, str(first_resultno + offset), f"Target {refno}-{offset}")
        for offset in range(count)
    ]


class LayoutPilotManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.identity = self.root / "identity.xlsx"
        self.pdf_dir = self.root / "pdfs"
        self.pdf_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_freezes_only_gte_eight_row_papers_with_unique_pdfs(self) -> None:
        rows = (
            paper_rows("10", 100, 8)
            + paper_rows("20", 200, 7)
            + paper_rows("30", 300, 9)
            + paper_rows("40", 400, 8)
        )
        write_identity(self.identity, rows)
        (self.pdf_dir / "10.pdf").write_bytes(b"%PDF pilot ten")
        (self.pdf_dir / "20.pdf").write_bytes(b"%PDF below threshold")
        (self.pdf_dir / "40.PDF").write_bytes(b"%PDF pilot forty")
        output = self.root / "frozen" / "pilot.json"

        manifest = freeze_layout_pilot_manifest(
            self.identity, self.pdf_dir, output
        )
        loaded = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(loaded, manifest)
        self.assertEqual(manifest["schema_version"], "r3_layout_pilot_manifest_v1")
        self.assertEqual(manifest["status"], "FROZEN_WITHOUT_LABEL_ACCESS")
        self.assertEqual({row["REFNO"] for row in manifest["rows"]}, {"10", "40"})
        self.assertEqual(len(manifest["rows"]), 16)
        self.assertTrue(
            all(tuple(row.keys()) == IDENTITY_FIELDS for row in manifest["rows"])
        )
        self.assertEqual(
            [paper["REFNO"] for paper in manifest["papers"]], ["10", "40"]
        )
        self.assertEqual(
            manifest["papers"][0]["pdf"]["sha256"],
            file_sha256(self.pdf_dir / "10.pdf"),
        )
        self.assertRegex(manifest["selection_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(manifest["selected_rows_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(manifest["selected_pdf_set_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            manifest["selection_rule"]["minimum_identity_rows_inclusive"], 8
        )
        self.assertEqual(manifest["audit"]["eligible_by_row_count_papers"], 3)
        self.assertEqual(manifest["audit"]["selected_papers"], 2)
        self.assertEqual(manifest["audit"]["excluded_below_threshold_papers"], 1)
        self.assertEqual(manifest["audit"]["excluded_missing_pdf_papers"], 1)
        self.assertEqual(manifest["audit"]["excluded_missing_pdf_refnos"], ["30"])
        self.assertEqual(manifest["audit"]["output_identity_fields"], list(IDENTITY_FIELDS))
        self.assertFalse(manifest["runtime_boundary"]["labels_read"])
        self.assertFalse(
            manifest["runtime_boundary"]["pmagresult_or_rockmag_read"]
        )
        self.assertFalse(manifest["runtime_boundary"]["prior_predictions_read"])
        self.assertFalse(manifest["runtime_boundary"]["evaluation_results_read"])
        self.assertEqual(manifest["audit"]["forbidden_sources_read"], [])

        rebuilt = build_layout_pilot_manifest(self.identity, self.pdf_dir)
        self.assertEqual(rebuilt["selection_sha256"], manifest["selection_sha256"])
        (self.pdf_dir / "10.pdf").write_bytes(b"%PDF changed bytes")
        changed = build_layout_pilot_manifest(self.identity, self.pdf_dir)
        self.assertNotEqual(changed["selection_sha256"], manifest["selection_sha256"])
        self.assertNotEqual(
            changed["selected_pdf_set_sha256"], manifest["selected_pdf_set_sha256"]
        )

    def test_rejects_extra_columns_formulas_multiple_sheets_and_duplicates(self) -> None:
        cases = []

        extra = self.root / "extra.xlsx"
        write_identity(
            extra,
            [("10", "1", "Target", 1)],
            header=("REFNO", "RESULTNO", "ROCKNAME", "R3"),
        )
        cases.append((extra, "exactly REFNO"))

        formula = self.root / "formula.xlsx"
        write_identity(formula, [("10", "1", "=A2")])
        cases.append((formula, "formulas are forbidden"))

        sheets = self.root / "sheets.xlsx"
        write_identity(sheets, [("10", "1", "Target")], second_sheet=True)
        cases.append((sheets, "exactly one worksheet"))

        duplicate = self.root / "duplicate.xlsx"
        write_identity(
            duplicate,
            [("10", "1", "Target A"), ("10", "1", "Target B")],
        )
        cases.append((duplicate, "duplicate RESULTNO"))

        for path, message in cases:
            with self.subTest(path=path.name):
                with self.assertRaisesRegex(LayoutPilotManifestError, message):
                    load_identity_rows(path)

    def test_fails_closed_when_eligible_refno_has_ambiguous_pdf_mapping(self) -> None:
        write_identity(self.identity, paper_rows("10", 100, 8))
        first_dir = self.root / "first"
        second_dir = self.root / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        first = first_dir / "10.pdf"
        second = second_dir / "10.PDF"
        first.write_bytes(b"first")
        second.write_bytes(b"second")

        with mock.patch.object(Path, "iterdir", return_value=iter((first, second))):
            with self.assertRaisesRegex(LayoutPilotManifestError, "multiple local PDF"):
                build_layout_pilot_manifest(self.identity, self.pdf_dir)

    def test_errors_when_no_eligible_paper_has_a_pdf(self) -> None:
        write_identity(self.identity, paper_rows("10", 100, 8))
        with self.assertRaisesRegex(LayoutPilotManifestError, "no REFNO satisfies"):
            build_layout_pilot_manifest(self.identity, self.pdf_dir)

    def test_cli_writes_manifest(self) -> None:
        write_identity(self.identity, paper_rows("10", 100, 8))
        (self.pdf_dir / "10.pdf").write_bytes(b"%PDF cli")
        output = self.root / "cli.json"
        status = main(
            [
                "--identity",
                str(self.identity),
                "--pdf-dir",
                str(self.pdf_dir),
                "--output",
                str(output),
            ]
        )
        self.assertEqual(status, 0)
        self.assertEqual(
            json.loads(output.read_text(encoding="utf-8"))["audit"]["selected_rows"],
            8,
        )


if __name__ == "__main__":
    unittest.main()
