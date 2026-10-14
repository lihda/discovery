import json
from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook

from scripts.filter_page_cache_for_identity import CacheFilterError, filter_cache


def write_manifest(path: Path, refnos: list[str]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["REFNO", "RESULTNO", "ROCKNAME"])
    for index, refno in enumerate(refnos, start=1):
        sheet.append([refno, str(index), f"Target {index}"])
    workbook.save(path)


class PageCacheFilterTests(unittest.TestCase):
    def test_filter_preserves_selected_raw_json_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            source = root / "all.jsonl"
            output = root / "selected.jsonl"
            write_manifest(manifest, ["10", "30"])
            lines = [
                json.dumps({"REFNO": refno, "pages": [{"page": 1, "text": refno}]}) + "\n"
                for refno in ("10", "20", "30")
            ]
            source.write_text("".join(lines), encoding="utf-8")

            audit = filter_cache(manifest, source, output)

            self.assertEqual(output.read_text(encoding="utf-8"), lines[0] + lines[2])
            self.assertEqual(audit["output_cache"]["unique_refnos"], 2)
            self.assertEqual(audit["ignored_extra_refnos"], 1)
            self.assertTrue(audit["raw_cache_rows_preserved"])
            self.assertFalse(audit["labels_read"])

    def test_filter_rejects_missing_manifest_refno(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "identity.xlsx"
            source = root / "all.jsonl"
            write_manifest(manifest, ["10", "missing"])
            source.write_text(json.dumps({"REFNO": "10", "pages": []}) + "\n")

            with self.assertRaisesRegex(CacheFilterError, "missing"):
                filter_cache(manifest, source, root / "selected.jsonl")


if __name__ == "__main__":
    unittest.main()
