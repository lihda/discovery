from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import build_r3_page_source_cache as source_cache


class PageSourceCacheTests(unittest.TestCase):
    def test_extractor_tradeoff_does_not_leave_empty_selection_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed = root / "parsed"
            pdfs = root / "pdfs"
            parsed.mkdir()
            pdfs.mkdir()
            (pdfs / "7.pdf").write_bytes(b"synthetic")
            pypdf = [{"page": 1, "text": "a" * 1000, "source": "pypdf"}]
            columns = [
                {"page": index, "text": "b" * 80, "source": "pdfplumber_columns"}
                for index in range(1, 11)
            ]
            with mock.patch.object(source_cache, "pypdf_pages", return_value=pypdf), mock.patch.object(
                source_cache,
                "pdfplumber_pages",
                return_value=columns,
            ):
                pages, metadata = source_cache.choose_pages(
                    "7",
                    parsed,
                    pdfs,
                    pdfplumber_mode="always",
                    pdftotext_mode="never",
                )

        self.assertEqual(metadata["source_kind"], "pdfplumber_columns")
        self.assertEqual(len(pages), 10)


if __name__ == "__main__":
    unittest.main()

