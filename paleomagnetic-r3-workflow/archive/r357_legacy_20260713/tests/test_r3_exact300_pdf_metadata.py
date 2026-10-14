from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import build_r3_exact300_pdf_metadata_cache as builder
from scripts import experiment_r3_exact300_pdf_metadata_auxiliary as evaluation


PDFINFO_SAMPLE = """Title: forbidden content title
Author: forbidden author
Creator: Acrobat 3.0 Capture Plug-in
Producer: Acrobat 3.0 Import Plug-in
CreationDate: 2002-06-06T07:06:08+08:00
ModDate: 2007-08-26T09:59:44+08:00
Custom Metadata: no
Metadata Stream: yes
Tagged: no
UserProperties: no
Suspects: no
JavaScript: no
Pages: 9
Encrypted: no
Page size: 548 x 742 pts
Page rot: 90
File size: 731556 bytes
Optimized: yes
PDF version: 1.3
"""


class R3Exact300PdfMetadataTests(unittest.TestCase):
    def test_parser_ignores_content_metadata_and_normalizes_pipeline(self) -> None:
        parsed = builder.parse_pdfinfo_output(PDFINFO_SAMPLE)
        self.assertNotIn("Title", parsed)
        self.assertNotIn("Author", parsed)
        self.assertEqual(builder.normalized_family(parsed["Creator"]), "acrobat_capture")
        self.assertEqual(builder.normalized_family(parsed["Producer"]), "acrobat_import")
        self.assertIn("capture", builder.normalized_tokens(parsed["Creator"]))

    def test_vector_encodes_structure_without_identifier_arguments(self) -> None:
        parsed = builder.parse_pdfinfo_output(PDFINFO_SAMPLE)
        vector, summary = builder.metadata_vector(parsed, stat_size=731556, pdfinfo_ok=True)
        values = dict(zip(builder.feature_names(), vector.tolist()))
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(vector.shape, (len(builder.feature_names()),))
        self.assertEqual(values["metadata_stream_yes"], 1.0)
        self.assertEqual(values["optimized_yes"], 1.0)
        self.assertEqual(values["page_rotation_90"], 1.0)
        self.assertEqual(values["scan_pipeline_hint"], 1.0)
        self.assertEqual(summary["creation_year"], 2002)
        self.assertEqual(summary["mod_year"], 2007)
        parameters = inspect.signature(builder.metadata_vector).parameters
        self.assertNotIn("refno", parameters)
        self.assertNotIn("resultno", parameters)
        self.assertNotIn("path", parameters)

    def test_builder_broadcasts_one_paper_vector_without_caching_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf_dir = Path(directory)
            (pdf_dir / "10.pdf").write_bytes(b"%PDF-1.4\nfirst")
            (pdf_dir / "20.pdf").write_bytes(b"%PDF-1.7\nsecond-longer")
            records = [
                {"REFNO": "10", "RESULTNO": "1", "ROCKNAME": "Alpha"},
                {"REFNO": "10", "RESULTNO": "2", "ROCKNAME": "Beta"},
                {"REFNO": "20", "RESULTNO": "3", "ROCKNAME": "Gamma"},
            ]

            def extractor(path: Path):
                return builder.parse_pdfinfo_output(PDFINFO_SAMPLE), True, ""

            matrix, audit = builder.build_features(records, pdf_dir, extractor=extractor)
            self.assertEqual(matrix.shape, (3, len(builder.feature_names())))
            np.testing.assert_array_equal(matrix[0], matrix[1])
            self.assertFalse(np.array_equal(matrix[0], matrix[2]))
            self.assertTrue(audit["paper_features_broadcast_to_result_rows"])
            self.assertEqual(audit["paper_count"], 2)
            self.assertEqual(audit["row_count"], 3)
            self.assertFalse(any(builder.FORBIDDEN_FEATURE_RE.search(name) for name in builder.feature_names()))

    def test_cache_schema_rejects_identifier_or_wrong_dtype(self) -> None:
        rows = 2
        good = np.zeros((rows, len(builder.feature_names())), dtype=np.float32)
        self.assertIn("pdf_metadata", builder.validate_arrays({"pdf_metadata": good}, rows))
        with self.assertRaisesRegex(ValueError, "schema"):
            builder.validate_arrays({"refnos": good}, rows)
        with self.assertRaisesRegex(ValueError, "not float32"):
            builder.validate_arrays({"pdf_metadata": good.astype(np.float64)}, rows)

    def test_freeze_builder_cannot_accept_label_sources(self) -> None:
        parameters = inspect.signature(builder.main).parameters
        self.assertNotIn("targets", parameters)
        source = inspect.getsource(builder)
        self.assertNotIn("load_training_labels", source)
        self.assertNotIn("DEFAULT_TARGETS", source)

    def test_candidate_screen_has_controls_and_metadata_models(self) -> None:
        specs = evaluation.candidate_specs()
        self.assertTrue(any(not spec.uses_pdf_metadata for spec in specs))
        self.assertTrue(any(spec.uses_pdf_metadata for spec in specs))
        self.assertTrue(all(spec.model.startswith(("rf", "hgb")) for spec in specs))


if __name__ == "__main__":
    unittest.main()
