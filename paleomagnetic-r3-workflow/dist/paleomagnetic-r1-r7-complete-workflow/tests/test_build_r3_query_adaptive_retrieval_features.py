from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.build_r3_query_adaptive_retrieval_features import (
    MODEL_REVISION,
    MODEL_RUNTIME_FILES,
    POOL_CATEGORIES,
    SUPPORT_CATEGORIES,
    RetrievalFeatureError,
    acquire_output_lock,
    cosine_similarity_matrix,
    inspect_model_snapshot,
    invalid_embedding_rows,
    load_query_kb,
    summarize_row,
)
from scripts.r3_scientific_evidence_chunks import EvidenceChunk


ROOT = Path(__file__).resolve().parents[1]


class QueryAdaptiveRetrievalBuilderTests(unittest.TestCase):
    def test_cosine_matrix_is_finite_bounded_and_aligned(self) -> None:
        left = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        right = np.asarray([[0.6, 0.8]], dtype=np.float32)
        np.testing.assert_allclose(
            cosine_similarity_matrix(left, right),
            np.asarray([[0.6], [0.8]], dtype=np.float32),
        )
        with self.assertRaises(RetrievalFeatureError):
            cosine_similarity_matrix(left, np.ones((1, 3), dtype=np.float32))

    def test_embedding_validation_rejects_finite_but_unnormalized_rows(self) -> None:
        values = np.asarray(
            [[1.0, 0.0], [0.6, 0.8], [1.0e20, 0.0], [np.nan, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            invalid_embedding_rows(values),
            np.asarray([False, False, True, True]),
        )

    def test_output_lock_rejects_a_second_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = acquire_output_lock(Path(temporary))
            try:
                with self.assertRaises(RetrievalFeatureError):
                    acquire_output_lock(Path(temporary))
            finally:
                first.close()

    def test_query_kb_is_narrow_and_label_free(self) -> None:
        payload = load_query_kb(
            ROOT / "data/knowledge/r3_query_adaptive_retrieval_v1.json"
        )
        self.assertEqual(set(payload["target_query_templates"]), set(SUPPORT_CATEGORIES))
        self.assertFalse(payload["runtime_boundary"]["labels_read"])
        self.assertFalse(payload["runtime_boundary"]["pmagresult_or_rockmag_read"])

    def test_model_snapshot_inspection_hashes_every_runtime_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / MODEL_REVISION
            snapshot.mkdir()
            for name in MODEL_RUNTIME_FILES:
                path = snapshot / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((name + "\n").encode("ascii"))
            audit = inspect_model_snapshot(snapshot)
            self.assertEqual(audit["revision"], MODEL_REVISION)
            self.assertEqual(len(audit["files"]), len(MODEL_RUNTIME_FILES))
            self.assertEqual(len(audit["snapshot_sha256"]), 64)

    def test_summarizer_is_finite_and_target_sensitive(self) -> None:
        chunks = [
            EvidenceChunk(
                chunk_id="a" * 64,
                scale="sentence_window",
                page=2,
                section="results",
                text="We measured IRM and identified magnetite as the carrier.",
                text_sha256="b" * 64,
                word_count=9,
                lexical_counts=(1, 0, 1, 0),
                own_study=True,
                result_language=True,
                citation_language=False,
                adverse_language=False,
            ),
            EvidenceChunk(
                chunk_id="c" * 64,
                scale="paragraph",
                page=3,
                section="discussion",
                text="The mineralogy supports a primary remanence origin.",
                text_sha256="d" * 64,
                word_count=8,
                lexical_counts=(0, 0, 0, 1),
                own_study=False,
                result_language=True,
                citation_language=False,
                adverse_language=False,
            ),
        ]
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        queries = [
            {"query_id": "m", "category": "method", "mode": "support", "text": "m"},
            {"query_id": "d", "category": "diagnostic", "mode": "support", "text": "d"},
            {"query_id": "c", "category": "carrier", "mode": "support", "text": "c"},
            {"query_id": "b", "category": "bridge", "mode": "support", "text": "b"},
            {"query_id": "n", "category": "counter", "mode": "counter", "text": "n"},
        ]
        query_embeddings = np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        target = {
            "method": np.asarray([1.0, 0.0], dtype=np.float32),
            "diagnostic": np.asarray([0.8, 0.2], dtype=np.float32),
            "carrier": np.asarray([1.0, 0.0], dtype=np.float32),
            "bridge": np.asarray([0.0, 1.0], dtype=np.float32),
        }
        stats = {
            f"{query['query_id']}::{scale}": {
                "documents": 10.0,
                "median": 0.3,
                "mad": 0.1,
                "q90": 0.7,
                "q95": 0.8,
            }
            for query in queries
            for scale in ("all", "sentence_window", "paragraph", "page_window", "section_window")
        }
        compact, names, pooled, receipt = summarize_row(
            chunks=chunks,
            embeddings=embeddings,
            query_specs=queries,
            query_embeddings=query_embeddings,
            target_embeddings=target,
            distractor_stats=stats,
            chunk_audit={
                "pages": 3,
                "body_characters": 500,
                "reference_characters_excluded": 100,
            },
            top_k=2,
            rrf_k=60,
        )
        self.assertEqual(compact.shape, (len(names),))
        self.assertTrue(np.isfinite(compact).all())
        self.assertEqual(pooled.shape, ((1 + len(POOL_CATEGORIES)) * 2,))
        self.assertTrue(np.isfinite(pooled).all())
        self.assertEqual(receipt["top_by_category"]["method"]["chunk_id"], "a" * 64)
        self.assertEqual(receipt["top_by_category"]["bridge"]["chunk_id"], "c" * 64)


if __name__ == "__main__":
    unittest.main()
