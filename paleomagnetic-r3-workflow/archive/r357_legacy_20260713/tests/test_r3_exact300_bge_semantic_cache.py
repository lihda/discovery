from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import build_r3_exact300_bge_semantic_cache as builder
from scripts import experiment_r3_exact300_auxiliary_training as auxiliary
from scripts import experiment_r3_exact300_bge_semantic_auxiliary as evaluation


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, documents, **kwargs):
        values = list(documents)
        self.calls.append((values, kwargs))
        return np.asarray(
            [[float(index), float(index + 1)] for index in range(len(values))],
            dtype=np.float32,
        )


class R3Exact300BgeSemanticCacheTests(unittest.TestCase):
    def test_encoder_deduplicates_text_and_restores_row_alignment(self) -> None:
        encoder = FakeEncoder()
        output = builder.encode_unique(encoder, ["alpha", "beta", "alpha"], 7)
        np.testing.assert_array_equal(output[0], output[2])
        self.assertFalse(np.array_equal(output[0], output[1]))
        documents, kwargs = encoder.calls[0]
        self.assertEqual(documents, ["alpha", "beta"])
        self.assertEqual(kwargs["batch_size"], 7)
        self.assertTrue(kwargs["normalize_embeddings"])
        self.assertTrue(kwargs["convert_to_numpy"])

    def test_input_loader_enforces_label_free_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "inputs.npz"
            manifest = root / "inputs.manifest.json"
            values = np.asarray(["evidence"] * builder.EXPECTED_ROWS)
            with cache.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    target=values,
                    abstract=values,
                    local_evidence=values,
                    paper_evidence=values,
                )
            receipt = {
                "schema_version": "r3_scientific_finetune_inputs_v2",
                "status": "PASS",
                "runtime_boundary": {
                    "labels_read": False,
                    "pmagresult_or_rockmag_read": False,
                    "prior_predictions_or_evaluations_read": False,
                    "cache_contains_record_identifiers": False,
                },
                "output": {"sha256": builder.sha256_file(cache)},
            }
            manifest.write_text(json.dumps(receipt), encoding="utf-8")
            with mock.patch.object(
                builder, "EXPECTED_INPUT_SHA256", builder.sha256_file(cache)
            ):
                arrays, _ = builder.load_inputs(cache, manifest)
                self.assertEqual(set(arrays), builder.EXPECTED_COMPONENTS)
                receipt["runtime_boundary"]["labels_read"] = True
                manifest.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "label-free boundary"):
                    builder.load_inputs(cache, manifest)

    def test_builder_has_no_label_or_identifier_input(self) -> None:
        source = inspect.getsource(builder.main).casefold()
        parser_arguments = source.split("args = parser.parse_args", maxsplit=1)[0]
        self.assertNotIn('"--targets"', parser_arguments)
        self.assertNotIn('"--identity"', parser_arguments)
        self.assertNotIn("load_training_labels", source)

    def test_frozen_cache_has_expected_schema_and_source_boundary(self) -> None:
        arrays, manifest = evaluation.validate_bge_cache(
            evaluation.DEFAULT_BGE_CACHE,
            evaluation.DEFAULT_BGE_MANIFEST,
            auxiliary.FULL_IDENTITY,
        )
        self.assertEqual(set(arrays), set(evaluation.EXPECTED_BGE_VIEWS))
        self.assertEqual(manifest["status"], "FROZEN_WITHOUT_LABEL_ACCESS")
        self.assertTrue(
            all(
                matrix.shape
                == (
                    auxiliary.EXPECTED_FULL_PDF_ROWS,
                    evaluation.EXPECTED_BGE_VIEWS[name],
                )
                for name, matrix in arrays.items()
            )
        )


if __name__ == "__main__":
    unittest.main()
