from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from openpyxl import Workbook

from scripts import build_r3_trace_page_graph_features as trace_freeze
from scripts.build_r3_nli_criterion_features import (
    FEATURE_NAMES,
    FOLD_QUOTAS,
    HYPOTHESES,
    NLICriterionFeatureError,
    _row_features,
    build_nli_criterion_features,
    inspect_model_snapshot,
    load_identity,
)
from scripts.r3_nli_premise_selector import select_nli_premises


def write_identity(
    path: Path,
    records: list[dict[str, str]],
    header: tuple[str, ...] = ("REFNO", "RESULTNO", "ROCKNAME"),
) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(header)
    for record in records:
        values = [record.get(name, "") for name in header]
        worksheet.append(values)
    workbook.save(path)
    workbook.close()


def write_minimal_safetensors(path: Path) -> None:
    header = json.dumps(
        {"classifier.bias": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0))


class NLICriterionFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_inputs(self) -> dict[str, Path | list[dict[str, str]]]:
        records: list[dict[str, str]] = []
        fold_ids: list[int] = []
        resultno = 1
        for fold, quota in FOLD_QUOTAS.items():
            for offset in range(quota):
                records.append(
                    {
                        "REFNO": f"{fold}{offset:02d}",
                        "RESULTNO": str(resultno),
                        "ROCKNAME": f"Formation {fold}-{offset}",
                    }
                )
                fold_ids.append(fold)
                resultno += 1
        for name in ("Member A", "Member B"):
            records.append(
                {"REFNO": "multi", "RESULTNO": str(resultno), "ROCKNAME": name}
            )
            fold_ids.append(1)
            resultno += 1

        identity = self.root / "identity.xlsx"
        write_identity(identity, records)
        fold_array = np.asarray(fold_ids, dtype=np.int8)
        alignment = trace_freeze._record_alignment_sha256(records)
        fold_rows = []
        for index, (record, fold) in enumerate(zip(records, fold_ids)):
            document = trace_freeze.json_sha256({"REFNO": record["REFNO"]})
            fold_rows.append(
                {
                    "row_index": index,
                    "REFNO": record["REFNO"],
                    "RESULTNO": record["RESULTNO"],
                    "fold": fold,
                    "document_sha256": document,
                }
            )
        fold_manifest = self.root / "folds.json"
        trace_freeze.atomic_json(
            fold_manifest,
            {
                "schema_version": "r3_trace_refno_fold_manifest_v4",
                "status": "FROZEN_WITHOUT_LABEL_ACCESS",
                "labels_read": False,
                "alignment_sha256": alignment,
                "fold_id_sha256": __import__("hashlib").sha256(
                    fold_array.tobytes()
                ).hexdigest(),
                "rows": fold_rows,
            },
        )

        page_cache = self.root / "pages.jsonl"
        cache_rows = []
        for record in records[:-2]:
            cache_rows.append(
                {
                    "REFNO": record["REFNO"],
                    "pages": [
                        {
                            "page": 1,
                            "section_hints": ["RESULTS"],
                            "text": (
                                f"{record['ROCKNAME']} specimens underwent IRM tests. "
                                "Magnetite carries the remanence and implies a primary origin."
                            ),
                        }
                    ],
                }
            )
        page_cache.write_text(
            "".join(json.dumps(row) + "\n" for row in cache_rows), encoding="utf-8"
        )

        criterion = self.root / "criterion.json"
        normative = self.root / "normative.json"
        criterion.write_text("{}\n", encoding="utf-8")
        normative.write_text("{}\n", encoding="utf-8")
        concepts = self.root / "concepts.json"
        trace_freeze.atomic_json(
            concepts,
            {
                "schema_version": "r3_trace_page_graph_concepts_v4",
                "status": "FROZEN_WITHOUT_LABEL_ACCESS",
                "criterion": {
                    "path": str(criterion),
                    "sha256": trace_freeze.file_sha256(criterion),
                },
                "normative_concepts": {
                    "path": str(normative),
                    "sha256": trace_freeze.file_sha256(normative),
                },
                "concepts": [
                    {"concept_id": "irm", "category": "method", "search_patterns": ["IRM"]},
                    {
                        "concept_id": "magnetite",
                        "category": "carrier",
                        "search_patterns": ["magnetite"],
                    },
                    {
                        "concept_id": "origin",
                        "category": "bridge",
                        "search_patterns": ["primary origin"],
                    },
                ],
            },
        )

        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "num_labels": 3,
                    "id2label": {
                        "0": "contradiction",
                        "1": "entailment",
                        "2": "neutral",
                    },
                }
            ),
            encoding="utf-8",
        )
        write_minimal_safetensors(model / "model.safetensors")
        for name in (
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "added_tokens.json",
        ):
            (model / name).write_text("{}\n", encoding="utf-8")
        (model / "spm.model").write_bytes(b"synthetic sentencepiece")
        return {
            "records": records,
            "identity": identity,
            "fold_manifest": fold_manifest,
            "page_cache": page_cache,
            "concepts": concepts,
            "model": model,
        }

    @staticmethod
    def selector(pages, concepts, target, **kwargs):
        if not isinstance(target, str):
            raise AssertionError("builder must pass the narrow string target interface")
        text = pages[0]["text"]
        premises = []
        for index, group in enumerate(
            ("METHOD_DIAGNOSTIC", "IDENTIFY", "SIGNIFICANCE_COUNTER")
        ):
            premises.append(
                {
                    "page": 1,
                    "section": "RESULTS",
                    "offset_start": index,
                    "offset_end": len(text),
                    "text": text,
                    "text_sha256": trace_freeze.file_sha256
                    if False
                    else "unused-by-builder",
                    "concepts": [group.casefold()],
                    "roles": ["reported_result"],
                    "evidence_mode": "support",
                    "group": group,
                    "score": 1.0 - index * 0.1,
                }
            )
        return {
            "schema_version": "r3_nli_premise_selection_v1",
            "premises": premises,
            "audit": {"selected_count": 3},
        }

    def test_end_to_end_freezes_before_mock_inference_and_keeps_nonpilot_zero(self) -> None:
        inputs = self.make_inputs()
        output = self.root / "output"
        calls = []

        def predictor(pairs, model_snapshot, config):
            self.assertTrue((output / "r3_nli_preinference_freeze_v1.json").is_file())
            self.assertFalse((output / "r3_nli_criterion_features_v1.npz").exists())
            calls.append(len(pairs))
            entailment = {
                "scope": 0.90,
                "method": 0.82,
                "identified": 0.76,
                "inconclusive": 0.30,
                "significance": 0.70,
                "external": 0.08,
                "not_performed": 0.04,
            }
            rows = []
            for pair in pairs:
                value = entailment[pair["hypothesis_id"]]
                rows.append([0.05, value, 0.95 - value])
            return np.asarray(rows, dtype=float)

        outputs = build_nli_criterion_features(
            identity_path=inputs["identity"],
            page_cache_path=inputs["page_cache"],
            fold_manifest_path=inputs["fold_manifest"],
            concept_snapshot_path=inputs["concepts"],
            model_snapshot_path=inputs["model"],
            output_dir=output,
            selector=self.selector,
            predictor=predictor,
            expected_rows=None,
        )
        self.assertEqual(calls, [64 * 13])
        with np.load(outputs["features"]) as archive:
            self.assertEqual(archive.files, ["nli"])
            matrix = archive["nli"]
        self.assertEqual(matrix.shape, (66, len(FEATURE_NAMES)))
        self.assertEqual(FEATURE_NAMES[:3], (
            "nli_pilot_selected",
            "nli_source_available",
            "nli_retrieval_available",
        ))
        self.assertTrue(np.all(matrix[:64, :3] == 1.0))
        self.assertTrue(np.all(matrix[64:] == 0.0))
        chain_index = FEATURE_NAMES.index("nli_criterion_chain_min_support")
        self.assertTrue(np.all(matrix[:64, chain_index] > 0.0))

        selection = json.loads(outputs["selection"].read_text(encoding="utf-8"))
        self.assertEqual(selection["selected_rows"], 64)
        self.assertEqual(
            Counter(row["fold"] for row in selection["rows"]),
            Counter(FOLD_QUOTAS),
        )
        candidate = json.loads(outputs["candidate"].read_text(encoding="utf-8"))
        prefreeze = json.loads(outputs["preinference"].read_text(encoding="utf-8"))
        manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
        self.assertEqual(candidate["matrix_sha256"], trace_freeze.array_sha256(matrix))
        self.assertEqual(
            candidate["preinference_candidate_spec_sha256"],
            prefreeze["candidate_spec_sha256"],
        )
        self.assertEqual(
            candidate["feature_ablation_candidates"]["flags_only"]["nli_columns"],
            [0, 1, 2],
        )
        self.assertEqual(
            {
                name: item["feature"]
                for name, item in candidate["zero_shot_candidates"].items()
            },
            {
                "method_only": "nli_method_only_support",
                "practice_chain": "nli_practice_chain_min_support",
                "normative_chain": "nli_normative_chain_min_support",
            },
        )
        criterion_index = FEATURE_NAMES.index("nli_criterion_chain_min_support")
        practice_index = FEATURE_NAMES.index("nli_practice_chain_min_support")
        np.testing.assert_array_equal(
            matrix[:, criterion_index], matrix[:, practice_index]
        )
        self.assertEqual(candidate["model_candidates"]["random_forest"]["random_state"], 35_720_260)
        self.assertEqual(candidate["model_candidates"]["logistic"]["C"], 1.0)
        self.assertEqual(manifest["matrix_sha256"], candidate["matrix_sha256"])
        self.assertEqual(audit["nli_pairs"], 64 * 13)
        for payload in (selection, candidate, prefreeze, manifest, audit):
            self.assertIs(
                payload["runtime_boundary"][
                    "outer_test_metrics_used_for_selection"
                ],
                False,
            )
        self.assertIs(
            prefreeze["candidate_spec"]["runtime_boundary"][
                "outer_test_metrics_used_for_selection"
            ],
            False,
        )
        self.assertTrue(audit["candidate_frozen_before_inference"])
        self.assertEqual(audit["nonpilot_nonzero_values"], 0)
        evidence = [
            json.loads(line)
            for line in outputs["evidence"].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(evidence), 64 * 13)
        self.assertEqual(
            set(evidence[0]["probabilities"]),
            {"contradiction", "entailment", "neutral"},
        )
        self.assertRegex(evidence[0]["text_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("label", evidence[0])

    def test_real_selector_accepts_the_builder_target_interface(self) -> None:
        payload = select_nli_premises(
            [
                {
                    "page": 1,
                    "section_hints": ["RESULTS"],
                    "text": (
                        "Formation 1-0 specimens underwent IRM tests. "
                        "Magnetite carries the remanence and implies a primary origin."
                    ),
                }
            ],
            [
                {"concept_id": "irm", "category": "method", "search_patterns": ["IRM"]},
                {
                    "concept_id": "magnetite",
                    "category": "carrier",
                    "search_patterns": ["magnetite"],
                },
                {
                    "concept_id": "origin",
                    "category": "bridge",
                    "search_patterns": ["primary origin"],
                },
            ],
            "Formation 1-0",
        )
        self.assertEqual(payload["schema_version"], "r3_nli_premise_selection_v1")
        self.assertGreater(len(payload["premises"]), 0)

    def test_external_premise_cannot_cancel_a_different_current_premise(self) -> None:
        def probability(entailment: float) -> np.ndarray:
            remainder = (1.0 - entailment) / 2.0
            return np.asarray([remainder, entailment, remainder], dtype=np.float64)

        premise_values = {
            0: {
                "scope": probability(0.98),
                "method": probability(0.90),
                "external": probability(0.01),
                "not_performed": probability(0.02),
            },
            1: {
                "scope": probability(0.98),
                "method": probability(0.95),
                "external": probability(0.995),
                "not_performed": probability(0.02),
            },
            2: {
                "scope": probability(0.80),
                "identified": probability(0.60),
                "inconclusive": probability(0.75),
                "external": probability(0.10),
                "not_performed": probability(0.01),
            },
            3: {
                "scope": probability(0.90),
                "significance": probability(0.80),
                "external": probability(0.20),
                "not_performed": probability(0.01),
            },
        }
        global_values = {}
        for hypothesis in HYPOTHESES:
            hypothesis_id = hypothesis["hypothesis_id"]
            global_values[hypothesis_id] = np.asarray(
                [
                    values[hypothesis_id]
                    for values in premise_values.values()
                    if hypothesis_id in values
                ],
                dtype=np.float64,
            ).reshape((-1, 3))

        features = _row_features(
            Counter(
                {
                    "METHOD_DIAGNOSTIC": 2,
                    "IDENTIFY": 1,
                    "SIGNIFICANCE_COUNTER": 1,
                }
            ),
            global_values,
            premise_values,
            source_available=True,
        )

        self.assertAlmostEqual(features["nli_negative_control_max_entailment"], 0.995)
        self.assertAlmostEqual(features["nli_scope_support"], 0.97)
        self.assertAlmostEqual(features["nli_method_scope_gated_support"], 0.90)
        self.assertAlmostEqual(features["nli_identify_scope_gated_support"], 0.70)
        self.assertAlmostEqual(features["nli_significance_scope_gated_support"], 0.70)
        self.assertAlmostEqual(features["nli_normative_chain_min_support"], 0.70)
        self.assertTrue(all("{target}" not in item["template"] for item in HYPOTHESES))

    def test_identity_with_label_column_is_rejected(self) -> None:
        path = self.root / "labelled.xlsx"
        write_identity(
            path,
            [{"REFNO": "1", "RESULTNO": "1", "ROCKNAME": "A", "R3": "1"}],
            header=("REFNO", "RESULTNO", "ROCKNAME", "R3"),
        )
        with self.assertRaisesRegex(NLICriterionFeatureError, "exactly REFNO"):
            load_identity(path)

    def test_model_snapshot_label_order_and_container_are_checked(self) -> None:
        inputs = self.make_inputs()
        info = inspect_model_snapshot(inputs["model"])
        self.assertTrue(info["safetensors"]["container_exact"])
        config = Path(inputs["model"]) / "config.json"
        config.write_text(
            json.dumps(
                {
                    "num_labels": 3,
                    "id2label": {"0": "contradiction", "1": "neutral", "2": "entailment"},
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(NLICriterionFeatureError, "0=contradiction"):
            inspect_model_snapshot(inputs["model"])


if __name__ == "__main__":
    unittest.main()
