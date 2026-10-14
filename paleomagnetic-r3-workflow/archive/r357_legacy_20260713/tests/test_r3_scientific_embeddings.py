from __future__ import annotations

import sys
from pathlib import Path
import hashlib
import re
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_scientific_embeddings as builder
import build_r3_scientific_finetune_inputs as finetune_inputs
import experiment_r3_scientific_embeddings as experiment
import experiment_r3_scientific_finetune as finetune


class R3ScientificEmbeddingTest(unittest.TestCase):
    def test_abstract_proxy_uses_summary_before_introduction(self) -> None:
        text = """[[PAGE 1]]
Journal header
Palaeomagnetism of the Example Formation
A. Author
Summary. IRM acquisition and Curie-temperature measurements identify
hematite as the remanence carrier.
1 Introduction
This later text must not enter the abstract proxy.
"""
        proxy = builder.extract_abstract_proxy(text, ["Example Formation"])

        self.assertIn("Example Formation [SEP]", proxy)
        self.assertIn("IRM acquisition", proxy)
        self.assertNotIn("later text", proxy)

    def test_criterion_pool_selects_query_nearest_chunks(self) -> None:
        embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32
        )
        query = np.asarray([1.0, 0.0], dtype=np.float32)

        mean, maximum, top1, top3 = builder.criterion_pool(
            embeddings, [0, 0, 1], 2, query
        )

        np.testing.assert_allclose(top1[0], [1.0, 0.0])
        np.testing.assert_allclose(top1[1], builder.l2_normalize([[0.8, 0.2]])[0])
        self.assertEqual(mean.shape, (2, 2))
        self.assertEqual(maximum.shape, (2, 2))
        self.assertEqual(top3.shape, (2, 2))

    def test_alignment_hash_changes_with_row_order(self) -> None:
        first = {"REFNO": "1", "RESULTNO": "2", "ROCKNAME": "Alpha"}
        second = {"REFNO": "1", "RESULTNO": "3", "ROCKNAME": "Beta"}
        self.assertNotEqual(
            builder.alignment_sha256([first, second]),
            builder.alignment_sha256([second, first]),
        )

    def test_source_stratified_pilot_is_deterministic_and_group_complete(self) -> None:
        records = [
            {"REFNO": str(paper), "RESULTNO": f"{paper}-1", "ROCKNAME": "Alpha"}
            for paper in range(8)
        ]
        records.extend(
            [
                {"REFNO": "8", "RESULTNO": "8-1", "ROCKNAME": "Alpha"},
                {"REFNO": "8", "RESULTNO": "8-2", "ROCKNAME": "Beta"},
                {"REFNO": "9", "RESULTNO": "9-1", "ROCKNAME": "Alpha"},
                {"REFNO": "9", "RESULTNO": "9-2", "ROCKNAME": "Beta"},
            ]
        )

        first, first_audit = builder.source_stratified_pilot(records, 5)
        second, second_audit = builder.source_stratified_pilot(records, 5)

        self.assertEqual(first, second)
        self.assertEqual(first_audit, second_audit)
        selected = {record["REFNO"] for record in first}
        self.assertEqual(len(selected), 5)
        self.assertEqual(
            first_audit["selected_strata"]["multi_target__rows_2"], 1
        )
        for refno in selected:
            self.assertEqual(
                sum(record["REFNO"] == refno for record in first),
                sum(record["REFNO"] == refno for record in records),
            )

    def test_expert_shortlist_keeps_taxonomy_rich_chunk(self) -> None:
        concepts = [
            {"patterns": [re.compile(r"\bIRM\b", re.I), re.compile("hysteresis", re.I)]}
        ]
        text = "\n\n".join(
            ["geology context"] * 7
            + ["IRM acquisition and hysteresis identify magnetite"]
        )

        chunks = builder.select_expert_chunks(
            text,
            concepts,
            target_prefix="Example Formation",
            target_chars=30,
            maximum_chunks=3,
        )

        self.assertEqual(len(chunks), 3)
        self.assertTrue(any("IRM acquisition" in chunk for chunk in chunks))
        self.assertTrue(all(chunk.startswith("Target: Example Formation") for chunk in chunks))

    def test_feature_cache_rejects_identifier_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            arrays = {
                name: np.zeros((2, 3), dtype=np.float32)
                for name in experiment.EXPECTED_ARRAYS
            }
            arrays["refnos"] = np.asarray(["1", "2"])
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "extra=.*refnos"):
                experiment.load_feature_views(path, 2)

    def test_probability_cache_rejects_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            with self.assertRaises(ValueError):
                experiment.write_probability_npz(
                    path, {"labels": np.asarray([0.2], dtype=np.float32)}
                )

    def test_frozen_small_screen_protocol_and_baseline_match(self) -> None:
        protocol = ROOT / "tmp/r3_small_screen_v1/screening_protocol.json"
        self.assertEqual(
            hashlib.sha256(protocol.read_bytes()).hexdigest(),
            experiment.SCREENING_PROTOCOL_SHA256,
        )
        baseline = experiment.load_taxonomy_baseline_report(
            ROOT / "tmp/r3_small_screen_v1/baseline.json",
            identity_path=ROOT / "tmp/r3_small_screen_v1/identity.xlsx",
            expected_rows=640,
        )
        self.assertEqual(baseline["best_full_coverage"]["accuracy"], 0.79375)
        self.assertEqual(len(baseline["best_folds"]), 5)

    def test_multi_target_paper_blend_does_not_change_single_target_paper(self) -> None:
        records = [
            {"REFNO": "a", "ROCKNAME": "Alpha"},
            {"REFNO": "a", "ROCKNAME": "Beta"},
            {"REFNO": "b", "ROCKNAME": "Gamma"},
        ]
        probability = np.asarray([0.2, 0.8, 0.7], dtype=np.float64)

        strategies = experiment.paper_propagation_strategies(
            records, {"semantic": probability}
        )

        np.testing.assert_allclose(
            strategies["semantic__multi_paper_050"], [0.35, 0.65, 0.7]
        )

    def test_finetune_abstract_keeps_summary_and_drops_introduction(self) -> None:
        text = """[[PAGE 1]]
Example title
Summary. IRM and hysteresis identify magnetite as the carrier.
1 Introduction
Later unrelated text.
"""
        value = finetune_inputs.compact_abstract(text)
        self.assertIn("IRM and hysteresis", value)
        self.assertNotIn("Later unrelated", value)

    def test_finetune_expert_chunk_uses_frozen_pattern_density(self) -> None:
        concepts = [{"patterns": [re.compile("IRM", re.I), re.compile("hysteresis", re.I)]}]
        text = "geology only\n\nIRM acquisition and hysteresis identify magnetite"
        self.assertIn(
            "IRM acquisition",
            finetune_inputs.highest_expert_chunk(
                text, concepts, target_chars=45
            ),
        )

    def test_component_token_budgets_preserve_final_evidence(self) -> None:
        class Tokenizer:
            cls_token_id = 101
            sep_token_id = 102
            pad_token_id = 0

            @staticmethod
            def encode(
                text: str,
                *,
                add_special_tokens: bool,
                truncation: bool,
                max_length: int,
            ) -> list[int]:
                self.assertFalse(add_special_tokens)
                self.assertTrue(truncation)
                return [7] * min(len(text.split()), max_length)

        components = {
            name: np.asarray(["word " * 500]) for name in finetune.COMPONENT_ORDER
        }
        ids, mask = finetune.encode_components(Tokenizer(), components)
        expected = 1 + sum(finetune.COMPONENT_TOKEN_BUDGET.values()) + 4
        self.assertEqual(int(mask.sum()), expected)
        self.assertEqual(int((ids[0] == Tokenizer.sep_token_id).sum()), 4)
        self.assertLessEqual(expected, finetune.MAX_SEQUENCE_LENGTH)

    def test_supervised_fold_parser_and_smoke_gate_are_frozen(self) -> None:
        self.assertEqual(finetune.parse_folds("1,3"), [1, 3])
        self.assertEqual(finetune.parse_folds("all"), [1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            finetune.parse_folds("0")
        self.assertFalse(finetune.fold1_smoke_required([1]))
        self.assertTrue(finetune.fold1_smoke_required([2]))
        self.assertTrue(finetune.fold1_smoke_required([1, 2, 3, 4, 5]))
        passed = finetune.smoke_decision(0.88333333, 0.85, stable=True)
        failed = finetune.smoke_decision(0.86666667, 0.85, stable=True)
        self.assertTrue(passed["continue_to_all_folds"])
        self.assertFalse(failed["continue_to_all_folds"])

    def test_supervised_probability_cache_rejects_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                finetune.probability_cache(
                    Path(directory) / "bad.npz",
                    {"fold_1_labels": np.asarray([0.2])},
                )

    def test_supervised_candidate_protocol_and_label_free_cache_are_frozen(self) -> None:
        protocol = ROOT / "tmp/r3_scientific_finetune/fold1_protocol_exact300_v2.json"
        screening = ROOT / "tmp/r3_exact_300_resultno_v2/screening_protocol.json"
        cache = ROOT / "tmp/r3_scientific_finetune/scincl_exact300_v2_inputs.npz"
        self.assertEqual(
            hashlib.sha256(protocol.read_bytes()).hexdigest(),
            finetune.CANDIDATE_PROTOCOL_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(screening.read_bytes()).hexdigest(),
            finetune.SCREENING_PROTOCOL_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(cache.read_bytes()).hexdigest(),
            "75cc13713f8f98693eb2a545921ba16e3ff317fb4788e214fdd958f857327ac8",
        )
        components = finetune.load_components(cache, 300)
        self.assertEqual(set(components), finetune.EXPECTED_COMPONENTS)

    def test_all_fold_gate_rejects_unbound_smoke_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema or status"):
            finetune.validate_fold1_smoke_report(
                {"smoke_gate": {"continue_to_all_folds": True}},
                expected_inputs={"identity_sha256": "identity"},
                expected_model={"epochs": 4},
                expected_rows=300,
                expected_groups=258,
                expected_test_rows=60,
                baseline_accuracy=0.85,
            )

    def test_all_fold_gate_binds_exact_fold1_artifacts(self) -> None:
        expected_inputs = {
            "identity_sha256": "identity",
            "model_files": {"model.safetensors": "weights"},
        }
        expected_model = {"epochs": 4, "upstream": "malteos/scincl"}
        report = {
            "schema_version": "r3_supervised_scincl_grouped_fold_v1",
            "status": "complete",
            "scope": {
                "evaluated_folds": [1],
                "pdf_only_rows": 300,
                "pdf_only_paper_groups": 258,
            },
            "inputs": dict(expected_inputs),
            "model": dict(expected_model),
            "data": {"rows": 300, "groups": 258},
            "folds": [
                {
                    "fold": 1,
                    "outer_test_rows": 60,
                    "training_stable": True,
                    "outer_test": {"accuracy": 0.88333333},
                }
            ],
            "smoke_gate": finetune.smoke_decision(
                0.88333333, 0.85, stable=True
            ),
        }

        finetune.validate_fold1_smoke_report(
            report,
            expected_inputs=expected_inputs,
            expected_model=expected_model,
            expected_rows=300,
            expected_groups=258,
            expected_test_rows=60,
            baseline_accuracy=0.85,
        )
        report["inputs"]["identity_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity_sha256"):
            finetune.validate_fold1_smoke_report(
                report,
                expected_inputs=expected_inputs,
                expected_model=expected_model,
                expected_rows=300,
                expected_groups=258,
                expected_test_rows=60,
                baseline_accuracy=0.85,
            )


if __name__ == "__main__":
    unittest.main()
