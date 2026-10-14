from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_r3_stable_production_predictions import (  # noqa: E402
    combine_with_paper_only,
    eligible_record_indices,
)
from r3_stable_production import predict_stable, validate_artifact  # noqa: E402


class ConstantModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        count = matrix.shape[0]
        positive = np.full(count, self.probability, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


class IdentityTransform:
    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix)


class DummyVectorizer:
    def transform(self, documents: list[str]) -> np.ndarray:
        return np.zeros((len(documents), 1), dtype=np.float64)


def artifact() -> dict:
    return {
        "schema_version": "r3_stable_production_model_v1",
        "threshold": 0.5,
        "domain_models": [
            {"name": "d1", "model": ConstantModel(0.4), "scaler": None},
            {"name": "d2", "model": ConstantModel(0.6), "scaler": None},
        ],
        "query_model": {
            "scaler": IdentityTransform(),
            "pca": IdentityTransform(),
            "model": ConstantModel(0.5),
        },
        "front_model": {"vectorizer": DummyVectorizer(), "model": ConstantModel(0.5)},
        "knowledge_models": {
            "k1": ConstantModel(0.8),
            "k2": ConstantModel(0.8),
            "k3": ConstantModel(0.8),
        },
        "fulltext_models": {
            "vectorizer": DummyVectorizer(),
            "models": {
                "f1": ConstantModel(0.9),
                "f2": ConstantModel(0.9),
                "f3": ConstantModel(0.9),
            },
        },
        "weights": {
            "base": {
                "domain_mean4": 0.4,
                "query_pca64_logreg": 0.3,
                "front_word_logreg": 0.3,
            },
            "v2": {"base": 0.5, "knowledge": 0.5},
            "v3": {"v2": 0.5, "fulltext": 0.5},
        },
        "dimensions": {
            "domain": 2,
            "v4": 1,
            "compact": 1,
            "pooled": 1,
            "query": 3,
            "knowledge": 2,
        },
        "training": {"historical_r3_labels_used": True},
    }


class StableProductionTests(unittest.TestCase):
    def test_single_and_multi_target_routes(self) -> None:
        predicted = predict_stable(
            artifact=artifact(),
            domain=np.zeros((2, 2)),
            v4=np.zeros((2, 1)),
            compact=np.zeros((2, 1)),
            pooled=np.zeros((2, 1)),
            front_documents=["single", "multi"],
            single_mask=np.asarray([True, False]),
            knowledge_single=np.zeros((1, 2)),
            full_documents_single=["single full text"],
        )
        self.assertEqual(predicted.route.tolist(), ["single_v3", "multi_no_strict"])
        self.assertAlmostEqual(float(predicted.probability[0]), 0.775)
        self.assertAlmostEqual(float(predicted.probability[1]), 0.5)

    def test_artifact_validation_fails_closed(self) -> None:
        value = artifact()
        del value["query_model"]
        with self.assertRaises(ValueError):
            validate_artifact(value)

    def test_pdf_and_nonempty_text_are_both_required(self) -> None:
        records = [
            {"REFNO": "10", "RESULTNO": "1"},
            {"REFNO": "20", "RESULTNO": "2"},
            {"REFNO": "30", "RESULTNO": "3"},
        ]
        cache = {
            "10": {"pages": [{"text": "evidence"}]},
            "20": {"pages": [{"text": ""}]},
            "30": {"pages": [{"text": "evidence"}]},
        }
        result = eligible_record_indices(records, cache, {"10", "20"})
        self.assertEqual(result.tolist(), [0])

    def test_missing_features_fall_back_to_paper_only(self) -> None:
        records = [
            {"REFNO": "10", "RESULTNO": "1", "ROCKNAME": "A"},
            {"REFNO": "20", "RESULTNO": "2", "ROCKNAME": "B"},
        ]
        paper_rows = [
            {
                "RESULTNO": "1",
                "R3": 0,
                "R3_RESULT_JSON": {"outcome": "NEGATIVE", "final_decision_reasoning": "paper 1"},
            },
            {
                "RESULTNO": "2",
                "R3": 1,
                "R3_RESULT_JSON": {"outcome": "POSITIVE", "final_decision_reasoning": "paper 2"},
            },
        ]
        rows, summary = combine_with_paper_only(
            records=records,
            paper_rows=paper_rows,
            probability_by_result={"1": 0.75},
            route_by_result={"1": "single_v3"},
            component_by_result={"1": {"base_probability": 0.6}},
            receipt_by_result={"1": {"query": "carrier"}},
            model_sha256="abc",
            grouped_oof_accuracy=2316 / 2751,
        )
        self.assertEqual([row["R3"] for row in rows], [1, 1])
        self.assertEqual(
            rows[1]["R3_RESULT_JSON"]["supervised_stable_model"]["status"],
            "FALLBACK_TO_PAPER_ONLY",
        )
        self.assertEqual(rows[0]["R3_RESULT_JSON"]["paper_only_audit"]["score"], 0)
        self.assertEqual(summary["paper_only_fallback_rows"], 1)
        self.assertEqual(summary["route_counts"], {"single_v3": 1})


if __name__ == "__main__":
    unittest.main()
