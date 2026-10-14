from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import build_r3_evidence_mil_bags as mil_freeze
from scripts import build_r3_trace_page_graph_features as trace_freeze
from scripts import build_r3_strict_trusted_bag_features as strict_summary


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = ROOT / "data/knowledge/r3_strict_trusted_summary_v1.json"
INSTANCE_NAMES = mil_freeze.instance_feature_names()
INSTANCE_INDEX = {name: index for index, name in enumerate(INSTANCE_NAMES)}


def make_instance(
    *,
    citation: bool = False,
    own_likely: bool = True,
    external: bool = False,
    future: bool = False,
    method_negation: bool = False,
    exclusions: tuple[str, ...] = (),
    trusted: tuple[bool, bool, bool, bool] = (True, False, True, False),
    lexical: tuple[int, int, int, int] = (1, 0, 1, 0),
    scale: str = "sentence_window",
    section: str = "methods",
    score: int = 4,
) -> tuple[dict[str, object], np.ndarray]:
    events = {
        "citation": citation,
        "definition": False,
        "external": external,
        "external_attribution": False,
        "figure_table": False,
        "first_person": True,
        "future": future,
        "method_negation": method_negation,
        "own_likely": own_likely,
        "own_strong": own_likely,
        "passive_execution": False,
        "procedure": True,
        "quantitative": True,
        "result_verb": True,
        "sample": True,
        "score": score,
        "target": False,
    }
    trusted_map = dict(zip(mil_freeze.CATEGORIES, trusted))
    lexical_map = dict(zip(mil_freeze.CATEGORIES, lexical))
    instance: dict[str, object] = {
        "lexical_counts": lexical_map,
        "language_flags": {
            "own_study": True,
            "result": True,
            "citation": citation,
            "adverse": False,
        },
        "event_signals": events,
        "trusted_categories": trusted_map,
        "exclusion_codes": list(exclusions),
        "scale": scale,
        "section_class": section,
    }
    vector = np.zeros(len(INSTANCE_NAMES), dtype=np.float32)
    lexical_total = sum(lexical)
    for category, value in lexical_map.items():
        vector[INSTANCE_INDEX[f"lexical_{category}_count_clip"]] = min(value, 4) / 4
    vector[INSTANCE_INDEX["lexical_total_log"]] = math.log1p(
        lexical_total
    ) / math.log1p(99 * len(mil_freeze.CATEGORIES))
    vector[INSTANCE_INDEX["lexical_category_coverage"]] = sum(
        value > 0 for value in lexical
    ) / len(mil_freeze.CATEGORIES)
    for name in mil_freeze.ALLOWED_SCALES:
        vector[INSTANCE_INDEX[f"scale_{name}"]] = float(scale == name)
    for name in ("methods", "results", "discussion"):
        vector[INSTANCE_INDEX[f"section_{name}"]] = float(section == name)
    vector[INSTANCE_INDEX["own_study_language"]] = 1.0
    vector[INSTANCE_INDEX["result_language"]] = 1.0
    vector[INSTANCE_INDEX["support_without_citation"]] = float(
        lexical_total > 0 and not citation
    )
    for name in strict_summary.EVENT_FEATURES_REVALIDATED:
        vector[INSTANCE_INDEX[f"event_{name}"]] = float(events[name])
    vector[INSTANCE_INDEX["event_score_scaled"]] = np.clip(score / 12, -1, 1)
    for category, value in trusted_map.items():
        vector[INSTANCE_INDEX[f"trusted_{category}"]] = float(value)
    vector[INSTANCE_INDEX["trusted_any"]] = float(any(trusted))
    vector[INSTANCE_INDEX["support_blocked"]] = float(
        lexical_total > 0 and not any(trusted)
    )
    return instance, vector


def attach_instance_identity(
    instance: dict[str, object], global_index: int, bag_offset: int
) -> dict[str, object]:
    value = dict(instance)
    value.update(
        {
            "global_instance_index": global_index,
            "bag_instance_index": bag_offset,
            "chunk_id": f"{global_index + 1:064x}",
            "text_sha256": f"{global_index + 101:064x}",
            "page": global_index + 1,
            "primary_route": "lexical_anchor",
        }
    )
    return value


def write_synthetic_source(
    mil_dir: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    strict_instance, strict_vector = make_instance(citation=True, own_likely=True)
    excluded_instance, excluded_vector = make_instance(
        citation=True,
        own_likely=False,
        exclusions=("CITATION_WITHOUT_OWNERSHIP",),
    )
    second_empty_instance, second_empty_vector = make_instance(
        trusted=(False, False, False, False),
    )
    raw_instances = [strict_instance, excluded_instance, second_empty_instance]
    features = np.vstack(
        [strict_vector, excluded_vector, second_empty_vector]
    ).astype(np.float32)
    indptr = np.asarray([0, 2, 3], dtype=np.int64)
    instances = [
        attach_instance_identity(value, index, index if index < 2 else 0)
        for index, value in enumerate(raw_instances)
    ]
    arrays = {
        "bag_indptr": indptr,
        "instance_features": features,
        "fold_ids": np.asarray([1, 2], dtype=np.int8),
        "source_row_indices": np.asarray([10, 20], dtype=np.int32),
        "refnos": np.asarray(["100", "200"], dtype="U32"),
        "resultnos": np.asarray(["1000", "2000"], dtype="U32"),
        "rocknames": np.asarray(["Alpha", "Beta"], dtype="U512"),
        "document_groups": np.asarray(["a" * 64, "b" * 64], dtype="U64"),
        "instance_bag_indices": np.asarray([0, 0, 1], dtype=np.int32),
        "instance_pages": np.asarray([1, 2, 3], dtype=np.int32),
        "instance_scale_ids": np.asarray([0, 0, 0], dtype=np.int8),
        "instance_route_ids": np.asarray([0, 0, 0], dtype=np.int8),
        "instance_chunk_ids": np.asarray(
            [value["chunk_id"] for value in instances], dtype="U64"
        ),
        "instance_text_sha256": np.asarray(
            [value["text_sha256"] for value in instances], dtype="U64"
        ),
    }
    receipts: list[dict[str, object]] = []
    for bag_index in range(2):
        start = int(indptr[bag_index])
        end = int(indptr[bag_index + 1])
        receipts.append(
            {
                "schema_version": "r3_evidence_mil_bag_receipt_v1",
                "bag_index": bag_index,
                "source_row_index": int(arrays["source_row_indices"][bag_index]),
                "REFNO": str(arrays["refnos"][bag_index]),
                "RESULTNO": str(arrays["resultnos"][bag_index]),
                "ROCKNAME": str(arrays["rocknames"][bag_index]),
                "fold": int(arrays["fold_ids"][bag_index]),
                "document_group": str(arrays["document_groups"][bag_index]),
                "bag_start": start,
                "bag_end": end,
                "instances": instances[start:end],
            }
        )
    feature_path = mil_dir / mil_freeze.FEATURE_NAME
    receipt_path = mil_dir / mil_freeze.RECEIPT_NAME
    manifest_path = mil_dir / mil_freeze.MANIFEST_NAME
    mil_freeze.atomic_npz(feature_path, **arrays)
    mil_freeze.atomic_jsonl(receipt_path, receipts)
    manifest = {
        "schema_version": mil_freeze.MANIFEST_SCHEMA,
        "feature_schema_version": mil_freeze.FEATURE_SCHEMA,
        "status": "COMPLETE_FROZEN_WITHOUT_LABEL_ACCESS",
        "bags": 2,
        "instances": 3,
        "instance_feature_count": len(INSTANCE_NAMES),
        "instance_feature_names": INSTANCE_NAMES,
        "instance_feature_names_sha256": mil_freeze.json_sha256(INSTANCE_NAMES),
        "array_sha256": {
            name: trace_freeze.array_sha256(value) for name, value in arrays.items()
        },
        "outputs": {
            "bags": {
                "path": str(feature_path),
                "sha256": trace_freeze.file_sha256(feature_path),
            },
            "receipt": {
                "path": str(receipt_path),
                "sha256": trace_freeze.file_sha256(receipt_path),
            },
        },
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_read": False,
            "evaluation_results_read": False,
        },
    }
    mil_freeze.atomic_json(manifest_path, manifest)
    return arrays, receipts


class StrictTrustedBagFeatureBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = strict_summary.load_summary_knowledge(KNOWLEDGE_PATH)

    def test_knowledge_freezes_28_inputs_and_62_label_blind_features(self) -> None:
        self.assertEqual(
            tuple(self.kb["summary"]["aggregate_instance_features"]),
            strict_summary.AGGREGATE_INSTANCE_FEATURES,
        )
        self.assertEqual(len(strict_summary.AGGREGATE_INSTANCE_FEATURES), 28)
        self.assertEqual(len(strict_summary.summary_feature_names(self.kb)), 62)
        self.assertLessEqual(
            len(strict_summary.summary_feature_names(self.kb)),
            self.kb["summary"]["maximum_feature_count"],
        )
        self.assertTrue(all(value is False for value in self.kb["runtime_boundary"].values()))

    def test_accepts_owned_citation_with_trusted_support(self) -> None:
        instance, vector = make_instance(citation=True, own_likely=True)
        self.assertTrue(
            strict_summary.is_strict_trusted_instance(
                instance, vector, INSTANCE_INDEX
            )
        )

    def test_rejects_any_exclusion_code(self) -> None:
        instance, vector = make_instance(exclusions=("DEMAG_ONLY",))
        self.assertFalse(
            strict_summary.is_strict_trusted_instance(
                instance, vector, INSTANCE_INDEX
            )
        )

    def test_rejects_blocked_external_future_and_method_negation(self) -> None:
        blocked, blocked_vector = make_instance(
            trusted=(False, False, False, False)
        )
        self.assertFalse(
            strict_summary.is_strict_trusted_instance(
                blocked, blocked_vector, INSTANCE_INDEX
            )
        )
        for field in ("external", "future", "method_negation"):
            with self.subTest(field=field):
                kwargs = {field: True}
                instance, vector = make_instance(**kwargs)
                self.assertFalse(
                    strict_summary.is_strict_trusted_instance(
                        instance, vector, INSTANCE_INDEX
                    )
                )

    def test_rejects_citation_without_likely_ownership(self) -> None:
        instance, vector = make_instance(citation=True, own_likely=False)
        self.assertFalse(
            strict_summary.is_strict_trusted_instance(
                instance, vector, INSTANCE_INDEX
            )
        )

    def test_two_bag_summary_retains_zero_row_and_counts_categories(self) -> None:
        strict_vector = make_instance()[1]
        excluded_vector = make_instance(exclusions=("DEMAG_ONLY",))[1]
        source = {
            "bag_indptr": np.asarray([0, 2, 3], dtype=np.int64),
            "instance_features": np.vstack(
                [strict_vector, excluded_vector, excluded_vector]
            ).astype(np.float32),
            "fold_ids": np.asarray([1, 2], dtype=np.int8),
            "source_row_indices": np.asarray([10, 20], dtype=np.int32),
            "refnos": np.asarray(["1", "2"], dtype="U32"),
            "resultnos": np.asarray(["11", "22"], dtype="U32"),
            "rocknames": np.asarray(["Alpha", "Beta"], dtype="U512"),
            "document_groups": np.asarray(["a" * 64, "b" * 64], dtype="U64"),
        }
        mask = np.asarray([True, False, False], dtype=np.bool_)
        output = strict_summary.build_summary_arrays(
            source, mask, INSTANCE_NAMES, self.kb
        )
        self.assertEqual(output["summary"].shape, (2, 62))
        self.assertEqual(output["available"].tolist(), [True, False])
        self.assertEqual(output["strict_trusted_counts"].tolist(), [1, 0])
        np.testing.assert_allclose(
            output["strict_trusted_fraction"], np.asarray([0.5, 0.0])
        )
        self.assertEqual(
            output["strict_trusted_category_counts"].tolist(),
            [[1, 0, 1, 0], [0, 0, 0, 0]],
        )
        self.assertEqual(output["strict_trusted_indptr"].tolist(), [0, 1, 1])
        self.assertEqual(output["strict_trusted_global_indices"].tolist(), [0])
        self.assertTrue(np.all(output["summary"][1] == 0.0))

    def test_cli_exposes_only_three_label_blind_inputs(self) -> None:
        options = {
            option
            for action in strict_summary.build_parser()._actions
            for option in action.option_strings
        }
        self.assertEqual(
            options,
            {"-h", "--help", "--mil-dir", "--knowledge", "--output-dir"},
        )

    def test_end_to_end_freeze_writes_hashed_self_describing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mil_dir = root / "mil"
            output_dir = root / "summary"
            mil_dir.mkdir()
            write_synthetic_source(mil_dir)
            paths = strict_summary.build_strict_trusted_summary(
                mil_dir=mil_dir,
                knowledge_path=KNOWLEDGE_PATH,
                output_dir=output_dir,
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["status"], "COMPLETE_FROZEN_WITHOUT_LABEL_ACCESS"
            )
            self.assertEqual(manifest["strict_trusted_instances"], 1)
            self.assertEqual(manifest["available_bags"], 1)
            self.assertEqual(manifest["unavailable_bags"], 1)
            with np.load(paths["features"], allow_pickle=False) as output:
                self.assertEqual(output["summary"].shape, (2, 62))
                self.assertEqual(output["available"].tolist(), [True, False])
                self.assertEqual(output["strict_trusted_counts"].tolist(), [1, 0])
                self.assertEqual(len(output["summary_feature_names"]), 62)
                for name in output.files:
                    self.assertEqual(
                        manifest["array_sha256"][name],
                        trace_freeze.array_sha256(output[name]),
                    )
            self.assertEqual(
                manifest["outputs"]["summary"]["sha256"],
                trace_freeze.file_sha256(paths["features"]),
            )
            self.assertEqual(
                manifest["outputs"]["receipt"]["sha256"],
                trace_freeze.file_sha256(paths["receipt"]),
            )

    def test_receipt_npz_identity_and_feature_drift_are_rejected(self) -> None:
        instance, vector = make_instance()
        drifted = vector.copy()
        drifted[INSTANCE_INDEX["trusted_carrier"]] = 0.0
        with self.assertRaisesRegex(
            strict_summary.StrictTrustedSummaryError, "receipt/NPZ feature drift"
        ):
            strict_summary.is_strict_trusted_instance(
                instance, drifted, INSTANCE_INDEX
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mil_dir = root / "mil"
            mil_dir.mkdir()
            _, receipts = write_synthetic_source(mil_dir)
            receipts[0]["REFNO"] = "wrong"
            receipt_path = mil_dir / mil_freeze.RECEIPT_NAME
            mil_freeze.atomic_jsonl(receipt_path, receipts)
            manifest_path = mil_dir / mil_freeze.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"]["receipt"]["sha256"] = trace_freeze.file_sha256(
                receipt_path
            )
            mil_freeze.atomic_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                strict_summary.StrictTrustedSummaryError, "receipt identity drift"
            ):
                strict_summary.build_strict_trusted_summary(
                    mil_dir=mil_dir,
                    knowledge_path=KNOWLEDGE_PATH,
                    output_dir=root / "summary",
                )


if __name__ == "__main__":
    unittest.main()
