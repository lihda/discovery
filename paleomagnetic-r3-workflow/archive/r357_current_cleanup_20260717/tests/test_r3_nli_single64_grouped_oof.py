from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import build_r3_trace_page_graph_features as freeze
from scripts.evaluate_r3_nli_single64_grouped_oof import (
    FLAG_FEATURE_NAMES,
    NliSingle64EvaluationError,
    RF_MODEL_NAMES,
    audit_saved_v4_random_forest,
    evaluation_masks,
    load_nli_inputs,
    normalize_nli_candidate,
    paired_changes,
    routing_invariants,
    run_grouped_oof,
    write_outputs,
)


def boundary() -> dict[str, bool]:
    return {
        "labels_read": False,
        "pmagresult_or_rockmag_read": False,
        "prior_predictions_read": False,
        "evaluation_results_read": False,
        "outer_test_metrics_used_for_selection": False,
    }


ZERO_SHOT_FEATURES = {
    "method_only": "nli_method_only_support",
    "practice_chain": "nli_practice_chain_min_support",
    "normative_chain": "nli_normative_chain_min_support",
}
ZERO_SHOT_STRATEGIES = tuple(
    f"selected_zero_shot_{name}" for name in ZERO_SHOT_FEATURES
)


def candidate(
    alignment: str,
    matrix_sha: str,
    names_sha: str,
    selection_sha: str,
    *,
    feature_names: list[str] | None = None,
    input_sha256: dict[str, str] | None = None,
    hypotheses_sha256: str = "e" * 64,
    selector: dict | None = None,
) -> dict:
    feature_names = feature_names or [
        *FLAG_FEATURE_NAMES,
        *ZERO_SHOT_FEATURES.values(),
        "nli_support",
    ]
    return {
        "schema_version": "r3_nli_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "alignment_sha256": alignment,
        "matrix_sha256": matrix_sha,
        "feature_names_sha256": names_sha,
        "feature_names": feature_names,
        "expected_matrix_shape": [12, len(feature_names)],
        "selection_sha256": selection_sha,
        "hypotheses_sha256": hypotheses_sha256,
        "input_sha256": input_sha256
        or {
            "identity": "1" * 64,
            "page_cache": "2" * 64,
            "fold_manifest": "3" * 64,
            "concept_snapshot": "4" * 64,
        },
        "selector": selector
        or {
            "kind": "module",
            "path": __file__,
            "sha256": freeze.file_sha256(Path(__file__)),
            "callable": "synthetic",
        },
        "fixed_decision_threshold": 0.5,
        "model_candidates": {
            "random_forest": {
                "n_estimators": 12,
                "max_features": 1.0,
                "min_samples_leaf": 1,
                "n_jobs": 1,
                "random_state": 35_720_260,
            },
            "logistic": {
                "C": 1.0,
                "standardize": True,
                "solver": "liblinear",
                "random_state": 35_720_260,
            },
        },
        "feature_ablation_candidates": {
            "base_v4": {"nli_columns": []},
            "flags_only": {
                "nli_columns": [0, 1, 2],
                "feature_names": list(FLAG_FEATURE_NAMES),
            },
            "full_nli": {"nli_columns": "all"},
        },
        "zero_shot_candidates": {
            name: {"feature": feature, "threshold": 0.5}
            for name, feature in ZERO_SHOT_FEATURES.items()
        },
        "runtime_boundary": boundary(),
    }


class NliSingle64GroupedOOFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_freeze(self, rows: int = 12, selected: int = 4):
        records = [
            {
                "REFNO": str(1000 + index),
                "RESULTNO": str(index + 1),
                "ROCKNAME": f"Target {index}",
            }
            for index in range(rows)
        ]
        fold_ids = np.asarray([(index % 2) + 1 for index in range(rows)], dtype=np.int8)
        selected_mask = np.zeros(rows, dtype=bool)
        selected_mask[:selected] = True
        alignment = freeze._record_alignment_sha256(records)

        input_paths = {}
        input_refs = {}
        for name in ("identity", "page_cache", "fold_manifest", "concept_snapshot"):
            path = self.root / f"{name}.input"
            path.write_text(f"synthetic {name}\n", encoding="utf-8")
            input_paths[name] = path
            input_refs[name] = {
                "path": str(path),
                "sha256": freeze.file_sha256(path),
            }

        salt = "r3-nli-single-target-pilot-v1"
        by_fold: dict[int, list[tuple[str, int]]] = {}
        for index in np.flatnonzero(selected_mask):
            record = records[index]
            digest = hashlib.sha256(
                f"{salt}|{record['REFNO']}|{record['RESULTNO']}".encode("utf-8")
            ).hexdigest()
            by_fold.setdefault(int(fold_ids[index]), []).append((digest, int(index)))
        selected_rows = []
        fold_quotas = {}
        for fold in sorted(by_fold):
            ranked = sorted(by_fold[fold])
            fold_quotas[str(fold)] = len(ranked)
            for rank, (digest, index) in enumerate(ranked, start=1):
                selected_rows.append(
                    {
                        "row_index": index,
                        **records[index],
                        "fold": fold,
                        "selection_rank_within_fold": rank,
                        "selection_digest": digest,
                    }
                )
        selection = {
            "schema_version": "r3_nli_single64_selection_v1",
            "status": "FROZEN_WITHOUT_LABEL_ACCESS",
            "selection_rule": (
                "within each frozen V4 fold, select the smallest SHA256 of "
                "salt|REFNO|RESULTNO among PDF-backed REFNO values with exactly one row"
            ),
            "salt": salt,
            "fold_quotas": fold_quotas,
            "eligible_single_target_rows": rows,
            "selected_rows": selected,
            "rows": selected_rows,
            "alignment_sha256": alignment,
            "fold_id_sha256": hashlib.sha256(fold_ids.tobytes()).hexdigest(),
            "inputs": {
                name: input_refs[name] for name in ("identity", "fold_manifest")
            },
            "runtime_boundary": boundary(),
        }
        selection["selection_sha256"] = freeze.json_sha256(
            {"salt": salt, "fold_quotas": fold_quotas, "rows": selected_rows}
        )
        selection_path = self.root / "selection.json"
        freeze.atomic_json(selection_path, selection)

        names = [
            *FLAG_FEATURE_NAMES,
            *ZERO_SHOT_FEATURES.values(),
            "nli_method_support",
            "nli_carrier_support",
        ]
        matrix = np.zeros((rows, len(names)), dtype=np.float32)
        selected_indices = np.flatnonzero(selected_mask)
        matrix[selected_indices, :3] = 1.0
        matrix[selected_indices, 3] = np.linspace(0.2, 0.8, selected)
        matrix[selected_indices, 4] = np.linspace(0.1, 0.9, selected)
        matrix[selected_indices, 5] = np.linspace(0.9, 0.1, selected)
        matrix[selected_indices, 6] = np.linspace(0.3, 0.7, selected)
        matrix[selected_indices, 7] = np.linspace(0.7, 0.3, selected)
        nli_path = self.root / "nli.npz"
        np.savez_compressed(nli_path, nli=matrix)
        matrix_sha = freeze.array_sha256(matrix)
        names_sha = freeze.json_sha256(names)

        hypotheses_path = self.root / "hypotheses.json"
        hypotheses_core = {"synthetic_hypotheses": list(ZERO_SHOT_FEATURES)}
        hypotheses_sha = freeze.json_sha256(hypotheses_core)
        freeze.atomic_json(
            hypotheses_path,
            {
                "schema_version": "r3_nli_criterion_hypotheses_v1",
                "status": "FROZEN_WITHOUT_LABEL_ACCESS",
                **hypotheses_core,
                "hypotheses_sha256": hypotheses_sha,
                "runtime_boundary": boundary(),
            },
        )
        evidence_path = self.root / "evidence.jsonl"
        evidence_path.write_text("{\"synthetic\":true}\n", encoding="utf-8")
        audit_path = self.root / "audit.json"
        freeze.atomic_json(audit_path, {"synthetic": True, "runtime_boundary": boundary()})
        builder_path = self.root / "builder.py"
        builder_path.write_text("# frozen synthetic builder\n", encoding="utf-8")
        selector_path = self.root / "selector.py"
        selector_path.write_text("# frozen synthetic selector\n", encoding="utf-8")
        selector = {
            "kind": "module",
            "path": str(selector_path),
            "sha256": freeze.file_sha256(selector_path),
            "callable": "select_nli_premises",
        }
        input_sha = {name: item["sha256"] for name, item in input_refs.items()}
        draft = candidate(
            alignment,
            matrix_sha,
            names_sha,
            selection["selection_sha256"],
            feature_names=names,
            input_sha256=input_sha,
            hypotheses_sha256=hypotheses_sha,
            selector=selector,
        )
        draft["expected_matrix_shape"] = list(matrix.shape)
        candidate_spec = copy.deepcopy(draft)
        candidate_spec["schema_version"] = "r3_nli_candidate_spec_v1"
        candidate_spec["status"] = "FROZEN_BEFORE_INFERENCE_AND_LABEL_ACCESS"
        candidate_spec.pop("matrix_sha256")

        preinference_path = self.root / "preinference.json"
        preinference = {
            "schema_version": "r3_nli_preinference_freeze_v1",
            "status": "FROZEN_BEFORE_INFERENCE_AND_LABEL_ACCESS",
            "candidate_spec": candidate_spec,
            "candidate_spec_sha256": freeze.json_sha256(candidate_spec),
            "selection": {
                "path": str(selection_path),
                "sha256": freeze.file_sha256(selection_path),
                "selection_sha256": selection["selection_sha256"],
            },
            "hypotheses": {
                "path": str(hypotheses_path),
                "sha256": freeze.file_sha256(hypotheses_path),
                "hypotheses_sha256": hypotheses_sha,
            },
            "builder": {
                "path": str(builder_path),
                "sha256": freeze.file_sha256(builder_path),
            },
            "runtime_boundary": boundary(),
        }
        freeze.atomic_json(preinference_path, preinference)

        frozen_candidate = {
            **candidate_spec,
            "schema_version": "r3_nli_candidate_freeze_v1",
            "status": "FROZEN_BEFORE_LABEL_ACCESS",
            "preinference_candidate_spec_sha256": preinference[
                "candidate_spec_sha256"
            ],
            "preinference_freeze": {
                "path": str(preinference_path),
                "sha256": freeze.file_sha256(preinference_path),
            },
            "matrix_sha256": matrix_sha,
            "evidence_sha256": freeze.file_sha256(evidence_path),
            "finalized_after_label_blind_inference": True,
        }
        candidate_path = self.root / "candidate.json"
        freeze.atomic_json(candidate_path, frozen_candidate)
        manifest = {
            "schema_version": "r3_nli_criterion_features_v1",
            "status": "FROZEN_WITHOUT_LABEL_ACCESS",
            "rows": rows,
            "features": len(names),
            "alignment_sha256": alignment,
            "matrix_sha256": matrix_sha,
            "feature_names": names,
            "feature_names_sha256": names_sha,
            "selection_sha256": selection["selection_sha256"],
            "hypotheses_sha256": hypotheses_sha,
            "contains_labels": False,
            "inputs": input_refs,
            "outputs": {
                name: {"path": str(path), "sha256": freeze.file_sha256(path)}
                for name, path in {
                    "selection": selection_path,
                    "hypotheses": hypotheses_path,
                    "preinference": preinference_path,
                    "features": nli_path,
                    "candidate": candidate_path,
                    "evidence": evidence_path,
                    "audit": audit_path,
                }.items()
            },
            "runtime_boundary": boundary(),
        }
        manifest_path = self.root / "manifest.json"
        freeze.atomic_json(manifest_path, manifest)
        return {
            "records": records,
            "fold_ids": fold_ids,
            "selected": selected,
            "selected_mask": selected_mask,
            "matrix": matrix,
            "names": names,
            "selection": selection,
            "selection_path": selection_path,
            "nli_path": nli_path,
            "candidate": frozen_candidate,
            "candidate_path": candidate_path,
            "preinference": preinference,
            "preinference_path": preinference_path,
            "audit_path": audit_path,
            "manifest": manifest,
            "manifest_path": manifest_path,
        }

    def test_loader_validates_hashes_alignment_selection_and_boundaries(self):
        fixture = self.make_freeze()
        matrix, names, mask, config, payloads = load_nli_inputs(
            records=fixture["records"],
            nli_path=fixture["nli_path"],
            manifest_path=fixture["manifest_path"],
            candidate_path=fixture["candidate_path"],
            selection_path=fixture["selection_path"],
            fold_ids=fixture["fold_ids"],
            expected_selected_rows=fixture["selected"],
        )
        np.testing.assert_array_equal(matrix, fixture["matrix"])
        self.assertEqual(names[:3], list(FLAG_FEATURE_NAMES))
        self.assertEqual(int(mask.sum()), fixture["selected"])
        np.testing.assert_array_equal(mask, fixture["selected_mask"])
        self.assertEqual(config["selected_only_logistic"]["class_weight"], None)
        self.assertEqual(config["selected_only_logistic"]["max_iter"], 2000)
        self.assertEqual(
            set(config["zero_shot_candidates"]), set(ZERO_SHOT_STRATEGIES)
        )
        self.assertEqual(payloads["selection"]["status"], "FROZEN_WITHOUT_LABEL_ACCESS")

        bad = dict(fixture["candidate"])
        bad["runtime_boundary"] = dict(
            bad["runtime_boundary"], prior_predictions_read=True
        )
        freeze.atomic_json(fixture["candidate_path"], bad)
        with self.assertRaisesRegex(NliSingle64EvaluationError, "prior_predictions"):
            load_nli_inputs(
                records=fixture["records"],
                nli_path=fixture["nli_path"],
                manifest_path=fixture["manifest_path"],
                candidate_path=fixture["candidate_path"],
                selection_path=fixture["selection_path"],
                fold_ids=fixture["fold_ids"],
                expected_selected_rows=fixture["selected"],
            )

    def test_loader_rejects_hash_mismatch_and_nonselected_features(self):
        fixture = self.make_freeze()
        changed = fixture["matrix"].copy()
        changed[-1, -1] = 1.0
        np.savez_compressed(fixture["nli_path"], nli=changed)
        with self.assertRaisesRegex(
            NliSingle64EvaluationError, "output features file hash"
        ):
            load_nli_inputs(
                records=fixture["records"],
                nli_path=fixture["nli_path"],
                manifest_path=fixture["manifest_path"],
                candidate_path=fixture["candidate_path"],
                selection_path=fixture["selection_path"],
                fold_ids=fixture["fold_ids"],
                expected_selected_rows=fixture["selected"],
            )

        matrix_sha = freeze.array_sha256(changed)
        candidate_changed = copy.deepcopy(fixture["candidate"])
        candidate_changed["matrix_sha256"] = matrix_sha
        freeze.atomic_json(fixture["candidate_path"], candidate_changed)
        manifest = copy.deepcopy(fixture["manifest"])
        manifest["matrix_sha256"] = matrix_sha
        manifest["outputs"]["features"]["sha256"] = freeze.file_sha256(
            fixture["nli_path"]
        )
        manifest["outputs"]["candidate"]["sha256"] = freeze.file_sha256(
            fixture["candidate_path"]
        )
        freeze.atomic_json(fixture["manifest_path"], manifest)
        with self.assertRaisesRegex(NliSingle64EvaluationError, "outside.*must be zero"):
            load_nli_inputs(
                records=fixture["records"],
                nli_path=fixture["nli_path"],
                manifest_path=fixture["manifest_path"],
                candidate_path=fixture["candidate_path"],
                selection_path=fixture["selection_path"],
                fold_ids=fixture["fold_ids"],
                expected_selected_rows=fixture["selected"],
            )

    def test_candidate_requires_fixed_logistic_and_zero_shot_contract(self):
        fixture = self.make_freeze()
        missing_outer_guard = copy.deepcopy(fixture["candidate"])
        missing_outer_guard["runtime_boundary"].pop(
            "outer_test_metrics_used_for_selection"
        )
        with self.assertRaisesRegex(
            NliSingle64EvaluationError,
            "outer_test_metrics_used_for_selection=false",
        ):
            normalize_nli_candidate(missing_outer_guard)

        bad = dict(fixture["candidate"])
        bad["model_candidates"] = dict(bad["model_candidates"])
        bad["model_candidates"]["logistic"] = dict(
            bad["model_candidates"]["logistic"], class_weight="balanced"
        )
        with self.assertRaisesRegex(NliSingle64EvaluationError, "class_weight=null"):
            normalize_nli_candidate(bad)
        bad_zero = dict(
            fixture["candidate"],
            zero_shot_candidates={
                "method_only": {"feature": "other", "threshold": 0.5}
            },
        )
        with self.assertRaisesRegex(NliSingle64EvaluationError, "invalid feature"):
            normalize_nli_candidate(bad_zero)

    def test_loader_rejects_tampered_selection_fold_and_manifest_output(self):
        fixture = self.make_freeze()
        wrong_folds = fixture["fold_ids"].copy()
        selected_index = int(np.flatnonzero(fixture["selected_mask"])[0])
        wrong_folds[selected_index] = 2 if wrong_folds[selected_index] == 1 else 1
        with self.assertRaisesRegex(NliSingle64EvaluationError, "fold-id hash mismatch"):
            load_nli_inputs(
                records=fixture["records"],
                nli_path=fixture["nli_path"],
                manifest_path=fixture["manifest_path"],
                candidate_path=fixture["candidate_path"],
                selection_path=fixture["selection_path"],
                fold_ids=wrong_folds,
                expected_selected_rows=fixture["selected"],
            )

        fixture = self.make_freeze()
        fixture["audit_path"].write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(
            NliSingle64EvaluationError, "output audit file hash"
        ):
            load_nli_inputs(
                records=fixture["records"],
                nli_path=fixture["nli_path"],
                manifest_path=fixture["manifest_path"],
                candidate_path=fixture["candidate_path"],
                selection_path=fixture["selection_path"],
                fold_ids=fixture["fold_ids"],
                expected_selected_rows=fixture["selected"],
            )

    def test_grouped_rf_selected_logistic_zero_shot_and_route(self):
        records = []
        labels = []
        folds = []
        selected = []
        for fold in range(1, 5):
            for offset in range(8):
                index = len(records)
                records.append(
                    {
                        "REFNO": str(1000 + index),
                        "RESULTNO": str(index + 1),
                        "ROCKNAME": f"Target {index}",
                    }
                )
                labels.append((offset + fold) % 2)
                folds.append(fold)
                selected.append(offset < 4)
        labels = np.asarray(labels, dtype=np.int8)
        folds = np.asarray(folds, dtype=np.int8)
        selected = np.asarray(selected, dtype=bool)
        base = np.column_stack(
            (np.linspace(0, 1, len(records)), labels * 0.25, folds / 4.0)
        )
        names = [*FLAG_FEATURE_NAMES, *ZERO_SHOT_FEATURES.values(), "nli_support"]
        nli = np.zeros((len(records), len(names)), dtype=np.float64)
        nli[selected, :3] = 1.0
        nli[selected, 3] = np.where(labels[selected] == 1, 0.8, 0.2)
        nli[selected, 4] = np.where(labels[selected] == 1, 0.7, 0.3)
        nli[selected, 5] = np.where(labels[selected] == 1, 0.6, 0.4)
        nli[selected, 6] = np.where(labels[selected] == 1, 1.0, -1.0)
        config = normalize_nli_candidate(
            candidate("a" * 64, "b" * 64, "c" * 64, "d" * 64)
        )
        probabilities, audit = run_grouped_oof(
            records=records,
            labels=labels,
            base_matrix=base,
            nli_matrix=nli,
            selected_mask=selected,
            fold_ids=folds,
            feature_names=names,
            model_configs=config,
        )
        expected_names = {
            *RF_MODEL_NAMES,
            "selected_only_standardized_logistic",
            *ZERO_SHOT_STRATEGIES,
        }
        self.assertEqual(set(probabilities), expected_names)
        for name in RF_MODEL_NAMES:
            self.assertTrue(np.isfinite(probabilities[name]).all())
        self.assertTrue(
            np.isfinite(probabilities["selected_only_standardized_logistic"][selected]).all()
        )
        self.assertTrue(
            np.isnan(probabilities["selected_only_standardized_logistic"][~selected]).all()
        )
        for offset, strategy in enumerate(ZERO_SHOT_STRATEGIES, start=3):
            np.testing.assert_allclose(
                probabilities[strategy][selected], nli[selected, offset]
            )
            self.assertTrue(np.isnan(probabilities[strategy][~selected]).all())
        route = routing_invariants(probabilities, selected)
        self.assertTrue(route["nonpilot_probabilities_equal_base_bitwise"])
        self.assertEqual(len(audit), 4)
        self.assertTrue(
            all(row["standardizer_fitted_on_selected_training_rows_only"] for row in audit)
        )

        masks = evaluation_masks(records, selected)
        self.assertEqual(int(masks["single_target_global"].sum()), len(records))
        change = paired_changes(
            labels,
            probabilities["base_random_forest"],
            probabilities["pilot_routed_full_nli_random_forest"],
            masks["overall"],
        )
        self.assertEqual(
            change["augmented_correct"] - change["base_correct"],
            change["net_repairs"],
        )

    def test_output_receipt_uses_null_outside_selected_scope(self):
        records = [
            {"REFNO": str(index), "RESULTNO": str(index), "ROCKNAME": f"T{index}"}
            for index in range(4)
        ]
        labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
        folds = np.asarray([1, 1, 2, 2], dtype=np.int8)
        expected_names = (
            *RF_MODEL_NAMES,
            "selected_only_standardized_logistic",
            *ZERO_SHOT_STRATEGIES,
        )
        probabilities = {
            name: np.asarray([0.1, 0.8, 0.2, 0.9], dtype=np.float32)
            for name in expected_names
        }
        for name in ("selected_only_standardized_logistic", *ZERO_SHOT_STRATEGIES):
            probabilities[name][2:] = np.nan
        report = {
            "schema_version": "synthetic_nli_report",
            "label_source": {"path": "synthetic", "sha256": "0" * 64},
        }
        paths = write_outputs(
            output_dir=self.root / "output",
            records=records,
            labels=labels,
            fold_ids=folds,
            probabilities=probabilities,
            report=report,
        )
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        self.assertIsNone(
            receipt["rows"][2]["probabilities"]["selected_only_standardized_logistic"]
        )
        for name in ZERO_SHOT_STRATEGIES:
            self.assertIsNone(receipt["rows"][2]["predictions"][name])
        self.assertEqual(
            receipt["report"]["sha256"], freeze.file_sha256(paths["report"])
        )

    def test_saved_v4_rf_is_postfit_audit_only(self):
        labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
        folds = np.asarray([1, 1, 2, 2], dtype=np.int8)
        rf = np.asarray([0.1, 0.9, 0.4, 0.6], dtype=np.float32)
        oof = self.root / "v4.npz"
        np.savez_compressed(oof, labels=labels, fold_ids=folds, random_forest=rf)
        report = self.root / "v4.json"
        freeze.atomic_json(
            report,
            {"output": {"path": str(oof), "sha256": freeze.file_sha256(oof)}},
        )
        audit = audit_saved_v4_random_forest(
            oof_path=oof,
            report_path=report,
            labels=labels,
            fold_ids=folds,
            reproduced=rf.copy(),
        )
        self.assertTrue(audit["exact_probability_array_match"])
        self.assertTrue(audit["loaded_after_all_nli_models_fitted"])
        self.assertFalse(audit["used_as_model_feature"])
        changed = rf.copy()
        changed[0] = 0.9
        with self.assertRaisesRegex(NliSingle64EvaluationError, "does not reproduce"):
            audit_saved_v4_random_forest(
                oof_path=oof,
                report_path=report,
                labels=labels,
                fold_ids=folds,
                reproduced=changed,
            )


if __name__ == "__main__":
    unittest.main()
