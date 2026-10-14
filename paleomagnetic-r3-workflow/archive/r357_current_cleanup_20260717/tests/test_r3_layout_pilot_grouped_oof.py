from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from openpyxl import Workbook

from scripts import build_r3_trace_page_graph_features as freeze
from scripts.build_r3_layout_pilot_manifest import freeze_layout_pilot_manifest
from scripts.evaluate_r3_layout_pilot_grouped_oof import (
    LayoutPilotEvaluationError,
    MODEL_NAMES,
    ablation_paired_report,
    audit_saved_v4_random_forest,
    evaluation_masks,
    load_layout_inputs,
    normalize_layout_candidate,
    paired_changes,
    routing_invariants,
    run_grouped_oof,
    write_outputs,
)


def write_identity(path: Path, records: list[dict[str, str]]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(("REFNO", "RESULTNO", "ROCKNAME"))
    for record in records:
        worksheet.append(
            (record["REFNO"], record["RESULTNO"], record["ROCKNAME"])
        )
    workbook.save(path)
    workbook.close()


def fixed_candidate(
    alignment_sha256: str,
    matrix_sha256: str,
    selection_sha256: str,
) -> dict:
    return {
        "schema_version": "r3_layout_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "alignment_sha256": alignment_sha256,
        "matrix_sha256": matrix_sha256,
        "pilot_selection_sha256": selection_sha256,
        "fixed_decision_threshold": 0.5,
        "model_candidates": {
            "random_forest": {
                "n_estimators": 12,
                "max_features": 1.0,
                "min_samples_leaf": 1,
                "n_jobs": 1,
                "random_state": 17,
            },
            "hgb": {
                "max_iter": 12,
                "learning_rate": 0.08,
                "max_leaf_nodes": 3,
                "min_samples_leaf": 2,
                "l2_regularization": 0.5,
                "random_state": 17,
            },
        },
        "feature_ablation_candidates": {
            "base_v4": {"layout_columns": []},
            "pilot_flags_only": {
                "layout_columns": [0, 1],
                "feature_names": [
                    "layout_pilot_selected",
                    "layout_document_available",
                ],
            },
            "full_layout": {"layout_columns": "all"},
        },
        "effect_acceptance_gates": {
            "require_nonpilot_rows_unchanged": True,
        },
        "deployment_route": {
            "layout_scope": "frozen_pilot_rows_only",
            "nonpilot_rows": "base_v4_unchanged",
            "membership_feature_is_not_evidence": True,
        },
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_read": False,
            "evaluation_results_read": False,
            "outer_test_metrics_used_for_selection": False,
        },
    }


class LayoutPilotGroupedOOFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_layout_freeze(self):
        records = [
            {"REFNO": "10", "RESULTNO": str(index + 1), "ROCKNAME": f"Target {index}"}
            for index in range(8)
        ]
        identity = self.root / "identity.xlsx"
        write_identity(identity, records)
        pdf_dir = self.root / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "10.pdf").write_bytes(b"%PDF synthetic")
        selection_path = self.root / "selection.json"
        selection = freeze_layout_pilot_manifest(identity, pdf_dir, selection_path)

        layout = np.asarray(
            [[float(index % 2), float(index) / 7.0] for index in range(8)],
            dtype=np.float32,
        )
        layout_path = self.root / "layout.npz"
        np.savez_compressed(layout_path, layout=layout)
        alignment = freeze._record_alignment_sha256(records)
        matrix_sha = freeze.array_sha256(layout)
        candidate = fixed_candidate(
            alignment, matrix_sha, selection["selection_sha256"]
        )
        candidate_path = self.root / "candidate.json"
        freeze.atomic_json(candidate_path, candidate)
        names = ["layout_pilot_selected", "layout_document_available"]
        manifest = {
            "schema_version": "r3_layout_features_v1",
            "status": "FROZEN_WITHOUT_LABEL_ACCESS",
            "alignment_sha256": alignment,
            "matrix_sha256": matrix_sha,
            "feature_names": names,
            "feature_names_sha256": freeze.json_sha256(names),
            "selection_sha256": selection["selection_sha256"],
            "inputs": {
                "selection_manifest": {
                    "path": str(selection_path),
                    "sha256": freeze.file_sha256(selection_path),
                    "selection_sha256": selection["selection_sha256"],
                }
            },
            "outputs": {
                "features": {
                    "path": str(layout_path),
                    "sha256": freeze.file_sha256(layout_path),
                },
                "candidate_freeze": {
                    "path": str(candidate_path),
                    "sha256": freeze.file_sha256(candidate_path),
                },
            },
            "runtime_boundary": {
                "labels_read": False,
                "pmagresult_or_rockmag_read": False,
                "prior_predictions_read": False,
                "evaluation_results_read": False,
            },
        }
        manifest_path = self.root / "manifest.json"
        freeze.atomic_json(manifest_path, manifest)
        return {
            "records": records,
            "layout": layout,
            "layout_path": layout_path,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "candidate": candidate,
            "candidate_path": candidate_path,
            "selection": selection,
            "selection_path": selection_path,
        }

    def test_layout_loader_checks_alignment_hashes_names_and_candidate_boundary(self) -> None:
        fixture = self.make_layout_freeze()
        matrix, names, mask, config, payloads = load_layout_inputs(
            records=fixture["records"],
            layout_path=fixture["layout_path"],
            manifest_path=fixture["manifest_path"],
            candidate_path=fixture["candidate_path"],
            selection_path=fixture["selection_path"],
            expected_pilot_rows=8,
        )
        np.testing.assert_array_equal(matrix, fixture["layout"])
        self.assertEqual(
            names, ["layout_pilot_selected", "layout_document_available"]
        )
        self.assertTrue(mask.all())
        self.assertEqual(config["random_forest"]["n_estimators"], 12)
        self.assertEqual(
            payloads["candidate"]["status"], "FROZEN_BEFORE_LABEL_ACCESS"
        )

        bad_candidate = dict(fixture["candidate"])
        bad_candidate["runtime_boundary"] = dict(
            bad_candidate["runtime_boundary"], prior_predictions_read=True
        )
        freeze.atomic_json(fixture["candidate_path"], bad_candidate)
        with self.assertRaisesRegex(LayoutPilotEvaluationError, "prior_predictions"):
            load_layout_inputs(
                records=fixture["records"],
                layout_path=fixture["layout_path"],
                manifest_path=fixture["manifest_path"],
                candidate_path=fixture["candidate_path"],
                selection_path=fixture["selection_path"],
                expected_pilot_rows=8,
            )

        unsupported = fixed_candidate(
            "a" * 64,
            "b" * 64,
            "c" * 64,
        )
        unsupported["model_candidates"]["random_forest"]["max_depth"] = 4
        with self.assertRaisesRegex(LayoutPilotEvaluationError, "unsupported fields"):
            normalize_layout_candidate(unsupported)

        no_ablation = fixed_candidate("a" * 64, "b" * 64, "c" * 64)
        del no_ablation["feature_ablation_candidates"]
        with self.assertRaisesRegex(
            LayoutPilotEvaluationError, "lacks feature_ablation_candidates"
        ):
            normalize_layout_candidate(no_ablation)

    def test_layout_loader_rejects_npz_hash_mismatch_and_nonzero_nonpilot_rows(self) -> None:
        fixture = self.make_layout_freeze()
        changed = fixture["layout"].copy()
        changed[0, 0] += 1.0
        np.savez_compressed(fixture["layout_path"], layout=changed)
        with self.assertRaisesRegex(LayoutPilotEvaluationError, "NPZ file hash"):
            load_layout_inputs(
                records=fixture["records"],
                layout_path=fixture["layout_path"],
                manifest_path=fixture["manifest_path"],
                candidate_path=fixture["candidate_path"],
                selection_path=fixture["selection_path"],
                expected_pilot_rows=8,
            )

        records = fixture["records"] + [
            {"REFNO": "20", "RESULTNO": "99", "ROCKNAME": "Outside"}
        ]
        alignment = freeze._record_alignment_sha256(records)
        matrix = np.vstack((fixture["layout"], np.asarray([[1.0, 0.0]], dtype=np.float32)))
        matrix_sha = freeze.array_sha256(matrix)
        np.savez_compressed(fixture["layout_path"], layout=matrix)
        candidate = fixed_candidate(
            alignment, matrix_sha, fixture["selection"]["selection_sha256"]
        )
        freeze.atomic_json(fixture["candidate_path"], candidate)
        manifest = dict(fixture["manifest"])
        manifest.update(
            {
                "alignment_sha256": alignment,
                "matrix_sha256": matrix_sha,
                "outputs": {
                    "features": {"sha256": freeze.file_sha256(fixture["layout_path"])},
                    "candidate_freeze": {
                        "sha256": freeze.file_sha256(fixture["candidate_path"])
                    },
                },
            }
        )
        freeze.atomic_json(fixture["manifest_path"], manifest)
        with self.assertRaisesRegex(LayoutPilotEvaluationError, "outside.*must be zero"):
            load_layout_inputs(
                records=records,
                layout_path=fixture["layout_path"],
                manifest_path=fixture["manifest_path"],
                candidate_path=fixture["candidate_path"],
                selection_path=fixture["selection_path"],
                expected_pilot_rows=8,
            )

    def test_fixed_grouped_models_strata_paired_changes_and_output_chain(self) -> None:
        records: list[dict[str, str]] = []
        labels: list[int] = []
        folds: list[int] = []
        for paper_index, refno in enumerate(("10", "20", "30", "40")):
            fold = 1 if paper_index < 2 else 2
            for target_index in range(9):
                identity_index = target_index if target_index < 8 else 0
                records.append(
                    {
                        "REFNO": refno,
                        "RESULTNO": str(len(records) + 1),
                        "ROCKNAME": f"Target {identity_index}",
                    }
                )
                labels.append((target_index + paper_index) % 2)
                folds.append(fold)
        labels_array = np.asarray(labels, dtype=np.int8)
        fold_ids = np.asarray(folds, dtype=np.int8)
        base = np.column_stack(
            (
                np.linspace(0.0, 1.0, len(records)),
                np.asarray(labels, dtype=float) * 0.4,
                np.asarray([int(record["REFNO"]) for record in records], dtype=float)
                / 40.0,
            )
        )
        layout = np.zeros((len(records), 2), dtype=float)
        layout[:9, 0] = np.asarray(labels[:9], dtype=float)
        layout[:9, 1] = np.linspace(0.1, 0.9, 9)
        pilot_mask = np.zeros(len(records), dtype=bool)
        pilot_mask[:9] = True
        candidate = fixed_candidate("a" * 64, "b" * 64, "c" * 64)
        config = normalize_layout_candidate(candidate)

        probabilities, fold_audit = run_grouped_oof(
            records=records,
            labels=labels_array,
            base_matrix=base,
            layout_matrix=layout,
            pilot_mask=pilot_mask,
            fold_ids=fold_ids,
            model_configs=config,
        )
        self.assertEqual(set(probabilities), set(MODEL_NAMES))
        self.assertTrue(all(values.dtype == np.float32 for values in probabilities.values()))
        self.assertTrue(all(np.isfinite(values).all() for values in probabilities.values()))
        np.testing.assert_array_equal(
            probabilities["pilot_routed_full_random_forest"][~pilot_mask],
            probabilities["base_random_forest"][~pilot_mask],
        )
        np.testing.assert_array_equal(
            probabilities["pilot_routed_full_random_forest"][pilot_mask],
            probabilities["augmented_random_forest"][pilot_mask],
        )
        self.assertEqual(len(fold_audit), 2)
        self.assertTrue(
            all(not row["outer_test_metrics_used_for_selection"] for row in fold_audit)
        )

        masks = evaluation_masks(records, labels_array, pilot_mask)
        self.assertEqual(int(masks["pilot_222"].sum()), 9)
        self.assertEqual(int(masks["pilot_distinct_canonical_gte_8"].sum()), 9)
        self.assertEqual(int(masks["pilot_duplicate_identity_ambiguity"].sum()), 2)
        self.assertEqual(int(masks["pilot_mixed_label_paper_diagnostic"].sum()), 9)
        paired = paired_changes(
            labels_array,
            probabilities["base_random_forest"],
            probabilities["augmented_random_forest"],
            masks["overall"],
        )
        self.assertEqual(
            paired["augmented_correct"] - paired["base_correct"],
            paired["net_repairs"],
        )
        ablations = ablation_paired_report(labels_array, probabilities, masks)
        self.assertIn(
            "flags_only_vs_full_layout", ablations["random_forest"]
        )
        self.assertEqual(
            ablations["random_forest"]["base_vs_full_layout"]["pilot_222"],
            ablations["random_forest"]["base_vs_pilot_routed_full"]["pilot_222"],
        )
        route_audit = routing_invariants(probabilities, pilot_mask)
        self.assertTrue(
            route_audit["random_forest"][
                "nonpilot_probabilities_equal_base_bitwise"
            ]
        )

        report = {
            "schema_version": "synthetic_layout_report",
            "label_source": {"path": "synthetic", "sha256": "0" * 64},
        }
        paths = write_outputs(
            output_dir=self.root / "outputs",
            records=records,
            labels=labels_array,
            fold_ids=fold_ids,
            probabilities=probabilities,
            report=report,
        )
        written_report = json.loads(paths["report"].read_text(encoding="utf-8"))
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        self.assertEqual(
            written_report["output"]["sha256"], freeze.file_sha256(paths["oof"])
        )
        self.assertEqual(
            receipt["report"]["sha256"], freeze.file_sha256(paths["report"])
        )
        with np.load(paths["oof"]) as archive:
            for name in MODEL_NAMES:
                self.assertEqual(
                    receipt["rows"][0]["probabilities"][name],
                    round(float(archive[name][0]), 8),
                )

    def test_saved_v4_rf_is_postfit_audit_and_mismatch_can_fail_closed(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
        folds = np.asarray([1, 1, 2, 2], dtype=np.int8)
        probabilities = np.asarray([0.1, 0.9, 0.4, 0.6], dtype=np.float32)
        oof = self.root / "v4.npz"
        np.savez_compressed(
            oof,
            labels=labels,
            fold_ids=folds,
            random_forest=probabilities,
        )
        report = self.root / "v4-report.json"
        freeze.atomic_json(
            report,
            {"output": {"path": str(oof), "sha256": freeze.file_sha256(oof)}},
        )
        audit = audit_saved_v4_random_forest(
            oof_path=oof,
            report_path=report,
            labels=labels,
            fold_ids=folds,
            reproduced=probabilities.copy(),
        )
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["loaded_after_all_layout_models_fitted"])
        self.assertFalse(audit["used_as_model_feature"])
        self.assertFalse(audit["used_for_candidate_selection"])

        changed = probabilities.copy()
        changed[0] = 0.8
        with self.assertRaisesRegex(LayoutPilotEvaluationError, "does not reproduce"):
            audit_saved_v4_random_forest(
                oof_path=oof,
                report_path=report,
                labels=labels,
                fold_ids=folds,
                reproduced=changed,
            )


if __name__ == "__main__":
    unittest.main()
