from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from openpyxl import Workbook

from scripts import build_r3_trace_page_graph_features as freeze
from scripts.build_r3_layout_pilot_manifest import freeze_layout_pilot_manifest
from scripts.evaluate_r3_entity_router_pilot import (
    ABLATION_GROUPS,
    EntityRouterEvaluationError,
    MODEL_NAMES,
    ablation_paired_report,
    assert_entity_freeze_unchanged,
    audit_saved_v4_random_forest,
    evaluation_masks,
    load_entity_inputs,
    normalize_entity_candidate,
    paired_changes,
    routing_invariants,
    run_grouped_oof,
    write_outputs,
)


BOUNDARY = {
    "labels_read": False,
    "pmagresult_or_rockmag_read": False,
    "prior_predictions_read": False,
    "evaluation_results_read": False,
    "outer_test_metrics_used_for_selection": False,
}


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


def ablation_declarations() -> dict[str, dict[str, list[int]]]:
    return {
        "flags_only": {"columns": [0, 1]},
        "layout_only": {"columns": [2, 3]},
        "entity_only": {"columns": [4, 5]},
        "full": {"columns": [0, 1, 2, 3, 4, 5]},
    }


def fixed_candidate(
    *,
    alignment_sha256: str,
    matrix_sha256: str,
    feature_names_sha256: str,
    selection_sha256: str,
) -> dict:
    return {
        "schema_version": "r3_entity_router_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "alignment_sha256": alignment_sha256,
        "matrix_sha256": matrix_sha256,
        "feature_names_sha256": feature_names_sha256,
        "pilot_selection_sha256": selection_sha256,
        "fixed_decision_threshold": 0.5,
        "model_candidates": {
            "random_forest": {
                "n_estimators": 12,
                "max_features": 1.0,
                "min_samples_leaf": 1,
                "n_jobs": 1,
                "random_state": 357,
            },
            "hgb": {
                "max_iter": 12,
                "learning_rate": 0.08,
                "max_leaf_nodes": 3,
                "min_samples_leaf": 2,
                "l2_regularization": 0.5,
                "random_state": 357,
            },
        },
        "feature_ablation_candidates": ablation_declarations(),
        "deployment_route": {
            "entity_scope": "frozen_pilot_rows_only",
            "nonpilot_rows": "base_v4_unchanged",
        },
        "runtime_boundary": dict(BOUNDARY),
    }


class EntityRouterPilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_freeze(self, name: str = "fixture") -> dict:
        root = self.root / name
        root.mkdir(parents=True)
        records = [
            {"REFNO": "10", "RESULTNO": str(index + 1), "ROCKNAME": f"Target {index}"}
            for index in range(8)
        ] + [
            {"REFNO": "20", "RESULTNO": "9", "ROCKNAME": "Outside A"},
            {"REFNO": "20", "RESULTNO": "10", "ROCKNAME": "Outside B"},
        ]
        identity_path = root / "identity.xlsx"
        write_identity(identity_path, records)
        pdf_dir = root / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "10.pdf").write_bytes(b"%PDF pilot")
        (pdf_dir / "20.pdf").write_bytes(b"%PDF outside")
        selection_path = root / "selection.json"
        selection = freeze_layout_pilot_manifest(
            identity_path, pdf_dir, selection_path
        )

        matrix = np.zeros((len(records), 6), dtype=np.float32)
        for index in range(8):
            matrix[index] = np.asarray(
                [1.0, 1.0, index / 7.0, (7 - index) / 7.0, index % 2, 0.25],
                dtype=np.float32,
            )
        entity_path = root / "entity.npz"
        np.savez_compressed(entity_path, entity=matrix)
        feature_names = [
            "entity_pilot_selected",
            "entity_document_available",
            "layout_same_row_mass",
            "layout_same_block_mass",
            "entity_target_mass_mean",
            "entity_margin_mean",
        ]
        alignment_sha = freeze._record_alignment_sha256(records)
        matrix_sha = freeze.array_sha256(matrix)
        names_sha = freeze.json_sha256(feature_names)
        candidate = fixed_candidate(
            alignment_sha256=alignment_sha,
            matrix_sha256=matrix_sha,
            feature_names_sha256=names_sha,
            selection_sha256=selection["selection_sha256"],
        )
        candidate_path = root / "candidate.json"
        freeze.atomic_json(candidate_path, candidate)
        audit_path = root / "audit.json"
        freeze.atomic_json(
            audit_path,
            {
                "schema_version": "synthetic_entity_audit_v1",
                "rows": len(records),
                "labels_read": False,
            },
        )
        evidence_graph_path = root / "evidence_graph.jsonl"
        evidence_graph_path.write_text(
            json.dumps({"REFNO": "10", "status": "exclusive"}) + "\n",
            encoding="utf-8",
        )
        code_path = root / "builder_snapshot.py"
        code_path.write_text("FEATURES = ('entity_margin_mean',)\n", encoding="utf-8")
        manifest = {
            "schema_version": "r3_entity_router_features_v1",
            "status": "FROZEN_WITHOUT_LABEL_ACCESS",
            "alignment_sha256": alignment_sha,
            "matrix_sha256": matrix_sha,
            "feature_names": feature_names,
            "feature_names_sha256": names_sha,
            "pilot_selection_sha256": selection["selection_sha256"],
            "feature_groups": ablation_declarations(),
            "inputs": {
                "selection": {
                    "path": str(selection_path),
                    "sha256": freeze.file_sha256(selection_path),
                    "selection_sha256": selection["selection_sha256"],
                },
                "code": [
                    {
                        "path": str(code_path),
                        "sha256": freeze.file_sha256(code_path),
                    }
                ],
            },
            "outputs": {
                "features": {
                    "path": str(entity_path),
                    "sha256": freeze.file_sha256(entity_path),
                },
                "candidate": {
                    "path": str(candidate_path),
                    "sha256": freeze.file_sha256(candidate_path),
                },
                "audit": {
                    "path": str(audit_path),
                    "sha256": freeze.file_sha256(audit_path),
                },
                "evidence_graph": {
                    "path": str(evidence_graph_path),
                    "sha256": freeze.file_sha256(evidence_graph_path),
                },
            },
            "runtime_boundary": dict(BOUNDARY),
        }
        manifest_path = root / "manifest.json"
        freeze.atomic_json(manifest_path, manifest)
        return {
            "root": root,
            "records": records,
            "matrix": matrix,
            "entity_path": entity_path,
            "feature_names": feature_names,
            "candidate": candidate,
            "candidate_path": candidate_path,
            "audit_path": audit_path,
            "evidence_graph_path": evidence_graph_path,
            "selection": selection,
            "selection_path": selection_path,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "code_path": code_path,
        }

    def load(self, fixture: dict):
        return load_entity_inputs(
            records=fixture["records"],
            entity_path=fixture["entity_path"],
            manifest_path=fixture["manifest_path"],
            candidate_path=fixture["candidate_path"],
            audit_path=fixture["audit_path"],
            evidence_graph_path=fixture["evidence_graph_path"],
            selection_path=fixture["selection_path"],
            expected_pilot_rows=8,
        )

    def rewrite_candidate_and_manifest(self, fixture: dict, candidate: dict) -> None:
        freeze.atomic_json(fixture["candidate_path"], candidate)
        manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
        manifest["outputs"]["candidate"]["sha256"] = freeze.file_sha256(
            fixture["candidate_path"]
        )
        freeze.atomic_json(fixture["manifest_path"], manifest)

    def test_valid_freeze_loads_all_groups_and_pilot_mask(self) -> None:
        fixture = self.make_freeze()
        matrix, names, mask, config, payloads = self.load(fixture)
        np.testing.assert_array_equal(matrix, fixture["matrix"])
        self.assertEqual(names, fixture["feature_names"])
        self.assertEqual(int(mask.sum()), 8)
        self.assertTrue(np.all(mask[:8]))
        self.assertTrue(np.all(~mask[8:]))
        self.assertEqual(
            tuple(config["feature_ablation_candidates"]), ABLATION_GROUPS
        )
        self.assertEqual(
            payloads["candidate"]["status"], "FROZEN_BEFORE_LABEL_ACCESS"
        )
        assert_entity_freeze_unchanged(payloads)

    def test_runtime_rewrite_after_initial_validation_is_rejected(self) -> None:
        fixture = self.make_freeze("runtime_rewrite")
        _matrix, _names, _mask, _config, payloads = self.load(fixture)
        fixture["audit_path"].write_text(
            fixture["audit_path"].read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            EntityRouterEvaluationError, "audit changed after validation"
        ):
            assert_entity_freeze_unchanged(payloads)

    def test_any_frozen_file_hash_tamper_is_rejected(self) -> None:
        targets = (
            "entity_path",
            "candidate_path",
            "audit_path",
            "evidence_graph_path",
            "selection_path",
            "code_path",
        )
        for index, key in enumerate(targets):
            with self.subTest(key=key):
                fixture = self.make_freeze(f"tamper_{index}")
                path = fixture[key]
                if path.suffix == ".json":
                    path.write_text(
                        path.read_text(encoding="utf-8") + " ", encoding="utf-8"
                    )
                else:
                    path.write_bytes(path.read_bytes() + b"tamper")
                with self.assertRaisesRegex(
                    EntityRouterEvaluationError, "hash mismatch"
                ):
                    self.load(fixture)

    def test_manifest_and_candidate_require_all_five_false_boundaries(self) -> None:
        fixture = self.make_freeze("manifest_boundary")
        manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
        del manifest["runtime_boundary"]["outer_test_metrics_used_for_selection"]
        freeze.atomic_json(fixture["manifest_path"], manifest)
        with self.assertRaisesRegex(
            EntityRouterEvaluationError,
            "outer_test_metrics_used_for_selection=false",
        ):
            self.load(fixture)

        fixture = self.make_freeze("candidate_boundary")
        candidate = dict(fixture["candidate"])
        candidate["runtime_boundary"] = dict(
            candidate["runtime_boundary"], prior_predictions_read=True
        )
        self.rewrite_candidate_and_manifest(fixture, candidate)
        with self.assertRaisesRegex(
            EntityRouterEvaluationError, "prior_predictions_read=false"
        ):
            self.load(fixture)

    def test_missing_required_file_reference_and_wrong_path_are_rejected(self) -> None:
        fixture = self.make_freeze("missing_ref")
        manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
        del manifest["outputs"]["audit"]
        freeze.atomic_json(fixture["manifest_path"], manifest)
        with self.assertRaisesRegex(EntityRouterEvaluationError, "audit reference"):
            self.load(fixture)

        fixture = self.make_freeze("wrong_path")
        manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
        other = fixture["root"] / "other.json"
        freeze.atomic_json(other, {"different": True})
        manifest["outputs"]["audit"] = {
            "path": str(other),
            "sha256": freeze.file_sha256(other),
        }
        freeze.atomic_json(fixture["manifest_path"], manifest)
        with self.assertRaisesRegex(EntityRouterEvaluationError, "audit path mismatch"):
            self.load(fixture)

    def test_ablation_columns_reject_duplicate_overlap_out_of_range_and_bad_full(self) -> None:
        mutations = (
            (
                "duplicates",
                lambda groups: groups["flags_only"].update(columns=[0, 0]),
                "duplicates",
            ),
            (
                "overlap",
                lambda groups: groups["layout_only"].update(columns=[1, 3]),
                "overlap",
            ),
            (
                "out_of_range",
                lambda groups: (
                    groups["entity_only"].update(columns=[4, 7]),
                    groups["full"].update(columns=[0, 1, 2, 3, 4, 7]),
                ),
                "out of range",
            ),
            (
                "bad_full",
                lambda groups: groups["full"].update(
                    columns=[0, 1, 2, 3, 5, 4]
                ),
                "full columns must equal",
            ),
        )
        for index, (name, mutate, expected) in enumerate(mutations):
            with self.subTest(name=name):
                fixture = self.make_freeze(f"columns_{index}")
                candidate = json.loads(
                    fixture["candidate_path"].read_text(encoding="utf-8")
                )
                mutate(candidate["feature_ablation_candidates"])
                self.rewrite_candidate_and_manifest(fixture, candidate)
                with self.assertRaisesRegex(EntityRouterEvaluationError, expected):
                    self.load(fixture)

        candidate = fixed_candidate(
            alignment_sha256="a" * 64,
            matrix_sha256="b" * 64,
            feature_names_sha256="c" * 64,
            selection_sha256="d" * 64,
        )
        candidate["model_candidates"]["random_forest"]["max_depth"] = 3
        with self.assertRaisesRegex(EntityRouterEvaluationError, "unsupported fields"):
            normalize_entity_candidate(candidate, width=6)

    def test_manifest_and_candidate_group_declarations_must_match(self) -> None:
        fixture = self.make_freeze("group_mismatch")
        manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
        manifest["feature_groups"]["layout_only"]["columns"] = [3, 2]
        manifest["feature_groups"]["full"]["columns"] = [0, 1, 3, 2, 4, 5]
        freeze.atomic_json(fixture["manifest_path"], manifest)
        with self.assertRaisesRegex(EntityRouterEvaluationError, "differ from builder"):
            self.load(fixture)

    def test_nonzero_nonpilot_matrix_is_rejected_after_valid_hash_updates(self) -> None:
        fixture = self.make_freeze("nonpilot")
        matrix = fixture["matrix"].copy()
        matrix[-1, 4] = 1.0
        np.savez_compressed(fixture["entity_path"], entity=matrix)
        matrix_sha = freeze.array_sha256(matrix)
        candidate = json.loads(
            fixture["candidate_path"].read_text(encoding="utf-8")
        )
        candidate["matrix_sha256"] = matrix_sha
        freeze.atomic_json(fixture["candidate_path"], candidate)
        manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
        manifest["matrix_sha256"] = matrix_sha
        manifest["outputs"]["features"]["sha256"] = freeze.file_sha256(
            fixture["entity_path"]
        )
        manifest["outputs"]["candidate"]["sha256"] = freeze.file_sha256(
            fixture["candidate_path"]
        )
        freeze.atomic_json(fixture["manifest_path"], manifest)
        with self.assertRaisesRegex(EntityRouterEvaluationError, "outside.*must be zero"):
            self.load(fixture)

    def test_grouped_oof_all_strategies_and_routing_invariants(self) -> None:
        records: list[dict[str, str]] = []
        labels: list[int] = []
        folds: list[int] = []
        for paper_index, refno in enumerate(("10", "20", "30", "40")):
            fold = 1 if paper_index < 2 else 2
            for target_index in range(8):
                records.append(
                    {
                        "REFNO": refno,
                        "RESULTNO": str(len(records) + 1),
                        "ROCKNAME": f"Target {target_index}",
                    }
                )
                labels.append((paper_index + target_index) % 2)
                folds.append(fold)
        labels_array = np.asarray(labels, dtype=np.int8)
        fold_ids = np.asarray(folds, dtype=np.int8)
        base = np.column_stack(
            (
                np.linspace(0.0, 1.0, len(records)),
                np.asarray(labels, dtype=float) * 0.3,
            )
        )
        entity = np.zeros((len(records), 6), dtype=float)
        entity[:8] = np.column_stack(
            (
                np.ones(8),
                np.ones(8),
                np.linspace(0.0, 1.0, 8),
                np.linspace(1.0, 0.0, 8),
                np.asarray(labels[:8], dtype=float),
                np.full(8, 0.25),
            )
        )
        pilot_mask = np.zeros(len(records), dtype=bool)
        pilot_mask[:8] = True
        candidate = fixed_candidate(
            alignment_sha256="a" * 64,
            matrix_sha256="b" * 64,
            feature_names_sha256="c" * 64,
            selection_sha256="d" * 64,
        )
        config = normalize_entity_candidate(candidate, width=6)
        probabilities, fold_audit = run_grouped_oof(
            records=records,
            labels=labels_array,
            base_matrix=base,
            entity_matrix=entity,
            pilot_mask=pilot_mask,
            fold_ids=fold_ids,
            model_configs=config,
        )
        self.assertEqual(set(probabilities), set(MODEL_NAMES))
        self.assertTrue(
            all(values.dtype == np.float32 for values in probabilities.values())
        )
        self.assertTrue(
            all(np.isfinite(values).all() for values in probabilities.values())
        )
        for family in ("random_forest", "hgb"):
            np.testing.assert_array_equal(
                probabilities[f"pilot_routed_full_{family}"][~pilot_mask],
                probabilities[f"base_{family}"][~pilot_mask],
            )
            np.testing.assert_array_equal(
                probabilities[f"pilot_routed_full_{family}"][pilot_mask],
                probabilities[f"full_{family}"][pilot_mask],
            )
        self.assertEqual(len(fold_audit), 2)
        self.assertTrue(
            all(not row["outer_test_metrics_used_for_selection"] for row in fold_audit)
        )
        route = routing_invariants(probabilities, pilot_mask)
        self.assertTrue(
            route["random_forest"]["nonpilot_probabilities_equal_base_bitwise"]
        )

        split_folds = fold_ids.copy()
        split_folds[0] = 2
        with self.assertRaisesRegex(EntityRouterEvaluationError, "split a REFNO"):
            run_grouped_oof(
                records=records,
                labels=labels_array,
                base_matrix=base,
                entity_matrix=entity,
                pilot_mask=pilot_mask,
                fold_ids=split_folds,
                model_configs=config,
            )

    def test_subsets_paired_repairs_degradations_and_report(self) -> None:
        records = [
            {"REFNO": "10", "RESULTNO": "1", "ROCKNAME": "Target A"},
            {"REFNO": "10", "RESULTNO": "2", "ROCKNAME": "Target A"},
            {"REFNO": "10", "RESULTNO": "3", "ROCKNAME": "Target B"},
            {"REFNO": "20", "RESULTNO": "4", "ROCKNAME": "Target C"},
        ]
        labels = np.asarray([0, 1, 1, 0], dtype=np.int8)
        pilot = np.asarray([True, True, True, False])
        masks = evaluation_masks(records, labels, pilot)
        self.assertEqual(int(masks["pilot_222"].sum()), 3)
        self.assertEqual(int(masks["pilot_identifiable_target"].sum()), 1)
        self.assertEqual(int(masks["pilot_duplicate_identity"].sum()), 2)
        self.assertEqual(int(masks["pilot_mixed_label_posthoc_only"].sum()), 3)

        base = np.asarray([0.6, 0.8, 0.8, 0.2], dtype=np.float32)
        augmented = np.asarray([0.2, 0.3, 0.8, 0.8], dtype=np.float32)
        paired = paired_changes(labels, base, augmented, masks["overall"])
        self.assertEqual(paired["repairs_base_wrong_augmented_right"], 1)
        self.assertEqual(paired["degradations_base_right_augmented_wrong"], 2)
        self.assertEqual(paired["net_repairs"], -1)
        self.assertEqual(paired["changed_predictions"], 3)

        probabilities = {
            name: (base.copy() if name.startswith("base_") else augmented.copy())
            for name in MODEL_NAMES
        }
        ablations = ablation_paired_report(labels, probabilities, masks)
        self.assertIn("base_vs_entity_only", ablations["random_forest"])
        self.assertIn(
            "pilot_mixed_label_posthoc_only",
            ablations["random_forest"]["base_vs_full"],
        )

    def test_saved_v4_oof_is_separate_postfit_audit_only(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
        folds = np.asarray([1, 1, 2, 2], dtype=np.int8)
        probabilities = np.asarray([0.1, 0.9, 0.4, 0.6], dtype=np.float32)
        oof_path = self.root / "v4.npz"
        np.savez_compressed(
            oof_path,
            labels=labels,
            fold_ids=folds,
            random_forest=probabilities,
        )
        report_path = self.root / "v4_report.json"
        freeze.atomic_json(
            report_path,
            {"output": {"path": str(oof_path), "sha256": freeze.file_sha256(oof_path)}},
        )
        audit = audit_saved_v4_random_forest(
            oof_path=oof_path,
            report_path=report_path,
            labels=labels,
            fold_ids=folds,
            reproduced=probabilities.copy(),
        )
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["loaded_after_all_entity_models_fitted"])
        self.assertFalse(audit["used_as_model_feature"])
        self.assertFalse(audit["used_for_candidate_selection"])

        changed = probabilities.copy()
        changed[0] = 0.9
        with self.assertRaisesRegex(EntityRouterEvaluationError, "does not reproduce"):
            audit_saved_v4_random_forest(
                oof_path=oof_path,
                report_path=report_path,
                labels=labels,
                fold_ids=folds,
                reproduced=changed,
            )

    def test_output_npz_report_and_receipt_cover_every_strategy(self) -> None:
        records = [
            {"REFNO": "10", "RESULTNO": "1", "ROCKNAME": "A"},
            {"REFNO": "20", "RESULTNO": "2", "ROCKNAME": "B"},
        ]
        labels = np.asarray([0, 1], dtype=np.int8)
        folds = np.asarray([1, 2], dtype=np.int8)
        probabilities = {
            name: np.asarray([0.2, 0.8], dtype=np.float32) for name in MODEL_NAMES
        }
        report = {
            "schema_version": "synthetic_entity_report",
            "label_source": {"path": "synthetic", "sha256": "0" * 64},
        }
        paths = write_outputs(
            output_dir=self.root / "outputs",
            records=records,
            labels=labels,
            fold_ids=folds,
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
            self.assertTrue(set(MODEL_NAMES).issubset(archive.files))
            for name in MODEL_NAMES:
                self.assertEqual(
                    receipt["rows"][0]["probabilities"][name],
                    round(float(archive[name][0]), 8),
                )


if __name__ == "__main__":
    unittest.main()
