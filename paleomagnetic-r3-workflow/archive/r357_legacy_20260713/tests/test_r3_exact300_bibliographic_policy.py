from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import audit_r3_exact300_bibliographic_policy_coverage as coverage
from scripts import experiment_r3_exact300_bibliographic_policy as experiment


def profile(refno: str, author: str) -> coverage.BibliographicProfile:
    return coverage.BibliographicProfile(
        refno=refno,
        title="Example",
        authors=frozenset({author}),
        year=2001,
        journal="jgr",
        publisher="agu",
        doi_prefixes=frozenset({"10.1029"}),
        source_dois=frozenset({f"10.1029/{refno}"}),
        citation_dois=frozenset(),
        citation_keys=frozenset(),
        source_author_year_keys=frozenset({f"{author}:2001"}),
    )


class R3Exact300BibliographicPolicyTests(unittest.TestCase):
    def test_candidate_family_is_bounded_and_predeclared(self) -> None:
        specs = experiment.candidate_specs()
        self.assertEqual(len(specs), 7)
        self.assertEqual(specs[0].name, "semantic_only")
        self.assertEqual(
            {spec.smoothing for spec in specs[1:]},
            set(experiment.SMOOTHING_STRENGTHS),
        )
        self.assertEqual(
            {spec.fusion_weight for spec in specs[1:]},
            set(experiment.FUSION_WEIGHTS),
        )

    def test_inner_fold_assignment_keeps_refno_together(self) -> None:
        records = [
            {"REFNO": refno}
            for refno in ("A", "A", "A", "B", "B", "C", "D", "E", "F", "G")
        ]
        folds = experiment.balanced_group_fold_ids(
            records, np.arange(len(records), dtype=np.int64)
        )
        by_refno: dict[str, set[int]] = {}
        for record, fold in zip(records, folds):
            by_refno.setdefault(record["REFNO"], set()).add(int(fold))
        self.assertTrue(all(len(values) == 1 for values in by_refno.values()))

    def test_empirical_bayes_author_posterior_is_smoothed(self) -> None:
        profiles = {
            "A": profile("A", "smith"),
            "B": profile("B", "smith"),
            "C": profile("C", "jones"),
        }
        encoder = experiment.fit_policy_encoder(
            ["A", "B", "C"], {"A": 1.0, "B": 1.0, "C": 0.0}, profiles
        )
        probability, confidence, audit = experiment.transform_policy_profile(
            encoder, profile("T", "smith"), 4.0
        )
        self.assertGreater(probability, encoder.prior)
        self.assertLess(probability, 1.0)
        self.assertGreater(confidence, 0.0)
        self.assertGreater(audit["supported_families"], 0)

    def test_confidence_gated_fusion_leaves_unsupported_row_unchanged(self) -> None:
        semantic = np.asarray([0.2, 0.8])
        policy = np.asarray([0.9, 0.1])
        confidence = np.asarray([0.0, 1.0])
        fused = experiment.fuse_probability(
            semantic, policy, confidence, 0.5
        )
        self.assertEqual(fused[0], semantic[0])
        self.assertAlmostEqual(fused[1], 0.45)

    def test_run_fold_does_not_accept_outer_labels(self) -> None:
        signature = inspect.signature(experiment.run_fold)
        self.assertNotIn("exact_labels", signature.parameters)
        self.assertNotIn("metric_record", inspect.getsource(experiment.run_fold))

    def test_oof_artifact_contains_no_identifier_or_label_array(self) -> None:
        original = experiment.EXPECTED_ROWS
        experiment.EXPECTED_ROWS = 2
        try:
            arrays = {
                name: np.asarray([0.2, 0.8])
                for name in (
                    "bibliographic_policy_fusion",
                    "semantic_hgb",
                    "policy_probability",
                    "policy_confidence",
                )
            }
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "oof.npz"
                experiment.write_oof(path, arrays)
                with np.load(path, allow_pickle=False) as payload:
                    self.assertNotIn("labels", payload.files)
                    self.assertNotIn("refnos", payload.files)
        finally:
            experiment.EXPECTED_ROWS = original

    def test_main_freezes_features_before_label_access(self) -> None:
        source = inspect.getsource(experiment.main)
        self.assertLess(
            source.index("base.atomic_json(freeze_path, freeze)"),
            source.index("base.load_training_labels"),
        )
        self.assertNotIn("baseline_probabilities", source)
        self.assertNotIn("oof_probabilities.npz", source)


if __name__ == "__main__":
    unittest.main()
