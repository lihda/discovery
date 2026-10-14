from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_r3_scope_fusion_all_oof_v1 as scope_eval


def test_scoped_routes_single_and_multi_rows() -> None:
    single = np.asarray([0.1, 0.2, 0.3])
    multi = np.asarray([0.7, 0.8, 0.9])
    mask = np.asarray([True, False, True])

    assert np.array_equal(scope_eval.scoped(single, multi, mask), [0.1, 0.8, 0.3])


def test_scoped_rejects_misaligned_arrays() -> None:
    with np.testing.assert_raises(scope_eval.ScopeFusionError):
        scope_eval.scoped(
            np.asarray([0.1]),
            np.asarray([0.2, 0.3]),
            np.asarray([True]),
        )


def test_delta_counts_repairs_and_degradations() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    baseline = np.asarray([0.2, 0.7, 0.8, 0.3])
    candidate = np.asarray([0.6, 0.2, 0.8, 0.7])

    assert scope_eval.delta(labels, baseline, candidate) == {
        "repairs": 2,
        "degradations": 1,
        "net_repairs": 1,
        "accuracy_delta": 0.25,
    }
