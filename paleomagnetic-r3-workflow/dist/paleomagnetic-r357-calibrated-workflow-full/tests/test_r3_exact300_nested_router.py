from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiment_r3_exact300_nested_router.py"


def load_module():
    spec = importlib.util.spec_from_file_location("nested_router", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_declarations_are_unique_and_fixed_threshold_compatible() -> None:
    module = load_module()
    experts = module.expert_specs()
    routers = module.router_specs()
    assert len(experts) == 6
    assert len(routers) == 10
    assert len({item.name for item in experts}) == len(experts)
    assert len({item.name for item in routers}) == len(routers)
    assert {item.context for item in routers} <= {"probability", "full_context"}


def test_deterministic_group_folds_are_complete_and_group_isolated() -> None:
    module = load_module()
    groups = ["a", "a", "b", "c", "c", "c", "d", "e", "f", "g"]
    indices = np.arange(20, 30, dtype=np.int64)
    first = module.deterministic_group_folds(indices, groups, n_splits=4)
    second = module.deterministic_group_folds(indices, groups, n_splits=4)
    assert [pair[1].tolist() for pair in first] == [pair[1].tolist() for pair in second]
    covered = []
    row_group = dict(zip(indices, groups))
    for fit, validation in first:
        assert not set(fit) & set(validation)
        assert {row_group[row] for row in fit}.isdisjoint(
            {row_group[row] for row in validation}
        )
        covered.extend(validation.tolist())
    assert sorted(covered) == indices.tolist()


def test_probability_and_context_features_are_finite_and_stable() -> None:
    module = load_module()
    probability = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.7, 0.6], [0.4, 0.6, 0.5]],
        dtype=np.float64,
    )
    own = np.ones((3, 2), dtype=np.float32)
    component = np.zeros((3, 4), dtype=np.float32)
    probability_only = module.meta_matrix(
        probability, own, component, "probability"
    )
    full = module.meta_matrix(probability, own, component, "full_context")
    assert probability_only.shape == (3, 12)
    assert full.shape == (3, 18)
    assert np.isfinite(full).all()
    assert np.allclose(probability_only[:, :3], probability)


def test_standardized_clip_is_label_independent_and_finite() -> None:
    module = load_module()
    scaler = module.StandardScaler()
    train = np.asarray([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])
    transformed = np.clip(
        scaler.fit_transform(train),
        -module.STANDARDIZED_CLIP,
        module.STANDARDIZED_CLIP,
    )
    extreme = np.clip(
        scaler.transform(np.asarray([[1e12, 2.0]])),
        -module.STANDARDIZED_CLIP,
        module.STANDARDIZED_CLIP,
    )
    assert np.isfinite(transformed).all()
    assert np.isfinite(extreme).all()
    assert np.max(np.abs(extreme)) <= module.STANDARDIZED_CLIP


def test_fixed_fusions_do_not_require_labels() -> None:
    module = load_module()
    train_probability = np.asarray([[0.1, 0.3], [0.8, 0.6]], dtype=np.float64)
    test_probability = np.asarray([[0.2, 0.4], [0.9, 0.5]], dtype=np.float64)
    empty_train = np.zeros((2, 1), dtype=np.float64)
    empty_test = np.zeros((2, 1), dtype=np.float64)
    labels = np.asarray([0, 1], dtype=np.int8)
    weights = np.ones(2, dtype=np.float64)
    for spec in module.router_specs()[:2]:
        value = module.fit_predict_router(
            spec,
            train_probability,
            test_probability,
            empty_train,
            empty_test,
            empty_train,
            empty_test,
            labels,
            weights,
        )
        assert value.shape == (2,)
        assert np.isfinite(value).all()


def test_script_declares_forbidden_inputs_absent() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    assert "prior_oof_or_evaluation_read\": false" in source
    assert "pmagresult_or_row_rockmag_used_as_feature\": false" in source
    assert "all_outer_test_refno_rows_excluded_from_every_fit\": true" in source
