from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiment_r3_exact300_nested_router_v2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("nested_router_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v2_candidates_are_unique_and_include_new_views() -> None:
    module = load_module()
    experts = module.expert_specs()
    routers = module.router_specs()
    names = {item.name for item in experts}
    assert len(experts) == 9
    assert len(names) == len(experts)
    assert "expert_hgb_field_signature" in names
    assert "expert_hgb_canonical_signature" in names
    assert "expert_hgb_component_bridge_metadata" in names
    assert {item.context_key for item in routers} == {
        "probability",
        "base",
        "field",
        "metadata",
        "all",
    }


def test_frozen_field_signature_cache_validates_without_labels() -> None:
    module = load_module()
    matrix, manifest = module.validate_field_cache(
        module.DEFAULT_FIELD_CACHE,
        module.DEFAULT_FIELD_MANIFEST,
        module.DEFAULT_FIELD_FREEZE,
        2172,
        "066aaddd5707f67b7a5183d3861d5301b0019f600f769263a17e959dd1e4c0e9",
    )
    assert matrix.shape == (2172, 531)
    assert np.isfinite(matrix).all()
    assert manifest["contains_labels"] is False
    assert manifest["contains_record_identifiers"] is False


def test_v2_matrices_and_contexts_are_aligned_and_finite() -> None:
    module = load_module()
    rows = 7
    numeric = {
        "combined": np.zeros((rows, 3), dtype=np.float32),
        "base_engineered": np.zeros((rows, 1), dtype=np.float32),
    }
    component = {
        "combined_plus_component_bridge": np.zeros((rows, 5), dtype=np.float32),
        "component_bridge": np.zeros((rows, 2), dtype=np.float32),
    }
    field = np.ones((rows, 4), dtype=np.float32)
    metadata = np.ones((rows, 6), dtype=np.float32)
    matrices = module.build_matrices(numeric, component, field, metadata)
    contexts = module.build_router_contexts(
        component["component_bridge"], field, metadata
    )
    assert matrices["canonical_plus_signature"].shape == (rows, 7)
    assert matrices["component_bridge_metadata"].shape == (rows, 11)
    assert contexts["all"].shape == (rows, 12)
    assert all(np.isfinite(value).all() for value in contexts.values())


def test_v2_source_does_not_read_prior_prediction_artifacts() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    assert "prior_oof_or_evaluation_read\": false" in source
    assert "nested_router_v1/report.json" not in source
    assert "nested_router_v1/oof_probabilities.npz" not in source
    assert "v1 report" in source
    assert "never reads" in source
