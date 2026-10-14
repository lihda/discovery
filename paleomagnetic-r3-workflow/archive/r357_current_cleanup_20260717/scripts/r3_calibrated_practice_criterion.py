#!/usr/bin/env python3
"""Validate the supervised R3 ROCKMAG-compatible practice criterion v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CRITERION = ROOT / "data/calibration/r3_rockmag_practice_criterion_v2.json"
SCHEMA_VERSION = "r3_calibrated_practice_criterion_schema_v2"
CRITERION_VERSION = "r3_rockmag_practice_criterion_v2"
EXPECTED_ROUTES = {
    "R3.PRACTICE.ROUTE.PERFORMED_METHOD",
    "R3.PRACTICE.ROUTE.SELF_AUTHENTICATING_RESULT",
    "R3.PRACTICE.ROUTE.CARRIER_ATTRIBUTION",
}
EXPECTED_OUTCOMES = {
    "DOES_NOT_MEET_PRACTICE_PROXY",
    "MEETS_PRACTICE_PROXY",
    "ABSTAIN",
    "SKIP_NO_SOURCE",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_RUNTIME = {
    "PMAGRESULT_at_runtime",
    "row_ROCKMAG_at_runtime",
    "R3_labels_at_runtime",
    "prior_predictions_at_runtime",
    "evaluation_results_at_runtime",
    "record_answer_mappings",
}


class R3CalibratedCriterionError(ValueError):
    """Raised when the calibrated criterion violates its declared contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R3CalibratedCriterionError(f"{where} must be an object")
    return value


def require_string_list(value: Any, where: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise R3CalibratedCriterionError(f"{where} must be a nonempty string list")
    if len(value) != len(set(value)):
        raise R3CalibratedCriterionError(f"{where} contains duplicates")
    return value


def validate_criterion(value: Mapping[str, Any]) -> dict[str, Any]:
    criterion = json.loads(json.dumps(require_mapping(value, "$"), ensure_ascii=False))
    required = {
        "schema_version",
        "criterion_version",
        "indicator",
        "criterion_type",
        "status",
        "title",
        "provenance",
        "source_basis",
        "semantic_revision",
        "evidence_routes",
        "exclusions",
        "scope_policy",
        "decision_policy",
        "dual_head_conflict_policy",
        "development_evidence",
        "runtime_policy",
    }
    if set(criterion) != required:
        raise R3CalibratedCriterionError(
            f"criterion top-level keys changed: missing={sorted(required - set(criterion))}, "
            f"unexpected={sorted(set(criterion) - required)}"
        )
    if criterion["schema_version"] != SCHEMA_VERSION:
        raise R3CalibratedCriterionError("unexpected calibrated criterion schema")
    if criterion["criterion_version"] != CRITERION_VERSION:
        raise R3CalibratedCriterionError("unexpected calibrated criterion version")
    if (
        criterion["indicator"] != "R3"
        or criterion["criterion_type"] != "historical_rockmag_compatibility"
        or criterion["status"] != "development_frozen"
    ):
        raise R3CalibratedCriterionError("criterion identity or status is invalid")

    provenance = require_mapping(criterion["provenance"], "provenance")
    for key in ("label_supervision", "rockmag_field_supervision", "evaluation_results_used"):
        if provenance.get(key) is not True:
            raise R3CalibratedCriterionError(f"provenance.{key} must be true")
    for key in (
        "contains_record_identifiers",
        "contains_record_answer_mapping",
        "contains_raw_rockmag_values",
        "contains_source_paper_quotes",
        "independent_external_validation",
    ):
        if provenance.get(key) is not False:
            raise R3CalibratedCriterionError(f"provenance.{key} must be false")

    bases = criterion["source_basis"]
    if not isinstance(bases, list) or len(bases) != 4:
        raise R3CalibratedCriterionError("source_basis must contain four frozen artifacts")
    basis_ids: set[str] = set()
    for index, raw in enumerate(bases):
        basis = require_mapping(raw, f"source_basis[{index}]")
        if set(basis) != {"basis_id", "repository_path", "sha256", "role"}:
            raise R3CalibratedCriterionError(f"source_basis[{index}] keys changed")
        basis_id = str(basis["basis_id"])
        if basis_id in basis_ids:
            raise R3CalibratedCriterionError(f"duplicate source basis {basis_id}")
        basis_ids.add(basis_id)
        path = Path(str(basis["repository_path"]))
        if path.is_absolute() or ".." in path.parts:
            raise R3CalibratedCriterionError("source basis path must be repository-relative")
        if not SHA256_RE.fullmatch(str(basis["sha256"])):
            raise R3CalibratedCriterionError("source basis sha256 is invalid")

    revision = require_mapping(criterion["semantic_revision"], "semantic_revision")
    revised = require_mapping(revision.get("revised_requirement"), "revised_requirement")
    if revised.get("operator") != "any" or set(revised.get("routes") or ()) != {
        "performed_eligible_characterization",
        "self_authenticating_diagnostic_result",
        "explicit_carrier_attribution",
    }:
        raise R3CalibratedCriterionError("semantic revision must use the three alternative routes")
    routes = criterion["evidence_routes"]
    if not isinstance(routes, list) or {
        str(require_mapping(route, "evidence_route").get("route_id")) for route in routes
    } != EXPECTED_ROUTES:
        raise R3CalibratedCriterionError("evidence routes changed")
    for route in routes:
        require_string_list(route.get("requires_all"), f"{route['route_id']}.requires_all")
    require_string_list(criterion["exclusions"], "exclusions")

    decision = require_mapping(criterion["decision_policy"], "decision_policy")
    selective = require_mapping(decision.get("selective_policy"), "selective_policy")
    if set(selective) != EXPECTED_OUTCOMES:
        raise R3CalibratedCriterionError("selective outcomes changed")
    negative = float(selective["DOES_NOT_MEET_PRACTICE_PROXY"]["maximum_probability"])
    positive = float(selective["MEETS_PRACTICE_PROXY"]["minimum_probability"])
    abstain = require_mapping(selective["ABSTAIN"], "selective_policy.ABSTAIN")
    if (
        negative != 0.30
        or positive != 0.70
        or float(abstain["minimum_exclusive_probability"]) != negative
        or float(abstain["maximum_exclusive_probability"]) != positive
    ):
        raise R3CalibratedCriterionError("selective thresholds must remain 0.30/0.70")
    if selective["SKIP_NO_SOURCE"] != "original PDF unavailable":
        raise R3CalibratedCriterionError("no-source policy changed")

    development = require_mapping(criterion["development_evidence"], "development_evidence")
    if not SHA256_RE.fullmatch(str(development.get("report_sha256"))):
        raise R3CalibratedCriterionError("development report sha256 is invalid")
    if (
        float(development.get("selective_decided_accuracy", 0.0)) < 0.90
        or not 0.0 < float(development.get("selective_coverage", 0.0)) < 1.0
        or "coverage must accompany accuracy" not in str(development.get("claim_boundary"))
    ):
        raise R3CalibratedCriterionError("development claim boundary is incomplete")

    runtime = require_mapping(criterion["runtime_policy"], "runtime_policy")
    if set(require_string_list(runtime.get("forbidden_sources"), "forbidden_sources")) != FORBIDDEN_RUNTIME:
        raise R3CalibratedCriterionError("runtime forbidden sources changed")
    if runtime.get("requires_model_package_before_deployment") is not True:
        raise R3CalibratedCriterionError("deployment must require a frozen model package")
    return criterion


def load_criterion(path: str | Path = DEFAULT_CRITERION) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R3CalibratedCriterionError(f"cannot load {source}: {error}") from error
    return validate_criterion(payload)


def decide_probability(
    criterion: Mapping[str, Any],
    probability: float,
    *,
    original_pdf_available: bool = True,
) -> str:
    """Apply the frozen selective policy to one model probability."""

    validated = validate_criterion(criterion)
    if not original_pdf_available:
        return "SKIP_NO_SOURCE"
    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise R3CalibratedCriterionError("probability must be within [0, 1]")
    selective = validated["decision_policy"]["selective_policy"]
    negative = float(
        selective["DOES_NOT_MEET_PRACTICE_PROXY"]["maximum_probability"]
    )
    positive = float(selective["MEETS_PRACTICE_PROXY"]["minimum_probability"])
    if value <= negative:
        return "DOES_NOT_MEET_PRACTICE_PROXY"
    if value >= positive:
        return "MEETS_PRACTICE_PROXY"
    return "ABSTAIN"


def verify_dependencies(
    criterion: Mapping[str, Any], root: str | Path = ROOT
) -> dict[str, Any]:
    validated = validate_criterion(criterion)
    repository = Path(root).resolve()
    checked: list[dict[str, str]] = []
    members = [
        *validated["source_basis"],
        {
            "basis_id": "R3.DEVELOPMENT.SELECTIVE_REPORT",
            "repository_path": validated["development_evidence"]["report_path"],
            "sha256": validated["development_evidence"]["report_sha256"],
        },
    ]
    for member in members:
        path = (repository / member["repository_path"]).resolve()
        try:
            path.relative_to(repository)
        except ValueError as error:
            raise R3CalibratedCriterionError("dependency escapes repository") from error
        observed = file_sha256(path)
        if observed != member["sha256"]:
            raise R3CalibratedCriterionError(
                f"dependency hash mismatch for {member['repository_path']}"
            )
        checked.append(
            {
                "basis_id": str(member["basis_id"]),
                "path": str(path),
                "sha256": observed,
                "status": "PASS",
            }
        )
    return {"status": "PASS", "dependencies": checked}


if __name__ == "__main__":
    document = load_criterion()
    print(json.dumps(verify_dependencies(document), ensure_ascii=False, indent=2))
