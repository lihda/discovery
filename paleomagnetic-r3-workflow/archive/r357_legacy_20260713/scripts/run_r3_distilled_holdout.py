#!/usr/bin/env python3
"""Build and freeze R3 distilled predictions on the preregistered holdout.

The command has three ordered phases:

1. derive an identity-only REFNO holdout and matching original-paper cache;
2. run the R3 builder behind the strict filesystem/network/process boundary;
3. validate and hash the predictions before any optional label evaluation.

Labels are never accepted by the scoring worker.  ``--evaluate-labels`` is an
explicit post-scoring convenience and is intentionally disabled by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from openpyxl import load_workbook

try:
    from .audit_r357_scoring_boundary import audit_scoring_boundary
    from .filter_identity_manifest_by_refno_split import (
        DEFAULT_HOLDOUT_FRACTION,
        DEFAULT_SALT,
        holdout_bucket,
        split_identity_manifest,
    )
    from .filter_page_cache_for_identity import filter_cache
    from .run_r357_strict_knowledge_workflow import (
        validate_identity_manifest,
        validate_page_cache,
    )
except ImportError:
    from audit_r357_scoring_boundary import audit_scoring_boundary
    from filter_identity_manifest_by_refno_split import (
        DEFAULT_HOLDOUT_FRACTION,
        DEFAULT_SALT,
        holdout_bucket,
        split_identity_manifest,
    )
    from filter_page_cache_for_identity import filter_cache
    from run_r357_strict_knowledge_workflow import (
        validate_identity_manifest,
        validate_page_cache,
    )


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/strict_scoring_worker.py"
BUILDER = ROOT / "scripts/build_r3_distilled_predictions.py"
REASONER = ROOT / "scripts/r3_rockmag_distilled_reasoner.py"
RETRIEVER = ROOT / "scripts/paper_text_retrieval.py"
CARRIER_CLAIM_STATUS = ROOT / "scripts/r3_carrier_claim_status.py"
EVALUATOR = ROOT / "scripts/evaluate_r3_distilled_predictions.py"
DEFAULT_ARTIFACT = ROOT / "data/calibration/r3_rockmag_text_distillation_v1.json"
DEFAULT_ARTIFACT_SHA256 = (
    "149df539a937678c8204aa6d6e60e77bd901e4a77975d26caa5362749aa18ad4"
)
IDENTITY_COLUMNS = ("REFNO", "RESULTNO", "ROCKNAME")
EXPECTED_ARTIFACT_SCHEMA = "r3_rockmag_text_distillation_v1"
EXPECTED_ARTIFACT_CLASS = "historically_supervised_distillation"
EXPECTED_ARTIFACT_PROFILE = "rockmag_text_distilled_v1"
FROZEN_RETRIEVAL_LIMIT = 30
FROZEN_PER_CATEGORY_LIMIT = 12
TARGET_SCOPE_POLICIES = (
    "sentence_only",
    "exclusive_scope",
    "exclusive_scope_plus_discourse",
)
DEFAULT_TARGET_SCOPE_POLICY = "exclusive_scope_plus_discourse"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class R3DistilledRunError(ValueError):
    """Raised when a run violates the preregistered scoring contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ensure_distinct(paths: Iterable[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(resolved) != len(set(resolved)):
        raise R3DistilledRunError("All run inputs and outputs must be distinct paths")


def _require_false(document: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    for key in keys:
        if document.get(key) is not False:
            raise R3DistilledRunError(f"{context}.{key} must be false")


def _positive_count(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise R3DistilledRunError(f"{location} must be a positive integer")
    return value


def validate_distilled_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Validate the supervised artifact without treating it as an unsupervised KB."""

    if not path.is_file() or path.suffix.casefold() != ".json":
        raise R3DistilledRunError(
            f"Frozen R3 distilled artifact must be an existing JSON file: {path}"
        )
    expected_sha256 = expected_sha256.strip().casefold()
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise R3DistilledRunError("--expected-artifact-sha256 must be 64 lowercase hex digits")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise R3DistilledRunError(
            "Frozen artifact SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise R3DistilledRunError(f"Frozen artifact is invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise R3DistilledRunError("Frozen artifact must be a JSON object")

    expected_fields = {
        "schema_version": EXPECTED_ARTIFACT_SCHEMA,
        "artifact_class": EXPECTED_ARTIFACT_CLASS,
        "profile": EXPECTED_ARTIFACT_PROFILE,
    }
    for key, expected in expected_fields.items():
        if document.get(key) != expected:
            raise R3DistilledRunError(
                f"Frozen artifact {key} must be {expected!r}; found {document.get(key)!r}"
            )

    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise R3DistilledRunError("Frozen artifact provenance must be an object")
    if provenance.get("phase") != "offline_training_only":
        raise R3DistilledRunError("Artifact provenance must be offline_training_only")
    if provenance.get("label_supervision") is not True:
        raise R3DistilledRunError(
            "This runner requires an explicitly declared historically supervised artifact"
        )
    if provenance.get("supervision_scope") != "pre_registered_training_partition_only":
        raise R3DistilledRunError("Artifact supervision must be training-partition-only")
    _require_false(
        provenance,
        (
            "raw_field_values_retained",
            "source_paper_quotes_retained",
            "row_identity_mapping_retained",
        ),
        "provenance",
    )

    split = document.get("split")
    if not isinstance(split, dict):
        raise R3DistilledRunError("Frozen artifact split must be an object")
    if split.get("salt") != DEFAULT_SALT:
        raise R3DistilledRunError("Frozen artifact uses a different REFNO split salt")
    if split.get("holdout_fraction") != DEFAULT_HOLDOUT_FRACTION:
        raise R3DistilledRunError("Frozen artifact uses a different holdout fraction")
    if split.get("unit") != "paper_identity":
        raise R3DistilledRunError("Frozen artifact split unit must be paper_identity")
    if split.get("held_out_supervision_accessed") is not False:
        raise R3DistilledRunError("Frozen artifact declares access to holdout supervision")
    counts = split.get("identity_counts")
    if not isinstance(counts, dict):
        raise R3DistilledRunError("Frozen artifact must declare split identity counts")
    count_summary: dict[str, dict[str, int]] = {}
    for partition in ("training", "holdout"):
        partition_counts = counts.get(partition)
        if not isinstance(partition_counts, dict):
            raise R3DistilledRunError(
                f"Frozen artifact identity_counts.{partition} must be an object"
            )
        count_summary[partition] = {
            "rows": _positive_count(
                partition_counts.get("rows"), f"identity_counts.{partition}.rows"
            ),
            "paper_groups": _positive_count(
                partition_counts.get("paper_groups"),
                f"identity_counts.{partition}.paper_groups",
            ),
        }

    runtime_policy = document.get("runtime_policy")
    if not isinstance(runtime_policy, dict):
        raise R3DistilledRunError("Frozen artifact runtime_policy must be an object")
    _require_false(
        runtime_policy,
        (
            "source_metadata_table_access",
            "supervised_field_access",
            "previous_prediction_access",
            "contains_record_answer_mapping",
        ),
        "runtime_policy",
    )
    allowed_sources = runtime_policy.get("allowed_scoring_sources")
    if not isinstance(allowed_sources, list) or len(allowed_sources) != 3:
        raise R3DistilledRunError(
            "Artifact runtime_policy must declare exactly three scoring source classes"
        )
    for taxonomy in ("method_taxonomy", "diagnostic_taxonomy", "carrier_taxonomy"):
        if not isinstance(document.get(taxonomy), list) or not document[taxonomy]:
            raise R3DistilledRunError(f"Frozen artifact {taxonomy} must be non-empty")
    if not isinstance(document.get("decision_policy"), dict):
        raise R3DistilledRunError("Frozen artifact decision_policy must be an object")

    # Record identifiers are forbidden even if a malformed artifact falsely declares
    # contains_record_answer_mapping=false.
    pending: list[Any] = [document]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            forbidden = {str(key).casefold() for key in value} & {"refno", "resultno"}
            if forbidden:
                raise R3DistilledRunError(
                    "Frozen artifact contains record identity key(s): "
                    + ", ".join(sorted(forbidden))
                )
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)

    return {
        "document": document,
        "sha256": actual_sha256,
        "counts": count_summary,
    }


def _identity_rows(path: Path) -> dict[str, tuple[str, str]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook.worksheets[0]
        rows = worksheet.iter_rows(values_only=True)
        header = tuple(_clean(value) for value in next(rows, ()))
        if header != IDENTITY_COLUMNS:
            raise R3DistilledRunError("Identity manifest columns changed after validation")
        identities: dict[str, tuple[str, str]] = {}
        for values in rows:
            cleaned = tuple(_clean(value) for value in values[:3])
            if not any(cleaned):
                continue
            refno, resultno, rockname = cleaned
            identities[resultno] = (refno, rockname)
        return identities
    finally:
        workbook.close()


def _validate_split_counts(
    source_info: Mapping[str, Any],
    holdout_info: Mapping[str, Any],
    artifact_counts: Mapping[str, Mapping[str, int]],
) -> None:
    expected_rows = sum(partition["rows"] for partition in artifact_counts.values())
    expected_refnos = sum(
        partition["paper_groups"] for partition in artifact_counts.values()
    )
    if source_info["rows"] != expected_rows or len(source_info["refnos"]) != expected_refnos:
        raise R3DistilledRunError(
            "Source identity manifest does not match the frozen artifact population counts: "
            f"rows={source_info['rows']}/{expected_rows}, "
            f"REFNOs={len(source_info['refnos'])}/{expected_refnos}"
        )
    expected_holdout = artifact_counts["holdout"]
    if (
        holdout_info["rows"] != expected_holdout["rows"]
        or len(holdout_info["refnos"]) != expected_holdout["paper_groups"]
    ):
        raise R3DistilledRunError(
            "Derived holdout does not match the frozen artifact counts: "
            f"rows={holdout_info['rows']}/{expected_holdout['rows']}, "
            f"REFNOs={len(holdout_info['refnos'])}/{expected_holdout['paper_groups']}"
        )


def _run_guarded(
    script: Path,
    script_args: Sequence[str],
    allowed_inputs: Sequence[Path],
    write_root: Path,
) -> None:
    command = [sys.executable, str(WORKER), "--script", str(script)]
    for path in allowed_inputs:
        command.extend(("--allow-read", str(path)))
    command.extend(("--write-root", str(write_root), "--", *script_args))
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TMPDIR": str(write_root),
            "TMP": str(write_root),
            "TEMP": str(write_root),
        }
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _run_evaluation(predictions: Path, labels: Path, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--predictions",
            str(predictions),
            "--labels",
            str(labels),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )


def validate_predictions(path: Path, identity_manifest: Path) -> dict[str, Any]:
    expected = _identity_rows(identity_manifest)
    seen: set[str] = set()
    outcomes: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise R3DistilledRunError(
                    f"Prediction line {line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise R3DistilledRunError(
                    f"Prediction line {line_number} must be an object"
                )
            required = {
                "REFNO",
                "RESULTNO",
                "ROCKNAME",
                "R3",
                "OUTCOME",
                "normative_r3",
                "rockmag_proxy_r3",
                "criterion_practice_conflict",
                "criterion_practice_comparison",
                "runtime_scoring_sources",
            }
            missing = required - set(row)
            if missing:
                raise R3DistilledRunError(
                    f"Prediction line {line_number} is missing: {sorted(missing)}"
                )
            refno = _clean(row["REFNO"])
            resultno = _clean(row["RESULTNO"])
            rockname = _clean(row["ROCKNAME"])
            if resultno in seen:
                raise R3DistilledRunError(f"Duplicate prediction RESULTNO: {resultno}")
            if resultno not in expected:
                raise R3DistilledRunError(f"Unexpected prediction RESULTNO: {resultno}")
            if (refno, rockname) != expected[resultno]:
                raise R3DistilledRunError(
                    f"Prediction identity mismatch for RESULTNO {resultno}"
                )
            r3 = row["R3"]
            if isinstance(r3, bool) or r3 not in (0, 1):
                raise R3DistilledRunError(
                    f"Prediction R3 must be binary at line {line_number}"
                )
            for field in (
                "normative_r3",
                "rockmag_proxy_r3",
                "criterion_practice_comparison",
            ):
                if not isinstance(row[field], dict):
                    raise R3DistilledRunError(
                        f"Prediction {field} must be an object at line {line_number}"
                    )
            normative = row["normative_r3"]
            proxy = row["rockmag_proxy_r3"]
            comparison = row["criterion_practice_comparison"]
            for field_name, track in (
                ("normative_r3", normative),
                ("rockmag_proxy_r3", proxy),
            ):
                track_decision = track.get("policy_decision")
                if isinstance(track_decision, bool) or track_decision not in (
                    0,
                    1,
                    "ABSTAIN",
                ):
                    raise R3DistilledRunError(
                        f"Prediction {field_name}.policy_decision is invalid "
                        f"at line {line_number}"
                    )
            normative_decision = normative["policy_decision"]
            proxy_decision = proxy["policy_decision"]
            expected_proxy_binary = proxy_decision if proxy_decision in (0, 1) else 0
            proxy_binary = proxy.get("binary_value")
            if isinstance(proxy_binary, bool) or proxy_binary != expected_proxy_binary:
                raise R3DistilledRunError(
                    f"rockmag_proxy_r3.binary_value is inconsistent at line {line_number}"
                )
            expected_proxy_outcome = (
                "POSITIVE"
                if proxy_decision == 1
                else "NEGATIVE"
                if proxy_decision == 0
                else "UNKNOWN"
            )
            if proxy.get("outcome") != expected_proxy_outcome:
                raise R3DistilledRunError(
                    f"rockmag_proxy_r3.outcome is inconsistent at line {line_number}"
                )
            if (
                r3 != expected_proxy_binary
                or _clean(row["OUTCOME"]).upper() != expected_proxy_outcome
            ):
                raise R3DistilledRunError(
                    f"Top-level R3/OUTCOME does not mirror proxy at line {line_number}"
                )
            conflict = row["criterion_practice_conflict"]
            if conflict is not True and conflict is not False and conflict is not None:
                raise R3DistilledRunError(
                    "criterion_practice_conflict must be true, false, or null "
                    f"at line {line_number}"
                )
            if proxy_decision == "ABSTAIN":
                expected_conflict, expected_status = None, "UNKNOWN"
            elif normative_decision == "ABSTAIN" and proxy_decision == 1:
                expected_conflict = True
                expected_status = "INCOMPLETE_NORMATIVE_CHAIN"
            elif normative_decision == "ABSTAIN" and proxy_decision == 0:
                expected_conflict, expected_status = False, "NO_PROXY_SUPPORT"
            elif normative_decision == proxy_decision:
                expected_conflict, expected_status = False, "AGREE"
            else:
                expected_conflict, expected_status = True, "TRACK_INCONSISTENCY"
            if (
                conflict is not expected_conflict
                or comparison.get("status") != expected_status
                or comparison.get("normative_policy_decision") != normative_decision
                or comparison.get("rockmag_proxy_policy_decision") != proxy_decision
            ):
                raise R3DistilledRunError(
                    f"Criterion/practice comparison is inconsistent at line {line_number}"
                )
            sources = row["runtime_scoring_sources"]
            if not isinstance(sources, list) or not sources or not all(
                isinstance(source, str) and source.strip() for source in sources
            ):
                raise R3DistilledRunError(
                    f"runtime_scoring_sources must be a non-empty string list at line {line_number}"
                )
            outcome = _clean(row["OUTCOME"]).upper()
            if outcome not in {"POSITIVE", "NEGATIVE", "ABSTAIN", "UNKNOWN"}:
                raise R3DistilledRunError(
                    f"Unsupported OUTCOME at prediction line {line_number}: {outcome}"
                )
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            seen.add(resultno)
    missing_resultnos = set(expected) - seen
    if missing_resultnos:
        raise R3DistilledRunError(
            f"Predictions are missing {len(missing_resultnos)} holdout RESULTNO values"
        )
    return {"rows": len(seen), "outcomes": dict(sorted(outcomes.items()))}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--page-cache", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--expected-artifact-sha256",
        default=DEFAULT_ARTIFACT_SHA256,
        help="Pinned SHA-256 for the frozen distilled artifact.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluate-labels",
        type=Path,
        help="Optional labels opened only after the prediction hash receipt is written.",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        help="Optional evaluation JSON path; must be inside --output-dir/evaluation.",
    )
    parser.add_argument(
        "--target-scope-policy",
        choices=TARGET_SCOPE_POLICIES,
        default=DEFAULT_TARGET_SCOPE_POLICY,
        help=(
            "Frozen target-binding policy passed to the guarded scorer: sentence_only, "
            "exclusive_scope, or exclusive_scope_plus_discourse."
        ),
    )
    return parser.parse_args(argv)


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    target_scope_policy = args.target_scope_policy
    if target_scope_policy not in TARGET_SCOPE_POLICIES:
        raise R3DistilledRunError(
            "--target-scope-policy must be sentence_only or exclusive_scope"
        )
    source_identity = args.identity_manifest.resolve(strict=False)
    source_cache = args.page_cache.resolve(strict=False)
    artifact = args.artifact.resolve(strict=False)
    output_dir = args.output_dir.resolve(strict=False)
    labels = (
        args.evaluate_labels.resolve(strict=False)
        if args.evaluate_labels is not None
        else None
    )
    if args.evaluation_output is not None and labels is None:
        raise R3DistilledRunError(
            "--evaluation-output requires the explicit --evaluate-labels option"
        )
    evaluation_output = (
        args.evaluation_output.resolve(strict=False)
        if args.evaluation_output is not None
        else output_dir / "evaluation/r3_distilled_holdout_evaluation.json"
    )
    distinct_paths = [source_identity, source_cache, artifact, output_dir]
    if labels is not None:
        distinct_paths.extend((labels, evaluation_output))
    _ensure_distinct(distinct_paths)
    if labels is not None and _is_within(labels, output_dir):
        raise R3DistilledRunError("--evaluate-labels must be outside --output-dir")
    evaluation_root = output_dir / "evaluation"
    if labels is not None and not _is_within(evaluation_output, evaluation_root):
        raise R3DistilledRunError(
            "--evaluation-output must be inside --output-dir/evaluation"
        )

    source_info = validate_identity_manifest(source_identity)
    artifact_info = validate_distilled_artifact(
        artifact, args.expected_artifact_sha256
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise R3DistilledRunError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    protected_inputs = (source_identity, source_cache, artifact)
    protected_hashes_before = {
        str(path): file_sha256(path) for path in protected_inputs
    }
    implementation_files = (
        RUNNER,
        WORKER,
        BUILDER,
        REASONER,
        RETRIEVER,
        CARRIER_CLAIM_STATUS,
    )
    implementation_hashes_before = {
        str(path): file_sha256(path) for path in implementation_files
    }
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir()
    holdout_identity = inputs_dir / "holdout_identity_manifest.xlsx"
    holdout_cache = inputs_dir / "holdout_page_cache.jsonl"
    split_audit_path = inputs_dir / "identity_split_audit.json"
    cache_audit_path = inputs_dir / "page_cache_filter_audit.json"
    split_audit = split_identity_manifest(
        source_identity,
        holdout_identity,
        partition="holdout",
        salt=DEFAULT_SALT,
        holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
    )
    cache_audit = filter_cache(holdout_identity, source_cache, holdout_cache)
    _write_json(split_audit_path, split_audit)
    _write_json(cache_audit_path, cache_audit)
    holdout_info = validate_identity_manifest(holdout_identity)
    cache_info = validate_page_cache(holdout_cache, holdout_info["refnos"])
    _validate_split_counts(source_info, holdout_info, artifact_info["counts"])

    runtime_inputs = (holdout_identity, holdout_cache, artifact)
    runtime_hashes_before = {str(path): file_sha256(path) for path in runtime_inputs}
    scoring_stage = output_dir / ".r3_distilled_scoring"
    scoring_stage.mkdir()
    staged_predictions = scoring_stage / "r3_distilled_predictions.jsonl"
    staged_summary = scoring_stage / "r3_distilled_summary.json"
    predictions = output_dir / staged_predictions.name
    builder_summary = output_dir / staged_summary.name
    _run_guarded(
        BUILDER,
        [
            "--identity-manifest",
            str(holdout_identity),
            "--page-cache",
            str(holdout_cache),
            "--artifact",
            str(artifact),
            "--expected-artifact-sha256",
            artifact_info["sha256"],
            "--output",
            str(staged_predictions),
            "--summary-json",
            str(staged_summary),
            "--retrieval-limit",
            str(FROZEN_RETRIEVAL_LIMIT),
            "--per-category-limit",
            str(FROZEN_PER_CATEGORY_LIMIT),
            "--target-scope-policy",
            target_scope_policy,
        ],
        runtime_inputs,
        scoring_stage,
    )
    expected_stage_files = {staged_predictions, staged_summary}
    actual_stage_files = {path for path in scoring_stage.iterdir()}
    if actual_stage_files != expected_stage_files or not all(
        path.is_file() for path in expected_stage_files
    ):
        raise R3DistilledRunError(
            "Guarded builder must produce exactly predictions JSONL and summary JSON"
        )
    staged_predictions.replace(predictions)
    staged_summary.replace(builder_summary)
    scoring_stage.rmdir()

    prediction_info = validate_predictions(predictions, holdout_identity)
    try:
        builder_summary_document = json.loads(builder_summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise R3DistilledRunError(f"Builder summary is invalid JSON: {exc}") from exc
    if not isinstance(builder_summary_document, dict):
        raise R3DistilledRunError("Builder summary must be a JSON object")
    expected_retrieval_policy = {
        "total_limit": FROZEN_RETRIEVAL_LIMIT,
        "per_category_limit": FROZEN_PER_CATEGORY_LIMIT,
        "target_scope_policy": target_scope_policy,
    }
    if builder_summary_document.get("retrieval_policy") != expected_retrieval_policy:
        raise R3DistilledRunError(
            "Builder summary does not match the frozen R3 retrieval policy"
        )

    runtime_hashes_after = {str(path): file_sha256(path) for path in runtime_inputs}
    protected_hashes_after = {str(path): file_sha256(path) for path in protected_inputs}
    implementation_hashes_after = {
        str(path): file_sha256(path) for path in implementation_files
    }
    if runtime_hashes_after != runtime_hashes_before:
        raise R3DistilledRunError("A guarded runtime input changed during scoring")
    if protected_hashes_after != protected_hashes_before:
        raise R3DistilledRunError("A source input changed during holdout preparation/scoring")
    if implementation_hashes_after != implementation_hashes_before:
        raise R3DistilledRunError("R3 scoring implementation changed during the run")

    boundary = audit_scoring_boundary([predictions], [])
    boundary_path = output_dir / "scoring_boundary_audit.json"
    _write_json(boundary_path, boundary)
    if not boundary["passed"]:
        codes = sorted({item["code"] for item in boundary["violations"]})
        raise R3DistilledRunError(
            f"Distilled predictions failed scoring boundary audit: {codes}"
        )

    prediction_sha256 = file_sha256(predictions)
    hash_receipt_path = output_dir / "r3_distilled_prediction_hash.json"
    hash_receipt = {
        "schema_version": "r3_distilled_holdout_prediction_freeze_v1",
        "created_at": _utc_now(),
        "phase": "before_optional_label_evaluation",
        "split": {
            "salt": DEFAULT_SALT,
            "holdout_fraction": DEFAULT_HOLDOUT_FRACTION,
        },
        "prediction": {
            "path": str(predictions),
            "sha256": prediction_sha256,
            **prediction_info,
        },
        "runtime_inputs": {
            "identity_manifest_sha256": runtime_hashes_before[str(holdout_identity)],
            "page_cache_sha256": runtime_hashes_before[str(holdout_cache)],
            "artifact_sha256": runtime_hashes_before[str(artifact)],
        },
        "implementation_sha256": implementation_hashes_before,
        "labels_opened": False,
    }
    _write_json(hash_receipt_path, hash_receipt)
    if file_sha256(predictions) != prediction_sha256:
        raise R3DistilledRunError("Predictions changed while their hash was being frozen")

    run_manifest_path = output_dir / "r3_distilled_run_manifest.json"
    summary: dict[str, Any] = {
        "schema_version": "r3_distilled_holdout_run_v1",
        "status": "PASS",
        "workflow_profile": "rockmag_text_distilled_v1",
        "split": {
            "method": "sha256_threshold_by_refno_v1",
            "salt": DEFAULT_SALT,
            "holdout_fraction": DEFAULT_HOLDOUT_FRACTION,
            "source_rows": source_info["rows"],
            "holdout_rows": holdout_info["rows"],
            "holdout_refnos": len(holdout_info["refnos"]),
        },
        "inputs": {
            "source_identity_manifest": {
                "path": str(source_identity),
                "sha256": protected_hashes_before[str(source_identity)],
            },
            "source_page_cache": {
                "path": str(source_cache),
                "sha256": protected_hashes_before[str(source_cache)],
            },
            "holdout_identity_manifest": {
                "path": str(holdout_identity),
                "sha256": runtime_hashes_before[str(holdout_identity)],
            },
            "holdout_page_cache": {
                "path": str(holdout_cache),
                "sha256": runtime_hashes_before[str(holdout_cache)],
                **cache_info,
            },
            "frozen_artifact": {
                "path": str(artifact),
                "sha256": artifact_info["sha256"],
                "schema_version": EXPECTED_ARTIFACT_SCHEMA,
                "artifact_class": EXPECTED_ARTIFACT_CLASS,
            },
        },
        "preparation_audits": {
            "identity_split": str(split_audit_path),
            "page_cache_filter": str(cache_audit_path),
        },
        "prediction": {
            "path": str(predictions),
            "sha256": prediction_sha256,
            **prediction_info,
            "builder_summary": {
                "path": str(builder_summary),
                "sha256": file_sha256(builder_summary),
            },
            "hash_receipt": {
                "path": str(hash_receipt_path),
                "sha256": file_sha256(hash_receipt_path),
                "written_before_labels_opened": True,
            },
        },
        "boundary_audit": {
            "path": str(boundary_path),
            "sha256": file_sha256(boundary_path),
            "status": boundary["status"],
        },
        "implementation": {
            "hash_algorithm": "SHA-256",
            "files": {
                path.name: {
                    "path": str(path),
                    "sha256": implementation_hashes_before[str(path)],
                }
                for path in implementation_files
            },
            "unchanged_during_scoring": True,
        },
        "scoring_contract": {
            "allowed_runtime_data_inputs": [
                "holdout identity manifest: REFNO/RESULTNO/ROCKNAME only",
                "holdout original-paper page cache",
                "pinned historically supervised aggregate artifact",
            ],
            "filesystem_read_guard": "enabled",
            "network_access": "blocked",
            "child_processes_inside_scoring": "blocked",
            "labels_accepted_by_scoring_stage": False,
            "source_tables_accepted_by_scoring_stage": False,
            "previous_evaluations_accepted_by_scoring_stage": False,
            "previous_predictions_accepted_by_scoring_stage": False,
            "frozen_retrieval_policy": expected_retrieval_policy,
        },
        "evaluation": {
            "status": "NOT_RUN" if labels is None else "REQUESTED_POST_SCORING",
            "labels_opened_during_scoring": False,
            "prediction_hash_fixed_before_evaluation": True,
            "command": "scripts/evaluate_r3_distilled_predictions.py",
        },
    }
    _write_json(run_manifest_path, summary)

    if labels is not None:
        evaluation_output.parent.mkdir(parents=True, exist_ok=True)
        frozen_hash = prediction_sha256
        try:
            _run_evaluation(predictions, labels, evaluation_output)
        except subprocess.CalledProcessError:
            summary["status"] = "POST_SCORING_EVALUATION_FAILED"
            summary["evaluation"]["status"] = "FAILED"
            _write_json(run_manifest_path, summary)
            raise
        if file_sha256(predictions) != frozen_hash:
            raise R3DistilledRunError("Predictions changed during post-scoring evaluation")
        evaluation_receipt = evaluation_output.with_suffix(
            evaluation_output.suffix + ".prediction_hashes.json"
        )
        if not evaluation_output.is_file() or not evaluation_receipt.is_file():
            raise R3DistilledRunError(
                "Post-scoring evaluator did not produce its report and hash receipt"
            )
        summary["evaluation"] = {
            "status": "COMPLETED_POST_SCORING",
            "labels_opened_during_scoring": False,
            "prediction_hash_fixed_before_evaluation": True,
            "report": {
                "path": str(evaluation_output),
                "sha256": file_sha256(evaluation_output),
            },
            "evaluator_prediction_hash_receipt": {
                "path": str(evaluation_receipt),
                "sha256": file_sha256(evaluation_receipt),
            },
        }
        _write_json(run_manifest_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_workflow(args)
    except (ValueError, OSError) as exc:
        print(f"R3 distilled run error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"R3 guarded subprocess failed with exit code {exc.returncode}", file=sys.stderr)
        return 3
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
