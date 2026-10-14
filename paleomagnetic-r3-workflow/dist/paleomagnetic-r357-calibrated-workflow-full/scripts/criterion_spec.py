#!/usr/bin/env python3
"""Load and validate the frozen, record-independent R3/R5/R7 criterion spec.

This module validates normative domain knowledge only.  It deliberately has no
API for paper records, row identifiers, predictions, or evaluation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CRITERION_SPEC = ROOT / "data/knowledge/r357_criterion_spec_v2.json"

INDICATORS = {"R3", "R5", "R7"}
LOGIC_OPERATORS = {"all", "any", "not", "implies"}
GROUP_ROLES = {
    "support_requirement",
    "conflict_gate",
    "comparison_gate",
    "conditional_requirement",
    "exception_guard",
}
EXCEPTION_OUTCOMES = {"MEETS", "DOES_NOT_MEET", "PERMITS_MEETING"}
SOURCE_KINDS = {"normative", "definition", "rationale"}

_ID_RE = re.compile(r"^[A-Z0-9]+(?:[._-][A-Z0-9]+)+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_VALUE_RE = re.compile(r"\b(?:REFNO|RESULTNO)\b", re.I)
_FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "answerkey",
    "answermap",
    "benchmarklabel",
    "expectedlabel",
    "expectedvalue",
    "gold",
    "goldlabel",
    "groundtruth",
    "label",
    "labels",
    "recordanswers",
    "rowanswers",
    "standardr3",
    "standardr5",
    "standardr7",
    "targetlabel",
}


class CriterionSpecValidationError(ValueError):
    """Raised when a criterion artifact violates schema or scoring boundaries."""


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _require_bilingual(value: Any, where: str) -> None:
    if not isinstance(value, Mapping):
        raise CriterionSpecValidationError(f"{where} must be an object")
    if set(value) != {"en", "zh"}:
        raise CriterionSpecValidationError(f"{where} must contain exactly en and zh")
    if not _clean(value.get("en")) or not _clean(value.get("zh")):
        raise CriterionSpecValidationError(f"{where} requires readable en and zh text")


def _require_object_keys(
    value: Mapping[str, Any],
    where: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    unexpected = set(value) - required - optional
    if missing:
        raise CriterionSpecValidationError(
            f"{where} is missing required keys: {', '.join(sorted(missing))}"
        )
    if unexpected:
        raise CriterionSpecValidationError(
            f"{where} contains unexpected keys: {', '.join(sorted(unexpected))}"
        )


def _walk_for_leakage(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, member in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            if "refno" in normalized or "resultno" in normalized:
                raise CriterionSpecValidationError(
                    f"forbidden record identifier key at {child_path}"
                )
            if normalized == "labelsupervision":
                if member is not False:
                    raise CriterionSpecValidationError(
                        f"label_supervision must be false at {child_path}"
                    )
            elif normalized == "containsanswermapping":
                if member is not False:
                    raise CriterionSpecValidationError(
                        f"contains_answer_mapping must be false at {child_path}"
                    )
            elif normalized in _FORBIDDEN_KEYS:
                raise CriterionSpecValidationError(f"forbidden answer key at {child_path}")
            _walk_for_leakage(member, child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, member in enumerate(value):
            _walk_for_leakage(member, f"{path}[{index}]")
        return
    if isinstance(value, str) and _FORBIDDEN_VALUE_RE.search(value):
        raise CriterionSpecValidationError(
            f"forbidden record identifier reference in text at {path}"
        )


def _require_id(value: Any, where: str, prefix: str | None = None) -> str:
    identifier = _clean(value)
    if not _ID_RE.fullmatch(identifier):
        raise CriterionSpecValidationError(f"{where} is not a valid stable identifier")
    if prefix and not identifier.startswith(f"{prefix}."):
        raise CriterionSpecValidationError(f"{where} must start with {prefix}.")
    return identifier


def _require_source_refs(
    value: Any,
    where: str,
    known_sources: Mapping[str, str],
    indicator: str,
) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not _clean(item) for item in value)
    ):
        raise CriterionSpecValidationError(f"{where} must be a non-empty string list")
    if len(value) != len(set(value)):
        raise CriterionSpecValidationError(f"{where} contains duplicate references")
    for source_ref in value:
        if source_ref not in known_sources:
            raise CriterionSpecValidationError(
                f"{where} references unknown source {source_ref}"
            )
        if known_sources[source_ref] != indicator:
            raise CriterionSpecValidationError(
                f"{where} references source from another criterion: {source_ref}"
            )
    return value


def _logic_refs(expression: Any, where: str, allowed_refs: set[str]) -> set[str]:
    if isinstance(expression, str):
        reference = _clean(expression)
        if reference not in allowed_refs:
            raise CriterionSpecValidationError(
                f"{where} references unknown logic identifier {reference or '<empty>'}"
            )
        return {reference}
    if not isinstance(expression, Mapping):
        raise CriterionSpecValidationError(
            f"{where} must be a reference string or logic-expression object"
        )
    if set(expression) != {"op", "args"}:
        raise CriterionSpecValidationError(f"{where} must contain exactly op and args")
    operator = _clean(expression.get("op")).casefold()
    if operator not in LOGIC_OPERATORS:
        raise CriterionSpecValidationError(f"{where}.op is invalid: {operator or '<empty>'}")
    arguments = expression.get("args")
    if not isinstance(arguments, list):
        raise CriterionSpecValidationError(f"{where}.args must be a list")
    expected_count = 1 if operator == "not" else 2 if operator == "implies" else None
    if expected_count is not None and len(arguments) != expected_count:
        raise CriterionSpecValidationError(
            f"{where}.{operator} requires exactly {expected_count} argument(s)"
        )
    if expected_count is None and not arguments:
        raise CriterionSpecValidationError(f"{where}.{operator} requires arguments")
    references: set[str] = set()
    for index, argument in enumerate(arguments):
        references.update(_logic_refs(argument, f"{where}.args[{index}]", allowed_refs))
    return references


def _check_group_cycles(
    group_expressions: Mapping[str, Any],
    group_ids: set[str],
    allowed_refs: set[str],
) -> None:
    dependencies = {
        group_id: _logic_refs(
            expression,
            f"required_groups[{group_id}].expression",
            allowed_refs,
        )
        & group_ids
        for group_id, expression in group_expressions.items()
    }
    state: dict[str, int] = {}

    def visit(group_id: str, trail: list[str]) -> None:
        marker = state.get(group_id, 0)
        if marker == 2:
            return
        if marker == 1:
            cycle = " -> ".join([*trail, group_id])
            raise CriterionSpecValidationError(f"cyclic required-group logic: {cycle}")
        state[group_id] = 1
        for dependency in dependencies[group_id]:
            visit(dependency, [*trail, group_id])
        state[group_id] = 2

    for group_id in group_expressions:
        visit(group_id, [])


def validate_criterion_spec(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a frozen criterion specification and return a detached copy."""

    if not isinstance(data, Mapping):
        raise CriterionSpecValidationError("criterion spec must be a JSON object")
    _walk_for_leakage(data)
    spec = json.loads(json.dumps(data, ensure_ascii=False))
    _require_object_keys(
        spec,
        "$",
        {
            "schema_version",
            "spec_version",
            "freeze_status",
            "title",
            "provenance",
            "source_artifact",
            "logic_language",
            "source_propositions",
            "criteria",
        },
    )

    if not _clean(spec.get("schema_version")):
        raise CriterionSpecValidationError("schema_version is required")
    if not _clean(spec.get("spec_version")):
        raise CriterionSpecValidationError("spec_version is required")
    if spec.get("freeze_status") != "frozen":
        raise CriterionSpecValidationError("freeze_status must be frozen")
    _require_bilingual(spec.get("title"), "title")

    provenance = spec.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CriterionSpecValidationError("provenance is required")
    _require_object_keys(
        provenance,
        "provenance",
        {
            "origin",
            "label_supervision",
            "contains_record_identifiers",
            "contains_answer_mapping",
            "runtime_record_lookup",
            "derivation",
        },
    )
    if not _clean(provenance.get("origin")):
        raise CriterionSpecValidationError("provenance.origin is required")
    if provenance.get("label_supervision") is not False:
        raise CriterionSpecValidationError("provenance.label_supervision=false is required")
    if provenance.get("contains_record_identifiers") is not False:
        raise CriterionSpecValidationError(
            "provenance.contains_record_identifiers=false is required"
        )
    if provenance.get("contains_answer_mapping") is not False:
        raise CriterionSpecValidationError(
            "provenance.contains_answer_mapping=false is required"
        )
    if provenance.get("runtime_record_lookup") is not False:
        raise CriterionSpecValidationError("provenance.runtime_record_lookup=false is required")
    _require_bilingual(provenance.get("derivation"), "provenance.derivation")

    logic_language = spec.get("logic_language")
    if not isinstance(logic_language, Mapping):
        raise CriterionSpecValidationError("logic_language is required")
    _require_object_keys(
        logic_language,
        "logic_language",
        {"truth_values", "semantics", "operators"},
    )
    if logic_language.get("truth_values") != ["true", "false", "unknown"]:
        raise CriterionSpecValidationError(
            "logic_language.truth_values must be true, false, unknown"
        )
    _require_bilingual(logic_language.get("semantics"), "logic_language.semantics")
    operators = logic_language.get("operators")
    if (
        not isinstance(operators, Mapping)
        or set(operators) != LOGIC_OPERATORS
        or any(not _clean(description) for description in operators.values())
    ):
        raise CriterionSpecValidationError(
            "logic_language.operators must describe all supported operators"
        )

    artifact = spec.get("source_artifact")
    if not isinstance(artifact, Mapping):
        raise CriterionSpecValidationError("source_artifact is required")
    _require_object_keys(
        artifact,
        "source_artifact",
        {"artifact_id", "kind", "original_name", "byte_size", "sha256"},
    )
    artifact_id = _require_id(artifact.get("artifact_id"), "source_artifact.artifact_id")
    if not _SHA256_RE.fullmatch(_clean(artifact.get("sha256"))):
        raise CriterionSpecValidationError("source_artifact.sha256 must be a SHA-256 digest")
    if not _clean(artifact.get("original_name")) or not _clean(artifact.get("kind")):
        raise CriterionSpecValidationError(
            "source_artifact.original_name and source_artifact.kind are required"
        )
    byte_size = artifact.get("byte_size")
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 1:
        raise CriterionSpecValidationError("source_artifact.byte_size must be a positive integer")

    source_propositions = spec.get("source_propositions")
    if not isinstance(source_propositions, list) or not source_propositions:
        raise CriterionSpecValidationError("source_propositions must be a non-empty list")
    known_sources: dict[str, str] = {}
    for index, source in enumerate(source_propositions):
        where = f"source_propositions[{index}]"
        if not isinstance(source, Mapping):
            raise CriterionSpecValidationError(f"{where} must be an object")
        _require_object_keys(
            source,
            where,
            {"source_id", "criterion", "artifact_ref", "kind", "locator", "statement"},
        )
        source_id = _require_id(source.get("source_id"), f"{where}.source_id", "SRC")
        if source_id in known_sources:
            raise CriterionSpecValidationError(f"duplicate source identifier {source_id}")
        indicator = _clean(source.get("criterion")).upper()
        if indicator not in INDICATORS:
            raise CriterionSpecValidationError(f"{where}.criterion is invalid")
        if source.get("artifact_ref") != artifact_id:
            raise CriterionSpecValidationError(f"{where}.artifact_ref is invalid")
        if source.get("kind") not in SOURCE_KINDS:
            raise CriterionSpecValidationError(f"{where}.kind is invalid")
        locator = source.get("locator")
        if (
            not isinstance(locator, Mapping)
            or not _clean(locator.get("section"))
            or not isinstance(locator.get("paragraph"), int)
            or isinstance(locator.get("paragraph"), bool)
            or locator["paragraph"] < 1
        ):
            raise CriterionSpecValidationError(
                f"{where}.locator requires section and positive paragraph"
            )
        _require_object_keys(locator, f"{where}.locator", {"section", "paragraph"})
        _require_bilingual(source.get("statement"), f"{where}.statement")
        known_sources[source_id] = indicator

    criteria = spec.get("criteria")
    if not isinstance(criteria, Mapping) or set(criteria) != INDICATORS:
        raise CriterionSpecValidationError("criteria must define exactly R3, R5, and R7")

    for indicator in sorted(INDICATORS):
        criterion = criteria[indicator]
        where = f"criteria.{indicator}"
        if not isinstance(criterion, Mapping):
            raise CriterionSpecValidationError(f"{where} must be an object")
        _require_object_keys(
            criterion,
            where,
            {"title", "source_refs", "propositions", "required_groups", "exceptions", "decision_logic"},
            {"eligible_methods"},
        )
        _require_bilingual(criterion.get("title"), f"{where}.title")
        _require_source_refs(
            criterion.get("source_refs"), f"{where}.source_refs", known_sources, indicator
        )

        propositions = criterion.get("propositions")
        if not isinstance(propositions, list) or not propositions:
            raise CriterionSpecValidationError(f"{where}.propositions must be non-empty")
        proposition_ids: set[str] = set()
        for index, proposition in enumerate(propositions):
            item_where = f"{where}.propositions[{index}]"
            if not isinstance(proposition, Mapping):
                raise CriterionSpecValidationError(f"{item_where} must be an object")
            _require_object_keys(
                proposition,
                item_where,
                {"proposition_id", "value_type", "statement", "source_refs"},
            )
            proposition_id = _require_id(
                proposition.get("proposition_id"),
                f"{item_where}.proposition_id",
                indicator,
            )
            if proposition_id in proposition_ids:
                raise CriterionSpecValidationError(
                    f"duplicate proposition identifier {proposition_id}"
                )
            proposition_ids.add(proposition_id)
            if proposition.get("value_type") != "boolean":
                raise CriterionSpecValidationError(
                    f"{item_where}.value_type must be boolean"
                )
            _require_bilingual(proposition.get("statement"), f"{item_where}.statement")
            _require_source_refs(
                proposition.get("source_refs"),
                f"{item_where}.source_refs",
                known_sources,
                indicator,
            )

        groups = criterion.get("required_groups")
        if not isinstance(groups, list) or not groups:
            raise CriterionSpecValidationError(f"{where}.required_groups must be non-empty")
        group_ids: set[str] = set()
        group_expressions: dict[str, Any] = {}
        for index, group in enumerate(groups):
            item_where = f"{where}.required_groups[{index}]"
            if not isinstance(group, Mapping):
                raise CriterionSpecValidationError(f"{item_where} must be an object")
            _require_object_keys(
                group,
                item_where,
                {"group_id", "role", "statement", "source_refs", "expression"},
            )
            group_id = _require_id(group.get("group_id"), f"{item_where}.group_id", indicator)
            if group_id in group_ids or group_id in proposition_ids:
                raise CriterionSpecValidationError(f"duplicate logic identifier {group_id}")
            group_ids.add(group_id)
            if group.get("role") not in GROUP_ROLES:
                raise CriterionSpecValidationError(f"{item_where}.role is invalid")
            _require_bilingual(group.get("statement"), f"{item_where}.statement")
            _require_source_refs(
                group.get("source_refs"),
                f"{item_where}.source_refs",
                known_sources,
                indicator,
            )
            group_expressions[group_id] = group.get("expression")

        exceptions = criterion.get("exceptions")
        if not isinstance(exceptions, list):
            raise CriterionSpecValidationError(f"{where}.exceptions must be a list")
        exception_ids: set[str] = set()
        for index, exception in enumerate(exceptions):
            item_where = f"{where}.exceptions[{index}]"
            if not isinstance(exception, Mapping):
                raise CriterionSpecValidationError(f"{item_where} must be an object")
            _require_object_keys(
                exception,
                item_where,
                {"exception_id", "statement", "source_refs", "trigger", "effect"},
            )
            exception_id = _require_id(
                exception.get("exception_id"), f"{item_where}.exception_id", indicator
            )
            if exception_id in proposition_ids | group_ids | exception_ids:
                raise CriterionSpecValidationError(f"duplicate logic identifier {exception_id}")
            exception_ids.add(exception_id)
            _require_bilingual(exception.get("statement"), f"{item_where}.statement")
            _require_source_refs(
                exception.get("source_refs"),
                f"{item_where}.source_refs",
                known_sources,
                indicator,
            )
            effect = exception.get("effect")
            if not isinstance(effect, Mapping) or effect.get("outcome") not in EXCEPTION_OUTCOMES:
                raise CriterionSpecValidationError(f"{item_where}.effect.outcome is invalid")
            _require_object_keys(effect, f"{item_where}.effect", {"outcome", "overrides"})
            overrides = effect.get("overrides")
            if not isinstance(overrides, list) or any(item not in group_ids for item in overrides):
                raise CriterionSpecValidationError(
                    f"{item_where}.effect.overrides must reference required groups"
                )

        all_ids = proposition_ids | group_ids | exception_ids
        for group_id, expression in group_expressions.items():
            _logic_refs(
                expression,
                f"{where}.required_groups[{group_id}].expression",
                all_ids,
            )
        _check_group_cycles(group_expressions, group_ids, all_ids)

        for index, exception in enumerate(exceptions):
            trigger_refs = _logic_refs(
                exception.get("trigger"),
                f"{where}.exceptions[{index}].trigger",
                all_ids,
            )
            if trigger_refs & exception_ids:
                raise CriterionSpecValidationError(
                    f"{where}.exceptions[{index}].trigger cannot reference exceptions"
                )

        decision = criterion.get("decision_logic")
        if not isinstance(decision, Mapping):
            raise CriterionSpecValidationError(f"{where}.decision_logic is required")
        _require_object_keys(
            decision,
            f"{where}.decision_logic",
            {"meets", "does_not_meet", "unknown_policy"},
        )
        for outcome in ("meets", "does_not_meet"):
            _logic_refs(
                decision.get(outcome),
                f"{where}.decision_logic.{outcome}",
                all_ids,
            )
        _require_bilingual(
            decision.get("unknown_policy"), f"{where}.decision_logic.unknown_policy"
        )

        eligible_methods = criterion.get("eligible_methods")
        if indicator == "R3":
            if (
                not isinstance(eligible_methods, list)
                or not eligible_methods
                or len(eligible_methods) != len(set(eligible_methods))
                or any(item not in proposition_ids for item in eligible_methods)
            ):
                raise CriterionSpecValidationError(
                    f"{where}.eligible_methods must reference unique R3 propositions"
                )
        elif eligible_methods is not None:
            raise CriterionSpecValidationError(
                f"{where}.eligible_methods is only valid for R3"
            )

    return spec


def load_criterion_spec(path: str | Path = DEFAULT_CRITERION_SPEC) -> dict[str, Any]:
    """Read and validate a criterion specification from disk."""

    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CriterionSpecValidationError(f"cannot load criterion spec {source}: {error}") from error
    return validate_criterion_spec(data)


def verify_source_artifact(
    spec: Mapping[str, Any], source_path: str | Path
) -> dict[str, Any]:
    """Verify that a local source file is exactly the artifact frozen in *spec*."""

    validated = validate_criterion_spec(spec)
    source = Path(source_path)
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise CriterionSpecValidationError(
            f"cannot read source artifact {source}: {error}"
        ) from error

    expected = validated["source_artifact"]
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected["sha256"]:
        raise CriterionSpecValidationError(
            "source artifact SHA-256 mismatch: "
            f"expected {expected['sha256']}, got {actual_sha256}"
        )
    expected_size = expected.get("byte_size")
    if expected_size is not None and size != expected_size:
        raise CriterionSpecValidationError(
            f"source artifact byte-size mismatch: expected {expected_size}, got {size}"
        )
    return {
        "status": "PASS",
        "artifact_id": expected["artifact_id"],
        "sha256": actual_sha256,
        "byte_size": size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_CRITERION_SPEC)
    parser.add_argument(
        "--source-artifact",
        type=Path,
        help="optionally verify the uploaded source bytes against the frozen digest",
    )
    args = parser.parse_args()
    spec = load_criterion_spec(args.spec)
    summary = {
        "status": "PASS",
        "spec_version": spec["spec_version"],
        "source_sha256": spec["source_artifact"]["sha256"],
        "criteria": sorted(spec["criteria"]),
        "source_propositions": len(spec["source_propositions"]),
    }
    if args.source_artifact is not None:
        summary["source_verification"] = verify_source_artifact(spec, args.source_artifact)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
