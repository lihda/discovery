#!/usr/bin/env python3
"""Validate the opt-in, non-normative R3 ROCKMAG practice criterion.

The practice criterion is deliberately separate from ``criterion_spec.py``.
Loading it cannot change the default normative R3/R5/R7 specification or any
prediction path.  This module validates the policy boundary and, optionally,
the hashes and terminology references of its frozen repository dependencies.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_R3_PRACTICE_CRITERION = (
    ROOT / "data/knowledge/r3_rockmag_practice_criterion_v1.json"
)

SCHEMA_VERSION = "r3_rockmag_practice_criterion_schema_v1"
CRITERION_VERSION = "r3_rockmag_practice_criterion_v1"
NORMATIVE_ARTIFACT = "r357_criterion_spec_v2.json"
NORMATIVE_PATH = "data/knowledge/r357_criterion_spec_v2.json"
LEXICON_PATH = "data/knowledge/r3_rockmag_lexicon_v1.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STABLE_ID_RE = re.compile(r"^[A-Z0-9]+(?:[._-][A-Z0-9]+)+$")
_CONCEPT_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_FORBIDDEN_IDENTIFIER_RE = re.compile(r"\b(?:REFNO|RESULTNO)\b", re.I)
_FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "answermap",
    "answermapping",
    "expectedlabel",
    "gold",
    "goldlabel",
    "groundtruth",
    "label",
    "labels",
    "recordanswers",
    "rowanswers",
    "targetlabel",
}
_FALSE_DECLARATIONS = {
    "containsanswermapping",
    "containsrecordidentifiers",
    "evaluationresultsused",
    "labelsupervision",
    "runtimerecordlookup",
}

_EXPECTED_FACETS = {
    "R3.F.SOURCE.ORIGINAL_PAPER": (
        "all",
        {"original_paper_text", "source_location_traceable"},
    ),
    "R3.F.CONTEXT.ELIGIBLE": (
        "all",
        {"non_reference_context", "not_excluded_context"},
    ),
    "R3.F.EXECUTION.AFFIRMED": (
        "any",
        {"method_performed", "measurement_reported", "diagnostic_result_reported"},
    ),
    "R3.F.CHARACTERIZATION.ELIGIBLE": (
        "any",
        {
            "eligible_method_family",
            "self_authenticating_diagnostic_result",
            "mineral_phase_characterization",
        },
    ),
    "R3.F.SCOPE.ATTRIBUTABLE": (
        "any",
        {
            "target_level_binding",
            "single_target_paper_scope",
            "explicit_shared_sample_scope",
        },
    ),
}

_EXPECTED_FAMILY_IDS = {
    "R3.METHOD.IRM_ACQUISITION",
    "R3.METHOD.THERMOMAGNETIC_CURVE",
    "R3.METHOD.TEMPERATURE_DEPENDENT_SUSCEPTIBILITY",
    "R3.METHOD.THREE_AXIS_OR_LOW_TEMPERATURE_IRM",
    "R3.METHOD.HYSTERESIS_FORC_DOMAIN",
    "R3.METHOD.COERCIVITY_OR_UNBLOCKING_SPECTRUM",
    "R3.METHOD.MAGNETIC_FABRIC",
    "R3.METHOD.PETROGRAPHY_OR_MICROIMAGING",
    "R3.METHOD.MINERAL_PHASE_ANALYSIS",
}

_EXPECTED_EXCLUSIONS = {
    "R3.X.STANDALONE.TUB": (
        "regex_search",
        "does_not_satisfy_episode_alone",
    ),
    "R3.X.STANDALONE.BARE_SUSC": (
        "regex_fullmatch",
        "does_not_satisfy_episode_alone",
    ),
    "R3.X.STANDALONE.BARE_AN": (
        "regex_fullmatch",
        "does_not_satisfy_episode_alone",
    ),
    "R3.X.STANDALONE.BARE_MINERAL_NAME": (
        "semantic_state",
        "does_not_satisfy_episode_alone",
    ),
    "R3.X.CONTEXT.REFERENCE_SECTION": (
        "context_flag",
        "reject_evidence_span",
    ),
    "R3.X.CONTEXT.NOT_IMPLEMENTED": (
        "context_flag",
        "reject_evidence_span",
    ),
}

_EXPECTED_CONTEXT_VALUES = {
    "R3.X.STANDALONE.BARE_MINERAL_NAME": {
        "carrier_name_without_method_or_diagnostic"
    },
    "R3.X.CONTEXT.REFERENCE_SECTION": {
        "bibliography",
        "reference_section",
        "citation_only",
    },
    "R3.X.CONTEXT.NOT_IMPLEMENTED": {
        "negated",
        "future_work",
        "hypothetical",
        "recommended_only",
        "method_definition_only",
    },
}

_EXCLUSION_MATCH_CASES = {
    "R3.X.STANDALONE.TUB": ("Tub=150-450C", "Tb > 600 C"),
    "R3.X.STANDALONE.BARE_SUSC": ("SUSC", "  SUSC  "),
    "R3.X.STANDALONE.BARE_AN": ("AN", "  AN "),
}
_EXCLUSION_NONMATCH_CASES = {
    "R3.X.STANDALONE.BARE_SUSC": ("SUSC-T", "SUSC measurements were performed"),
    "R3.X.STANDALONE.BARE_AN": ("ANOVA", "AN measurements were performed"),
}

_EXPECTED_SHARED_SCOPE_REQUIREMENTS = {
    "evidence_and_target_share_sample_set",
    "episode_applies_to_shared_samples",
    "no_competing_target_in_scope",
}
_EXPECTED_PROPAGATION_CONDITIONS = {
    "R3.SCOPE.SINGLE_TARGET_PAPER": {"paper_has_single_target"},
    "R3.SCOPE.EXPLICIT_SHARED_SAMPLE": {
        "explicit_shared_sample_scope",
        "scope_includes_target",
        "episode_applies_to_shared_samples",
    },
}
_EXPECTED_PROPAGATION_BLOCKS = {
    "multiple_targets_without_shared_sample_scope",
    "competing_target_in_evidence_scope",
    "target_identity_ambiguous",
}


class R3PracticeCriterionValidationError(ValueError):
    """Raised when the practice criterion violates its frozen contract."""


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _require_exact_keys(
    value: Mapping[str, Any], where: str, expected: set[str]
) -> None:
    missing = expected - set(value)
    unexpected = set(value) - expected
    if missing:
        raise R3PracticeCriterionValidationError(
            f"{where} is missing required keys: {', '.join(sorted(missing))}"
        )
    if unexpected:
        raise R3PracticeCriterionValidationError(
            f"{where} contains unexpected keys: {', '.join(sorted(unexpected))}"
        )


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R3PracticeCriterionValidationError(f"{where} must be an object")
    return value


def _require_bilingual(value: Any, where: str) -> None:
    member = _require_mapping(value, where)
    _require_exact_keys(member, where, {"en", "zh"})
    if not _clean(member.get("en")) or not _clean(member.get("zh")):
        raise R3PracticeCriterionValidationError(
            f"{where} requires readable en and zh text"
        )


def _string_list(value: Any, where: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not _clean(item) for item in value)
    ):
        raise R3PracticeCriterionValidationError(
            f"{where} must be a non-empty string list"
        )
    if len(value) != len(set(value)):
        raise R3PracticeCriterionValidationError(f"{where} contains duplicates")
    return value


def _require_sha256(value: Any, where: str) -> str:
    digest = _clean(value)
    if not _SHA256_RE.fullmatch(digest):
        raise R3PracticeCriterionValidationError(
            f"{where} must be a lowercase SHA-256 digest"
        )
    return digest


def _walk_boundary(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, member in value.items():
            normalized = _normalized_key(key)
            child = f"{path}.{key}"
            if "refno" in normalized or "resultno" in normalized:
                raise R3PracticeCriterionValidationError(
                    f"forbidden record identifier key at {child}"
                )
            if normalized in _FALSE_DECLARATIONS:
                if member is not False:
                    raise R3PracticeCriterionValidationError(
                        f"{key} must be false at {child}"
                    )
            elif normalized in _FORBIDDEN_KEYS:
                raise R3PracticeCriterionValidationError(
                    f"forbidden answer or label key at {child}"
                )
            _walk_boundary(member, child)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, member in enumerate(value):
            _walk_boundary(member, f"{path}[{index}]")
        return
    if isinstance(value, str) and _FORBIDDEN_IDENTIFIER_RE.search(value):
        raise R3PracticeCriterionValidationError(
            f"forbidden record identifier reference at {path}"
        )


def _validate_source_basis(
    value: Any, normative_sha256: str
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != 2:
        raise R3PracticeCriterionValidationError(
            "source_basis must contain exactly the normative and terminology artifacts"
        )
    output: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(value):
        where = f"source_basis[{index}]"
        basis = _require_mapping(raw, where)
        _require_exact_keys(
            basis,
            where,
            {
                "basis_id",
                "kind",
                "repository_path",
                "sha256",
                "label_supervision",
                "use",
            },
        )
        basis_id = _clean(basis.get("basis_id"))
        if not _STABLE_ID_RE.fullmatch(basis_id) or not basis_id.startswith("R3.BASIS."):
            raise R3PracticeCriterionValidationError(
                f"{where}.basis_id is not a stable R3 basis identifier"
            )
        if basis_id in output:
            raise R3PracticeCriterionValidationError(f"duplicate basis_id {basis_id}")
        if basis.get("label_supervision") is not False:
            raise R3PracticeCriterionValidationError(
                f"{where}.label_supervision must be false"
            )
        path = Path(_clean(basis.get("repository_path")))
        if path.is_absolute() or ".." in path.parts:
            raise R3PracticeCriterionValidationError(
                f"{where}.repository_path must be repository-relative"
            )
        _require_sha256(basis.get("sha256"), f"{where}.sha256")
        _require_bilingual(basis.get("use"), f"{where}.use")
        output[basis_id] = basis

    expected = {
        "R3.BASIS.NORMATIVE.V2": ("normative_criterion", NORMATIVE_PATH),
        "R3.BASIS.ROCKMAG.LEXICON.V1": (
            "deidentified_aggregate_terminology",
            LEXICON_PATH,
        ),
    }
    if set(output) != set(expected):
        raise R3PracticeCriterionValidationError(
            "source_basis must define the frozen normative and ROCKMAG lexicon dependencies"
        )
    for basis_id, (kind, path) in expected.items():
        basis = output[basis_id]
        if basis.get("kind") != kind or basis.get("repository_path") != path:
            raise R3PracticeCriterionValidationError(
                f"{basis_id} kind or repository_path is invalid"
            )
    if output["R3.BASIS.NORMATIVE.V2"]["sha256"] != normative_sha256:
        raise R3PracticeCriterionValidationError(
            "normative dependency digest disagrees with normative_relationship"
        )
    return output


def _validate_evidence_episode(value: Any) -> set[str]:
    episode = _require_mapping(value, "evidence_episode")
    _require_exact_keys(
        episode,
        "evidence_episode",
        {
            "episode_id",
            "unit",
            "positive_semantics",
            "required_facets",
            "eligible_families",
            "not_required_by_practice_proxy",
        },
    )
    if episode.get("episode_id") != "R3.PRACTICE.EPISODE.CHARACTERIZATION":
        raise R3PracticeCriterionValidationError("evidence_episode.episode_id is invalid")
    if episode.get("unit") != "target_bound_implemented_characterization_episode":
        raise R3PracticeCriterionValidationError(
            "evidence_episode.unit must require a target-bound implemented episode"
        )
    _require_bilingual(episode.get("positive_semantics"), "evidence_episode.positive_semantics")

    facets = episode.get("required_facets")
    if not isinstance(facets, list) or not facets:
        raise R3PracticeCriterionValidationError(
            "evidence_episode.required_facets must be non-empty"
        )
    seen_facets: set[str] = set()
    for index, raw in enumerate(facets):
        where = f"evidence_episode.required_facets[{index}]"
        facet = _require_mapping(raw, where)
        _require_exact_keys(facet, where, {"facet_id", "operator", "accepted_states"})
        facet_id = _clean(facet.get("facet_id"))
        if facet_id in seen_facets:
            raise R3PracticeCriterionValidationError(f"duplicate facet_id {facet_id}")
        seen_facets.add(facet_id)
        if facet_id not in _EXPECTED_FACETS:
            raise R3PracticeCriterionValidationError(f"unsupported facet_id {facet_id}")
        expected_operator, expected_states = _EXPECTED_FACETS[facet_id]
        states = set(_string_list(facet.get("accepted_states"), f"{where}.accepted_states"))
        if facet.get("operator") != expected_operator or states != expected_states:
            raise R3PracticeCriterionValidationError(
                f"{facet_id} does not preserve the frozen practice semantics"
            )
    if seen_facets != set(_EXPECTED_FACETS):
        raise R3PracticeCriterionValidationError(
            "required_facets must include source, context, execution, characterization, and scope"
        )

    families = episode.get("eligible_families")
    if not isinstance(families, list) or not families:
        raise R3PracticeCriterionValidationError(
            "evidence_episode.eligible_families must be non-empty"
        )
    family_ids: set[str] = set()
    for index, raw in enumerate(families):
        where = f"evidence_episode.eligible_families[{index}]"
        family = _require_mapping(raw, where)
        _require_exact_keys(family, where, {"family_id", "evidence_modes", "concept_refs"})
        family_id = _clean(family.get("family_id"))
        if not _STABLE_ID_RE.fullmatch(family_id) or not family_id.startswith("R3.METHOD."):
            raise R3PracticeCriterionValidationError(
                f"{where}.family_id is not a stable R3 method identifier"
            )
        if family_id in family_ids:
            raise R3PracticeCriterionValidationError(f"duplicate family_id {family_id}")
        family_ids.add(family_id)
        modes = set(_string_list(family.get("evidence_modes"), f"{where}.evidence_modes"))
        if modes != {"implemented_method", "reported_result"}:
            raise R3PracticeCriterionValidationError(
                f"{family_id} must require an implemented method or reported result"
            )
        concepts = _string_list(family.get("concept_refs"), f"{where}.concept_refs")
        if any(not _CONCEPT_ID_RE.fullmatch(concept) for concept in concepts):
            raise R3PracticeCriterionValidationError(
                f"{where}.concept_refs contains an invalid concept identifier"
            )
    if family_ids != _EXPECTED_FAMILY_IDS:
        raise R3PracticeCriterionValidationError(
            "eligible_families does not preserve the frozen method coverage"
        )

    not_required = set(
        _string_list(
            episode.get("not_required_by_practice_proxy"),
            "evidence_episode.not_required_by_practice_proxy",
        )
    )
    if not_required != {
        "explicit_carrier_identification",
        "explicit_remanence_significance_interpretation",
    }:
        raise R3PracticeCriterionValidationError(
            "practice downgrade must be limited to carrier identification and significance"
        )
    return seen_facets


def _validate_exclusions(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise R3PracticeCriterionValidationError("exclusions must be a non-empty list")
    exclusions: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(value):
        where = f"exclusions[{index}]"
        exclusion = _require_mapping(raw, where)
        _require_exact_keys(
            exclusion,
            where,
            {"exclusion_id", "matcher", "effect", "statement"},
        )
        exclusion_id = _clean(exclusion.get("exclusion_id"))
        if exclusion_id in exclusions:
            raise R3PracticeCriterionValidationError(
                f"duplicate exclusion_id {exclusion_id}"
            )
        if exclusion_id not in _EXPECTED_EXCLUSIONS:
            raise R3PracticeCriterionValidationError(
                f"unsupported exclusion_id {exclusion_id}"
            )
        matcher = _require_mapping(exclusion.get("matcher"), f"{where}.matcher")
        _require_exact_keys(matcher, f"{where}.matcher", {"kind", "values"})
        values = _string_list(matcher.get("values"), f"{where}.matcher.values")
        expected_kind, expected_effect = _EXPECTED_EXCLUSIONS[exclusion_id]
        if matcher.get("kind") != expected_kind or exclusion.get("effect") != expected_effect:
            raise R3PracticeCriterionValidationError(
                f"{exclusion_id} matcher or effect is invalid"
            )
        if expected_kind.startswith("regex_"):
            try:
                patterns = [re.compile(pattern, re.I) for pattern in values]
            except re.error as error:
                raise R3PracticeCriterionValidationError(
                    f"{exclusion_id} contains an invalid regex: {error}"
                ) from error
            matcher_function = re.Pattern.fullmatch if expected_kind == "regex_fullmatch" else re.Pattern.search
            for sample in _EXCLUSION_MATCH_CASES[exclusion_id]:
                if not any(matcher_function(pattern, sample) for pattern in patterns):
                    raise R3PracticeCriterionValidationError(
                        f"{exclusion_id} does not recognize required standalone case {sample!r}"
                    )
            for sample in _EXCLUSION_NONMATCH_CASES.get(exclusion_id, ()):
                if any(matcher_function(pattern, sample) for pattern in patterns):
                    raise R3PracticeCriterionValidationError(
                        f"{exclusion_id} incorrectly rejects contextual case {sample!r}"
                    )
        elif set(values) != _EXPECTED_CONTEXT_VALUES[exclusion_id]:
            raise R3PracticeCriterionValidationError(
                f"{exclusion_id} does not preserve the required semantic states"
            )
        _require_bilingual(exclusion.get("statement"), f"{where}.statement")
        exclusions[exclusion_id] = exclusion
    if set(exclusions) != set(_EXPECTED_EXCLUSIONS):
        raise R3PracticeCriterionValidationError(
            "exclusions must cover Tub, bare SUSC, bare AN, bare mineral, references, and non-execution"
        )


def _validate_scope_policy(value: Any) -> None:
    scope = _require_mapping(value, "scope_policy")
    _require_exact_keys(
        scope,
        "scope_policy",
        {
            "evidence_resolution_order",
            "target_level",
            "shared_sample_scope",
            "paper_level_propagation",
        },
    )
    if scope.get("evidence_resolution_order") != [
        "target_level",
        "explicit_shared_sample_scope",
        "single_target_paper_scope",
    ]:
        raise R3PracticeCriterionValidationError(
            "scope_policy must resolve target-level evidence before propagation"
        )

    target = _require_mapping(scope.get("target_level"), "scope_policy.target_level")
    _require_exact_keys(
        target,
        "scope_policy.target_level",
        {"priority", "requires_explicit_or_local_binding", "paper_propagation_needed"},
    )
    if (
        target.get("priority") != 1
        or target.get("requires_explicit_or_local_binding") is not True
        or target.get("paper_propagation_needed") is not False
    ):
        raise R3PracticeCriterionValidationError(
            "scope_policy.target_level must be the direct-evidence priority"
        )

    shared = _require_mapping(
        scope.get("shared_sample_scope"), "scope_policy.shared_sample_scope"
    )
    _require_exact_keys(
        shared,
        "scope_policy.shared_sample_scope",
        {"must_be_explicit", "requires_all"},
    )
    if shared.get("must_be_explicit") is not True or set(
        _string_list(shared.get("requires_all"), "scope_policy.shared_sample_scope.requires_all")
    ) != _EXPECTED_SHARED_SCOPE_REQUIREMENTS:
        raise R3PracticeCriterionValidationError(
            "shared-sample propagation requires explicit sample linkage and no competitor"
        )

    propagation = _require_mapping(
        scope.get("paper_level_propagation"),
        "scope_policy.paper_level_propagation",
    )
    _require_exact_keys(
        propagation,
        "scope_policy.paper_level_propagation",
        {"default", "allow_if_any", "forbid_if_any", "unresolved_scope_outcome"},
    )
    if propagation.get("default") != "forbidden":
        raise R3PracticeCriterionValidationError(
            "paper-level propagation must be forbidden by default"
        )
    if propagation.get("unresolved_scope_outcome") != "ABSTAIN":
        raise R3PracticeCriterionValidationError(
            "unresolved paper-level scope must abstain"
        )
    conditions = propagation.get("allow_if_any")
    if not isinstance(conditions, list) or not conditions:
        raise R3PracticeCriterionValidationError(
            "paper-level propagation requires explicit allow conditions"
        )
    found: dict[str, set[str]] = {}
    for index, raw in enumerate(conditions):
        where = f"scope_policy.paper_level_propagation.allow_if_any[{index}]"
        condition = _require_mapping(raw, where)
        _require_exact_keys(condition, where, {"condition_id", "requires_all"})
        condition_id = _clean(condition.get("condition_id"))
        if condition_id in found:
            raise R3PracticeCriterionValidationError(
                f"duplicate propagation condition {condition_id}"
            )
        found[condition_id] = set(
            _string_list(condition.get("requires_all"), f"{where}.requires_all")
        )
    if found != _EXPECTED_PROPAGATION_CONDITIONS:
        raise R3PracticeCriterionValidationError(
            "paper-level propagation is allowed only for a single target or explicit shared samples"
        )
    if set(
        _string_list(
            propagation.get("forbid_if_any"),
            "scope_policy.paper_level_propagation.forbid_if_any",
        )
    ) != _EXPECTED_PROPAGATION_BLOCKS:
        raise R3PracticeCriterionValidationError(
            "paper-level propagation blocks do not preserve target isolation"
        )


def _validate_decision_logic(value: Any, facet_ids: set[str]) -> None:
    decision = _require_mapping(value, "decision_logic")
    _require_exact_keys(
        decision,
        "decision_logic",
        {
            "outcomes",
            "positive_requires_all",
            "negative_requires_all",
            "abstain_if_any",
            "skip_if",
        },
    )
    if decision.get("outcomes") != [
        "MEETS_PRACTICE_PROXY",
        "DOES_NOT_MEET_PRACTICE_PROXY",
        "ABSTAIN",
        "SKIP_NO_SOURCE",
    ]:
        raise R3PracticeCriterionValidationError("decision_logic.outcomes is invalid")
    if set(
        _string_list(
            decision.get("positive_requires_all"),
            "decision_logic.positive_requires_all",
        )
    ) != facet_ids:
        raise R3PracticeCriterionValidationError(
            "positive decision must require every evidence facet"
        )
    if set(
        _string_list(
            decision.get("negative_requires_all"),
            "decision_logic.negative_requires_all",
        )
    ) != {
        "original_paper_available",
        "extraction_sufficient",
        "no_eligible_attributable_episode_after_exclusions",
    }:
        raise R3PracticeCriterionValidationError(
            "negative decision must require a sufficiently extracted original paper"
        )
    if set(
        _string_list(decision.get("abstain_if_any"), "decision_logic.abstain_if_any")
    ) != {
        "source_extraction_incomplete",
        "target_binding_ambiguous",
        "shared_sample_scope_ambiguous",
    }:
        raise R3PracticeCriterionValidationError(
            "decision_logic.abstain_if_any is invalid"
        )
    if decision.get("skip_if") != "original_paper_unavailable":
        raise R3PracticeCriterionValidationError(
            "targets without an original paper must be skipped"
        )


def _validate_runtime_policy(value: Any) -> None:
    runtime = _require_mapping(value, "runtime_policy")
    _require_exact_keys(
        runtime,
        "runtime_policy",
        {
            "integration_mode",
            "emit_separate_practice_decision",
            "overwrite_normative_r3",
            "allowed_scoring_sources",
            "forbidden_scoring_sources",
        },
    )
    if (
        runtime.get("integration_mode") != "explicit_opt_in_only"
        or runtime.get("emit_separate_practice_decision") is not True
        or runtime.get("overwrite_normative_r3") is not False
    ):
        raise R3PracticeCriterionValidationError(
            "practice criterion must remain opt-in and separate from normative R3"
        )
    if set(
        _string_list(
            runtime.get("allowed_scoring_sources"),
            "runtime_policy.allowed_scoring_sources",
        )
    ) != {
        "original_paper_text",
        "target_identity_and_scope_manifest",
        "this_frozen_practice_criterion",
        "frozen_deidentified_terminology_artifact",
    }:
        raise R3PracticeCriterionValidationError(
            "runtime_policy.allowed_scoring_sources is invalid"
        )
    if set(
        _string_list(
            runtime.get("forbidden_scoring_sources"),
            "runtime_policy.forbidden_scoring_sources",
        )
    ) != {
        "source_metadata_result_table",
        "existing_evaluation_results",
        "previous_predictions",
        "record_answer_mappings",
    }:
        raise R3PracticeCriterionValidationError(
            "runtime_policy must forbid result tables, evaluations, prior predictions, and answer maps"
        )


def validate_r3_practice_criterion(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached copy of the frozen practice criterion."""

    if not isinstance(data, Mapping):
        raise R3PracticeCriterionValidationError("practice criterion must be an object")
    _walk_boundary(data)
    criterion = json.loads(json.dumps(data, ensure_ascii=False))
    _require_exact_keys(
        criterion,
        "$",
        {
            "schema_version",
            "criterion_version",
            "freeze_status",
            "indicator",
            "title",
            "provenance",
            "normative_relationship",
            "source_basis",
            "evidence_episode",
            "exclusions",
            "scope_policy",
            "decision_logic",
            "runtime_policy",
        },
    )
    if criterion.get("schema_version") != SCHEMA_VERSION:
        raise R3PracticeCriterionValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}"
        )
    if criterion.get("criterion_version") != CRITERION_VERSION:
        raise R3PracticeCriterionValidationError(
            f"criterion_version must be {CRITERION_VERSION!r}"
        )
    if criterion.get("freeze_status") != "frozen":
        raise R3PracticeCriterionValidationError("freeze_status must be frozen")
    if criterion.get("indicator") != "R3":
        raise R3PracticeCriterionValidationError("indicator must be R3")
    _require_bilingual(criterion.get("title"), "title")

    provenance = _require_mapping(criterion.get("provenance"), "provenance")
    _require_exact_keys(
        provenance,
        "provenance",
        {
            "derivation",
            "label_supervision",
            "evaluation_results_used",
            "contains_record_identifiers",
            "contains_answer_mapping",
            "runtime_record_lookup",
        },
    )
    _require_bilingual(provenance.get("derivation"), "provenance.derivation")
    for key in (
        "label_supervision",
        "evaluation_results_used",
        "contains_record_identifiers",
        "contains_answer_mapping",
        "runtime_record_lookup",
    ):
        if provenance.get(key) is not False:
            raise R3PracticeCriterionValidationError(f"provenance.{key} must be false")

    normative = _require_mapping(
        criterion.get("normative_relationship"), "normative_relationship"
    )
    _require_exact_keys(
        normative,
        "normative_relationship",
        {
            "normative_artifact",
            "normative_sha256",
            "criterion",
            "relationship",
            "replaces_normative",
            "default_scoring_enabled",
            "statement",
        },
    )
    normative_sha256 = _require_sha256(
        normative.get("normative_sha256"),
        "normative_relationship.normative_sha256",
    )
    if (
        normative.get("normative_artifact") != NORMATIVE_ARTIFACT
        or normative.get("criterion") != "R3"
        or normative.get("relationship") != "non_normative_practice_proxy"
        or normative.get("replaces_normative") is not False
        or normative.get("default_scoring_enabled") is not False
    ):
        raise R3PracticeCriterionValidationError(
            "normative_relationship must preserve normative v2 and disable default scoring"
        )
    _require_bilingual(normative.get("statement"), "normative_relationship.statement")

    _validate_source_basis(criterion.get("source_basis"), normative_sha256)
    facet_ids = _validate_evidence_episode(criterion.get("evidence_episode"))
    _validate_exclusions(criterion.get("exclusions"))
    _validate_scope_policy(criterion.get("scope_policy"))
    _validate_decision_logic(criterion.get("decision_logic"), facet_ids)
    _validate_runtime_policy(criterion.get("runtime_policy"))
    return criterion


def load_r3_practice_criterion(
    path: str | Path = DEFAULT_R3_PRACTICE_CRITERION,
) -> dict[str, Any]:
    """Load a practice criterion without activating it in any scoring path."""

    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R3PracticeCriterionValidationError(
            f"cannot load practice criterion {source}: {error}"
        ) from error
    return validate_r3_practice_criterion(data)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_r3_practice_dependencies(
    criterion: Mapping[str, Any], repository_root: str | Path = ROOT
) -> dict[str, Any]:
    """Verify dependency bytes and cross-check all terminology references."""

    validated = validate_r3_practice_criterion(criterion)
    root = Path(repository_root).resolve()
    dependency_results: list[dict[str, Any]] = []
    dependency_payloads: dict[str, Any] = {}
    for basis in validated["source_basis"]:
        relative = Path(basis["repository_path"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
            actual_sha256 = _file_sha256(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise R3PracticeCriterionValidationError(
                f"cannot verify dependency {relative}: {error}"
            ) from error
        if actual_sha256 != basis["sha256"]:
            raise R3PracticeCriterionValidationError(
                f"dependency SHA-256 mismatch for {relative}: "
                f"expected {basis['sha256']}, got {actual_sha256}"
            )
        dependency_payloads[basis["basis_id"]] = payload
        dependency_results.append(
            {
                "basis_id": basis["basis_id"],
                "repository_path": basis["repository_path"],
                "sha256": actual_sha256,
                "status": "PASS",
            }
        )

    normative = dependency_payloads["R3.BASIS.NORMATIVE.V2"]
    if (
        normative.get("spec_version") != "r357_criterion_spec_v2"
        or "R3" not in (normative.get("criteria") or {})
    ):
        raise R3PracticeCriterionValidationError(
            "normative dependency is not the expected R3 criterion specification"
        )

    lexicon = dependency_payloads["R3.BASIS.ROCKMAG.LEXICON.V1"]
    if lexicon.get("knowledge_version") != "r3_rockmag_lexicon_v1":
        raise R3PracticeCriterionValidationError(
            "terminology dependency has an unexpected knowledge version"
        )
    lexicon_provenance = lexicon.get("provenance") or {}
    if lexicon_provenance.get("label_supervision") is not False:
        raise R3PracticeCriterionValidationError(
            "terminology dependency must be label-independent"
        )
    available_concepts = {
        _clean(item.get("concept_id"))
        for item in lexicon.get("concepts") or []
        if isinstance(item, Mapping)
    }
    required_concepts = {
        concept
        for family in validated["evidence_episode"]["eligible_families"]
        for concept in family["concept_refs"]
    }
    missing_concepts = sorted(required_concepts - available_concepts)
    if missing_concepts:
        raise R3PracticeCriterionValidationError(
            f"terminology dependency is missing concept refs: {missing_concepts}"
        )
    return {
        "status": "PASS",
        "criterion_version": validated["criterion_version"],
        "dependencies": dependency_results,
        "concept_refs_verified": len(required_concepts),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--criterion",
        type=Path,
        default=DEFAULT_R3_PRACTICE_CRITERION,
    )
    parser.add_argument(
        "--verify-dependencies",
        action="store_true",
        help="verify the pinned normative and terminology artifact bytes",
    )
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    criterion = load_r3_practice_criterion(args.criterion)
    result: dict[str, Any] = {
        "status": "PASS",
        "criterion_version": criterion["criterion_version"],
        "relationship": criterion["normative_relationship"]["relationship"],
        "default_scoring_enabled": criterion["normative_relationship"][
            "default_scoring_enabled"
        ],
        "exclusions": len(criterion["exclusions"]),
        "eligible_families": len(criterion["evidence_episode"]["eligible_families"]),
    }
    if args.verify_dependencies:
        result["dependency_verification"] = verify_r3_practice_dependencies(
            criterion, args.repository_root
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
