#!/usr/bin/env python3
"""Source-only R3 reasoning with a frozen, supervised ROCKMAG taxonomy.

The offline artifact may be distilled with R3 labels, but it must contain only
aggregate terminology and policy.  Runtime scoring is deliberately narrower:
it accepts target-bound hits produced from original-paper text and never reads
PMAGRESULT, ROCKUNIT, labels, evaluation results, or record-level answer maps.

The accepted artifact is the generated
``r3_rockmag_text_distillation_v1`` document.  It declares
``artifact_class=historically_supervised_distillation`` and contains
``method_taxonomy``, ``carrier_taxonomy``, and ``diagnostic_taxonomy`` lists.
Only each concept's ``runtime_original_text_patterns`` is used at scoring
time; training statistics remain provenance and never become row lookups.

The returned decision is dual-track. ``policy_decision`` preserves the formal
three-valued result (1, 0, or ``ABSTAIN``), while ``proxy_decision`` is the
binary compatibility value expected by legacy R3 workbooks.  An abstention is
therefore never misrepresented as positive evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
from typing import Any

try:
    from .r3_carrier_claim_status import carrier_claim_is_affirmed
except ImportError:
    from r3_carrier_claim_status import carrier_claim_is_affirmed


ENGINE_VERSION = "r3_rockmag_distilled_reasoner_v2"
ARTIFACT_CLASS = "historically_supervised_distillation"
ARTIFACT_SCHEMA_VERSION = "r3_rockmag_text_distillation_v1"
MINIMUM_TARGET_RELEVANCE = 0.2
MAXIMUM_PAGE_SPAN = 1
DEFAULT_ROCKMAG_PROXY_POLICY = "connected_evidence"
ROCKMAG_PROXY_POLICIES = {
    DEFAULT_ROCKMAG_PROXY_POLICY,
    "distilled_method_or_self_authenticating_result",
}
DEFAULT_IDENTITY_GRANULARITY = "result_component"
IDENTITY_GRANULARITIES = {
    DEFAULT_IDENTITY_GRANULARITY,
    "canonical_target",
}


class R3RockmagArtifactValidationError(ValueError):
    """Raised when the frozen taxonomy violates its declared boundary."""


class R3RockmagRuntimeBoundaryError(ValueError):
    """Raised when a runtime retrieval contains a forbidden table or answer field."""


_ANSWER_DECLARATION_KEYS = {
    "containsanswermapping",
    "containsrecordanswermapping",
    "containsrowanswermapping",
}
_FORBIDDEN_ARTIFACT_KEYS = {
    "answer",
    "answers",
    "answerbyid",
    "answermap",
    "answermapping",
    "expectedr3",
    "gold",
    "goldlabel",
    "groundtruth",
    "groundtruthlabel",
    "perrecordanswers",
    "recordanswers",
    "refno",
    "resultno",
    "r3answer",
    "r3answers",
    "r3label",
    "r3labels",
    "rowanswers",
    "rowlabels",
}
_FORBIDDEN_RUNTIME_KEYS = _FORBIDDEN_ARTIFACT_KEYS | {
    "pmagresult",
    "priorprediction",
    "priorpredictions",
    "rockmag",
    "rockunit",
    "r3",
    "r3score",
    "r3value",
}
_QUALITY_RISK_FLAGS = {
    "extractionincomplete",
    "glyphorcidtext",
    "lowocrquality",
    "missingpages",
    "noextractabletext",
    "ocrrisk",
    "possiblyincompleteextraction",
    "poorocr",
    "shorttext",
    "truncatedtext",
}
_REFERENCE_SECTION_RE = re.compile(
    r"\b(?:bibliograph(?:y|ies)|literature cited|references|works cited)\b",
    re.I,
)
_NON_EXECUTION_RE = re.compile(
    r"\b(?:did\s+not|do\s+not|does\s+not|not|never)\s+"
    r"(?:[a-z-]+\s+){0,3}(?:conduct(?:ed)?|perform(?:ed)?|carry\s+out|"
    r"carried\s+out|undertak(?:e|en)|measure(?:d)?|analy[sz](?:e|ed)|"
    r"examin(?:e|ed)|investigat(?:e|ed))\b|"
    r"\b(?:was|were|is|are)\s+not\s+"
    r"(?:conducted|performed|carried\s+out|undertaken|measured|analy[sz]ed|"
    r"examined|investigated)\b|"
    r"\b(?:IRM|SIRM|hysteresis|FORC|thermomagnetic|susceptibility|"
    r"microscop(?:y|ic)|petrograph(?:y|ic)|SEM|TEM|XRD)\b.{0,55}"
    r"\b(?:not\s+performed|not\s+conducted|not\s+available|was\s+not\s+done)\b|"
    r"\bwithout\s+(?:an?\s+)?(?:IRM|SIRM|hysteresis|FORC|thermomagnetic|"
    r"susceptibility|microscop(?:y|ic)|petrograph(?:y|ic)|SEM|TEM|XRD)\b|"
    r"\b(?:future|further)\s+(?:work|stud(?:y|ies))\b.{0,70}"
    r"\b(?:should|must|need(?:s|ed)?)\s+(?:be\s+)?(?:perform|conduct|measure|"
    r"investigate|examine)|"
    r"\b(?:impossible|unable)\b.{0,70}\b(?:perform|conduct|measure|analy[sz]e|"
    r"investigate|examine)\b",
    re.I | re.S,
)
_NEGATED_ALTERATION_RE = re.compile(
    r"\bno\s+(?:evidence\s+(?:for|of)\s+)?(?:mineralogical\s+|magnetic\s+)?"
    r"alteration\b|\b(?:alteration|oxidation)\b.{0,35}"
    r"\b(?:was|is)\s+not\s+(?:identified|detected|observed|found)\b",
    re.I | re.S,
)
_ALTERATION_RESULT_RE = re.compile(
    r"\b(?:alter(?:ation|ed)|oxidi[sz](?:ation|ed)|exsolution|replacement)\b",
    re.I,
)
_AUXILIARY_DIAGNOSTIC_RE = re.compile(
    r"\bno\s+saturation\b|"
    r"\b(?:do|does|did)\s+not\s+saturate\b|"
    r"\b(?:not|never)\s+(?:fully\s+|completely\s+)?saturated\b|"
    r"\b(?:coercivity|unblocking)\s+(?:spectrum|spectra|distribution)\b|"
    r"\b(?:Hcr|Bcr)\s*/\s*(?:Hc|Bc)\b|\b(?:Mrs|Jrs)\s*/\s*(?:Ms|Js)\b",
    re.I,
)
_TEMPERATURE_VALUE_PATTERN = (
    r"[~\u2248\u223c<>\u2264\u2265]?\s*\d{2,4}(?:\.\d+)?"
    r"(?:\s*(?:-|\u2013|\u2014|to|and)\s*"
    r"[~\u2248\u223c<>\u2264\u2265]?\s*\d{2,4}(?:\.\d+)?)?"
    r"\s*(?:\u00b0|\u00ba|\u25e6)?\s*[CK]\b"
)
_FIELD_VALUE_PATTERN = (
    r"[~\u2248\u223c<>\u2264\u2265]?\s*\d+(?:\.\d+)?"
    r"\s*(?:mT|T|kOe|Oe|gauss)\b"
)
_CURIE_QUANTIFIED_RESULT_RE = re.compile(
    rf"(?:\bCurie\s+(?:points?|temperatures?)\b\s*"
    rf"(?:(?:is|was)\s+estimated\s+(?:at|as)|estimated\s+(?:at|as)|"
    rf"(?:is|was)(?:\s+(?:about|around|approximately))?|at|near|about|"
    rf"around|approximately|below|above|between|"
    rf"rang(?:e|es|ed|ing)(?:\s+from)?|of(?:\s+about)?|=)\s*"
    rf"{_TEMPERATURE_VALUE_PATTERN}|{_TEMPERATURE_VALUE_PATTERN}"
    rf"\s*(?:is|was|as|for)?\s*(?:the\s+)?"
    rf"Curie\s+(?:points?|temperatures?)\b)",
    re.I,
)
_SATURATION_ENDPOINT_LINK_PATTERN = (
    r"(?:at|by|above|below|before|until|beyond|up\s+to|"
    r"in\s+(?:the\s+)?(?:maximum\s+)?(?:applied\s+)?fields?(?:\s+of)?)"
)
_SATURATION_FIELD_ENDPOINT_RE = re.compile(
    rf"(?:\b(?:saturat(?:e|ed|ion)|non[- ]saturation|unsaturat(?:ed|ion)?)\b"
    rf"[^.;!?]{{0,60}}{_SATURATION_ENDPOINT_LINK_PATTERN}\s*"
    rf"{_FIELD_VALUE_PATTERN}|{_SATURATION_ENDPOINT_LINK_PATTERN}\s*"
    rf"{_FIELD_VALUE_PATTERN}[^.;!?]{{0,60}}"
    rf"\b(?:saturat(?:e|ed|ion)|non[- ]saturation|"
    rf"unsaturat(?:ed|ion)?)\b)",
    re.I,
)
_HYSTERESIS_RATIO_PATTERN = (
    r"(?:\b(?:Hcr|Bcr)\s*/\s*(?:Hc|Bc)\b|"
    r"\b(?:Mrs|Jrs)\s*/\s*(?:Ms|Js)\b)"
)
_HYSTERESIS_RATIO_QUANTIFIED_RE = re.compile(
    rf"{_HYSTERESIS_RATIO_PATTERN}[^.;!?]{{0,55}}"
    r"(?:=\s*\d+(?:\.\d+)?|(?:values?\s+)?between\s+\d+(?:\.\d+)?"
    r"\s+and\s+\d+(?:\.\d+)?|rang(?:e|es|ed|ing)\s+(?:from\s+)?"
    r"\d+(?:\.\d+)?\s+(?:to|\u2013|-)\s+\d+(?:\.\d+)?|"
    r"(?:value|ratio)\s+(?:of\s+)?\d+(?:\.\d+)?)",
    re.I,
)
_DOMAIN_STATE_PATTERN = (
    r"(?:\b(?:SD|PSD|MD)\b|\bsingle[- ]domain\b|"
    r"\bpseudo[- ]?single[- ]domain\b|\bmulti[- ]domain\b)"
)
_DOMAIN_PLOT_RESULT_RE = re.compile(
    rf"\b(?:samples?|data|points?|parameters?)\b[^.;!?]{{0,90}}"
    rf"\b(?:plot|plotted|fall|falls|fell|lie|lies|cluster|clustered)\b"
    rf"[^.;!?]{{0,90}}{_DOMAIN_STATE_PATTERN}(?:\s+(?:field|region))?",
    re.I,
)
_HYSTERESIS_CONTEXT_RE = re.compile(
    rf"\b(?:hysteresis|FORC|Day\s+plot)\b|{_HYSTERESIS_RATIO_PATTERN}",
    re.I,
)
_WEAK_ONLY_TEXT_RE = re.compile(
    r"\bT(?:ub|b)\s*(?:=|>|<|:)|^\s*(?:SUSC|AN|Qn)\s*(?:\([^)]*\)|=.*)?\s*$",
    re.I,
)
_SIGNIFICANCE_SUBTYPES = {
    "carrier_significance",
    "explicit_carrier_bridge",
    "mineral_remanence_bridge",
    "origin_bridge",
}
_MICROSCOPY_CONCEPTS = {
    "mineralogical_phase_analysis",
    "petrographic_microscopy",
}


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _clean(value: Any) -> str:
    return " ".join("" if value is None else str(value).split())


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise R3RockmagArtifactValidationError(
            f"artifact must contain JSON-compatible finite values: {exc}"
        ) from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _walk_artifact_boundary(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        normalized_seen: set[str] = set()
        for key, member in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            if not normalized or normalized in normalized_seen:
                raise R3RockmagArtifactValidationError(
                    f"empty or duplicate normalized artifact key at {child_path}"
                )
            normalized_seen.add(normalized)
            if normalized in _ANSWER_DECLARATION_KEYS:
                if member is not False:
                    raise R3RockmagArtifactValidationError(
                        f"{child_path} must be false"
                    )
            elif normalized in _FORBIDDEN_ARTIFACT_KEYS:
                raise R3RockmagArtifactValidationError(
                    f"record identifier, label payload, or answer mapping is forbidden at {child_path}"
                )
            if normalized == "labelsupervision" and child_path != "$.provenance.label_supervision":
                raise R3RockmagArtifactValidationError(
                    "label_supervision is allowed only at $.provenance.label_supervision"
                )
            _walk_artifact_boundary(member, child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, member in enumerate(value):
            _walk_artifact_boundary(member, f"{path}[{index}]")


def _walk_runtime_boundary(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        normalized_seen: set[str] = set()
        for key, member in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            if not normalized or normalized in normalized_seen:
                raise R3RockmagRuntimeBoundaryError(
                    f"empty or duplicate normalized retrieval key at {child_path}"
                )
            normalized_seen.add(normalized)
            if normalized in _FORBIDDEN_RUNTIME_KEYS:
                raise R3RockmagRuntimeBoundaryError(
                    f"forbidden runtime table, identifier, label, or answer field at {child_path}"
                )
            _walk_runtime_boundary(member, child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, member in enumerate(value):
            _walk_runtime_boundary(member, f"{path}[{index}]")


def _required_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise R3RockmagArtifactValidationError(f"{key} must be an object")
    return value


def _required_bool(parent: Mapping[str, Any], key: str, expected: bool, path: str) -> None:
    if parent.get(key) is not expected:
        raise R3RockmagArtifactValidationError(f"{path}.{key} must be {str(expected).lower()}")


def _string_list(
    parent: Mapping[str, Any],
    key: str,
    path: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    raw = parent.get(key)
    if not isinstance(raw, list) or (not raw and not allow_empty):
        requirement = "a list" if allow_empty else "a non-empty list"
        raise R3RockmagArtifactValidationError(f"{path}.{key} must be {requirement}")
    cleaned = tuple(_clean(member).casefold() for member in raw)
    if any(not member for member in cleaned) or len(set(cleaned)) != len(cleaned):
        raise R3RockmagArtifactValidationError(
            f"{path}.{key} must contain unique non-empty strings"
        )
    return cleaned


def _compile_pattern_list(raw: Any, path: str) -> tuple[re.Pattern[str], ...]:
    if not isinstance(raw, list) or not raw:
        raise R3RockmagArtifactValidationError(f"{path} must be a non-empty list")
    patterns: list[re.Pattern[str]] = []
    seen: set[str] = set()
    for index, expression in enumerate(raw):
        if not isinstance(expression, str) or not expression.strip() or len(expression) > 600:
            raise R3RockmagArtifactValidationError(
                f"{path}[{index}] must be a non-empty regex of at most 600 characters"
            )
        if expression in seen:
            raise R3RockmagArtifactValidationError(f"{path} contains duplicate regexes")
        seen.add(expression)
        try:
            compiled = re.compile(expression, re.I | re.S)
        except re.error as exc:
            raise R3RockmagArtifactValidationError(
                f"invalid regex at {path}[{index}]: {exc}"
            ) from exc
        if compiled.search(""):
            raise R3RockmagArtifactValidationError(
                f"{path}[{index}] must not match empty text"
            )
        patterns.append(compiled)
    return tuple(patterns)


def _taxonomy_entries(
    artifact: Mapping[str, Any],
    key: str,
) -> tuple[tuple[str, tuple[re.Pattern[str], ...]], ...]:
    raw = artifact.get(key)
    if not isinstance(raw, list) or not raw:
        raise R3RockmagArtifactValidationError(f"{key} must be a non-empty list")
    output: list[tuple[str, tuple[re.Pattern[str], ...]]] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        path = f"{key}[{index}]"
        if not isinstance(entry, Mapping):
            raise R3RockmagArtifactValidationError(f"{path} must be an object")
        concept_id = _clean(entry.get("concept_id")).casefold()
        if not concept_id or concept_id in seen:
            raise R3RockmagArtifactValidationError(
                f"{path}.concept_id must be unique and non-empty"
            )
        seen.add(concept_id)
        patterns = _compile_pattern_list(
            entry.get("runtime_original_text_patterns"),
            f"{path}.runtime_original_text_patterns",
        )
        if not isinstance(entry.get("training_statistics"), Mapping):
            raise R3RockmagArtifactValidationError(
                f"{path}.training_statistics must be an aggregate object"
            )
        output.append((concept_id, patterns))
    return tuple(output)


def validate_r3_rockmag_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return an isolated JSON copy of a supervised aggregate artifact."""

    if not isinstance(artifact, Mapping):
        raise R3RockmagArtifactValidationError("artifact must be a JSON object")
    _walk_artifact_boundary(artifact)
    copy = json.loads(_canonical_json(artifact))

    if copy.get("artifact_class") != ARTIFACT_CLASS:
        raise R3RockmagArtifactValidationError(
            f"artifact_class must be {ARTIFACT_CLASS!r}"
        )
    if copy.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise R3RockmagArtifactValidationError(
            f"schema_version must be {ARTIFACT_SCHEMA_VERSION!r}"
        )
    if copy.get("profile") != "rockmag_text_distilled_v1":
        raise R3RockmagArtifactValidationError(
            "profile must be 'rockmag_text_distilled_v1'"
        )

    provenance = _required_mapping(copy, "provenance")
    _required_bool(provenance, "label_supervision", True, "provenance")
    _required_bool(provenance, "raw_field_values_retained", False, "provenance")
    _required_bool(provenance, "row_identity_mapping_retained", False, "provenance")
    _required_bool(provenance, "source_paper_quotes_retained", False, "provenance")
    if provenance.get("phase") != "offline_training_only":
        raise R3RockmagArtifactValidationError(
            "provenance.phase must be 'offline_training_only'"
        )
    if provenance.get("supervision_scope") != "pre_registered_training_partition_only":
        raise R3RockmagArtifactValidationError(
            "provenance.supervision_scope must be 'pre_registered_training_partition_only'"
        )

    split = _required_mapping(copy, "split")
    _required_bool(split, "held_out_supervision_accessed", False, "split")
    if split.get("unit") != "paper_identity":
        raise R3RockmagArtifactValidationError("split.unit must be 'paper_identity'")

    runtime = _required_mapping(copy, "runtime_policy")
    for key in (
        "contains_record_answer_mapping",
        "previous_prediction_access",
        "source_metadata_table_access",
        "supervised_field_access",
    ):
        _required_bool(runtime, key, False, "runtime_policy")
    allowed = set(_string_list(runtime, "allowed_scoring_sources", "runtime_policy"))
    normalized_allowed = {_normalized_key(source) for source in allowed}
    required_allowed = {
        "identitymanifest",
        "originalpaperpagetext",
        "thisfrozenaggregateartifact",
    }
    if not required_allowed.issubset(normalized_allowed):
        missing = sorted(required_allowed - normalized_allowed)
        raise R3RockmagArtifactValidationError(
            f"runtime_policy.allowed_scoring_sources is missing {missing}"
        )

    _taxonomy_entries(copy, "method_taxonomy")
    _taxonomy_entries(copy, "carrier_taxonomy")
    _taxonomy_entries(copy, "diagnostic_taxonomy")
    decision_policy = _required_mapping(copy, "decision_policy")
    _string_list(decision_policy, "always_required", "decision_policy")
    _string_list(decision_policy, "positive_requires_any", "decision_policy")
    _string_list(decision_policy, "standalone_non_support", "decision_policy")
    proxy_policy = decision_policy.get(
        "rockmag_proxy_policy",
        DEFAULT_ROCKMAG_PROXY_POLICY,
    )
    if proxy_policy not in ROCKMAG_PROXY_POLICIES:
        raise R3RockmagArtifactValidationError(
            "decision_policy.rockmag_proxy_policy is unsupported"
        )
    identity_granularity = decision_policy.get(
        "identity_granularity",
        DEFAULT_IDENTITY_GRANULARITY,
    )
    if identity_granularity not in IDENTITY_GRANULARITIES:
        raise R3RockmagArtifactValidationError(
            "decision_policy.identity_granularity is unsupported"
        )
    if decision_policy.get("missing_or_ambiguous_source") != "ABSTAIN":
        raise R3RockmagArtifactValidationError(
            "decision_policy.missing_or_ambiguous_source must be 'ABSTAIN'"
        )
    exclusions = copy.get("standalone_exclusions")
    if not isinstance(exclusions, list) or not exclusions:
        raise R3RockmagArtifactValidationError(
            "standalone_exclusions must be a non-empty list"
        )
    return copy


def _source_flags(value: Sequence[str] | Mapping[str, Any] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        members = [key for key, present in value.items() if present]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        members = list(value)
    else:
        raise TypeError("source_flags must be a sequence of strings or a mapping")
    if any(not isinstance(member, str) for member in members):
        raise TypeError("source_flags must contain strings")
    return tuple(sorted({_clean(member) for member in members if _clean(member)}))


def _hit_page(hit: Mapping[str, Any]) -> int | None:
    value = hit.get("page")
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _hit_relevance(hit: Mapping[str, Any]) -> float:
    value = hit.get("target_relevance", hit.get("binding", 0.0))
    if isinstance(value, bool):
        return 0.0
    try:
        relevance = float(value)
    except (TypeError, ValueError):
        return 0.0
    return relevance if math.isfinite(relevance) else 0.0


def _hit_text(hit: Mapping[str, Any]) -> str:
    return _clean(hit.get("claim_text") or hit.get("matched_text"))


def _iter_hits(retrieval: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[tuple[str | None, Any]] = []
    for hit in retrieval.get("hits") or []:
        candidates.append((None, hit))
    categories = retrieval.get("categories") or {}
    if isinstance(categories, Mapping):
        for category, hits in categories.items():
            if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes, bytearray)):
                candidates.extend((str(category), hit) for hit in hits)
    chain_hits = ((retrieval.get("best_chain") or {}).get("hits") or {})
    if isinstance(chain_hits, Mapping):
        candidates.extend((str(category), hit) for category, hit in chain_hits.items())

    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for category_hint, raw in candidates:
        if not isinstance(raw, Mapping):
            continue
        hit = dict(raw)
        category = _clean(hit.get("category") or category_hint).casefold()
        hit["category"] = category
        key = (
            category,
            _clean(hit.get("subtype")).casefold(),
            _hit_page(hit),
            _clean(hit.get("quote")),
            _clean(hit.get("matched_text")),
            hit.get("start"),
            hit.get("end"),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(hit)
    return output


def _base_rejection_reason(
    hit: Mapping[str, Any],
    *,
    multi_target: bool,
    minimum_relevance: float,
) -> str | None:
    if hit.get("reference_noise") is True or _REFERENCE_SECTION_RE.search(
        _clean(hit.get("section"))
    ):
        return "reference_or_bibliography"
    if _hit_page(hit) is None:
        return "missing_page"
    if _hit_relevance(hit) < minimum_relevance:
        return "low_target_relevance"
    if multi_target and hit.get("local_target") is not True:
        return "cross_target_or_unbound"
    status = _normalized_key(hit.get("claim_status") or hit.get("status") or "")
    if status in {"negated", "notexecuted", "hypothetical", "reference", "rejected"}:
        return "negated_or_not_executed"
    if hit.get("executed") is False or hit.get("affirmed") is False:
        return "negated_or_not_executed"
    if _NON_EXECUTION_RE.search(_hit_text(hit)):
        return "negated_or_not_executed"
    return None


def _compact_hit(hit: Mapping[str, Any], role: str) -> dict[str, Any]:
    quote = _clean(
        hit.get("claim_text") or hit.get("matched_text") or hit.get("quote")
    )
    return {
        "role": role,
        "category": _clean(hit.get("category")).casefold(),
        "subtype": _clean(hit.get("subtype")).casefold(),
        "page": _hit_page(hit),
        "section": _clean(hit.get("section")),
        "target_relevance": round(_hit_relevance(hit), 4),
        "local_target": bool(hit.get("local_target")),
        "target_anchor": _clean(hit.get("target_anchor")),
        "scope_id": _clean(hit.get("scope_id")) or None,
        "scope_kind": _clean(hit.get("scope_kind")) or None,
        "matched_concepts": list(hit.get("_matched_concepts") or []),
        "quote": quote[:500],
        "quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest() if quote else None,
    }


def _matches_any(patterns: Sequence[re.Pattern[str]], text: str) -> bool:
    return bool(text and any(pattern.search(text) for pattern in patterns))


def _matching_concepts(
    taxonomy: Sequence[tuple[str, Sequence[re.Pattern[str]]]],
    text: str,
) -> tuple[str, ...]:
    return tuple(
        concept_id
        for concept_id, patterns in taxonomy
        if _matches_any(patterns, text)
    )


def _with_concepts(hit: Mapping[str, Any], concepts: Sequence[str]) -> dict[str, Any]:
    enriched = dict(hit)
    enriched["_matched_concepts"] = sorted(set(concepts))
    return enriched


def _claim_identity(hit: Mapping[str, Any]) -> tuple[int | None, str]:
    claim = _clean(
        hit.get("claim_text") or hit.get("quote") or hit.get("matched_text")
    ).casefold()
    return _hit_page(hit), claim


def _self_authenticating_diagnostic(hit: Mapping[str, Any]) -> bool:
    text = _hit_text(hit)
    if not text:
        return False
    if _CURIE_QUANTIFIED_RESULT_RE.search(text):
        return True
    if _SATURATION_FIELD_ENDPOINT_RE.search(text):
        return True
    if _HYSTERESIS_RATIO_QUANTIFIED_RE.search(text):
        return True
    return bool(
        _HYSTERESIS_CONTEXT_RE.search(text)
        and _DOMAIN_PLOT_RESULT_RE.search(text)
    )


_PROMOTED_SCOPE_ANCHORS = {
    "target_same_paragraph",
    "target_same_table_row",
    "target_same_discourse",
}


def _scope_compatible(hits: Sequence[Mapping[str, Any]], multi_target: bool) -> bool:
    if not multi_target:
        return True
    if not any(_clean(hit.get("target_anchor")) in _PROMOTED_SCOPE_ANCHORS for hit in hits):
        return True
    scope_ids = [_clean(hit.get("scope_id")) for hit in hits]
    return bool(scope_ids) and all(scope_id == scope_ids[0] for scope_id in scope_ids)


def _best_pair(
    methods: Sequence[Mapping[str, Any]],
    companions: Sequence[Mapping[str, Any]],
    maximum_page_span: int,
    multi_target: bool,
    *,
    diagnostic_pair: bool = False,
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    candidates: list[tuple[tuple[float, ...], Mapping[str, Any], Mapping[str, Any]]] = []
    for method in methods:
        method_page = _hit_page(method)
        if method_page is None:
            continue
        for companion in companions:
            if not _scope_compatible((method, companion), multi_target):
                continue
            if (
                diagnostic_pair
                and _claim_identity(method) == _claim_identity(companion)
                and not _self_authenticating_diagnostic(companion)
            ):
                continue
            companion_page = _hit_page(companion)
            if companion_page is None:
                continue
            page_span = abs(method_page - companion_page)
            if page_span > maximum_page_span:
                continue
            rank = (
                float(-page_span),
                float(bool(method.get("local_target")) + bool(companion.get("local_target"))),
                float(
                    diagnostic_pair
                    and _clean(companion.get("category")).casefold() == "diagnostic"
                ),
                float(
                    diagnostic_pair
                    and _clean(method.get("category")).casefold() == "method"
                ),
                _hit_relevance(method) + _hit_relevance(companion),
                -float(method_page),
                -float(companion_page),
            )
            candidates.append((rank, method, companion))
    if not candidates:
        return None
    _, method, companion = max(candidates, key=lambda candidate: candidate[0])
    return method, companion


def _best_single(
    hits: Sequence[Mapping[str, Any]],
    *,
    preferred_category: str,
) -> Mapping[str, Any] | None:
    candidates = [hit for hit in hits if _hit_page(hit) is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda hit: (
            _clean(hit.get("category")).casefold() == preferred_category,
            bool(hit.get("local_target")),
            _hit_relevance(hit),
            -int(_hit_page(hit) or 0),
        ),
    )


def _best_triple(
    methods: Sequence[Mapping[str, Any]],
    carriers: Sequence[Mapping[str, Any]],
    significance: Sequence[Mapping[str, Any]],
    maximum_page_span: int,
    multi_target: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None:
    candidates: list[
        tuple[
            tuple[float, ...],
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
        ]
    ] = []
    for method in methods:
        for carrier in carriers:
            for bridge in significance:
                if not _scope_compatible((method, carrier, bridge), multi_target):
                    continue
                pages = (_hit_page(method), _hit_page(carrier), _hit_page(bridge))
                if any(page is None for page in pages):
                    continue
                span = max(int(page) for page in pages) - min(int(page) for page in pages)
                if span > maximum_page_span:
                    continue
                rank = (
                    float(-span),
                    float(sum(bool(hit.get("local_target")) for hit in (method, carrier, bridge))),
                    sum(_hit_relevance(hit) for hit in (method, carrier, bridge)),
                    -float(min(int(page) for page in pages)),
                )
                candidates.append((rank, method, carrier, bridge))
    if not candidates:
        return None
    _, method, carrier, bridge = max(candidates, key=lambda candidate: candidate[0])
    return method, carrier, bridge


class R3RockmagDistilledReasoner:
    """Reconstruct a ROCKMAG-equivalent evidence episode from original text."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.artifact = validate_r3_rockmag_artifact(artifact)
        self.artifact_sha256 = _sha256_json(self.artifact)
        self.method_taxonomy = _taxonomy_entries(self.artifact, "method_taxonomy")
        self.carrier_taxonomy = _taxonomy_entries(self.artifact, "carrier_taxonomy")
        self.diagnostic_taxonomy = _taxonomy_entries(self.artifact, "diagnostic_taxonomy")
        decision_policy = self.artifact["decision_policy"]
        self.rockmag_proxy_policy = decision_policy.get(
            "rockmag_proxy_policy",
            DEFAULT_ROCKMAG_PROXY_POLICY,
        )
        self.identity_granularity = decision_policy.get(
            "identity_granularity",
            DEFAULT_IDENTITY_GRANULARITY,
        )
        self.minimum_relevance = MINIMUM_TARGET_RELEVANCE
        self.maximum_page_span = MAXIMUM_PAGE_SPAN

    def evaluate(
        self,
        retrieval: Mapping[str, Any],
        *,
        source_flags: Sequence[str] | Mapping[str, Any] | None = None,
        multi_target: bool = False,
        identity_ambiguous: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(retrieval, Mapping):
            raise TypeError("retrieval must be a mapping")
        if not isinstance(multi_target, bool) or not isinstance(identity_ambiguous, bool):
            raise TypeError("multi_target and identity_ambiguous must be booleans")
        _walk_runtime_boundary(retrieval)
        flags = _source_flags(source_flags)
        normalized_flags = {_normalized_key(flag) for flag in flags}
        quality_risks = sorted(normalized_flags & _QUALITY_RISK_FLAGS)
        source_available = bool(retrieval.get("source_available"))

        all_hits = _iter_hits(retrieval)
        rejection_counts: Counter[str] = Counter()
        eligible: list[dict[str, Any]] = []
        for hit in all_hits:
            if hit.get("category") == "verdict":
                rejection_counts["embedded_score_or_verdict"] += 1
                continue
            reason = _base_rejection_reason(
                hit,
                multi_target=multi_target,
                minimum_relevance=self.minimum_relevance,
            )
            if reason:
                rejection_counts[reason] += 1
                continue
            eligible.append(hit)

        methods: list[dict[str, Any]] = []
        microscopy: list[dict[str, Any]] = []
        carriers: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        alterations: list[dict[str, Any]] = []
        significance: list[dict[str, Any]] = []
        weak_only_count = 0
        for hit in eligible:
            subtype = _clean(hit.get("subtype")).casefold()
            text = _hit_text(hit)
            method_concepts = _matching_concepts(self.method_taxonomy, text)
            carrier_concepts = _matching_concepts(self.carrier_taxonomy, text)
            diagnostic_concepts = _matching_concepts(self.diagnostic_taxonomy, text)
            if subtype in {
                "bare_anisotropy",
                "bare_susceptibility",
                "demagnetization",
                "unblocking_temperature_only",
            } or _WEAK_ONLY_TEXT_RE.search(text):
                weak_only_count += 1
            if method_concepts:
                method_hit = _with_concepts(hit, method_concepts)
                methods.append(method_hit)
                if set(method_concepts) & _MICROSCOPY_CONCEPTS:
                    microscopy.append(method_hit)
            if carrier_concepts and carrier_claim_is_affirmed(hit):
                carriers.append(_with_concepts(hit, carrier_concepts))
            if diagnostic_concepts or _AUXILIARY_DIAGNOSTIC_RE.search(text):
                concepts = diagnostic_concepts or ("criterion_diagnostic_result",)
                diagnostics.append(_with_concepts(hit, concepts))
            if _ALTERATION_RESULT_RE.search(text) and not _NEGATED_ALTERATION_RE.search(text):
                alterations.append(_with_concepts(hit, ("mineral_alteration",)))
            if (
                hit.get("category") == "bridge"
                and subtype in _SIGNIFICANCE_SUBTYPES
                and carrier_claim_is_affirmed(hit)
            ):
                significance.append(_with_concepts(hit, ("carrier_significance",)))

        microscopy_pair = _best_pair(
            microscopy,
            [*carriers, *alterations],
            self.maximum_page_span,
            multi_target,
        )
        method_diagnostic_pair = _best_pair(
            methods,
            diagnostics,
            self.maximum_page_span,
            multi_target,
            diagnostic_pair=True,
        )
        method_carrier_pair = _best_pair(
            methods,
            carriers,
            self.maximum_page_span,
            multi_target,
        )
        normative_triple = _best_triple(
            methods,
            carriers,
            significance,
            self.maximum_page_span,
            multi_target,
        )
        self_authenticating_diagnostics = [
            hit for hit in diagnostics if _self_authenticating_diagnostic(hit)
        ]

        selected_pair: tuple[Mapping[str, Any], Mapping[str, Any]] | None
        companion_role: str
        if microscopy_pair:
            selected_pair = microscopy_pair
            companion_role = "carrier_or_alteration"
            decision_path = "MICROSCOPY_WITH_CARRIER_OR_ALTERATION"
        elif method_diagnostic_pair:
            selected_pair = method_diagnostic_pair
            companion_role = "diagnostic_result"
            decision_path = "ACCEPTED_METHOD_WITH_DIAGNOSTIC_RESULT"
        elif method_carrier_pair:
            selected_pair = method_carrier_pair
            companion_role = "carrier_identification"
            decision_path = "ACCEPTED_METHOD_WITH_CARRIER_IDENTIFICATION"
        else:
            selected_pair = None
            companion_role = ""
            decision_path = "NO_CONNECTED_STRONG_EVIDENCE"

        proxy_single: Mapping[str, Any] | None = None
        proxy_single_role = ""
        if (
            selected_pair is None
            and self.rockmag_proxy_policy
            == "distilled_method_or_self_authenticating_result"
        ):
            proxy_single = _best_single(methods, preferred_category="method")
            if proxy_single is not None:
                proxy_single_role = "distilled_rockmag_method"
                decision_path = "ACCEPTED_DISTILLED_ROCKMAG_METHOD"
            else:
                proxy_single = _best_single(
                    self_authenticating_diagnostics,
                    preferred_category="diagnostic",
                )
                if proxy_single is not None:
                    proxy_single_role = "self_authenticating_diagnostic_result"
                    decision_path = "ACCEPTED_SELF_AUTHENTICATING_DIAGNOSTIC_RESULT"

        proxy_support = bool(selected_pair or proxy_single)
        identity_blocked = bool(
            identity_ambiguous
            and self.identity_granularity == DEFAULT_IDENTITY_GRANULARITY
        )

        if not source_available:
            policy_decision: int | str = "ABSTAIN"
            normative_decision: int | str = "ABSTAIN"
            normative_reason_code = "SOURCE_UNAVAILABLE"
            reason_code = "SOURCE_UNAVAILABLE"
            reason = "No original-paper text was available for source-only R3 reasoning."
        elif identity_blocked:
            policy_decision = "ABSTAIN"
            normative_decision = "ABSTAIN"
            normative_reason_code = "IDENTITY_AMBIGUOUS"
            reason_code = "IDENTITY_AMBIGUOUS"
            reason = "The strict identity manifest cannot distinguish the requested result component."
        elif quality_risks:
            policy_decision = "ABSTAIN"
            normative_decision = "ABSTAIN"
            normative_reason_code = "SOURCE_QUALITY_RISK"
            reason_code = "SOURCE_QUALITY_RISK"
            reason = "Original-paper extraction quality is insufficient for a closed-world R3 decision."
        elif proxy_support:
            policy_decision = 1
            normative_decision = 1 if normative_triple else "ABSTAIN"
            normative_reason_code = (
                "METHOD_CARRIER_SIGNIFICANCE_CHAIN"
                if normative_triple
                else "NORMATIVE_CHAIN_INCOMPLETE"
            )
            reason_code = decision_path
            if selected_pair:
                reason = (
                    "A target-bound accepted carrier-characterization method and a same-page or "
                    "adjacent-page diagnostic result, carrier identification, or microscopy result "
                    "were recovered from the original paper."
                )
            else:
                reason = (
                    "The original paper reports a target-bound performed method or quantified "
                    "diagnostic result whose semantic class was distilled from the frozen "
                    "ROCKMAG expert taxonomy."
                )
        else:
            policy_decision = 0
            normative_decision = 1 if normative_triple else "ABSTAIN"
            normative_reason_code = (
                "METHOD_CARRIER_SIGNIFICANCE_CHAIN"
                if normative_triple
                else "NORMATIVE_CHAIN_INCOMPLETE"
            )
            reason_code = "NO_TARGET_BOUND_STRONG_ROCKMAG_EVIDENCE"
            reason = (
                "Complete high-quality source text contained no connected target-bound strong "
                "ROCKMAG-equivalent evidence; weak or rejected evidence cannot trigger R3."
            )

        proxy_decision = 1 if policy_decision == 1 else 0
        outcome = (
            "POSITIVE" if policy_decision == 1 else "NEGATIVE" if policy_decision == 0 else "UNKNOWN"
        )
        selected_evidence: dict[str, Any] = {}
        selected_page_span: int | None = None
        if selected_pair:
            method, companion = selected_pair
            selected_evidence = {
                "method": _compact_hit(method, "accepted_method"),
                "companion": _compact_hit(companion, companion_role),
            }
            selected_page_span = abs(int(_hit_page(method)) - int(_hit_page(companion)))
        elif proxy_single:
            selected_evidence = {
                "proxy": _compact_hit(proxy_single, proxy_single_role),
            }
            selected_page_span = 0

        normative_evidence: dict[str, Any] = {}
        normative_page_span: int | None = None
        if normative_triple:
            method, carrier, bridge = normative_triple
            normative_evidence = {
                "method": _compact_hit(method, "accepted_method"),
                "carrier": _compact_hit(carrier, "carrier_identification"),
                "significance": _compact_hit(bridge, "carrier_significance"),
            }
            normative_pages = [
                int(_hit_page(method)),
                int(_hit_page(carrier)),
                int(_hit_page(bridge)),
            ]
            normative_page_span = max(normative_pages) - min(normative_pages)

        normative_outcome = (
            "POSITIVE"
            if normative_decision == 1
            else "NEGATIVE"
            if normative_decision == 0
            else "UNKNOWN"
        )
        if not source_available or identity_blocked or quality_risks:
            criterion_practice_conflict: bool | None = None
            comparison_status = "UNKNOWN"
        elif policy_decision == 1 and normative_decision == "ABSTAIN":
            criterion_practice_conflict = True
            comparison_status = "INCOMPLETE_NORMATIVE_CHAIN"
        elif policy_decision == normative_decision == 1:
            criterion_practice_conflict = False
            comparison_status = "AGREE"
        elif normative_decision == 1 and policy_decision == 0:
            criterion_practice_conflict = True
            comparison_status = "TRACK_INCONSISTENCY"
        else:
            criterion_practice_conflict = False
            comparison_status = "NO_PROXY_SUPPORT"

        conditions = {
            "source_available": source_available,
            "source_quality_sufficient": not quality_risks,
            "identity_clear": not identity_blocked,
            "result_component_identity_ambiguous": identity_ambiguous,
            "identity_granularity": self.identity_granularity,
            "rockmag_proxy_policy": self.rockmag_proxy_policy,
            "accepted_method": bool(methods),
            "diagnostic_result": bool(diagnostics),
            "carrier_identification": bool(carriers),
            "carrier_significance": bool(significance),
            "microscopy_method": bool(microscopy),
            "alteration_result": bool(alterations),
            "connected_positive_path": bool(selected_pair),
            "practice_proxy_positive_path": proxy_support,
            "self_authenticating_diagnostic_result": bool(
                self_authenticating_diagnostics
            ),
            "connected_normative_chain": bool(normative_triple),
            "scope_compatibility_required": multi_target,
            "maximum_page_span": self.maximum_page_span,
            "selected_page_span": selected_page_span,
            "normative_page_span": normative_page_span,
        }
        normative_track = {
            "policy_decision": normative_decision,
            "value": normative_decision if normative_decision in {0, 1} else None,
            "binary_value": 1 if normative_decision == 1 else 0,
            "outcome": normative_outcome,
            "decision_path": (
                "METHOD_CARRIER_SIGNIFICANCE_CHAIN"
                if normative_decision == 1
                else normative_reason_code
            ),
            "evidence": normative_evidence,
        }
        proxy_track = {
            "policy_decision": policy_decision,
            "value": policy_decision if policy_decision in {0, 1} else None,
            "binary_value": proxy_decision,
            "outcome": outcome,
            "decision_path": decision_path if policy_decision == 1 else reason_code,
            "evidence": selected_evidence,
        }
        return {
            "schema_version": "r3_rockmag_distilled_decision_v1",
            "engine_version": ENGINE_VERSION,
            "indicator": "R3",
            "artifact_class": ARTIFACT_CLASS,
            "artifact_sha256": self.artifact_sha256,
            "evidence_state": "SUPPORTED_ONLY" if policy_decision == 1 else "NEITHER",
            "policy_decision": policy_decision,
            "proxy_decision": proxy_decision,
            "binary_compatibility_value": proxy_decision,
            "normative_r3": normative_track,
            "rockmag_proxy_r3": proxy_track,
            "criterion_practice_conflict": criterion_practice_conflict,
            "criterion_practice_comparison": {
                "status": comparison_status,
                "normative_policy_decision": normative_decision,
                "rockmag_proxy_policy_decision": policy_decision,
            },
            "outcome": outcome,
            "manual_review": (
                policy_decision == "ABSTAIN"
                or criterion_practice_conflict is True
                or identity_ambiguous
            ),
            "reason_code": reason_code,
            "reason": reason,
            "decision_path": decision_path if policy_decision == 1 else reason_code,
            "conditions": conditions,
            "selected_evidence": selected_evidence,
            "evidence_counts": {
                "retrieved": len(all_hits),
                "eligible": len(eligible),
                "accepted_methods": len(methods),
                "microscopy_methods": len(microscopy),
                "diagnostic_results": len(diagnostics),
                "carrier_identifications": len(carriers),
                "carrier_significance": len(significance),
                "alteration_results": len(alterations),
                "weak_only": weak_only_count,
            },
            "rejected_evidence_counts": dict(sorted(rejection_counts.items())),
            "source_flags": list(flags),
            "runtime_scoring_sources": [
                "identity manifest",
                "original paper page text",
                "this frozen aggregate artifact",
            ],
            "forbidden_runtime_scoring_sources": [
                "PMAGRESULT",
                "ROCKUNIT",
                "R3 labels",
                "evaluation results",
                "prior predictions",
                "record-level answer mappings",
            ],
            "provenance": {
                "offline_artifact_label_supervision": True,
                "runtime_label_access": False,
                "runtime_source_workbook_access": False,
                "contains_record_answer_mapping": False,
            },
        }


def reason_r3_rockmag_distilled(
    retrieval: Mapping[str, Any],
    *,
    source_flags: Sequence[str] | Mapping[str, Any] | None,
    multi_target: bool,
    identity_ambiguous: bool,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Functional API for one source-only R3 proxy decision."""

    return R3RockmagDistilledReasoner(artifact).evaluate(
        retrieval,
        source_flags=source_flags,
        multi_target=multi_target,
        identity_ambiguous=identity_ambiguous,
    )


__all__ = [
    "ARTIFACT_CLASS",
    "ENGINE_VERSION",
    "R3RockmagArtifactValidationError",
    "R3RockmagDistilledReasoner",
    "R3RockmagRuntimeBoundaryError",
    "reason_r3_rockmag_distilled",
    "validate_r3_rockmag_artifact",
]
