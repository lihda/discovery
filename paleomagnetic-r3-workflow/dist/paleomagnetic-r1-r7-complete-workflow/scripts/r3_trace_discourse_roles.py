#!/usr/bin/env python3
"""Label-free discourse-role features for R3 evidence windows.

The rules encode the R3 criterion's distinction between an experiment that was
actually executed and text that merely cites, negates, conditions, or proposes
one.  They operate only on original-text strings and never inspect database
rows, labels, predictions, or evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any


DISCOURSE_ROLE_NAMES = (
    "performed_execution",
    "reported_result",
    "background_citation_prior_work",
    "negated",
    "prospective",
    "conditional",
    "contrast",
    "cause",
    "interpretation",
)

DISCOURSE_FEATURE_NAMES = (
    "discourse_sentence_count_log",
    *(
        f"discourse_{role}_{statistic}"
        for role in DISCOURSE_ROLE_NAMES
        for statistic in ("present", "sentence_count_log", "sentence_ratio")
    ),
)


_SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?;][\"')\]]?\s+|\n\s*\n)")
_ABBREVIATION_RE = re.compile(
    r"(?:\b(?:fig|figs|eq|eqs|e\.g|i\.e|cf|ca|approx|dr|mr|mrs|ms|vs|"
    r"no|nos|vol|pp|p|ref|refs|inc)|\bet\s+al)\.$",
    re.I,
)
_INITIALS_RE = re.compile(r"(?:(?:\b[A-Z]\.){2,4}|,\s*[A-Z]\.)$")
_CLAUSE_BOUNDARY_RE = re.compile(
    r";|,\s*(?=(?:but|yet|whereas|although|though|however|nevertheless|"
    r"nonetheless|conversely|we|our|this\s+study|the\s+present\s+study)\b)",
    re.I,
)

_ACTIVE_EXECUTION_RE = re.compile(
    r"\b(?:we|I)\s+(?:not\s+only\s+)?"
    r"(?:(?:also|then|subsequently|first|further)\s+){0,2}"
    r"(?:perform(?:ed)?|conduct(?:ed)?|carr(?:y|ied)\s+out|measure(?:d)?|"
    r"analy[sz](?:e|ed)|examine(?:d)?|investigate(?:d)?|collect(?:ed)?|"
    r"apply|applied|use|used|acquire(?:d)?|obtain(?:ed)?|heat(?:ed)?|"
    r"cool(?:ed)?|demagneti[sz](?:e|ed)|subject(?:ed)?)\b",
    re.I,
)
_CURRENT_STUDY_EXECUTION_RE = re.compile(
    r"\b(?:this|the\s+present)\s+(?:study|work|investigation)\s+"
    r"(?:(?:also|then)\s+)?(?:perform(?:s|ed)?|conduct(?:s|ed)?|"
    r"measure(?:s|d)?|analy[sz](?:es|ed)?|examine(?:s|d)?|"
    r"investigate(?:s|d)?|apply|applies|applied|use|uses|used|"
    r"acquire(?:s|d)?|obtain(?:s|ed)?)\b",
    re.I,
)
_EXECUTION_SUBJECT = (
    r"samples?|specimens?|cores?|sites?|rocks?|materials?|aliquots?|chips?|"
    r"powders?|data|measurements?|experiments?|analyses?|tests?|curves?|"
    r"loops?|acquisitions?|demagneti[sz]ations?|heating|cooling|microscopy|"
    r"susceptibilit(?:y|ies)"
)
_PASSIVE_EXECUTION_RE = re.compile(
    rf"\b(?:the(?:se)?\s+)?(?:[A-Za-z0-9()+/-]+\s+){{0,4}}"
    rf"(?:{_EXECUTION_SUBJECT})\s+"
    r"(?:was|were|has\s+been|have\s+been)\s+"
    r"(?:(?:also|then|subsequently|first|individually|separately)\s+){0,2}"
    r"(?:performed|conducted|carried\s+out|measured|analy[sz]ed|examined|"
    r"investigated|collected|applied|used|acquired|obtained|heated|cooled|"
    r"demagneti[sz]ed|subjected|tested|made)\b",
    re.I,
)
_MEASUREMENTS_MADE_RE = re.compile(
    r"\b(?:we\s+made|measurements?\s+(?:was|were)\s+made)\b", re.I
)
_CONDITIONAL_OPEN_RE = re.compile(
    r"^\s*(?:if|unless|whether|assuming|provided\s+that|in\s+case)\b", re.I
)
_PRIOR_EXECUTION_SCOPE_RE = re.compile(
    r"\b(?:previous|previously|prior|earlier|published)\b.{0,90}"
    r"\b(?:performed|conducted|measured|analy[sz]ed|examined|investigated|"
    r"applied|used|acquired|obtained)\b",
    re.I | re.S,
)
_EXECUTED_BY_CITATION_RE = re.compile(
    r"\b(?:performed|conducted|measured|analy[sz]ed|examined|investigated)"
    r"\s+by\s+[A-Z][A-Za-z'-]+(?:\s+et\s+al\.?)?",
)

_REPORTED_RESULT_RE = re.compile(
    r"\b(?:results?|data|measurements?|observations?|curves?|loops?|analyses?|"
    r"experiments?|tests?)\s+(?:clearly\s+|also\s+)?"
    r"(?:show|shows|showed|indicate|indicates|indicated|reveal|reveals|"
    r"revealed|demonstrate|demonstrates|demonstrated|suggest|suggests|"
    r"suggested|yield|yields|yielded|confirm|confirms|confirmed)\b|"
    r"\b(?:we|this\s+study)\s+(?:found|find|observed|observe|identified|"
    r"identify|determined|determine|detected|detect|confirmed|confirm)\b|"
    r"\b(?:was|were)\s+(?:identified|detected|observed|found|determined|"
    r"confirmed|resolved|estimated|calculated|reached)\b|"
    r"\b(?:yielded|gave|produced)\s+(?:a|an|the|values?\s+of)\b",
    re.I,
)

_AUTHOR_YEAR_RE = re.compile(
    r"(?:\b[A-Z][A-Za-z'-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'-]+|"
    r"&\s*[A-Z][A-Za-z'-]+))?\s*,?\s*\(?(?:18|19|20)\d{2}[a-z]?\)?|"
    r"\([A-Z][A-Za-z'-]+(?:\s+et\s+al\.?)?\s*,\s*"
    r"(?:18|19|20)\d{2}[a-z]?\))"
)
_NUMERIC_CITATION_RE = re.compile(r"\[(?:\d{1,3}(?:\s*[-,]\s*\d{1,3})*)\]")
_PRIOR_WORK_RE = re.compile(
    r"\b(?:previous|previously|earlier|prior|published|existing)\s+"
    r"(?:work|study|studies|research|investigations?|measurements?|results?|"
    r"observations?|authors?|workers?)\b|"
    r"\b(?:according\s+to|as\s+(?:reported|described|shown|suggested|"
    r"proposed|demonstrated)\s+by)\b|"
    r"\b(?:has|have|had)\s+(?:also\s+)?been\s+(?:reported|described|shown|"
    r"suggested|proposed|demonstrated|documented)\b|"
    r"\b(?:the\s+literature|earlier\s+workers?)\b",
    re.I,
)

_NEGATED_RE = re.compile(
    r"\b(?:no|never|neither|nor|without|cannot|can't|could\s+not|"
    r"wasn't|weren't|isn't|aren't|didn't|doesn't|fail(?:ed|s)?\s+to|"
    r"unable\s+to|lack(?:s|ed|ing)?|insufficient)\b|"
    r"\bnot\b(?!\s+only\b)|\blittle\s+(?:evidence|support)\b",
    re.I,
)
_PROSPECTIVE_RE = re.compile(
    r"\b(?:will|shall)\s+(?:be\s+)?(?:perform(?:ed)?|conduct(?:ed)?|"
    r"measure(?:d)?|analy[sz](?:e|ed)|examine(?:d)?|investigate(?:d)?|"
    r"test(?:ed)?|assess(?:ed)?|determine(?:d)?|identif(?:y|ied)|"
    r"address(?:ed)?|explore(?:d)?)\b|"
    r"\b(?:future|further)\s+(?:work|study|studies|research|experiments?|"
    r"measurements?|analyses?|investigations?)\b|"
    r"\b(?:plan|plans|planned|intend|intends|intended)\s+to\b|"
    r"\b(?:remains?|remain)\s+to\s+be\b|"
    r"\b(?:should|needs?\s+to|must)\s+(?:be\s+)?(?:performed|conducted|"
    r"measured|analy[sz]ed|examined|investigated|tested|assessed|determined|"
    r"identified)|\b(?:is|are)\s+(?:still\s+)?(?:needed|required|recommended)\b",
    re.I,
)
_CONDITIONAL_RE = re.compile(
    r"\b(?:if|unless|whether|assuming|provided\s+that|depending\s+on|"
    r"in\s+case)\b|"
    r"\b(?:may|might|would|could(?!\s+not\b)|can)\s+(?:possibly\s+)?"
    r"(?:be\s+|have\s+|indicate(?:d)?|reflect(?:ed)?|represent(?:ed)?|"
    r"suggest(?:ed)?|impl(?:y|ied)|identif(?:y|ied)|determine(?:d)?|show(?:n)?|"
    r"result|arise|originate|carry|account|explain|support(?:ed)?|"
    r"correspond|belong|derive|record|preserve|yield|provide)",
    re.I,
)
_CONTRAST_RE = re.compile(
    r"\b(?:however|although|though|whereas|but|yet|nevertheless|nonetheless|"
    r"conversely|despite|in\s+contrast|by\s+contrast|on\s+the\s+other\s+hand|"
    r"unlike)\b",
    re.I,
)
_CAUSE_RE = re.compile(
    r"\b(?:because(?:\s+of)?|due\s+to|owing\s+to|as\s+a\s+result(?:\s+of)?|"
    r"caused\s+by|result(?:s|ed)?\s+from|arises?\s+from|attribut(?:e|ed|able)\s+to|"
    r"responsible\s+for|therefore|thus|hence|consequently)\b",
    re.I,
)
_INTERPRETATION_RE = re.compile(
    r"\b(?:we\s+)?(?:interpret|interprets|interpreted|infer|infers|inferred|"
    r"conclude|concludes|concluded)\b|"
    r"\b(?:is|are|was|were)\s+(?:interpreted|inferred|attributed|regarded|"
    r"considered)\s+as\b|"
    r"\b(?:consistent|inconsistent|compatible|incompatible)\s+with\b|"
    r"\b(?:suggest|suggests|suggested|imply|implies|implied|support|supports|"
    r"supported|indicate|indicates|indicated|appears?|seems?|likely|probably|"
    r"presumably|indicative\s+of|reflect|reflects|reflected|represents?|"
    r"corresponds?\s+to)\b",
    re.I,
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("discourse-role input must be an original-text string")
    return value


def _sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(text):
        if text[boundary.start() : boundary.start() + 1] == ".":
            prefix = text[max(0, boundary.start() - 20) : boundary.start() + 1]
            if _ABBREVIATION_RE.search(prefix) or _INITIALS_RE.search(prefix):
                continue
        end = boundary.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return tuple(spans)


def _execution_is_affirmed(sentence: str) -> bool:
    for clause in _CLAUSE_BOUNDARY_RE.split(sentence):
        if not clause.strip() or _CONDITIONAL_OPEN_RE.search(clause):
            continue
        execution = (
            _ACTIVE_EXECUTION_RE.search(clause)
            or _CURRENT_STUDY_EXECUTION_RE.search(clause)
            or _PASSIVE_EXECUTION_RE.search(clause)
            or _MEASUREMENTS_MADE_RE.search(clause)
        )
        if not execution:
            continue
        prefix = clause[: execution.start()]
        local = clause[max(0, execution.start() - 45) : execution.end() + 35]
        if _NEGATED_RE.search(local) or _PROSPECTIVE_RE.search(local):
            continue
        if _PRIOR_EXECUTION_SCOPE_RE.search(clause) and not re.search(
            r"\b(?:we|our|this\s+study|the\s+present\s+study)\b", clause, re.I
        ):
            continue
        if _EXECUTED_BY_CITATION_RE.search(clause):
            continue
        if re.search(r"\b(?:if|unless|whether|assuming)\b", prefix, re.I):
            continue
        return True
    return False


def _roles_for_sentence(sentence: str) -> tuple[str, ...]:
    flags = {
        "performed_execution": _execution_is_affirmed(sentence),
        "reported_result": bool(_REPORTED_RESULT_RE.search(sentence)),
        "background_citation_prior_work": bool(
            _AUTHOR_YEAR_RE.search(sentence)
            or _NUMERIC_CITATION_RE.search(sentence)
            or _PRIOR_WORK_RE.search(sentence)
        ),
        "negated": bool(_NEGATED_RE.search(sentence)),
        "prospective": bool(_PROSPECTIVE_RE.search(sentence)),
        "conditional": bool(_CONDITIONAL_RE.search(sentence)),
        "contrast": bool(_CONTRAST_RE.search(sentence)),
        "cause": bool(_CAUSE_RE.search(sentence)),
        "interpretation": bool(_INTERPRETATION_RE.search(sentence)),
    }
    return tuple(role for role in DISCOURSE_ROLE_NAMES if flags[role])


@dataclass(frozen=True)
class SentenceRoleAnnotation:
    """One sentence and its deterministic, potentially multi-label roles."""

    start: int
    end: int
    text: str
    roles: tuple[str, ...]

    @property
    def vector(self) -> tuple[int, ...]:
        return tuple(int(role in self.roles) for role in DISCOURSE_ROLE_NAMES)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "roles": list(self.roles),
            "role_vector": list(self.vector),
        }


def annotate_discourse_window(text: Any) -> tuple[SentenceRoleAnnotation, ...]:
    """Split an original-text window and attach zero or more roles per sentence."""

    source = _text(text)
    annotations = []
    for start, end in _sentence_spans(source):
        sentence = source[start:end].strip()
        if sentence:
            annotations.append(
                SentenceRoleAnnotation(start, end, sentence, _roles_for_sentence(sentence))
            )
    return tuple(annotations)


def discourse_role_counts(text: Any) -> dict[str, int]:
    """Return sentence counts for all roles, including roles with zero hits."""

    counts = dict.fromkeys(DISCOURSE_ROLE_NAMES, 0)
    for annotation in annotate_discourse_window(text):
        for role in annotation.roles:
            counts[role] += 1
    return counts


def discourse_role_features(text: Any) -> dict[str, float]:
    """Return fixed, bounded-scale numeric features for one text window."""

    annotations = annotate_discourse_window(text)
    sentence_count = len(annotations)
    counts = dict.fromkeys(DISCOURSE_ROLE_NAMES, 0)
    for annotation in annotations:
        for role in annotation.roles:
            counts[role] += 1
    features = {"discourse_sentence_count_log": math.log1p(sentence_count)}
    for role in DISCOURSE_ROLE_NAMES:
        count = counts[role]
        features[f"discourse_{role}_present"] = float(count > 0)
        features[f"discourse_{role}_sentence_count_log"] = math.log1p(count)
        features[f"discourse_{role}_sentence_ratio"] = (
            count / sentence_count if sentence_count else 0.0
        )
    return features


def discourse_role_vector(text: Any) -> tuple[float, ...]:
    """Return features in the exact order declared by DISCOURSE_FEATURE_NAMES."""

    features = discourse_role_features(text)
    return tuple(features[name] for name in DISCOURSE_FEATURE_NAMES)


__all__ = [
    "DISCOURSE_FEATURE_NAMES",
    "DISCOURSE_ROLE_NAMES",
    "SentenceRoleAnnotation",
    "annotate_discourse_window",
    "discourse_role_counts",
    "discourse_role_features",
    "discourse_role_vector",
]
