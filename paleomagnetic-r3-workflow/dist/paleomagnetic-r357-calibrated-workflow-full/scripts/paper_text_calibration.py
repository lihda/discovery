#!/usr/bin/env python3
"""Frozen lexical calibration over target-bound original-paper text.

Runtime scoring uses only a target identity, a paper page-cache row, retrieved
paper descriptions, and a frozen calibration artifact.  Labels and source
metadata tables are deliberately outside this module's runtime boundary.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_STOPWORDS = {
    "the", "and", "that", "with", "from", "this", "were", "was", "are", "for",
    "have", "has", "had", "into", "their", "these", "those", "which", "than", "then",
    "also", "between", "during", "after", "before", "been", "being", "using", "used",
    "show", "shown", "result", "results", "data", "study", "figure", "table", "paper",
    "sample", "samples", "site", "sites", "rock", "rocks", "formation", "group",
    "complex", "component", "components", "magnetic", "palaeomagnetic", "paleomagnetic",
}

_GENERIC_TARGET_TOKENS = {
    "formation", "group", "member", "suite", "complex", "rocks", "rock", "beds",
    "series", "combined", "component", "components", "dykes", "dikes", "basalts",
    "granites", "volcanics", "intrusives", "sediments", "sandstones", "limestones",
}

_R5_CALIBRATION_PATTERNS = {
    "missing_paleohorizontal": r"no palaeohorizontal control|no paleohorizontal control",
    "no_structural_correction": r"without (?:any )?structural correction|not structurally corrected|no structural correction",
    "unknown_attitude": r"attitudes? (?:are )?not known|bedding (?:is|was) unknown|positions? are not known with certainty",
    "rotation": r"vertical[- ]axis rotation|block rotation|rotated clockwise|rotated counter[- ]?clockwise|clockwise rotation|counter[- ]clockwise rotation",
    "mobile_block": r"allochthonous|translated block|displaced block|mobile belt|rift block",
    "fold_timing": r"syn[- ]?fold(?:ing)?|post[- ]?fold(?:ing)?|magneti[sz]ation acquired (?:during|after) folding",
    "inclination_risk": r"inclination shallowing|flattening correction|corrected inclination",
    "structural_rescue": r"tilt[- ]corrected|tilt correction|structural correction|corrected for bedding|restored to palaeohorizontal|restored to paleohorizontal",
    "attitude_rescue": r"bedding|strikes?\s+[^.;]{0,60}dips?|dip(?:s|ping)?\s+(?:at|between|from|of)?\s*\d{1,2}",
    "field_test_rescue": r"positive (?:fold|contact|baked contact|conglomerate) test|fold test.{0,100}(?:positive|passed)|baked contact test",
    "flat_rescue": r"flat[- ]lying|sub[- ]?horizontal|nearly horizontal|almost horizontal|gently dipping",
}

_R7_CALIBRATION_PATTERNS = {
    "younger_pole": r"(?:palaeopole|paleopole|pole).{0,220}(?:younger|younger segment of the APWP)|younger.{0,220}(?:palaeopole|paleopole|pole)",
    "pole_mismatch": r"(?:palaeomagnetic|paleomagnetic) pole does not match contemporaneous poles|does not match (?:the )?(?:coeval|contemporaneous) poles",
    "remagnetization": r"remagneti[sz](?:ed|ation)|secondary (?:component|magneti[sz]ation|remanence)|overprint(?:ed|ing)?",
    "present_overprint": r"present[- ]day overprint|recent overprint|viscous remanent magneti[sz]ation|\bVRM\b",
    "reset": r"pseudotachylite|hydrothermal|metamorphic resetting|thermal resetting|reset(?:ting)? of (?:the )?magneti[sz]ation|locally remagnetized after",
    "age_ambiguity": r"older and younger ages|age spread|multi[- ]phase|multiple intrusive episodes|much younger.{0,100}overprints",
    "primary_rescue": r"primary (?:remanence|magneti[sz]ation|origin)|primary ChRM|believed to be primary|acquired (?:during|at) (?:cooling|emplacement|deposition)",
    "field_test_rescue": r"positive (?:fold|reversal|baked contact|contact|conglomerate) test|fold test.{0,100}(?:positive|passed)|reversal test.{0,100}(?:positive|passed)",
    "age_rescue": r"U[- ]Pb|baddeleyite|zircon|40Ar/39Ar|Ar[- ]Ar|radiometric age|well dated|independent age",
    "pole_match_rescue": r"(?:consistent|agrees?|matches?).{0,160}(?:contemporaneous|coeval|same age).{0,100}(?:pole|poles|APWP)",
}


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def target_tokens(target: Any) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", clean(target).casefold())
    return list(dict.fromkeys(
        token
        for token in tokens
        if token not in _GENERIC_TARGET_TOKENS and len(token) >= 4
    ))


def bernoulli_features(text: str, pattern_flags: Mapping[str, Any] | None = None) -> set[str]:
    words = [
        word
        for word in re.findall(r"[a-z][a-z0-9'-]{2,}", text.casefold())
        if word not in _STOPWORDS
    ]
    features = {f"w:{word}" for word in words}
    features.update(
        f"b:{left}_{right}"
        for left, right in zip(words, words[1:])
        if left != right
    )
    features.update(
        f"flag:{name}"
        for name, value in (pattern_flags or {}).items()
        if value and name in _R5_CALIBRATION_PATTERNS
    )
    return features


def _safe_page_number(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _excerpt(text: str, start: int, end: int, radius: int = 520) -> str:
    return clean(text[max(0, start - radius):min(len(text), end + radius)])[:1400]


def _target_windows(text: str, tokens: Sequence[str], limit: int = 4) -> list[str]:
    windows: list[tuple[int, str]] = []
    seen: set[int] = set()
    for token in tokens:
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])", re.I)
        for match in pattern.finditer(text):
            bucket = match.start() // 420
            if bucket in seen:
                continue
            seen.add(bucket)
            windows.append((match.start(), _excerpt(text, match.start(), match.end())))
            if len(windows) >= limit:
                break
        if len(windows) >= limit:
            break
    return [window for _, window in sorted(windows)]


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(value).casefold()).strip()


def _calibration_anchor_pages(
    pages: Sequence[Mapping[str, Any]],
    target: Any,
) -> tuple[list[Mapping[str, Any]], bool]:
    target_norm = _normalized(target)
    tokens = target_tokens(target)
    selected: list[Mapping[str, Any]] = []
    exact = False
    for page in pages:
        page_norm = _normalized(page.get("text"))
        hits = sum(token in page_norm for token in tokens)
        exact_here = bool(target_norm and target_norm in page_norm)
        majority = bool(tokens and hits >= max(1, math.ceil(len(tokens) * 0.6)))
        if exact_here or majority:
            selected.append(page)
            exact = exact or exact_here
    return selected, exact


def _calibration_windows(
    text: str,
    tokens: Sequence[str],
    *,
    radius: int = 430,
    limit: int = 14,
) -> list[tuple[int, str]]:
    patterns = {**_R5_CALIBRATION_PATTERNS, **_R7_CALIBRATION_PATTERNS}
    found: list[tuple[int, str]] = []
    seen: set[int] = set()
    for pattern in patterns.values():
        for match in re.finditer(pattern, text, flags=re.I | re.S):
            bucket = match.start() // 350
            if bucket in seen:
                continue
            seen.add(bucket)
            found.append(
                (
                    match.start(),
                    clean(text[max(0, match.start() - radius):match.end() + radius]),
                )
            )
    for token in tokens:
        for match in re.finditer(re.escape(token), text, flags=re.I):
            bucket = match.start() // 350
            if bucket in seen:
                continue
            seen.add(bucket)
            found.append(
                (
                    match.start(),
                    clean(text[max(0, match.start() - radius):match.end() + radius]),
                )
            )
            if len(found) >= limit * 2:
                break
    found.sort(key=lambda item: item[0])
    return [(position, value[:1100]) for position, value in found[:limit]]


def _mask_target_terms(text: str, target: Any) -> str:
    masked = text
    for token in sorted(target_tokens(target), key=len, reverse=True):
        masked = re.sub(
            r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])",
            "TARGETTERM",
            masked,
            flags=re.I,
        )
    return masked


def build_calibration_input(cache_row: Mapping[str, Any], target: Any) -> dict[str, Any]:
    """Reproduce the held-out target-window feature construction used for R5."""

    raw_pages = cache_row.get("pages") or []
    pages = [page for page in raw_pages if isinstance(page, Mapping)]
    if not pages:
        return {
            "text": "source unavailable",
            "pattern_flags": {"source_missing": 1},
            "passages": [],
            "source_available": False,
            "exact_target_anchor": False,
            "target_page_count": 0,
        }
    anchored, exact = _calibration_anchor_pages(pages, target)
    selected = anchored[:5] if anchored else pages[:2]
    tokens = target_tokens(target)
    chunks = [f"target {clean(target)}", clean(pages[0].get("text"))[:1200]]
    passages: list[dict[str, Any]] = []
    for fallback, page in enumerate(selected, start=1):
        page_number = _safe_page_number(page.get("page"), fallback)
        window_limit = 10 if anchored else 7
        for _, window in _calibration_windows(
            str(page.get("text") or ""),
            tokens,
            limit=window_limit,
        ):
            chunks.append(window)
            passages.append(
                {
                    "text": window,
                    "page": page_number,
                    "section": "Target-centered calibration window",
                    "kind": "calibration_window",
                    "source": str(page.get("source") or "original_paper"),
                }
            )
    full_document = "\n".join(dict.fromkeys(chunk for chunk in chunks if chunk))[:14000]
    pattern_flags = {
        name: int(re.search(pattern, full_document, flags=re.I | re.S) is not None)
        for name, pattern in {**_R5_CALIBRATION_PATTERNS, **_R7_CALIBRATION_PATTERNS}.items()
    }
    pattern_flags.update(
        {
            "exact_target_anchor": int(exact),
            "target_page_count_ge2": int(len(anchored) >= 2),
            "source_missing": 0,
        }
    )
    feature_text = "\n".join(full_document.splitlines()[2:]).strip()
    feature_text = _mask_target_terms(feature_text, target) or "source unavailable"
    return {
        "text": feature_text,
        "pattern_flags": pattern_flags,
        "passages": passages,
        "source_available": True,
        "exact_target_anchor": exact,
        "target_page_count": len(anchored),
    }


def build_target_document(
    cache_row: Mapping[str, Any],
    target: Any,
    retrieval: Mapping[str, Any],
    *,
    limit: int = 18000,
) -> dict[str, Any]:
    """Build a target-centered document and retain passage provenance."""

    raw_pages = cache_row.get("pages") or []
    pages = [page for page in raw_pages if isinstance(page, Mapping)]
    anchor = retrieval.get("target_anchor") or {}
    anchor_pages = {
        int(page)
        for page in (anchor.get("pages") or [])
        if str(page).strip().lstrip("-").isdigit()
    }
    tokens = target_tokens(target)
    calibration_input = build_calibration_input(cache_row, target)
    passages: list[dict[str, Any]] = [
        {
            "text": f"target {clean(target)}",
            "page": None,
            "section": "Identity manifest",
            "kind": "identity",
            "source": "identity_manifest",
        }
    ]

    if pages:
        first = pages[0]
        passages.append(
            {
                "text": clean(str(first.get("text") or ""))[:1400],
                "page": _safe_page_number(first.get("page"), 1),
                "section": "Paper opening",
                "kind": "paper_opening",
                "source": str(first.get("source") or "original_paper"),
            }
        )

    for hit in retrieval.get("hits") or []:
        if not isinstance(hit, Mapping) or hit.get("reference_noise"):
            continue
        quote = clean(hit.get("quote"))
        if not quote:
            continue
        passages.append(
            {
                "text": quote,
                "page": hit.get("page"),
                "section": clean(hit.get("section")) or "Unknown",
                "kind": f"retrieved_{clean(hit.get('category')) or 'evidence'}",
                "source": str(hit.get("source") or "original_paper"),
                "target_binding": hit.get("target_anchor"),
                "target_relevance": hit.get("target_relevance"),
            }
        )

    for offset, page in enumerate(pages, start=1):
        page_number = _safe_page_number(page.get("page"), offset)
        if page_number not in anchor_pages:
            continue
        page_text = str(page.get("text") or "")
        for window in _target_windows(page_text, tokens):
            passages.append(
                {
                    "text": window,
                    "page": page_number,
                    "section": "Target context",
                    "kind": "target_window",
                    "source": str(page.get("source") or "original_paper"),
                }
            )

    passages.extend(calibration_input["passages"])

    unique: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    used = 0
    for passage in passages:
        text = clean(passage.get("text"))
        key = text.casefold()
        if not text or key in seen_text:
            continue
        remaining = limit - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]
        copy = dict(passage)
        copy["text"] = text
        unique.append(copy)
        seen_text.add(key)
        used += len(text) + 1

    return {
        "target": clean(target),
        "text": "\n".join(passage["text"] for passage in unique),
        "calibration_text": calibration_input["text"],
        "calibration_pattern_flags": calibration_input["pattern_flags"],
        "passages": unique,
        "source_available": bool(pages),
        "target_anchor": dict(anchor),
        "provenance": "identity manifest plus target-bound original-paper passages",
    }


def _feature_terms(feature: str) -> tuple[str, ...]:
    if feature.startswith("w:"):
        terms = (feature[2:],)
        return tuple(term for term in terms if term != "targetterm")
    if feature.startswith("b:"):
        return tuple(
            part
            for part in feature[2:].split("_")
            if part and part != "targetterm"
        )
    return ()


def _passage_for_feature(
    feature: str,
    passages: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    terms = _feature_terms(feature)
    if not terms:
        return None
    fallback: dict[str, Any] | None = None
    for passage in passages:
        sample = clean(passage.get("text"))
        folded = sample.casefold()
        if not all(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", folded) for term in terms):
            continue
        candidate = {
            "quote": sample[:600],
            "section_or_page": (
                f"page {passage.get('page')} | {clean(passage.get('section'))}"
                if passage.get("page") not in {None, ""}
                else clean(passage.get("section")) or "identity manifest"
            ),
            "source": passage.get("source"),
            "passage_kind": passage.get("kind"),
        }
        if passage.get("kind") != "identity":
            return candidate
        fallback = candidate
    return fallback


class FrozenPaperTextModel:
    """Load and apply a frozen, auditable paper-text calibration artifact."""

    def __init__(self, path: Path, indicator: str = "R5") -> None:
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        models = data.get("models") or {}
        name = str(indicator).strip().upper()
        if name not in models:
            raise ValueError(f"{self.path} does not contain a {name} model")
        model = models[name]
        self.indicator = name
        self.version = str(data.get("version") or self.path.stem)
        self.scoring = str(model.get("scoring") or "tfidf_linear")
        self.bias = float(model["bias"])
        self.threshold = float(model["threshold"])
        if self.scoring == "bernoulli_presence":
            self.parameters = {
                str(feature): (1.0, float(weight))
                for feature, weight in model.get("feature_parameters") or []
            }
        else:
            self.parameters = {
                str(feature): (float(idf), float(weight))
                for feature, idf, weight in model.get("feature_parameters") or []
            }
        self.audit = data.get("audit") or {}
        if not self.parameters:
            raise ValueError(f"{self.path} contains no {name} feature parameters")

    def predict(self, document: Mapping[str, Any], contribution_limit: int = 3) -> dict[str, Any]:
        if self.scoring == "bernoulli_presence":
            present = bernoulli_features(
                str(document.get("calibration_text") or "source unavailable"),
                document.get("calibration_pattern_flags") or {},
            )
            contributions = [
                (feature, self.parameters[feature][1])
                for feature in present
                if feature in self.parameters
            ]
        else:
            counts: dict[str, int] = {}
            for feature in bernoulli_features(str(document.get("text") or "")):
                counts[feature] = counts.get(feature, 0) + 1
            weighted: list[tuple[str, float, float]] = []
            squared_norm = 0.0
            for feature, count in counts.items():
                parameter = self.parameters.get(feature)
                if parameter is None:
                    continue
                idf, coefficient = parameter
                value = (1.0 + math.log(count)) * idf
                squared_norm += value * value
                weighted.append((feature, value, coefficient))
            norm = math.sqrt(squared_norm) or 1.0
            contributions = [
                (feature, coefficient * value / norm)
                for feature, value, coefficient in weighted
            ]
        risk_score = self.bias + sum(value for _, value in contributions)
        candidate_score = 0 if risk_score >= self.threshold else 1
        passages = document.get("passages") or []

        def explain(items: list[tuple[str, float]]) -> list[dict[str, Any]]:
            result = []
            for feature, contribution in items[:contribution_limit]:
                evidence = _passage_for_feature(feature, passages)
                result.append(
                    {
                        "feature": feature,
                        "contribution": round(contribution, 6),
                        "evidence": evidence,
                    }
                )
            return result

        risk = sorted(
            ((feature, value) for feature, value in contributions if value > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        support = sorted(
            ((feature, value) for feature, value in contributions if value < 0),
            key=lambda item: item[1],
        )
        return {
            "candidate_score": candidate_score,
            "risk_score": round(risk_score, 6),
            "threshold": self.threshold,
            "margin": round(abs(risk_score - self.threshold), 6),
            "model_version": self.version,
            "model_path": str(self.path),
            "source_available": bool(document.get("source_available")),
            "target_anchor": document.get("target_anchor") or {},
            "risk_contributions": explain(risk),
            "support_contributions": explain(support),
            "runtime_scoring_sources": [
                "identity manifest",
                "target-bound original paper text",
                "frozen lexical calibration parameters",
            ],
        }
