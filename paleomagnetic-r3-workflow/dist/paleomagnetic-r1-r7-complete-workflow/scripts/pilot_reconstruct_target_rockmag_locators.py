#!/usr/bin/env python3
"""Reconstruct ROCKMAG using a frozen locator-only identity manifest and paper text."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable

from paper_text_retrieval import PaperTextRetriever, canonical_target_identity
from pilot_reconstruct_target_rockmag import (
    CARRIER_PATTERNS,
    DIAGNOSTIC_RE,
    DOMAIN_PATTERNS,
    METHOD_PATTERNS,
    build_prediction,
    claim_text,
    clean,
    extract_carriers,
    extract_codes,
    file_sha256,
    format_rockmag,
    parse_pages,
    unique_hits,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs/rockmag_multitarget_pilot_v2_locators_20260717/identity_manifest.locators.json"
DEFAULT_PARSED_DIR = ROOT / "data/parsed_mds"
DEFAULT_OUTPUT = ROOT / "outputs/rockmag_multitarget_pilot_v2_locators_20260717/predictions.locator.v22.blind.jsonl"

IDENTITY_FIELDS = (
    "REFNO",
    "RESULTNO",
    "ROCKUNITNO",
    "ROCKNAME",
    "RLAT",
    "RLONG",
    "COMPONENT",
    "SLAT",
    "SLONG",
    "DEC",
    "INC",
    "PLAT",
    "PLONG",
    "PTYPE",
)
FORBIDDEN_SOURCES = (
    "PMAGRESULT.ROCKMAG",
    "PMAGRESULT.TESTS",
    "data/targets.xlsx::R3",
    "PMAGRESULT.LABDETAILS",
    "PMAGRESULT.COMMENTS",
    "prior predictions",
    "evaluation reports",
)


def numeric(value: Any) -> float | None:
    try:
        return float(clean(value))
    except (TypeError, ValueError):
        return None


def normalized_component(value: Any) -> str:
    text = canonical_target_identity(value)
    text = re.sub(r"\bcomponent\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rounded_locator(value: Any) -> str:
    number = numeric(value)
    return "" if number is None else f"{number:.3f}"


def metadata_key(identity: dict[str, str]) -> tuple[str, ...]:
    return (
        normalized_component(identity.get("COMPONENT")),
        rounded_locator(identity.get("DEC")),
        rounded_locator(identity.get("INC")),
        rounded_locator(identity.get("PLAT")),
        rounded_locator(identity.get("PLONG")),
        clean(identity.get("PTYPE")).upper(),
    )


def component_pattern(value: Any) -> re.Pattern[str] | None:
    raw = clean(value)
    label = normalized_component(raw)
    if not label:
        return None
    if label == "combined result":
        return re.compile(r"\bcombined\s+(?:result|mean|categor(?:y|ies)|class(?:es)?)\b", re.I)
    compact = re.sub(r"[^a-z0-9]+", "", label)
    if re.fullmatch(r"[a-z]\d?", compact):
        escaped = re.escape(compact)
        return re.compile(
            rf"\b(?:component\s*\(?\s*{escaped}\s*\)?|{escaped}\s+component|"
            rf"components?\b.{{0,32}}\b{escaped})\b",
            re.I,
        )
    return re.compile(r"\b" + r"\s+".join(re.escape(part) for part in label.split()) + r"\b", re.I)


def number_pattern(value: Any, *, cyclic: bool = False) -> str | None:
    number = numeric(value)
    if number is None:
        return None
    if cyclic:
        number %= 360.0
    sign = r"[-\u2212\u2013]\s*" if number < 0 else r"\+?\s*"
    absolute = abs(number)
    if abs(absolute - round(absolute)) <= 0.55:
        body = rf"{int(round(absolute))}(?:\.0+)?"
    else:
        body = rf"{absolute:.1f}".replace(".", r"\.") + r"\d*"
    return sign + body


def pair_patterns(first: Any, second: Any, *, cyclic_first: bool = False) -> list[re.Pattern[str]]:
    left = number_pattern(first, cyclic=cyclic_first)
    right = number_pattern(second)
    if not left or not right:
        return []
    separator = r".{0,140}?"
    return [
        re.compile(rf"(?<![\d.]){left}(?![\d.]){separator}(?<![\d.]){right}(?![\d.])", re.I | re.S),
    ]


def pole_patterns(latitude: Any, longitude: Any) -> list[re.Pattern[str]]:
    lat = numeric(latitude)
    lon = numeric(longitude)
    if lat is None or lon is None:
        return []
    candidates = [(lat, lon % 360.0), (-lat, (lon + 180.0) % 360.0)]
    patterns: list[re.Pattern[str]] = []
    for candidate_lat, candidate_lon in candidates:
        lat_pattern = number_pattern(candidate_lat)
        lon_pattern = number_pattern(candidate_lon, cyclic=True)
        if not lat_pattern or not lon_pattern:
            continue
        separator = r".{0,140}?"
        patterns.extend(
            [
                re.compile(rf"(?<![\d.]){lat_pattern}(?![\d.]){separator}(?<![\d.]){lon_pattern}(?![\d.])", re.I | re.S),
                re.compile(rf"(?<![\d.]){lon_pattern}(?![\d.]){separator}(?<![\d.]){lat_pattern}(?![\d.])", re.I | re.S),
            ]
        )
    return patterns


def excerpt(text: str, start: int, end: int, radius: int = 700) -> str:
    return clean(text[max(0, start - radius) : min(len(text), end + radius)])[:1800]


def pattern_hits(
    pages: list[dict[str, Any]],
    patterns: Iterable[tuple[str, re.Pattern[str]]],
    allowed_pages: set[int] | None = None,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for page in pages:
        page_number = int(page["page"])
        if allowed_pages is not None and page_number not in allowed_pages:
            continue
        page_text = str(page.get("text") or "")
        for locator_type, pattern in patterns:
            for match in pattern.finditer(page_text):
                quote = excerpt(page_text, match.start(), match.end())
                key = (locator_type, page_number, quote)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    {
                        "locator_type": locator_type,
                        "page": page_number,
                        "matched_text": clean(match.group(0))[:260],
                        "quote": quote,
                    }
                )
    return hits


def locator_resolution(
    identity: dict[str, str],
    siblings: list[dict[str, str]],
    pages: list[dict[str, Any]],
    target_anchor: dict[str, Any],
) -> dict[str, Any]:
    metadata_unique = len({metadata_key(row) for row in siblings}) == len(siblings)
    anchor_pages = set(target_anchor.get("exact_pages") or []) | set(target_anchor.get("token_pages") or [])
    adjacent_pages = {
        page
        for anchor_page in anchor_pages
        for page in (int(anchor_page) - 1, int(anchor_page), int(anchor_page) + 1)
        if page > 0
    }
    patterns: list[tuple[str, re.Pattern[str]]] = []
    component = component_pattern(identity.get("COMPONENT"))
    component_unique = len({normalized_component(row.get("COMPONENT")) for row in siblings}) == len(siblings)
    if component is not None and component_unique:
        patterns.append(("component", component))
    for pattern in pair_patterns(identity.get("DEC"), identity.get("INC"), cyclic_first=True):
        patterns.append(("direction", pattern))
    for pattern in pole_patterns(identity.get("PLAT"), identity.get("PLONG")):
        patterns.append(("pole", pattern))
    paper_hits = pattern_hits(pages, patterns, adjacent_pages or None)
    if paper_hits:
        status = "PAPER_ROW_MATCH"
    elif metadata_unique:
        status = "METADATA_UNIQUE"
    else:
        status = "AMBIGUOUS"
    return {
        "status": status,
        "metadata_unique": metadata_unique,
        "component_unique": component_unique,
        "anchor_pages": sorted(anchor_pages),
        "locator_pages": sorted({int(hit["page"]) for hit in paper_hits}),
        "matches": paper_hits[:10],
    }


def manual_component_evidence(
    identity: dict[str, str],
    siblings: list[dict[str, str]],
    paper_rows: list[dict[str, str]],
    pages: list[dict[str, Any]],
    locator: dict[str, Any],
) -> list[dict[str, Any]]:
    pattern = component_pattern(identity.get("COMPONENT"))
    component_unique = len({normalized_component(row.get("COMPONENT")) for row in siblings}) == len(siblings)
    if pattern is None or not component_unique:
        return []
    component_value = normalized_component(identity.get("COMPONENT"))
    component_owner_names = {
        canonical_target_identity(row.get("ROCKNAME"))
        for row in paper_rows
        if normalized_component(row.get("COMPONENT")) == component_value
    }
    component_unique_in_paper = len(component_owner_names) == 1
    target_folded = canonical_target_identity(identity.get("ROCKNAME"))
    anchor_pages = set(locator.get("anchor_pages") or [])
    allowed_pages = {
        page
        for anchor_page in anchor_pages
        for page in (int(anchor_page) - 1, int(anchor_page), int(anchor_page) + 1)
        if page > 0
    }
    contexts = pattern_hits(pages, [("component", pattern)], allowed_pages or None)
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for context in contexts:
        quote = context["quote"]
        if not component_unique_in_paper and target_folded not in canonical_target_identity(quote):
            continue
        categories: list[str] = []
        if extract_codes([quote], METHOD_PATTERNS):
            categories.append("method")
        if extract_carriers([quote]):
            categories.append("carrier")
        if DIAGNOSTIC_RE.search(quote):
            categories.append("diagnostic")
        if re.search(r"\b(?:carried\s+by|carrier|carries)\b", quote, re.I):
            categories.append("bridge")
        for category in categories:
            key = (category, int(context["page"]), quote)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "category": category,
                    "page": int(context["page"]),
                    "target_anchor": "locator_component_context",
                    "matched_text": context["matched_text"],
                    "quote": quote[:900],
                    "source": "original_paper_text",
                }
            )
    return evidence


def merge_codes(*groups: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for group in groups for item in group if item))


def build_locator_prediction(
    identity: dict[str, str],
    retrieval: dict[str, Any],
    siblings: list[dict[str, str]],
    paper_rows: list[dict[str, str]],
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = build_prediction(identity, retrieval, len(siblings), "practical")
    locator = locator_resolution(identity, siblings, pages, retrieval.get("target_anchor") or {})
    anchor_status = clean((retrieval.get("target_anchor") or {}).get("status"))
    manual_scope_enabled = len(siblings) > 1 or anchor_status not in {"exact", "token"}
    manual_evidence = (
        manual_component_evidence(identity, siblings, paper_rows, pages, locator)
        if manual_scope_enabled
        else []
    )
    manual_texts = [clean(item.get("quote")) for item in manual_evidence]
    manual_methods = extract_codes(manual_texts, METHOD_PATTERNS)
    manual_carriers = extract_carriers(manual_texts)
    manual_domains = extract_codes(manual_texts, DOMAIN_PATTERNS)
    methods = merge_codes(baseline.get("methods") or [], manual_methods)
    carriers = merge_codes(baseline.get("carriers") or [], manual_carriers)
    domains = merge_codes(baseline.get("domains") or [], manual_domains)
    diagnostic = bool(baseline.get("has_diagnostic_result") or any(DIAGNOSTIC_RE.search(text) for text in manual_texts))
    candidate = format_rockmag(methods, carriers, domains)
    enough_evidence = bool(methods and (carriers or diagnostic))
    duplicate = len(siblings) > 1

    if duplicate and locator["status"] == "AMBIGUOUS":
        status = "ABSTAIN_IDENTITY"
    elif enough_evidence and (anchor_status in {"exact", "token"} or locator["status"] == "PAPER_ROW_MATCH"):
        status = "PRESENT"
    elif anchor_status not in {"exact", "token"} and locator["status"] != "PAPER_ROW_MATCH":
        status = "ABSTAIN_ANCHOR"
    elif baseline.get("prediction_status") == "ABSENT" and not manual_evidence:
        status = "ABSENT"
    else:
        status = "ABSTAIN_INSUFFICIENT"

    if status == "PRESENT" and locator["status"] == "PAPER_ROW_MATCH" and carriers:
        confidence = "HIGH"
    elif status == "PRESENT":
        confidence = "MEDIUM"
    elif status == "ABSENT":
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    locator_evidence = [
        {
            "category": f"locator_{hit['locator_type']}",
            "page": hit["page"],
            "target_anchor": "result_locator",
            "matched_text": hit["matched_text"],
            "quote": hit["quote"][:900],
            "source": "original_paper_text",
        }
        for hit in locator.get("matches") or []
    ]
    evidence = unique_hits([*(baseline.get("evidence") or []), *manual_evidence, *locator_evidence])[:14]
    return {
        **baseline,
        "prediction_status": status,
        "predicted_rockmag": candidate if status == "PRESENT" else "",
        "candidate_rockmag": candidate,
        "direct_r3": int(status == "PRESENT"),
        "confidence": confidence,
        "methods": methods,
        "carriers": carriers,
        "domains": domains,
        "has_diagnostic_result": diagnostic,
        "evidence": evidence,
        "locator_resolution": locator["status"],
        "locator_metadata_unique": locator["metadata_unique"],
        "locator_component_unique": locator["component_unique"],
        "locator_pages": locator["locator_pages"],
        "locator_match_types": sorted({hit["locator_type"] for hit in locator.get("matches") or []}),
        "manual_component_scope_enabled": manual_scope_enabled,
        "provenance": "locator-only identity manifest plus original parsed paper text",
        "binding_policy": "practical+result-locators-v2",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.identity_manifest.read_text(encoding="utf-8"))
    identities = manifest.get("identities")
    if not isinstance(identities, list) or not identities:
        raise SystemExit("identity manifest must contain identities")
    normalized: list[dict[str, str]] = []
    for row in identities:
        if not isinstance(row, dict) or tuple(row.keys()) != IDENTITY_FIELDS:
            raise SystemExit("identity manifest fields or order do not match the v2 allowlist")
        normalized.append({field: clean(row.get(field)) for field in IDENTITY_FIELDS})

    by_refno: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in normalized:
        by_refno[row["REFNO"]].append(row)
    input_hashes = {str(args.identity_manifest): file_sha256(args.identity_manifest)}
    predictions: list[dict[str, Any]] = []
    for refno, rows in by_refno.items():
        text_path = args.parsed_dir / f"{refno}.txt"
        if not text_path.exists():
            raise SystemExit(f"missing parsed paper text: {text_path}")
        input_hashes[str(text_path)] = file_sha256(text_path)
        pages = parse_pages(text_path)
        retriever = PaperTextRetriever({"pages": pages, "source": {"source_quality_flags": []}})
        names = list(dict.fromkeys(row["ROCKNAME"] for row in rows))
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            groups[canonical_target_identity(row["ROCKNAME"])].append(row)
        for row in rows:
            competitors = [
                name
                for name in names
                if canonical_target_identity(name) != canonical_target_identity(row["ROCKNAME"])
            ]
            retrieval = retriever.retrieve(
                row["ROCKNAME"],
                multi_target=True,
                indicator="R3",
                competing_targets=competitors,
                allow_structured_scope=True,
                allow_discourse_scope=True,
                limit=40,
                per_category_limit=12,
            )
            siblings = groups[canonical_target_identity(row["ROCKNAME"])]
            predictions.append(build_locator_prediction(row, retrieval, siblings, rows, pages))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    receipt_path = args.receipt or args.output.with_suffix(args.output.suffix + ".receipt.json")
    receipt = {
        "schema_version": "rockmag_multitarget_locator_prediction_receipt_v2",
        "prediction_engine": "pilot_reconstruct_target_rockmag_locators_v2.2",
        "papers": len(by_refno),
        "targets": len(predictions),
        "status_counts": dict(sorted(Counter(row["prediction_status"] for row in predictions).items())),
        "locator_resolution_counts": dict(sorted(Counter(row["locator_resolution"] for row in predictions).items())),
        "forbidden_sources_not_read": list(FORBIDDEN_SOURCES),
        "input_sha256": input_hashes,
        "prediction_sha256": file_sha256(args.output),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
