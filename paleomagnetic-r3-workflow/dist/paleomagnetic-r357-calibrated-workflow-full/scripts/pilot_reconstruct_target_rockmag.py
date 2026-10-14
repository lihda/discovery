#!/usr/bin/env python3
"""Reconstruct target-specific ROCKMAG codes from original-paper text only.

This pilot intentionally does not read PMAGRESULT, historical R3 labels, prior
predictions, or evaluation artifacts. It emits a frozen JSONL prediction file
that can be compared with historical metadata only in a later evaluation step.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable

from paper_text_retrieval import PaperTextRetriever, canonical_target_identity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs/rockmag_multitarget_pilot_20260717/identity_manifest.json"
DEFAULT_PARSED_DIR = ROOT / "data/parsed_mds"
DEFAULT_OUTPUT = ROOT / "outputs/rockmag_multitarget_pilot_20260717/predictions.blind.jsonl"

PAGE_RE = re.compile(r"(?m)^###\s+page_(\d+)\s*$")
ACCEPTED_BINDINGS = {
    "target_same_sentence",
    "target_same_paragraph",
    "target_same_table_row",
    "target_same_discourse",
    "target_same_section",
}

METHOD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("IRM", re.compile(r"\b(?:S?IRM)\b|isothermal\s+remanent\s+magneti[sz]ation", re.I)),
    ("3-axis Jrs-T", re.compile(r"three[- ]axis.{0,80}(?:IRM|Jrs)|3[- ]axis.{0,80}(?:IRM|Jrs)|\bLowrie(?:[-– ](?:Fuller\s+)?)?test\b", re.I | re.S)),
    ("SUSC-T", re.compile(r"temperature[- ]dependent\s+(?:low[- ]field\s+)?magnetic\s+susceptibility|temperature\s+dependence\s+of.{0,60}magnetic\s+susceptibility|temperature\s+depen-\s*.{0,100}dence\s+of.{0,80}magnetic\s+susceptibility|temperature\s+changes\s+of\s+susceptibility|magnetic\s+susceptibility.{0,70}(?:heating|cooling|temperature)|susceptibility.{0,30}(?:versus|vs\.?|-)\s*temperature|thermomagnetic\s+(?:curve|analysis|analyses)|Curie\s+(?:point|temperature)", re.I | re.S)),
    ("HYST", re.compile(r"\bhysteresis\b|hysteresis\s+(?:loop|parameter|propert)", re.I)),
    ("FORC", re.compile(r"\bFORC\b|first[- ]order\s+reversal\s+curve", re.I)),
    ("SEM", re.compile(r"\bSEM\b|scanning\s+electron\s+microscop", re.I)),
    ("TEM", re.compile(r"transmission\s+electron\s+microscop", re.I)),
    ("OP", re.compile(r"reflected[- ]light|optical\s+microscop|opaque[- ]mineral|petrographic|thin[- ]section", re.I)),
    ("Low-T", re.compile(r"low[- ]temperature\s+(?:IRM|magnetic|experiment|measurement)|Verwey\s+transition|Morin\s+transition", re.I)),
    ("X-RAY", re.compile(r"\bXRD\b|x[- ]ray\s+diffraction", re.I)),
    ("MICROPROBE", re.compile(r"electron\s+microprobe|\bEPMA\b|energy[- ]dispersive\s+(?:x[- ]ray\s+)?spect", re.I)),
)

CARRIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("TM", re.compile(r"\btitano[- ]?magnetite\b|\btitanomagnetite\b|\bTi[- ]magnetite\b", re.I)),
    ("MAGH", re.compile(r"\bmaghemite\b", re.I)),
    ("MAGN", re.compile(r"(?<!titano)\bmagnetite\b", re.I)),
    ("HEM", re.compile(r"\bhematite\b|\bhaematite\b|\bspecularite\b|\bspecular\s+hematite\b", re.I)),
    ("PYRR", re.compile(r"\bpyrrhotite\b", re.I)),
    ("GREIG", re.compile(r"\bgreigite\b", re.I)),
    ("GOETH", re.compile(r"\bgoethite\b", re.I)),
    ("ILM", re.compile(r"\bilmenite\b", re.I)),
)

DOMAIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PSD", re.compile(r"\bpseudo[- ]single[- ]domain\b|\bPSD\b", re.I)),
    ("SD", re.compile(r"(?<!pseudo[- ])\bsingle[- ]domain\b|\bSD\b", re.I)),
    ("MD", re.compile(r"\bmulti[- ]?domain\b|\bMD\b", re.I)),
)

DIAGNOSTIC_RE = re.compile(
    r"(?:not\s+)?saturat(?:e|ed|ion)|Curie\s+(?:point|temperature)|Hopkinson|"
    r"coerciv(?:e|ity)|unblocking\s+temperature|remanence[- ]coercivity\s+spectr|"
    r"Verwey\s+transition|Morin\s+transition|\b(?:SD|PSD|MD)\b|"
    r"single[- ]domain|multi[- ]domain|pseudo[- ]single[- ]domain",
    re.I,
)
NEGATED_CARRIER_RE = re.compile(r"\b(?:no|not|without|absent|absence\s+of)\b.{0,35}$", re.I | re.S)


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pages(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(PAGE_RE.finditer(text))
    if not matches:
        return [{"page": 1, "text": text, "source": "parsed_mds"}] if clean(text) else []
    pages: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages.append(
            {
                "page": int(match.group(1)),
                "text": text[match.end() : end].strip(),
                "source": "parsed_mds",
            }
        )
    return pages


def usable_hit(
    hit: dict[str, Any],
    anchor: dict[str, Any],
    target: str,
    binding_policy: str,
) -> bool:
    binding = clean(hit.get("target_anchor"))
    strong_pages = set(anchor.get("exact_pages") or []) | set(anchor.get("token_pages") or [])
    if binding_policy == "practical":
        return bool(
            (bool(hit.get("local_target")) and binding in ACCEPTED_BINDINGS)
            or (
                clean(anchor.get("status")) in {"exact", "token"}
                and not anchor.get("common_target")
                and binding in {"near_target", "same_target_page", "target_context"}
                and hit.get("page") in strong_pages
            )
        )
    claim_folded = canonical_target_identity(claim_text(hit))
    target_folded = canonical_target_identity(target)
    target_mentioned = bool(target_folded) and target_folded in claim_folded
    return bool(
        clean(anchor.get("status")) in {"exact", "token"}
        and not anchor.get("common_target")
        and binding in ACCEPTED_BINDINGS | {"near_target", "same_target_page", "target_context"}
        and hit.get("page") in strong_pages
        and target_mentioned
    )


def unique_hits(hits: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for hit in hits:
        key = (hit.get("page"), clean(hit.get("claim_text")), clean(hit.get("matched_text")))
        if key in seen:
            continue
        seen.add(key)
        output.append(hit)
    return output


def claim_text(hit: dict[str, Any]) -> str:
    return clean(hit.get("claim_text") or hit.get("quote") or hit.get("matched_text"))


def extract_codes(texts: Iterable[str], patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> list[str]:
    joined = "\n".join(clean(text) for text in texts if clean(text))
    return [code for code, pattern in patterns if pattern.search(joined)]


def extract_carriers(texts: Iterable[str]) -> list[str]:
    carriers: list[str] = []
    for text in texts:
        value = clean(text)
        for code, pattern in CARRIER_PATTERNS:
            for match in pattern.finditer(value):
                if NEGATED_CARRIER_RE.search(value[max(0, match.start() - 45) : match.start()]):
                    continue
                carriers.append(code)
                break
    return list(dict.fromkeys(carriers))


def format_rockmag(methods: list[str], carriers: list[str], domains: list[str]) -> str:
    carrier_codes = list(carriers)
    if len(carrier_codes) == 1 and domains:
        carrier_codes = [f"{'/'.join(domains)} {carrier_codes[0]}"]
    method_text = ",".join(methods)
    carrier_text = ",".join(carrier_codes)
    if method_text and carrier_text:
        return f"{method_text}({carrier_text})"
    return method_text or carrier_text


def build_prediction(
    identity: dict[str, str],
    retrieval: dict[str, Any],
    identity_count: int,
    binding_policy: str,
) -> dict[str, Any]:
    category_hits = retrieval.get("categories") or {}
    anchor = retrieval.get("target_anchor") or {}
    raw_hits = unique_hits(
        hit
        for category in ("method", "diagnostic", "carrier", "bridge")
        for hit in category_hits.get(category, [])
    )
    eligible_hits = [
        hit
        for hit in raw_hits
        if usable_hit(hit, anchor, identity["ROCKNAME"], binding_policy)
    ]

    method_hits = [
        hit for hit in eligible_hits if extract_codes([claim_text(hit)], METHOD_PATTERNS)
    ]
    method_pages = {int(hit["page"]) for hit in method_hits if hit.get("page") is not None}
    strong_pages = set(anchor.get("exact_pages") or []) | set(anchor.get("token_pages") or [])
    supplemental_hits = [
        hit
        for hit in raw_hits
        if hit not in eligible_hits
        and hit.get("page") in method_pages
        and hit.get("page") in strong_pages
        and clean(hit.get("target_anchor"))
        in ACCEPTED_BINDINGS | {"near_target", "same_target_page", "target_context"}
        and (extract_carriers([claim_text(hit)]) or DIAGNOSTIC_RE.search(claim_text(hit)))
    ]
    eligible_hits = unique_hits([*eligible_hits, *supplemental_hits])
    carrier_hits = [hit for hit in eligible_hits if extract_carriers([claim_text(hit)])]
    diagnostic_hits = [hit for hit in eligible_hits if DIAGNOSTIC_RE.search(claim_text(hit))]
    companion_hits = [
        hit
        for hit in unique_hits([*carrier_hits, *diagnostic_hits])
        if hit.get("page") is not None
        and any(abs(int(hit["page"]) - method_page) <= 1 for method_page in method_pages)
    ]
    companion_pages = {int(hit["page"]) for hit in companion_hits if hit.get("page") is not None}
    linked_method_hits = [
        hit
        for hit in method_hits
        if hit.get("page") is not None
        and any(abs(int(hit["page"]) - page) <= 1 for page in companion_pages)
    ]
    chain_hits = unique_hits([*linked_method_hits, *companion_hits])
    evidence_texts = [claim_text(hit) for hit in chain_hits]
    methods = extract_codes([claim_text(hit) for hit in linked_method_hits], METHOD_PATTERNS)
    carriers = extract_carriers([claim_text(hit) for hit in companion_hits])
    domains = extract_codes([claim_text(hit) for hit in companion_hits], DOMAIN_PATTERNS)
    diagnostic = any(DIAGNOSTIC_RE.search(claim_text(hit)) for hit in companion_hits)
    bridge = any(hit.get("category") == "bridge" for hit in chain_hits)
    candidate = format_rockmag(methods, carriers, domains)

    anchor_status = clean(anchor.get("status"))
    anchor_strength = clean(anchor.get("strength"))
    duplicate_identity = identity_count > 1
    enough_evidence = bool(methods and (carriers or diagnostic))

    if duplicate_identity:
        status = "ABSTAIN_IDENTITY"
    elif anchor_status not in {"exact", "token"}:
        status = "ABSTAIN_ANCHOR"
    elif enough_evidence:
        status = "PRESENT"
    elif not eligible_hits:
        status = "ABSENT"
    else:
        status = "ABSTAIN_INSUFFICIENT"

    if status == "PRESENT" and methods and carriers and anchor_strength == "strong":
        confidence = "HIGH"
    elif status == "PRESENT":
        confidence = "MEDIUM"
    elif status == "ABSENT" and anchor_strength == "strong":
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    selected_evidence = []
    for hit in chain_hits[:8]:
        selected_evidence.append(
            {
                "category": clean(hit.get("category")),
                "page": hit.get("page"),
                "target_anchor": clean(hit.get("target_anchor")),
                "matched_text": clean(hit.get("matched_text"))[:240],
                "quote": claim_text(hit)[:900],
                "source": "original_paper_text",
            }
        )

    return {
        **identity,
        "prediction_status": status,
        "predicted_rockmag": candidate if status == "PRESENT" else "",
        "candidate_rockmag": candidate,
        "direct_r3": int(status == "PRESENT"),
        "confidence": confidence,
        "methods": methods,
        "carriers": carriers,
        "domains": domains,
        "has_diagnostic_result": diagnostic,
        "has_carrier_bridge": bridge,
        "target_anchor_status": anchor_status,
        "target_anchor_strength": anchor_strength,
        "target_anchor_pages": anchor.get("pages") or [],
        "duplicate_target_identity": duplicate_identity,
        "evidence": selected_evidence,
        "provenance": "identity manifest plus original parsed paper text only",
        "binding_policy": binding_policy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--binding-policy", choices=["strict", "practical"], default="strict")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.identity_manifest.read_text(encoding="utf-8"))
    identities = manifest.get("identities")
    if not isinstance(identities, list) or not identities:
        raise SystemExit("identity manifest must contain a non-empty identities list")
    required = {"REFNO", "RESULTNO", "ROCKUNITNO", "ROCKNAME"}
    normalized: list[dict[str, str]] = []
    for row in identities:
        if not isinstance(row, dict) or set(row) != required:
            raise SystemExit("identity rows must contain only REFNO, RESULTNO, ROCKUNITNO, ROCKNAME")
        normalized.append({key: clean(row[key]) for key in required})

    by_refno: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in normalized:
        by_refno[row["REFNO"]].append(row)
    identity_counts = Counter(
        (row["REFNO"], canonical_target_identity(row["ROCKNAME"])) for row in normalized
    )

    predictions: list[dict[str, Any]] = []
    input_hashes = {str(args.identity_manifest): file_sha256(args.identity_manifest)}
    for refno, rows in by_refno.items():
        text_path = args.parsed_dir / f"{refno}.txt"
        if not text_path.exists():
            raise SystemExit(f"missing parsed paper text: {text_path}")
        input_hashes[str(text_path)] = file_sha256(text_path)
        retriever = PaperTextRetriever(
            {"pages": parse_pages(text_path), "source": {"source_quality_flags": []}}
        )
        names = list(dict.fromkeys(row["ROCKNAME"] for row in rows))
        for row in rows:
            competitors = [name for name in names if canonical_target_identity(name) != canonical_target_identity(row["ROCKNAME"])]
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
            predictions.append(
                build_prediction(
                    row,
                    retrieval,
                    identity_counts[(refno, canonical_target_identity(row["ROCKNAME"]))],
                    args.binding_policy,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    receipt_path = args.receipt or args.output.with_suffix(args.output.suffix + ".receipt.json")
    receipt = {
        "schema_version": "rockmag_multitarget_blind_prediction_receipt_v1",
        "prediction_engine": f"pilot_reconstruct_target_rockmag_v4_{args.binding_policy}",
        "binding_policy": args.binding_policy,
        "papers": len(by_refno),
        "targets": len(predictions),
        "status_counts": dict(sorted(Counter(row["prediction_status"] for row in predictions).items())),
        "forbidden_sources_not_read": [
            "data/source_metadata/PMAGRESULT.xlsx",
            "data/targets.xlsx::R3",
            "prior predictions",
            "evaluation reports",
        ],
        "input_sha256": input_hashes,
        "prediction_sha256": file_sha256(args.output),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
