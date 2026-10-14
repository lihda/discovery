#!/usr/bin/env python3
"""Build label-free R3 structured evidence JSONL from fresh evidence packs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXTRACTOR = "Codex chat model (structured evidence extraction)"

DIAGNOSTIC_ROCKMAG_PATTERN = (
    r"\bIRM\b|SIRM|3[- ]axis|Jrs[-/]?T|Js[- ]?T|SUSC[- ]?T|SUSC\(T\)|X/T|"
    r"HYST|FORC|\bOP\b|SEM|TEM|EDS|microprobe|Lowrie|CT\(|Curie|Hcr|Hc,|"
    r"Jrs/Js|Hcr/Hc|Low temp|low temperature|thermomagnetic"
)
CARRIER_ROCKMAG_PATTERN = (
    r"MAGN|HEM|MAGH|TM|ILM|PYRR|goeth|magnetite|hematite|haematite|"
    r"titanomag|greig|pyrrhot"
)
INDEPENDENT_ROCKMAG_METHOD_PATTERN = (
    r"\bIRM\b|SIRM|3[- ]axis|Jrs[-/]?T|Jrs\(?T\)?|\bJrs\b|"
    r"Js[- ]?T|Js\(?T\)?|\bJs\b|\bJn\b|"
    r"SUSC[- ]?T|SUSC\(T\)|HYST|FORC|Lowrie|thermomagnetic|"
    r"\bOP\b|SEM|TEM|EDS|microprobe|petrograph|thin sect|"
    r"mineralogical|miner\.analysis|magnetic mineralogy"
)
TUB_ONLY_FERRIMAGNETIC_CARRIER_PATTERN = r"\b(MAGN|MAGH|TM|PYRR|ILM)\b"
BARE_CARRIER_CODE_PATTERN = (
    r"^\s*(MAGN|HEM|MAGH|TM|ILM|PYRR)"
    r"(\s*[,;/]\s*(MAGN|HEM|MAGH|TM|ILM|PYRR))*\s*$"
)
SECONDARY_COMPONENT_CONTEXT_PATTERN = (
    r"post[- ]folding|secondary|overprint|remagneti[sz]|post[- ]tectonic"
)
OP_ALTERATION_ONLY_PATTERN = (
    r"\bOP\([^)]*(?:alter|oxid|exsolv|scarcity|complete range|low-Ti|TM\+ILM)[^)]*\)"
)
SOURCE_NEGATIVE_REMANENCE_CONTEXT_PATTERN = (
    r"most likely (?:a )?remagneti[sz]ation|not primary|"
    r"spurious magneti[sz]ation|failed to isolate|difficult to isolate|"
    r"may be younger|local tectonics|post[- ]tectonic remagneti[sz]ation|"
    r"post[- ]tectonic|younger remagneti[sz]ation|"
    r"component is related to formation of authigenic minerals due to rock remagnetiz"
)
SOURCE_LAB_ARTIFACT_REMAGNETIZATION_PATTERN = r"remagnetization due to the magnetic saw"
SOURCE_LOCAL_REMAGNETIZATION_PATTERN = r"\blocally remagnetized\b"
SOURCE_AUTHIGENIC_REMAGNETIZATION_PATTERN = r"authigenic minerals due to rock remagneti[sz]"
OP_HYST_ROCKMAG_PATTERN = r"(?=.*\bOP\b)(?=.*\bHYST\b)"
REVIEW_METHOD_PATTERN = (
    r"\bIRM\b|\bSIRM\b|3[- ]axis|Jrs[-/]?T|Js[- ]?T|SUSC[- ]?T|SUSC\(T\)|"
    r"\bhysteresis\b|\bHYST\b|\bFORC\b|Lowrie|thermomagnetic|Curie|CT\(|"
    r"\bSEM\b|\bTEM\b|\bEPMA\b|microprobe|microscop|petrograph|"
    r"reflected[- ]light|thin section|ore microscop"
)
REVIEW_CARRIER_PATTERN = (
    r"\bmagnetite\b|\btitanomagnetite\b|titano[- ]magnetite|\bhematite\b|"
    r"\bhaematite\b|\bmaghemite\b|\bgoethite\b|\bpyrrhotite\b|\bgreigite\b|"
    r"Fe[- ]Ti oxide|\bilmenite\b|\bferrimagnetic\b|\bantiferromagnetic\b|"
    r"\bMAGN\b|\bHEM\b|\bMAGH\b|\bTM\b|\bILM\b|\bPYRR\b"
)
REVIEW_SIGNIFICANCE_PATTERN = (
    r"carr(?:y|ies|ied|ier|iers).{0,180}(ChRM|characteristic|remanence|component|magneti[sz]ation)|"
    r"(ChRM|characteristic|remanence|component|magneti[sz]ation).{0,180}carr(?:y|ies|ied|ier|iers)|"
    r"(primary|secondary|overprint|remagneti[sz]|CRM|TRM|DRM|VRM|detrital|chemical|authigenic|alteration).{0,220}"
    r"(magnetite|hematite|haematite|carrier|component|remanence)|"
    r"(magnetite|hematite|haematite|carrier|component).{0,220}"
    r"(primary|secondary|overprint|remagneti[sz]|CRM|TRM|DRM|VRM|stability|stable|reliab)"
)


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def field(status: str, quote: str, reason: str, **extra: str) -> dict[str, str]:
    return {"status": status, "quote": quote, "reason": reason, **extra}


def risk_dict(**values: bool) -> dict[str, bool]:
    base = {
        "directional_substitution_risk": False,
        "multi_target_transfer_risk": False,
        "demagnetization_only_risk": False,
        "source_quality_risk": False,
        "evaluator_inference_risk": False,
        "separate_evidence_family_risk": False,
    }
    base.update(values)
    return base


def pack_text(pack: dict[str, Any], categories: list[str]) -> str:
    parts = []
    evidence = pack.get("candidate_evidence", {})
    for category in categories:
        for item in evidence.get(category, []) or []:
            parts.append(str(item.get("matched_text", "")))
            parts.append(str(item.get("excerpt", "")))
    return clean(" ".join(parts))


def first_quote(pack: dict[str, Any], category: str) -> str:
    items = pack.get("candidate_evidence", {}).get(category, []) or []
    for item in items:
        text = clean(item.get("excerpt") or item.get("matched_text") or "")
        if text:
            return text[:950]
    return ""


def first_item(pack: dict[str, Any], category: str) -> dict[str, Any] | None:
    items = pack.get("candidate_evidence", {}).get(category, []) or []
    return items[0] if items else None


def has(text: str, pattern: str) -> bool:
    return re.search(pattern, text or "", re.I | re.S) is not None


def tub_only_ferrimagnetic_inference(rockmag: str) -> bool:
    """Treat bare ferrimagnetic Tub inference as weak without an independent method.

    Tub/unblocking-temperature notes can support R3 when paired with a real
    rock-magnetic or mineralogical pathway, but a terse "MAGN (Tub)"-style
    code is only a carrier inference from demagnetization behavior.
    """

    return bool(
        has(rockmag, r"\bTub\b")
        and has(rockmag, TUB_ONLY_FERRIMAGNETIC_CARRIER_PATTERN)
        and not has(rockmag, INDEPENDENT_ROCKMAG_METHOD_PATTERN)
    )


def bare_secondary_carrier_code(rockmag: str, comments: str, component: str) -> bool:
    """Reject a bare mineral code when the component context is secondary."""

    return bool(
        has(rockmag, BARE_CARRIER_CODE_PATTERN)
        and has(f"{comments} {component}", SECONDARY_COMPONENT_CONTEXT_PATTERN)
    )


def review_hit_text(pack: dict[str, Any]) -> str:
    hits = source_text_review_hits(pack).get("hits", [])
    return clean(
        " ".join(
            f"{hit.get('quote', '')} {hit.get('matched_text', '')}"
            for hit in hits
            if isinstance(hit, dict)
        )
    )


def op_alteration_with_negative_source_context(pack: dict[str, Any], rockmag: str) -> bool:
    """Treat petrographic alteration shorthand as weak when source context is negative."""

    return bool(
        has(rockmag, OP_ALTERATION_ONLY_PATTERN)
        and has(review_hit_text(pack), SOURCE_NEGATIVE_REMANENCE_CONTEXT_PATTERN)
    )


def source_lab_artifact_remagnetization(pack: dict[str, Any]) -> bool:
    """Reject carrier evidence when the source context points to preparation artifacts."""

    return has(review_hit_text(pack), SOURCE_LAB_ARTIFACT_REMAGNETIZATION_PATTERN)


def op_hyst_with_local_remagnetization(pack: dict[str, Any], rockmag: str) -> bool:
    """Treat OP+HYST carrier evidence as weak when the source flags local remagnetization."""

    return bool(
        has(rockmag, OP_HYST_ROCKMAG_PATTERN)
        and has(review_hit_text(pack), SOURCE_LOCAL_REMAGNETIZATION_PATTERN)
    )


def op_without_carrier_hit_authigenic_remagnetization(pack: dict[str, Any], rockmag: str) -> bool:
    """Reject OP shorthand when source snippets only support authigenic remagnetization."""

    review = source_text_review_hits(pack)
    counts = review.get("counts", {})
    return bool(
        has(rockmag, r"\bOP\(")
        and int(counts.get("magnetic_carrier", 0) or 0) == 0
        and has(review_hit_text(pack), SOURCE_AUTHIGENIC_REMAGNETIZATION_PATTERN)
    )


def field_text(*fields: dict[str, str]) -> str:
    return clean(" ".join(field.get("quote", "") + " " + field.get("reason", "") for field in fields))


def explicit_carrier_remanence_language(text: str) -> bool:
    return has(
        text,
        r"carr(?:y|ies|ied|ier|iers).{0,140}(ChRM|characteristic|remanence|component|magneti[sz]ation)|"
        r"(ChRM|characteristic|remanence|component|magneti[sz]ation).{0,140}carr(?:y|ies|ied|ier|iers)|"
        r"held by|holds? the (?:ChRM|characteristic|remanence|component)|resides? in|"
        r"(CRM|TRM|DRM|VRM).{0,140}(magnetite|hematite|haematite|carrier|component)|"
        r"(magnetite|hematite|haematite|carrier|component).{0,140}(CRM|TRM|DRM|VRM|overprint|remagnet)",
    )


def directional_validation_language(text: str) -> bool:
    return has(
        text,
        r"fold test|reversal test|contact test|baked contact|conglomerate test|"
        r"APWP|apparent polar wander|palaeopole|paleopole|pole.{0,80}(agree|resemble|similar)|"
        r"polarity|normal and reverse|dual[- ]polarity",
    )


def pmagresult_metadata(pack: dict[str, Any]) -> dict[str, str]:
    metadata = pack.get("source_metadata", {}).get("pmagresult", {})
    if not isinstance(metadata, dict):
        return {}
    return {str(key): clean(str(value)) for key, value in metadata.items() if clean(str(value))}


def metadata_rockmag(pack: dict[str, Any]) -> dict[str, Any]:
    metadata = pmagresult_metadata(pack)
    rockmag = clean(metadata.get("ROCKMAG"))
    labdetails = clean(metadata.get("LABDETAILS"))
    treatment = clean(metadata.get("TREATMENT"))
    comments = clean(metadata.get("COMMENTS"))
    component = clean(metadata.get("COMPONENT"))
    quote_parts = []
    if rockmag:
        quote_parts.append(f"PMAGRESULT.ROCKMAG={rockmag}")
    if labdetails:
        quote_parts.append(f"LABDETAILS={labdetails}")
    if treatment:
        quote_parts.append(f"TREATMENT={treatment}")
    if comments:
        quote_parts.append(f"COMMENTS={comments}")
    quote = "; ".join(quote_parts)[:950]

    if not rockmag:
        return {
            "status": "absent",
            "quote": quote,
            "reason": "PMAGRESULT.ROCKMAG is blank for this RESULTNO.",
            "method_type": "none",
            "identity_type": "none",
            "valid_r3_evidence": False,
            "rockmag_code": "",
            "labdetails": labdetails,
            "treatment": treatment,
            "comments": comments,
            "component": component,
        }

    has_diagnostic = has(rockmag, DIAGNOSTIC_ROCKMAG_PATTERN)
    has_carrier = has(rockmag, CARRIER_ROCKMAG_PATTERN)
    weak_reasons: list[str] = []
    if tub_only_ferrimagnetic_inference(rockmag):
        weak_reasons.append(
            "PMAGRESULT.ROCKMAG infers a ferrimagnetic carrier from Tub/unblocking temperature only, "
            "without an independent rock-magnetic or mineralogical method."
        )
    if bare_secondary_carrier_code(rockmag, comments, component):
        weak_reasons.append(
            "PMAGRESULT.ROCKMAG is only a bare carrier code while COMMENTS/COMPONENT indicate a "
            "secondary, overprint, remagnetized, or post-folding component."
        )
    if op_alteration_with_negative_source_context(pack, rockmag):
        weak_reasons.append(
            "PMAGRESULT.ROCKMAG uses opaque-mineral petrography shorthand for alteration/oxidation, "
            "while source-text review snippets contain negative remanence context; this requires a "
            "target-specific carrier-remanence bridge."
        )
    if source_lab_artifact_remagnetization(pack):
        weak_reasons.append(
            "Source-text review snippets describe remagnetization due to a magnetic saw, so the "
            "carrier/mineralogy metadata is treated as preparation-artifact contaminated unless "
            "there is a clearer target-specific carrier-remanence bridge."
        )
    if op_hyst_with_local_remagnetization(pack, rockmag):
        weak_reasons.append(
            "PMAGRESULT.ROCKMAG combines opaque-mineral petrography with hysteresis, but source-text "
            "review snippets flag local remagnetization; this is weak without a clearer "
            "target-specific carrier-remanence bridge."
        )
    if op_without_carrier_hit_authigenic_remagnetization(pack, rockmag):
        weak_reasons.append(
            "PMAGRESULT.ROCKMAG is an opaque-mineral shorthand, but source-text review snippets have "
            "no named magnetic-carrier hit and instead link the component to authigenic minerals from "
            "rock remagnetization."
        )
    valid = has_diagnostic or has_carrier
    if weak_reasons:
        valid = False
    if has_diagnostic:
        method_type = "source_metadata_rock_magnetic_or_petrographic"
    elif has_carrier:
        method_type = "source_metadata_carrier_code"
    else:
        method_type = "source_metadata_weak_code"
    if has_carrier:
        identity_type = "named_mineral_or_mineral_code"
    elif has_diagnostic:
        identity_type = "diagnostic_method_without_named_mineral"
    else:
        identity_type = "weak_or_unclassified"

    return {
        "status": "present" if valid else "weak",
        "quote": quote,
        "reason": (
            "PMAGRESULT.ROCKMAG contains diagnostic rock-magnetic, petrographic, "
            "or carrier-mineral code evidence for this RESULTNO."
            if valid
            else (
                " ".join(weak_reasons)
                if weak_reasons
                else "PMAGRESULT.ROCKMAG is non-empty but only contains a weak/unclassified code."
            )
        ),
        "method_type": method_type,
        "identity_type": identity_type,
        "valid_r3_evidence": valid,
        "weak_r3_evidence_reasons": weak_reasons,
        "rockmag_code": rockmag,
        "labdetails": labdetails,
        "treatment": treatment,
        "comments": comments,
        "component": component,
    }


def review_hit_item(item: dict[str, Any], hit_type: str, pattern: str) -> dict[str, Any] | None:
    haystack = clean(f"{item.get('matched_text', '')} {item.get('excerpt', '')}")
    match = re.search(pattern, haystack, re.I | re.S)
    if not match:
        return None
    return {
        "hit_type": hit_type,
        "matched_term": clean(match.group(0))[:180],
        "matched_text": clean(str(item.get("matched_text", "")))[:220],
        "quote": clean(str(item.get("excerpt", "")))[:950],
        "line": item.get("line"),
        "page": item.get("page"),
        "section_hint": clean(str(item.get("section_hint", "")))[:220],
        "page_source": item.get("page_source"),
        "target_token_nearby": bool(item.get("target_token_nearby")),
        "candidate_category": str(item.get("category", "")),
    }


def source_text_review_hits(pack: dict[str, Any], *, max_per_type: int = 4) -> dict[str, Any]:
    """Surface source-text/PDF snippets for manual R3 audit without changing the score."""

    categories = [
        ("carrier_investigation_candidates", "rock_magnetic_method", REVIEW_METHOD_PATTERN),
        ("carrier_identity_candidates", "magnetic_carrier", REVIEW_CARRIER_PATTERN),
        ("carrier_investigation_candidates", "magnetic_carrier", REVIEW_CARRIER_PATTERN),
        ("carrier_significance_candidates", "carrier_remanence_significance", REVIEW_SIGNIFICANCE_PATTERN),
    ]
    evidence = pack.get("candidate_evidence", {})
    hits_by_type: dict[str, list[dict[str, Any]]] = {
        "rock_magnetic_method": [],
        "magnetic_carrier": [],
        "carrier_remanence_significance": [],
    }
    seen: set[tuple[str, str, str, str]] = set()
    for category, hit_type, pattern in categories:
        for item in evidence.get(category, []) or []:
            if len(hits_by_type[hit_type]) >= max_per_type:
                continue
            if not isinstance(item, dict):
                continue
            hit = review_hit_item(item, hit_type, pattern)
            if hit is None:
                continue
            key = (
                hit_type,
                str(hit.get("page", "")),
                str(hit.get("line", "")),
                str(hit.get("quote", ""))[:160].casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            hits_by_type[hit_type].append(hit)

    all_hits = [hit for hit_type in hits_by_type for hit in hits_by_type[hit_type]]
    return {
        "status": "present" if all_hits else "absent",
        "source_kind": str(pack.get("source", {}).get("source_kind", "")),
        "note": (
            "Source-text/PDF snippets are included for manual audit only; "
            "metadata_v1 scoring remains controlled by RESULTNO-specific PMAGRESULT.ROCKMAG."
        ),
        "counts": {hit_type: len(hits) for hit_type, hits in hits_by_type.items()},
        "hits": all_hits,
    }


def target_anchor(pack: dict[str, Any]) -> dict[str, str]:
    anchors = pack.get("candidate_evidence", {}).get("target_context", []) or []
    if anchors:
        return field(
            "present",
            clean(anchors[0].get("excerpt", ""))[:950],
            "Fresh evidence pack contains target-name context.",
            anchor_type="exact_unit",
        )
    if pack.get("multi_target_refno"):
        return field("group_inferred", "", "No exact target quote in pack, but REFNO has multiple target rows.", anchor_type="combined_group")
    return field("unclear", "", "No target anchor candidate was extracted.", anchor_type="unclear")


def carrier_investigation(pack: dict[str, Any]) -> dict[str, str]:
    text = pack_text(pack, ["carrier_investigation_candidates"])
    quote = first_quote(pack, "carrier_investigation_candidates")
    if not quote:
        return field("absent", "", "No carrier-investigation candidate was extracted.", method_type="none")
    if has(text, r"IRM|isothermal|Lowrie|three[- ]axis|thermomagnetic|susceptibility|hysteresis|FORC|SEM|TEM|EDS|EPMA|microscop|petrograph|mineral separation|rock magnetic"):
        method = "mixed" if has(text, r"thermal demagnet|AF demagnet") else "rock_magnetic"
        return field("present", quote, "Evidence pack includes rock-magnetic, microscopic, or mineralogical carrier investigation.", method_type=method)
    if has(text, r"thermal demagnet|AF demagnet|unblocking|blocking temperature|coerciv"):
        return field("demagnetization_only", quote, "Only demagnetization/coercivity-style carrier evidence was extracted.", method_type="thermal_af_cleaning_only")
    return field("unclear", quote, "Carrier-investigation candidate is ambiguous.", method_type="unclear")


def carrier_identity(pack: dict[str, Any]) -> dict[str, str]:
    text = pack_text(pack, ["carrier_identity_candidates", "carrier_investigation_candidates"])
    quote = first_quote(pack, "carrier_identity_candidates") or first_quote(pack, "carrier_investigation_candidates")
    if has(text, r"magnetite|titanomagnetite|hematite|haematite|maghemite|goethite|pyrrhotite|greigite|Fe[- ]Ti oxide"):
        return field("present", quote, "Evidence pack names magnetic carrier minerals.", identity_type="named_mineral")
    if has(text, r"low[- ]coercivity|high[- ]coercivity|unblocking|blocking temperature|single[- ]domain|pseudo[- ]single[- ]domain|domain state"):
        return field("present", quote, "Evidence pack identifies carrier class/domain/coercivity behavior.", identity_type="coercivity_or_unblocking_class")
    if quote:
        return field("inferred_only", quote, "Carrier identity would require inference from candidate text.", identity_type="unclear")
    return field("absent", "", "No carrier identity candidate was extracted.", identity_type="none")


def remanence_bridge(pack: dict[str, Any]) -> dict[str, str]:
    text = pack_text(pack, ["carrier_significance_candidates"])
    quote = first_quote(pack, "carrier_significance_candidates")
    if not quote:
        directional = first_quote(pack, "directional_or_weak_substitution_warnings")
        if directional:
            return field("directional_only", directional, "Only directional/age-validation warning text was extracted.", bridge_type="directional_age_validation")
        return field("absent", "", "No carrier-remanence bridge candidate was extracted.", bridge_type="none")
    if has(text, r"primary|secondary|overprint|remagnet|CRM|TRM|DRM|VRM|detrital|chemical|authigenic|alteration"):
        return field("present", quote, "Carrier/mineralogy evidence is linked to remanence origin or overprint interpretation.", bridge_type="primary_or_secondary_origin")
    if has(text, r"ChRM|characteristic|component|carr(?:y|ies|ied|ier)|stable|stability|reliab|unblocking|coerciv"):
        return field("present", quote, "Candidate text links carrier behavior to remanence/component interpretation.", bridge_type="component_isolation")
    return field("inferred_only", quote, "Bridge candidate is present but requires inference.", bridge_type="unclear")


def evidence_family(
    pack: dict[str, Any],
    a: dict[str, str],
    b: dict[str, str],
    c: dict[str, str],
    *,
    line_window: int,
    target_near_crossref: bool,
) -> dict[str, str]:
    if a["quote"] and b["quote"] and c["quote"]:
        if a["quote"] == b["quote"] or b["quote"] == c["quote"] or a["quote"] == c["quote"]:
            return field("same_section_or_figure", a["quote"] or b["quote"] or c["quote"], "A/B/C quotes overlap in the fresh evidence pack.")
        items = [
            first_item(pack, "carrier_investigation_candidates"),
            first_item(pack, "carrier_identity_candidates"),
            first_item(pack, "carrier_significance_candidates"),
        ]
        lines = [item.get("line") for item in items if isinstance(item, dict) and isinstance(item.get("line"), int)]
        pages = [item.get("page") for item in items if isinstance(item, dict) and isinstance(item.get("page"), int)]
        nearby_target = [
            bool(item.get("target_token_nearby"))
            for item in items
            if isinstance(item, dict) and "target_token_nearby" in item
        ]
        if len(lines) >= 2 and max(lines) - min(lines) <= line_window:
            return field(
                "same_section_or_figure",
                (a["quote"] + " " + b["quote"] + " " + c["quote"])[:950],
                f"A/B/C candidates occur within a {line_window}-line window in the fresh evidence pack.",
            )
        if len(pages) >= 2 and max(pages) - min(pages) <= 1 and nearby_target and sum(nearby_target) >= 2:
            return field(
                "same_section_or_figure",
                (a["quote"] + " " + b["quote"] + " " + c["quote"])[:950],
                "A/B/C candidates occur on the same or adjacent source page with target anchoring.",
            )
        combined = field_text(a, b, c)
        if (
            target_near_crossref
            and nearby_target
            and sum(nearby_target) >= 2
            and explicit_carrier_remanence_language(combined)
            and len(lines) >= 2
            and max(lines) - min(lines) <= max(line_window * 4, 120)
        ):
            return field(
                "explicit_cross_reference",
                (a["quote"] + " " + b["quote"] + " " + c["quote"])[:950],
                "At least two A/B/C candidates are target-anchored and contain explicit carrier-remanence language.",
            )
        return field("separate_stitched", (a["quote"] + " " + b["quote"] + " " + c["quote"])[:950], "A/B/C candidates are present but come from separate extracted snippets.")
    return field("unclear", "", "Incomplete A/B/C candidate chain.",)


def source_quality(pack: dict[str, Any]) -> dict[str, str]:
    source = pack.get("source", {})
    flags = source.get("source_quality_flags") or []
    if not flags:
        status = "complete"
    elif "no_local_text" in flags or "no_local_pdf" in flags or "no_extractable_text" in flags:
        status = "missing"
    elif "glyph_or_cid_text" in flags or "short_text" in flags or "possibly_incomplete_extraction" in flags:
        status = "partial"
    else:
        status = "partial"
    return {"status": status, "source_kind": str(source.get("source_kind", "parsed_text")), "notes": "; ".join(flags) if flags else "no source-quality flags"}


def structured(pack: dict[str, Any], *, line_window: int, target_near_crossref: bool) -> dict[str, Any]:
    a = carrier_investigation(pack)
    b = carrier_identity(pack)
    c = remanence_bridge(pack)
    fam = evidence_family(
        pack,
        a,
        b,
        c,
        line_window=line_window,
        target_near_crossref=target_near_crossref,
    )
    sq = source_quality(pack)
    rockmag = metadata_rockmag(pack)
    review_hits = source_text_review_hits(pack)
    directional = bool((pack.get("candidate_evidence", {}).get("directional_or_weak_substitution_warnings") or []))
    combined_text = field_text(a, b, c, fam)
    target_status = target_anchor(pack)["status"]
    demag_or_directional_only = (
        a["status"] == "demagnetization_only"
        or (
            a.get("method_type") == "mixed"
            and not has(
                combined_text,
                r"IRM|isothermal|Lowrie|three[- ]axis|thermomagnetic|hysteresis|FORC|SEM|TEM|EDS|EPMA|microscop|petrograph|mineral separation",
            )
        )
    )
    directional_substitution = (
        c["status"] == "directional_only"
        or (
            directional_validation_language(combined_text)
            and not explicit_carrier_remanence_language(combined_text)
        )
        or (
            directional
            and c["status"] != "present"
        )
    )
    risks = risk_dict(
        directional_substitution_risk=directional_substitution,
        multi_target_transfer_risk=bool(pack.get("multi_target_refno")) and (
            target_status != "present" or fam["status"] in {"separate_stitched", "unclear"}
        ),
        demagnetization_only_risk=demag_or_directional_only,
        source_quality_risk=sq["status"] != "complete",
        evaluator_inference_risk=(
            b["status"] == "inferred_only"
            or c["status"] == "inferred_only"
            or (c["status"] == "present" and not explicit_carrier_remanence_language(combined_text))
        ),
        separate_evidence_family_risk=fam["status"] == "separate_stitched",
    )
    return {
        "resultno": str(pack["RESULTNO"]),
        "target": str(pack["TARGET"]),
        "extractor": EXTRACTOR,
        "source_quality": sq,
        "target_anchor": target_anchor(pack),
        "carrier_investigation": a,
        "carrier_identity": b,
        "remanence_use_bridge": c,
        "evidence_family": fam,
        "metadata_rockmag": rockmag,
        "source_text_review_hits": review_hits,
        "risks": risks,
        "high_risk_route": any(risks.values()) or sq["status"] != "complete",
        "extractor_notes": (
            "Generated from fresh R3 evidence pack; no labels or old structured extraction used. "
            f"family_line_window={line_window}; target_near_crossref={target_near_crossref}; "
            "PMAGRESULT.ROCKMAG is included as RESULTNO-specific source metadata when available."
        ),
        "manual_review": sq["status"] in {"missing"}
        or target_anchor(pack)["status"] in {"unclear", "absent", "wrong_target", "group_inferred"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--line-window",
        type=int,
        default=35,
        help="Maximum line span for treating A/B/C evidence as same-section evidence.",
    )
    parser.add_argument(
        "--target-near-crossref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow two target-near A/B/C snippets to count as explicit cross-reference evidence.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.packs_jsonl.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as out:
        for line in source:
            if not line.strip():
                continue
            pack = json.loads(line)
            row = {
                "REFNO": str(pack["REFNO"]),
                "RESULTNO": str(pack["RESULTNO"]),
                "TARGET": str(pack["TARGET"]),
                "STRUCTURED_EVIDENCE": structured(
                    pack,
                    line_window=args.line_window,
                    target_near_crossref=args.target_near_crossref,
                ),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"rows": count, "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
