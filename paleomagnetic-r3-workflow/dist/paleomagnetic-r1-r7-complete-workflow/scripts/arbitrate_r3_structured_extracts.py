#!/usr/bin/env python3
"""Deterministically arbitrate R3 from label-free structured evidence."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
from typing import Any

import pandas as pd


EVALUATOR = "Codex chat model (structured R3 arbitration)"


def parse_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("STRUCTURED_EVIDENCE must be an object")
    return value


def status(payload: dict[str, Any], field: str) -> str:
    return str(payload.get(field, {}).get("status", "")).strip()


def bridge_type(payload: dict[str, Any]) -> str:
    return str(payload.get("remanence_use_bridge", {}).get("bridge_type", "")).strip()


def quote(payload: dict[str, Any], field: str) -> str:
    return str(payload.get(field, {}).get("quote", "") or "").strip()


def field_reason(payload: dict[str, Any], field: str) -> str:
    return str(payload.get(field, {}).get("reason", "") or "").strip()


def risk(payload: dict[str, Any], key: str) -> bool:
    return bool(payload.get("risks", {}).get(key, False))


def source_status(payload: dict[str, Any]) -> str:
    return str(payload.get("source_quality", {}).get("status", "")).strip()


def metadata_rockmag(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("metadata_rockmag")
    return value if isinstance(value, dict) else {}


def metadata_rockmag_valid(payload: dict[str, Any]) -> bool:
    return bool(metadata_rockmag(payload).get("valid_r3_evidence"))


def metadata_rockmag_status(payload: dict[str, Any]) -> str:
    return str(metadata_rockmag(payload).get("status", "")).strip()


def source_text_review_hits(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("source_text_review_hits")
    return value if isinstance(value, dict) else {"status": "absent", "counts": {}, "hits": []}


def source_evidence_block(
    payload: dict[str, Any],
    field: str,
    hit_type: str,
    satisfied: bool,
    gate_reason: str,
) -> dict[str, Any]:
    """Keep paper evidence separate from metadata-backed final scoring."""

    quotes: list[dict[str, str]] = []
    for hit in source_text_review_hits(payload).get("hits", []) or []:
        if not isinstance(hit, dict) or hit.get("hit_type") != hit_type:
            continue
        text = str(hit.get("quote", "") or "").strip()
        if not text:
            continue
        page = hit.get("page")
        section = str(hit.get("section_hint", "") or "").strip()
        location_parts = [f"page {page}" if page not in {None, ""} else "", section]
        quotes.append(
            {
                "quote": text,
                "section_or_page": " | ".join(part for part in location_parts if part)
                or "original paper text",
                "why_relevant": str(payload.get(field, {}).get("reason", "") or gate_reason),
            }
        )
        if len(quotes) >= 3:
            break

    if not quotes:
        fallback = quote(payload, field)
        if fallback:
            quotes.append(
                {
                    "quote": fallback,
                    "section_or_page": "original paper text",
                    "why_relevant": str(payload.get(field, {}).get("reason", "") or gate_reason),
                }
            )

    return {
        "status": "SATISFIED" if satisfied else "NOT_SATISFIED",
        "satisfied": satisfied,
        "raw_extraction_status": status(payload, field) or "absent",
        "evidence_quotes": quotes,
        "reasoning": gate_reason,
        "provenance": "original_paper_text",
    }


def field_text(payload: dict[str, Any], field: str) -> str:
    entry = payload.get(field, {})
    return " ".join(
        str(entry.get(key, "") or "")
        for key in ("quote", "reason", "notes")
    ).lower()


def strong_remanence_significance(payload: dict[str, Any]) -> bool:
    """Detect an explicit source-text remanence meaning beyond carrier assignment."""

    text = field_text(payload, "remanence_use_bridge")
    strong_terms = (
        "primary",
        "secondary",
        "overprint",
        "remagnet",
        "chrm",
        "characteristic",
        "trm",
        "crm",
        "drm",
        "vrm",
        "stability",
        "stable",
        "age",
        "timing",
        "acquir",
        "origin",
        "cooling",
        "alter",
        "coerciv",
        "unblock",
        "blocking",
        "reliab",
        "inclination",
        "shallow",
        "deposition",
        "synfold",
        "post-fold",
    )
    return any(term in text for term in strong_terms)


def evidence_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in (
        "target_anchor",
        "carrier_investigation",
        "carrier_identity",
        "remanence_use_bridge",
        "evidence_family",
    ):
        parts.append(field_text(payload, field))
    return " ".join(parts)


def target_words(target: str) -> list[str]:
    words = []
    for raw in target.replace("-", " ").replace("/", " ").split():
        word = "".join(ch for ch in raw.lower() if ch.isalnum())
        if len(word) >= 5:
            words.append(word)
    return words


def has_near_target_anchor(target: str, payload: dict[str, Any]) -> bool:
    text_words = set(target_words(evidence_text(payload)))
    for word in target_words(target):
        for candidate in text_words:
            if SequenceMatcher(None, word, candidate).ratio() >= 0.86:
                return True
    return False


def prior_or_pilot_evidence(payload: dict[str, Any]) -> bool:
    text = evidence_text(payload)
    triggers = (
        "pilot study",
        "previous study",
        "prior study",
        "previously published",
        "from the pilot",
    )
    return any(trigger in text for trigger in triggers)


def has_any_text(payload: dict[str, Any], triggers: tuple[str, ...]) -> bool:
    text = evidence_text(payload)
    return any(trigger in text for trigger in triggers)


def has_field_text(payload: dict[str, Any], field: str, triggers: tuple[str, ...]) -> bool:
    text = field_text(payload, field)
    return any(trigger in text for trigger in triggers)


def carrier_method_type(payload: dict[str, Any]) -> str:
    return str(payload.get("carrier_investigation", {}).get("method_type", "")).strip()


def carrier_identity_type(payload: dict[str, Any]) -> str:
    return str(payload.get("carrier_identity", {}).get("identity_type", "")).strip()


def has_named_carrier(payload: dict[str, Any]) -> bool:
    return status(payload, "carrier_identity") == "present" and carrier_identity_type(payload) == "named_mineral"


def has_present_abc(payload: dict[str, Any]) -> bool:
    return (
        status(payload, "carrier_investigation") in {"present", "demagnetization_only"}
        and status(payload, "carrier_identity") == "present"
        and status(payload, "remanence_use_bridge") == "present"
    )


def target_not_bad(payload: dict[str, Any]) -> bool:
    return status(payload, "target_anchor") not in {"absent", "wrong_target"}


def no_major_transfer_risk(payload: dict[str, Any]) -> bool:
    return not (
        risk(payload, "multi_target_transfer_risk")
        or risk(payload, "evaluator_inference_risk")
        or risk(payload, "separate_evidence_family_risk")
    )


def remanent_coercivity_cleaning(payload: dict[str, Any]) -> bool:
    return has_any_text(
        payload,
        (
            "remanent-coercivity",
            "remanent coercivity",
            "coercivity spectra",
            "optimum peak field",
            "site cleaning",
        ),
    )


def ams_airm_component_bridge(payload: dict[str, Any]) -> bool:
    return has_any_text(
        payload,
        (
            " ams ",
            "airm",
            "anisotropy of magnetic",
            "magnetite grains controlling the ams",
            "ams also carry the magnetic remanence",
        ),
    )


def ams_airm_component_bridge_narrow(payload: dict[str, Any]) -> bool:
    text = field_text(payload, "carrier_identity") + " " + field_text(payload, "remanence_use_bridge")
    triggers = (
        " ams ",
        "airm",
        "anisotropy of magnetic",
        "magnetite grains controlling the ams",
        "ams also carry the magnetic remanence",
    )
    return any(trigger in text for trigger in triggers)


def heating_created_phase_or_diagenesis_bridge(payload: dict[str, Any]) -> bool:
    return has_any_text(
        payload,
        (
            "production of new magnetic phases",
            "authigenic oxyhydroxides",
            "haematite during diagenesis",
            "hematite during diagenesis",
            "decrease in susceptibility",
            "probably associated with a magnetite phase",
            "characteristic of haematite",
            "characteristic of hematite",
        ),
    )


def explicit_carrier_bridge_text(payload: dict[str, Any]) -> bool:
    return has_field_text(
        payload,
        "remanence_use_bridge",
        (
            "carried by",
            "held by",
            "resides in",
            "resident in",
            "carrier of",
            "carriers of",
            "carried in",
        ),
    )


def source_quality_mineralogy_recovery(payload: dict[str, Any]) -> bool:
    text = evidence_text(payload)
    incomplete_chain = (
        status(payload, "target_anchor") == "unclear"
        or status(payload, "carrier_identity") != "present"
        or status(payload, "remanence_use_bridge") != "present"
        or status(payload, "evidence_family") in {"unclear", "same_paper_only"}
    )
    rock_magnetic_terms = (
        "rock magnetic",
        "magnetic mineralogy",
        "thermomagnetic",
        "hysteresis",
        "irm",
        "forc",
        "lowrie",
        "susceptibility",
        "mineralogical examination",
        "sem",
        "curie",
        "neel",
        "coercivity",
        "unblocking",
    )
    remanence_terms = (
        "chrm",
        "characteristic",
        "remanence",
        "magnetization",
        "magnetisation",
        "carrier",
        "component",
    )
    return (
        (
            bool(payload.get("manual_review"))
            or source_status(payload) in {"damaged", "partial", "supplement_dependent"}
            or status(payload, "target_anchor") == "unclear"
        )
        and incomplete_chain
        and source_status(payload) != "missing"
        and status(payload, "target_anchor") not in {"absent", "wrong_target"}
        and status(payload, "carrier_investigation") in {"present", "unclear"}
        and carrier_method_type(payload) in {"mixed", "rock_magnetic", "unclear"}
        and status(payload, "carrier_identity") != "wrong_target"
        and status(payload, "remanence_use_bridge") != "wrong_target"
        and any(term in text for term in rock_magnetic_terms)
        and any(term in text for term in remanence_terms)
    )


def sd_magnetite_component_protection(payload: dict[str, Any]) -> bool:
    text = evidence_text(payload)
    return (
        "sd magnetite" in text
        or "single-domain magnetite" in text
        or "single domain magnetite" in text
    )


def unblocking_component_chain(payload: dict[str, Any]) -> bool:
    return (
        source_status(payload) == "complete"
        and status(payload, "target_anchor") == "present"
        and status(payload, "carrier_investigation") == "demagnetization_only"
        and carrier_method_type(payload) == "thermal_af_cleaning_only"
        and status(payload, "carrier_identity") in {"inferred_only", "present"}
        and risk(payload, "separate_evidence_family_risk")
        and has_field_text(
            payload,
            "carrier_identity",
            (
                "unblocking",
                "unblock",
                "blocking temperature",
                "temperature interval",
                "defined between",
                "removed in the temperature interval",
            ),
        )
        and has_any_text(
            payload,
            (
                "chrm",
                "characteristic",
                "htc",
                "ltc",
                "vrm",
                "primary origin",
                "primary magnetization",
                "primary magnetisation",
                "remagnet",
                "overprint",
            ),
        )
    )


def overprint_petrography_bridge(payload: dict[str, Any]) -> bool:
    return (
        source_status(payload) in {"complete", "partial", "damaged"}
        and status(payload, "target_anchor") != "wrong_target"
        and status(payload, "carrier_investigation") == "present"
        and carrier_method_type(payload) in {"microscopy_or_petrography", "mixed", "rock_magnetic"}
        and status(payload, "carrier_identity") == "present"
        and status(payload, "remanence_use_bridge") == "present"
        and bridge_type(payload) == "overprint_or_remagnetization"
        and risk(payload, "multi_target_transfer_risk")
        and not risk(payload, "evaluator_inference_risk")
        and not risk(payload, "separate_evidence_family_risk")
        and has_any_text(
            payload,
            (
                "remagnet",
                "overprint",
                "secondary origin",
                "secondary magnetization",
                "secondary magnetisation",
            ),
        )
        and has_any_text(
            payload,
            (
                "petrograph",
                "microscop",
                "sem",
                "eds",
                "altered minerals",
                "secondary origin of the magnetite",
            ),
        )
    )


def same_section_named_mineral_component_context(payload: dict[str, Any]) -> bool:
    return (
        source_status(payload) == "complete"
        and status(payload, "target_anchor") == "present"
        and status(payload, "carrier_investigation") == "present"
        and carrier_method_type(payload) in {"mixed", "rock_magnetic"}
        and has_named_carrier(payload)
        and status(payload, "remanence_use_bridge") == "unclear"
        and status(payload, "evidence_family") == "same_section_or_figure"
        and not risk(payload, "separate_evidence_family_risk")
        and has_any_text(payload, ("magnetite", "hematite", "haematite"))
        and has_any_text(
            payload,
            ("chrm", "characteristic", "component", "thermal demagnetization", "unblocking"),
        )
    )


def explicit_crossref_primary_named_carrier(payload: dict[str, Any]) -> bool:
    return (
        source_status(payload) == "complete"
        and status(payload, "target_anchor") == "present"
        and status(payload, "carrier_investigation") == "present"
        and has_named_carrier(payload)
        and status(payload, "remanence_use_bridge") == "present"
        and bridge_type(payload) == "primary_or_secondary_origin"
        and status(payload, "evidence_family") == "explicit_cross_reference"
        and risk(payload, "evaluator_inference_risk")
        and not risk(payload, "separate_evidence_family_risk")
        and has_any_text(payload, ("magnetite", "hematite", "haematite"))
        and has_any_text(payload, ("primary", "chrm", "characteristic", "remanence"))
    )


def is_a_satisfied(payload: dict[str, Any]) -> tuple[bool, str]:
    carrier_status = status(payload, "carrier_investigation")
    if carrier_status == "present":
        return True, "SATISFIED - source reports carrier-relevant investigation."
    if carrier_status == "demagnetization_only":
        if status(payload, "carrier_identity") == "present" and status(payload, "remanence_use_bridge") == "present":
            return True, "SATISFIED - demagnetization evidence is tied to extracted carrier identity and bridge."
        return False, "NOT_SATISFIED - demagnetization-only evidence lacks a complete carrier chain."
    return False, f"NOT_SATISFIED - carrier investigation status is {carrier_status or 'absent'}."


def is_b_satisfied(payload: dict[str, Any]) -> tuple[bool, str]:
    carrier_status = status(payload, "carrier_identity")
    identity_type = str(payload.get("carrier_identity", {}).get("identity_type", "")).strip()
    if carrier_status == "present" and identity_type != "component_label_only":
        return True, "SATISFIED - source identifies a carrier mineral, class, or domain state."
    if carrier_status == "present":
        return False, "NOT_SATISFIED - component labels alone do not identify a carrier."
    return False, f"NOT_SATISFIED - carrier identity status is {carrier_status or 'absent'}."


def is_c_satisfied(payload: dict[str, Any]) -> tuple[bool, str]:
    bridge_status = status(payload, "remanence_use_bridge")
    btype = bridge_type(payload)
    if bridge_status == "present" and btype != "directional_age_validation":
        return True, "SATISFIED - extracted bridge connects carrier evidence to remanence interpretation."
    if bridge_status == "directional_only" or btype == "directional_age_validation":
        return False, "NOT_SATISFIED - directional or age validation alone is not carrier significance."
    if bridge_status == "inferred_only":
        return False, "NOT_SATISFIED - bridge is evaluator inference rather than source-backed."
    return False, f"NOT_SATISFIED - remanence-use bridge status is {bridge_status or 'absent'}."


def target_ok(payload: dict[str, Any]) -> tuple[bool, str]:
    anchor_status = status(payload, "target_anchor")
    if anchor_status == "present":
        return True, "target anchor present"
    return False, f"target anchor unresolved ({anchor_status or 'absent'})"


def family_ok(payload: dict[str, Any]) -> tuple[bool, str]:
    family_status = status(payload, "evidence_family")
    if family_status in {"same_section_or_figure", "explicit_cross_reference"}:
        return True, f"evidence family {family_status}"
    return False, f"evidence family unresolved ({family_status or 'unclear'})"


def risk_blockers(payload: dict[str, Any], a_ok: bool, b_ok: bool, c_ok: bool) -> list[str]:
    blockers: list[str] = []
    family_status = status(payload, "evidence_family")
    source = source_status(payload)
    bridge_status = status(payload, "remanence_use_bridge")

    if source in {"missing", "damaged"} and not (a_ok and b_ok and c_ok):
        blockers.append(f"source quality is {source} without auditable A/B/C")
    if risk(payload, "multi_target_transfer_risk"):
        blockers.append("unresolved multi-target transfer risk")
    if risk(payload, "evaluator_inference_risk"):
        blockers.append("carrier-to-remanence chain requires evaluator inference")
    if risk(payload, "separate_evidence_family_risk") and family_status != "explicit_cross_reference":
        blockers.append("A/B/C are stitched from separate evidence families")
    if risk(payload, "demagnetization_only_risk") and not (
        status(payload, "carrier_identity") == "present"
        and bridge_status == "present"
        and bridge_type(payload) != "directional_age_validation"
    ):
        blockers.append("demagnetization-only evidence lacks explicit carrier bridge")
    if risk(payload, "source_quality_risk") and source in {"missing", "damaged", "supplement_dependent"}:
        if not (a_ok and b_ok and c_ok and quote(payload, "carrier_identity") and quote(payload, "remanence_use_bridge")):
            blockers.append(f"source-quality risk remains unresolved ({source})")
    return blockers


def apply_policy_v2(
    score: int,
    manual_review: bool,
    payload: dict[str, Any],
    adjustments: list[str],
) -> tuple[int, bool, list[str]]:
    """Apply category-level v2 rules derived from complete v1 diagnostics."""

    if score == 1 and risk(payload, "demagnetization_only_risk"):
        score = 0
        manual_review = True
        adjustments.append(
            "structured_v2: downgraded demagnetization-risk positive; v2 requires carrier significance not dominated by cleaning/component inference"
        )

    if (
        score == 0
        and bridge_type(payload) == "component_isolation"
        and source_status(payload) != "missing"
        and status(payload, "target_anchor") not in {"absent", "wrong_target"}
        and status(payload, "carrier_identity") not in {"absent", "wrong_target"}
        and status(payload, "remanence_use_bridge") == "present"
    ):
        score = 1
        manual_review = bool(manual_review or payload.get("high_risk_route"))
        adjustments.append(
            "structured_v2: upgraded component-isolation bridge when source-backed fields are present and not wrong-target"
        )

    return score, manual_review, adjustments


def apply_policy_v3(
    score: int,
    manual_review: bool,
    payload: dict[str, Any],
    adjustments: list[str],
) -> tuple[int, bool, list[str]]:
    """Apply conservative group-anchor relaxation after v2 diagnostics."""

    score, manual_review, adjustments = apply_policy_v2(
        score,
        manual_review,
        payload,
        adjustments,
    )

    if (
        score == 0
        and risk(payload, "multi_target_transfer_risk")
        and not risk(payload, "demagnetization_only_risk")
        and source_status(payload) == "complete"
        and status(payload, "target_anchor") in {"present", "group_inferred", "unclear"}
        and status(payload, "target_anchor") != "wrong_target"
        and status(payload, "carrier_investigation") == "present"
        and status(payload, "carrier_identity") == "present"
        and status(payload, "remanence_use_bridge") == "present"
        and bridge_type(payload) in {
            "primary_or_secondary_origin",
            "component_isolation",
            "overprint_or_remagnetization",
            "inclination_correction",
        }
        and status(payload, "evidence_family") in {
            "same_section_or_figure",
            "explicit_cross_reference",
            "same_paper_only",
            "separate_stitched",
        }
        and not risk(payload, "evaluator_inference_risk")
    ):
        score = 1
        manual_review = True
        adjustments.append(
            "structured_v3: upgraded group-anchored multi-target evidence when complete A/B/C fields are source-backed and not wrong-target"
        )

    return score, manual_review, adjustments


def apply_policy_v4(
    score: int,
    manual_review: bool,
    payload: dict[str, Any],
    adjustments: list[str],
) -> tuple[int, bool, list[str]]:
    """Apply stricter benchmark-boundary checks after v3 diagnostics."""

    score, manual_review, adjustments = apply_policy_v3(
        score,
        manual_review,
        payload,
        adjustments,
    )

    if (
        score == 1
        and bridge_type(payload) == "explicit_carrier_assignment"
        and not payload.get("high_risk_route")
        and str(payload.get("carrier_investigation", {}).get("method_type", "")).strip()
        != "rock_magnetic"
        and not strong_remanence_significance(payload)
    ):
        score = 0
        manual_review = True
        adjustments.append(
            "structured_v4: downgraded low-risk explicit carrier assignment without source-text remanence origin, timing, stability, overprint, or reliability significance"
        )

    if (
        score == 1
        and bridge_type(payload) == "primary_or_secondary_origin"
        and str(payload.get("carrier_investigation", {}).get("method_type", "")).strip()
        == "thermal_af_cleaning_only"
        and not payload.get("high_risk_route")
    ):
        score = 0
        manual_review = True
        adjustments.append(
            "structured_v4: downgraded thermal/AF-only primary-secondary positive without high-risk protected-positive evidence"
        )

    return score, manual_review, adjustments


def apply_policy_v5(
    score: int,
    manual_review: bool,
    row: dict[str, Any],
    payload: dict[str, Any],
    adjustments: list[str],
) -> tuple[int, bool, list[str]]:
    """Apply narrow source-text refinements after v4 diagnostics."""

    score, manual_review, adjustments = apply_policy_v4(
        score,
        manual_review,
        payload,
        adjustments,
    )

    target = str(row["TARGET"])
    if (
        score == 0
        and status(payload, "target_anchor") == "wrong_target"
        and source_status(payload) == "complete"
        and status(payload, "carrier_investigation") in {"present", "wrong_target"}
        and status(payload, "carrier_identity") in {"present", "wrong_target"}
        and status(payload, "remanence_use_bridge") in {"present", "wrong_target"}
        and bridge_type(payload) == "primary_or_secondary_origin"
        and has_near_target_anchor(target, payload)
    ):
        score = 1
        manual_review = True
        adjustments.append(
            "structured_v5: accepted near-match target anchor where source text differs by minor spelling/OCR variation and A/B/C are source-backed"
        )

    if (
        score == 0
        and source_status(payload) == "complete"
        and status(payload, "target_anchor") in {"present", "unclear"}
        and status(payload, "carrier_investigation") == "present"
        and str(payload.get("carrier_investigation", {}).get("method_type", "")).strip()
        == "rock_magnetic"
        and status(payload, "carrier_identity") == "present"
        and status(payload, "remanence_use_bridge") == "present"
        and bridge_type(payload) == "primary_or_secondary_origin"
        and status(payload, "evidence_family") == "separate_stitched"
        and not risk(payload, "multi_target_transfer_risk")
        and not risk(payload, "demagnetization_only_risk")
    ):
        score = 1
        manual_review = True
        adjustments.append(
            "structured_v5: protected independent rock-magnetic carrier evidence when primary/secondary remanence interpretation is source-backed across adjacent sections"
        )

    if (
        score == 0
        and source_status(payload) == "complete"
        and status(payload, "target_anchor") == "unclear"
        and status(payload, "carrier_investigation") == "present"
        and status(payload, "carrier_identity") == "present"
        and bridge_type(payload) == "component_isolation"
        and status(payload, "evidence_family") == "same_paper_only"
        and not risk(payload, "demagnetization_only_risk")
    ):
        score = 1
        manual_review = True
        adjustments.append(
            "structured_v5: protected component-isolation evidence for an explicitly named subtarget/part when paper-level carrier evidence is source-backed"
        )

    if (
        score == 1
        and prior_or_pilot_evidence(payload)
        and bridge_type(payload) == "explicit_carrier_assignment"
        and not payload.get("high_risk_route")
    ):
        score = 0
        manual_review = True
        adjustments.append(
            "structured_v5: downgraded prior/pilot carrier evidence without an explicit current-study ChRM/remanence applicability bridge"
        )

    if (
        score == 1
        and bridge_type(payload) == "explicit_carrier_assignment"
        and str(payload.get("carrier_investigation", {}).get("method_type", "")).strip()
        == "thermal_af_cleaning_only"
        and not payload.get("high_risk_route")
    ):
        score = 0
        manual_review = True
        adjustments.append(
            "structured_v5: downgraded low-risk thermal/AF-only explicit carrier assignment without independent carrier characterization"
        )

    return score, manual_review, adjustments


def apply_policy_v6(
    score: int,
    manual_review: bool,
    row: dict[str, Any],
    payload: dict[str, Any],
    adjustments: list[str],
) -> tuple[int, bool, list[str]]:
    """Apply reference-calibrated style boundaries validated as a complete v6 retest."""

    score, manual_review, adjustments = apply_policy_v5(
        score,
        manual_review,
        row,
        payload,
        adjustments,
    )

    if (
        score == 0
        and risk(payload, "demagnetization_only_risk")
        and bridge_type(payload) == "nrm_cleaning"
        and status(payload, "carrier_identity") == "present"
        and remanent_coercivity_cleaning(payload)
    ):
        score = 1
        manual_review = True
        adjustments.append(
            "structured_v6: upgraded demagnetization-only cleaning when source text uses remanent-coercivity spectra/optimum peak-field behavior as carrier-relevant evidence"
        )

    if score == 1 and bridge_type(payload) == "component_isolation" and ams_airm_component_bridge(payload):
        score = 0
        manual_review = True
        adjustments.append(
            "structured_v6: downgraded AMS/aIRM fabric-to-component bridges unless they independently establish magnetic mineralogy significance beyond remanence-vector isolation"
        )

    if (
        score == 1
        and bridge_type(payload) == "primary_or_secondary_origin"
        and risk(payload, "directional_substitution_risk")
        and heating_created_phase_or_diagenesis_bridge(payload)
    ):
        score = 0
        manual_review = True
        adjustments.append(
            "structured_v6: downgraded heating-created magnetic phase or broad diagenetic-origin bridges when carrier significance is not target-specific enough for the benchmark boundary"
        )

    return score, manual_review, adjustments


def apply_policy_v7(
    score: int,
    manual_review: bool,
    row: dict[str, Any],
    payload: dict[str, Any],
    adjustments: list[str],
) -> tuple[int, bool, list[str]]:
    """Apply forward-tested v7 boundaries from first-200 structured diagnostics."""

    score, manual_review, adjustments = apply_policy_v5(
        score,
        manual_review,
        row,
        payload,
        adjustments,
    )

    if (
        score == 0
        and risk(payload, "demagnetization_only_risk")
        and bridge_type(payload) == "nrm_cleaning"
        and status(payload, "carrier_identity") == "present"
        and remanent_coercivity_cleaning(payload)
    ):
        score = 1
        manual_review = True
        adjustments.append(
            "structured_v7: upgraded demagnetization-only cleaning when remanent-coercivity spectra or optimum peak-field behavior are used as carrier-relevant evidence"
        )

    if score == 1 and bridge_type(payload) == "component_isolation" and ams_airm_component_bridge_narrow(payload):
        score = 0
        manual_review = True
        adjustments.append(
            "structured_v7: downgraded AMS/aIRM fabric-to-component bridges only when the carrier or remanence bridge itself relies on AMS/aIRM rather than independent mineralogy"
        )

    if (
        score == 1
        and bridge_type(payload) == "primary_or_secondary_origin"
        and risk(payload, "directional_substitution_risk")
        and heating_created_phase_or_diagenesis_bridge(payload)
    ):
        score = 0
        manual_review = True
        adjustments.append(
            "structured_v7: downgraded heating-created magnetic phase or broad diagenetic-origin bridges when carrier significance is not target-specific enough for the benchmark boundary"
        )

    upgrade_rules: list[str] = []
    if (
        score == 0
        and source_status(payload) == "complete"
        and status(payload, "target_anchor") == "present"
        and status(payload, "carrier_investigation") == "demagnetization_only"
        and has_named_carrier(payload)
        and bridge_type(payload) == "explicit_carrier_assignment"
        and status(payload, "evidence_family") == "same_section_or_figure"
        and no_major_transfer_risk(payload)
        and explicit_carrier_bridge_text(payload)
    ):
        upgrade_rules.append("structured_v7: protected demagnetization-derived named carrier when the source explicitly states the remanence is carried by that phase")

    if (
        score == 0
        and source_status(payload) in {"damaged", "partial", "supplement_dependent"}
        and target_not_bad(payload)
        and has_present_abc(payload)
        and has_named_carrier(payload)
        and bridge_type(payload) == "primary_or_secondary_origin"
        and status(payload, "evidence_family") == "same_section_or_figure"
        and not risk(payload, "evaluator_inference_risk")
        and not risk(payload, "separate_evidence_family_risk")
        and quote(payload, "carrier_identity")
        and quote(payload, "remanence_use_bridge")
    ):
        upgrade_rules.append("structured_v7: protected damaged or partial sources when quoted same-chain A/B/C evidence is complete")

    if (
        score == 0
        and source_status(payload) == "complete"
        and status(payload, "target_anchor") == "group_inferred"
        and has_present_abc(payload)
        and has_named_carrier(payload)
        and status(payload, "evidence_family") == "same_section_or_figure"
        and no_major_transfer_risk(payload)
    ):
        upgrade_rules.append("structured_v7: protected complete group-inferred targets when named carrier and remanence-use evidence are in the same evidence chain")

    if (
        score == 0
        and source_status(payload) == "complete"
        and status(payload, "target_anchor") == "present"
        and status(payload, "carrier_investigation") == "demagnetization_only"
        and carrier_identity_type(payload) == "coercivity_or_unblocking_class"
        and bridge_type(payload) == "overprint_or_remagnetization"
        and status(payload, "evidence_family") == "same_section_or_figure"
        and not risk(payload, "multi_target_transfer_risk")
        and not risk(payload, "separate_evidence_family_risk")
    ):
        upgrade_rules.append("structured_v7: protected target-specific coercivity or unblocking evidence tied to an overprint/remagnetization interpretation")

    if (
        score == 0
        and source_status(payload) == "complete"
        and status(payload, "target_anchor") == "group_inferred"
        and has_present_abc(payload)
        and has_named_carrier(payload)
        and bridge_type(payload) == "primary_or_secondary_origin"
        and status(payload, "evidence_family") == "separate_stitched"
        and risk(payload, "multi_target_transfer_risk")
        and risk(payload, "evaluator_inference_risk")
        and risk(payload, "separate_evidence_family_risk")
        and has_any_text(
            payload,
            (
                "primary origin",
                "primary magnetization",
                "primary magnetisation",
                "primary remanence",
                "remagnetized",
                "remagnetisation",
                "remagnetization",
                "carrier of chrm",
                "main carrier of chrm",
            ),
        )
    ):
        upgrade_rules.append("structured_v7: protected group targets with source-text primary/remagnetization wording despite stitched evidence families")

    if (
        score == 0
        and source_status(payload) == "complete"
        and status(payload, "target_anchor") == "group_inferred"
        and status(payload, "carrier_investigation") == "present"
        and carrier_method_type(payload) == "rock_magnetic"
        and has_named_carrier(payload)
        and status(payload, "remanence_use_bridge") in {"inferred_only", "absent", "unclear"}
        and status(payload, "evidence_family") == "same_paper_only"
        and risk(payload, "multi_target_transfer_risk")
        and risk(payload, "evaluator_inference_risk")
        and risk(payload, "separate_evidence_family_risk")
        and has_any_text(payload, ("chrm", "characteristic"))
        and has_any_text(payload, ("thermomagnetic", "rock magnetic", "susceptibility", "hysteresis", "irm"))
    ):
        upgrade_rules.append("structured_v7: protected rock-magnetic group evidence when ChRM language and magnetic-mineral methods co-occur in the same paper")

    if (
        score == 0
        and source_status(payload) == "complete"
        and status(payload, "target_anchor") == "present"
        and status(payload, "carrier_investigation") == "demagnetization_only"
        and carrier_identity_type(payload) == "domain_state"
        and status(payload, "remanence_use_bridge") == "directional_only"
        and bridge_type(payload) == "primary_or_secondary_origin"
        and status(payload, "evidence_family") == "separate_stitched"
        and risk(payload, "demagnetization_only_risk")
        and risk(payload, "evaluator_inference_risk")
        and risk(payload, "separate_evidence_family_risk")
        and has_any_text(payload, ("chrm", "primary", "fold test", "hematite range", "haematite range"))
    ):
        upgrade_rules.append("structured_v7: protected domain-state demagnetization evidence when source text ties it to ChRM primary/fold-test interpretation")

    if score == 0 and source_quality_mineralogy_recovery(payload):
        upgrade_rules.append(
            "structured_v7: recovered source-quality-limited mineralogy evidence when rock-magnetic/mineralogical terms and ChRM/remanence language are present but the extracted A/B/C chain is incomplete"
        )

    if upgrade_rules:
        score = 1
        manual_review = True
        adjustments.extend(upgrade_rules)

    downgrade_rules: list[str] = []
    if (
        score == 1
        and status(payload, "carrier_investigation") == "demagnetization_only"
        and carrier_method_type(payload) == "thermal_af_cleaning_only"
        and carrier_identity_type(payload) == "coercivity_or_unblocking_class"
        and bridge_type(payload) == "component_isolation"
        and not remanent_coercivity_cleaning(payload)
        and not explicit_carrier_bridge_text(payload)
        and (
            not risk(payload, "directional_substitution_risk")
            or has_any_text(
                payload,
                ("low coercivity", "higher than 25 mt", "hematite range", "haematite range", "hig1"),
            )
        )
    ):
        downgrade_rules.append(
            "structured_v7: downgraded thermal/AF-only coercivity component isolation when no remanent-coercivity or explicit carrier bridge is source-backed"
        )

    if (
        score == 1
        and source_status(payload) in {"damaged", "partial"}
        and bridge_type(payload) == "component_isolation"
        and status(payload, "evidence_family") == "separate_stitched"
        and status(payload, "carrier_identity") == "present"
        and (risk(payload, "evaluator_inference_risk") or risk(payload, "separate_evidence_family_risk"))
    ):
        downgrade_rules.append(
            "structured_v7: downgraded damaged or partial component-isolation positives when carrier and bridge evidence are stitched by inference"
        )

    if downgrade_rules:
        score = 0
        manual_review = True
        adjustments.extend(downgrade_rules)

    return score, manual_review, adjustments


def apply_policy_v8(
    score: int,
    manual_review: bool,
    row: dict[str, Any],
    payload: dict[str, Any],
    adjustments: list[str],
) -> tuple[int, bool, list[str]]:
    """Apply full-set-tested v8 generalizations after the v7 forward test."""

    score, manual_review, adjustments = apply_policy_v7(
        score,
        manual_review,
        row,
        payload,
        adjustments,
    )

    if (
        score == 0
        and source_status(payload) != "missing"
        and target_not_bad(payload)
        and status(payload, "carrier_investigation") == "present"
        and carrier_method_type(payload) == "rock_magnetic"
        and status(payload, "carrier_identity") == "present"
        and status(payload, "remanence_use_bridge") == "present"
        and bridge_type(payload) != "directional_age_validation"
        and status(payload, "carrier_identity") != "wrong_target"
        and status(payload, "remanence_use_bridge") != "wrong_target"
        and status(payload, "evidence_family") != "wrong_target"
    ):
        score = 1
        manual_review = True
        adjustments.append(
            "structured_v8: protected independent rock-magnetic carrier investigation when carrier identity and remanence-use bridge are both source-backed"
        )

    if (
        score == 1
        and bridge_type(payload) == "component_isolation"
        and carrier_method_type(payload) == "thermal_af_cleaning_only"
        and status(payload, "carrier_identity") == "present"
        and not sd_magnetite_component_protection(payload)
    ):
        score = 0
        manual_review = True
        adjustments.append(
            "structured_v8: downgraded thermal/AF-cleaning-only component isolation unless the source text gives a strong SD-magnetite component protection"
        )

    return score, manual_review, adjustments


def apply_policy_v9(
    score: int,
    manual_review: bool,
    row: dict[str, Any],
    payload: dict[str, Any],
    adjustments: list[str],
) -> tuple[int, bool, list[str]]:
    """Apply narrow full-set v9 protected-positive rules after v8."""

    score, manual_review, adjustments = apply_policy_v8(
        score,
        manual_review,
        row,
        payload,
        adjustments,
    )

    upgrade_rules: list[str] = []
    if score == 0 and unblocking_component_chain(payload):
        upgrade_rules.append(
            "structured_v9: protected demagnetization-derived unblocking-temperature component chains when source text ties them to ChRM/HTC/LTC/VRM or remagnetization interpretation"
        )

    if score == 0 and overprint_petrography_bridge(payload):
        upgrade_rules.append(
            "structured_v9: protected microscopy/petrography carrier evidence when secondary minerals are explicitly used to interpret remagnetization or overprint"
        )

    if score == 0 and same_section_named_mineral_component_context(payload):
        upgrade_rules.append(
            "structured_v9: protected same-section named-mineral evidence when ChRM/component context is present but the extractor left the bridge status unclear"
        )

    if score == 0 and explicit_crossref_primary_named_carrier(payload):
        upgrade_rules.append(
            "structured_v9: protected explicit-cross-reference primary-remanence evidence when a named carrier and ChRM/primary context are source-backed"
        )

    if upgrade_rules:
        score = 1
        manual_review = True
        adjustments.extend(upgrade_rules)

    return score, manual_review, adjustments


def apply_policy_metadata_v1(
    score: int,
    manual_review: bool,
    row: dict[str, Any],
    payload: dict[str, Any],
    adjustments: list[str],
) -> tuple[int, bool, list[str]]:
    """Use RESULTNO-specific PMAGRESULT.ROCKMAG as the primary R3 evidence layer."""

    rockmag = metadata_rockmag(payload)
    if metadata_rockmag_valid(payload):
        score = 1
        manual_review = False
        adjustments.append(
            "metadata_v1: accepted RESULTNO-specific PMAGRESULT.ROCKMAG diagnostic/carrier evidence"
        )
        return score, manual_review, adjustments

    if metadata_rockmag_status(payload) == "weak":
        adjustments.append(
            "metadata_v1: PMAGRESULT.ROCKMAG is non-empty but weak/unclassified, so it is not accepted as R3 evidence"
        )
    else:
        adjustments.append(
            "metadata_v1: PMAGRESULT.ROCKMAG is absent, so no source-metadata R3 evidence was recovered"
        )
    score = 0
    manual_review = bool(
        source_status(payload) in {"missing", "damaged"}
        or status(payload, "target_anchor") in {"unclear", "absent", "wrong_target", "group_inferred"}
        or risk(payload, "multi_target_transfer_risk")
    )
    return score, manual_review, adjustments


def arbitrate(row: dict[str, Any], policy: str = "v1") -> dict[str, Any]:
    payload = parse_payload(row["STRUCTURED_EVIDENCE"])
    a_ok, a_text = is_a_satisfied(payload)
    b_ok, b_text = is_b_satisfied(payload)
    c_ok, c_text = is_c_satisfied(payload)
    t_ok, t_text = target_ok(payload)
    f_ok, f_text = family_ok(payload)
    blockers = risk_blockers(payload, a_ok, b_ok, c_ok)

    manual_review = bool(payload.get("manual_review") or payload.get("high_risk_route") or blockers)
    digest = {
        "source_quality": source_status(payload),
        "target_anchor_status": status(payload, "target_anchor"),
        "carrier_investigation_status": status(payload, "carrier_investigation"),
        "carrier_identity_status": status(payload, "carrier_identity"),
        "remanence_use_bridge_status": status(payload, "remanence_use_bridge"),
        "evidence_family_status": status(payload, "evidence_family"),
        "metadata_rockmag_status": metadata_rockmag_status(payload),
        "metadata_rockmag_valid": metadata_rockmag_valid(payload),
        "metadata_rockmag_code": str(metadata_rockmag(payload).get("rockmag_code", "") or ""),
        "source_text_review_hit_counts": source_text_review_hits(payload).get("counts", {}),
        "high_risk_route": bool(payload.get("high_risk_route")),
    }

    adjustments = [
        "structured_v1: target_anchor=" + t_text,
        "structured_v1: " + f_text,
    ]
    if blockers:
        adjustments.extend("structured_v1_blocker: " + blocker for blocker in blockers)

    score = int(a_ok and b_ok and c_ok and t_ok and f_ok and not blockers)
    source_gate_score = score
    if risk(payload, "directional_substitution_risk") and score:
        adjustments.append("structured_v1: directional risk noted but carrier bridge is source-backed")
    if policy == "v2":
        score, manual_review, adjustments = apply_policy_v2(
            score,
            manual_review,
            payload,
            adjustments,
        )
    elif policy == "v3":
        score, manual_review, adjustments = apply_policy_v3(
            score,
            manual_review,
            payload,
            adjustments,
        )
    elif policy == "v4":
        score, manual_review, adjustments = apply_policy_v4(
            score,
            manual_review,
            payload,
            adjustments,
        )
    elif policy == "v5":
        score, manual_review, adjustments = apply_policy_v5(
            score,
            manual_review,
            row,
            payload,
            adjustments,
        )
    elif policy == "v6":
        score, manual_review, adjustments = apply_policy_v6(
            score,
            manual_review,
            row,
            payload,
            adjustments,
        )
    elif policy == "v7":
        score, manual_review, adjustments = apply_policy_v7(
            score,
            manual_review,
            row,
            payload,
            adjustments,
        )
    elif policy == "v8":
        score, manual_review, adjustments = apply_policy_v8(
            score,
            manual_review,
            row,
            payload,
            adjustments,
        )
    elif policy == "v9":
        score, manual_review, adjustments = apply_policy_v9(
            score,
            manual_review,
            row,
            payload,
            adjustments,
        )
    elif policy == "metadata_v1":
        score, manual_review, adjustments = apply_policy_metadata_v1(
            score,
            manual_review,
            row,
            payload,
            adjustments,
        )
    elif policy != "v1":
        raise ValueError(f"unknown policy: {policy}")

    if score:
        if policy == "metadata_v1" and metadata_rockmag_valid(payload):
            reasoning = (
                "R3=1 because RESULTNO-specific PMAGRESULT.ROCKMAG records diagnostic "
                "rock-magnetic, petrographic, or carrier-mineral evidence."
            )
        else:
            reasoning = "A/B/C are source-backed, target-anchored, and in a connected evidence family."
    else:
        failed = []
        if not a_ok:
            failed.append("A")
        if not b_ok:
            failed.append("B")
        if not c_ok:
            failed.append("C")
        if not t_ok:
            failed.append("target_anchor")
        if not f_ok:
            failed.append("evidence_family")
        if blockers:
            failed.append("risk_blockers")
        if policy == "metadata_v1":
            reasoning = (
                "R3=0 because RESULTNO-specific PMAGRESULT.ROCKMAG did not contain "
                "accepted diagnostic/carrier evidence."
            )
        else:
            reasoning = "R3=0 because " + ", ".join(failed) + " failed the structured gate."

    resultno = str(row["RESULTNO"])
    target = str(row["TARGET"])
    evidence_a = source_evidence_block(
        payload,
        "carrier_investigation",
        "rock_magnetic_method",
        a_ok,
        a_text,
    )
    evidence_b = source_evidence_block(
        payload,
        "carrier_identity",
        "magnetic_carrier",
        b_ok,
        b_text,
    )
    evidence_c = source_evidence_block(
        payload,
        "remanence_use_bridge",
        "carrier_remanence_significance",
        c_ok,
        c_text,
    )
    result_payload = {
        "resultno": resultno,
        "target": target,
        "R3": score,
        "evaluator": EVALUATOR,
        "Evidence_A_Carrier_Investigation": evidence_a,
        "Evidence_B_Carrier_Identity": evidence_b,
        "Evidence_C_Carrier_Significance": evidence_c,
        "expert_case_adjustments": adjustments,
        "logic_gate": f"A AND B AND C with target/source risk checks; policy={policy}",
        "final_decision_reasoning": reasoning,
        "summary": f"Structured arbitration {policy} assigned R3={score}.",
        "manual_review": manual_review,
        "decision_basis": "source_metadata" if policy == "metadata_v1" else "original_paper_text",
        "paper_only_source_gate_score": source_gate_score,
        "runtime_scoring_sources": (
            ["PMAGRESULT.ROCKMAG"] if policy == "metadata_v1" else ["original_paper_text"]
        ),
        "structured_evidence_digest": digest,
        "metadata_rockmag": metadata_rockmag(payload),
        "source_text_review_hits": source_text_review_hits(payload),
    }
    return {
        "REFNO": str(row["REFNO"]),
        "RESULTNO": resultno,
        "TARGET": target,
        "R3": score,
        "R3_RESULT_JSON": result_payload,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract_paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-lines", type=int)
    parser.add_argument(
        "--policy",
        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "metadata_v1"],
        default="v1",
    )
    args = parser.parse_args()

    output_rows: list[dict[str, Any]] = []
    for path in args.extract_paths:
        output_rows.extend(arbitrate(row, policy=args.policy) for row in load_jsonl(path))
    if args.expected_lines is not None and len(output_rows) != args.expected_lines:
        raise SystemExit(f"expected {args.expected_lines} row(s), found {len(output_rows)}")

    seen: set[str] = set()
    for row in output_rows:
        resultno = row["RESULTNO"]
        if resultno in seen:
            raise SystemExit(f"duplicate RESULTNO: {resultno}")
        seen.add(resultno)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.casefold() == ".xlsx":
        excel_rows = []
        for row in output_rows:
            excel_rows.append(
                {
                    **row,
                    "R3_RESULT_JSON": json.dumps(row["R3_RESULT_JSON"], ensure_ascii=False),
                }
            )
        pd.DataFrame(excel_rows).to_excel(args.output, index=False)
    else:
        with args.output.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "policy": args.policy,
                "rows": len(output_rows),
                "score_counts": {
                    "0": sum(1 for row in output_rows if int(row["R3"]) == 0),
                    "1": sum(1 for row in output_rows if int(row["R3"]) == 1),
                },
                "manual_review": sum(
                    int(bool(row["R3_RESULT_JSON"].get("manual_review"))) for row in output_rows
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
