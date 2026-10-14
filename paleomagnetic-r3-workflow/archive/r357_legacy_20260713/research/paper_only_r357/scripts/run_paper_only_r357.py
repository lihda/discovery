#!/usr/bin/env python3
"""Evaluate R3/R5/R7 from original-paper text without source metadata tables.

Runtime scoring reads only REFNO, RESULTNO, ROCKNAME and a label-free page
cache. PMAGRESULT, ROCKUNIT and R3/R5/R7 labels are not scoring inputs. When a
label workbook is supplied, labels are loaded only after every prediction has
been frozen so the run can be evaluated for research purposes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "research/paper_only_r357/outputs"


@dataclass(frozen=True)
class PatternSpec:
    field: str
    polarity: str
    label: str
    pattern: str
    weight: float


@dataclass
class RawNode:
    field: str
    polarity: str
    label: str
    matched_text: str
    quote: str
    page: int
    base_weight: float
    reference_noise: bool


@dataclass
class EvidenceNode:
    field: str
    polarity: str
    label: str
    matched_text: str
    quote: str
    page: int
    binding: float
    source_reliability: float
    base_weight: float
    score: float
    target_anchor: str
    reference_noise: bool


R3_PATTERNS = (
    PatternSpec("r3_method_strong", "support", "IRM or coercivity experiment", r"\bIRM\b|isothermal remanent magneti[sz]ation|back[- ]field|coerciv", 1.0),
    PatternSpec("r3_method_strong", "support", "Lowrie or three-axis experiment", r"Lowrie|three[- ]axis|3[- ]axis", 1.0),
    PatternSpec("r3_method_strong", "support", "thermomagnetic or Curie experiment", r"thermomagnetic|Curie (?:point|temperature)|susceptibility.{0,80}temperature", 1.0),
    PatternSpec("r3_method_strong", "support", "hysteresis or domain-state experiment", r"hysteresis|Day plot|FORC|domain state|single[- ]domain|pseudo[- ]single[- ]domain|multi[- ]domain", 1.0),
    PatternSpec("r3_method_strong", "support", "microscopy or petrography", r"reflected[- ]light|ore microscop|thin section|petrograph|\bSEM\b|\bTEM\b|electron microprobe|\bEPMA\b|opaque mineral", 0.95),
    PatternSpec("r3_method_demag", "support", "demagnetization or unblocking experiment", r"unblocking temperature|blocking temperature|thermal demagneti[sz]ation|AF demagneti[sz]ation|alternating field demagneti[sz]ation", 0.55),
    PatternSpec("r3_carrier", "support", "named magnetic carrier", r"magnetite|titanomagnetite|titano[- ]magnetite|hematite|haematite|maghemite|goethite|pyrrhotite|greigite|Fe[- ]Ti oxide", 1.0),
    PatternSpec("r3_bridge", "support", "carrier linked to remanence", r"carr(?:y|ies|ied|ier|iers).{0,180}(?:ChRM|characteristic|remanence|component|magneti[sz]ation)|(?:ChRM|characteristic|remanence|component|magneti[sz]ation).{0,180}carr(?:y|ies|ied|ier|iers)", 1.2),
    PatternSpec("r3_bridge", "support", "carrier linked to primary or secondary origin", r"(?:primary|secondary|overprint|remagneti[sz]|CRM|TRM|DRM|VRM|detrital|chemical|authigenic).{0,220}(?:magnetite|hematite|haematite|carrier|remanence)|(?:magnetite|hematite|haematite|carrier).{0,220}(?:primary|secondary|overprint|remagneti[sz]|CRM|TRM|DRM|VRM)", 1.05),
    PatternSpec("r3_directional_only", "noise", "directional evidence without carrier link", r"fold test|reversal test|APWP|apparent polar wander|palaeopole|paleopole|primary direction", 0.45),
)

R5_PATTERNS = (
    PatternSpec("structural_control_missing", "oppose", "explicit lack of paleohorizontal control", r"no palaeohorizontal control|no paleohorizontal control", 1.45),
    PatternSpec("structural_control_missing", "oppose", "no structural correction", r"without (?:any )?structural correction|not structurally corrected|no structural correction", 1.35),
    PatternSpec("structural_attitude_unknown", "oppose", "unknown site attitude", r"positions? are not known with certainty|attitudes? (?:are )?not known|bedding (?:is|was) unknown", 1.25),
    PatternSpec("block_rotation", "oppose", "vertical-axis or block rotation", r"vertical[- ]axis rotation|block rotation|rotated clockwise|rotated counter[- ]?clockwise|clockwise rotation|counter[- ]clockwise rotation", 1.0),
    PatternSpec("mobile_setting", "oppose", "allochthonous or translated block", r"allochthonous|translated block|displaced block|noticeably shifted|mobile belt|rift block", 0.85),
    PatternSpec("inclination_correction_risk", "oppose", "inclination shallowing correction", r"inclination shallowing|flattening correction|corrected inclination", 0.75),
    PatternSpec("fold_timing_risk", "oppose", "syn-folding or post-folding magnetization", r"syn[- ]?fold(?:ing)?|post[- ]?fold(?:ing)?|magneti[sz]ation acquired during folding|magneti[sz]ation acquired after folding", 0.65),
    PatternSpec("structural_attitude", "rescue", "bedding strike or dip reported", r"bedding|strikes?\s+[^.;]{0,60}dips?|dip(?:s|ping)?\s+(?:at|between|from|of)?\s*\d{1,2}", 0.65),
    PatternSpec("structural_correction", "rescue", "tilt or structural correction", r"tilt[- ]corrected|tilt correction|structural correction|corrected for bedding|restored to palaeohorizontal|restored to paleohorizontal", 1.05),
    PatternSpec("paleohorizontal", "rescue", "flat or subhorizontal target", r"flat[- ]lying|sub[- ]?horizontal|nearly horizontal|almost horizontal|gently dipping", 0.8),
    PatternSpec("field_test", "rescue", "positive fold or contact test", r"positive (?:fold|contact|baked contact|conglomerate) test|fold test.{0,100}(?:positive|passed)|baked contact test", 1.15),
)

R7_PATTERNS = (
    PatternSpec("younger_pole", "oppose", "pole resembles a younger reference", r"(?:palaeopole|paleopole|pole).{0,220}(?:younger|Bushveld|younger segment of the APWP)|(?:younger|Bushveld).{0,220}(?:palaeopole|paleopole|pole)", 1.15),
    PatternSpec("pole_age_mismatch", "oppose", "pole mismatches contemporaneous poles", r"(?:palaeomagnetic|paleomagnetic) pole does not match contemporaneous poles|does not match (?:the )?(?:coeval|contemporaneous) poles", 1.35),
    PatternSpec("remagnetization", "oppose", "explicit remagnetization or secondary overprint", r"remagneti[sz](?:ed|ation)|secondary (?:component|magneti[sz]ation|remanence)|overprint(?:ed|ing)?", 1.1),
    PatternSpec("present_day_overprint", "oppose", "present-day or viscous overprint", r"present[- ]day overprint|recent overprint|viscous remanent magneti[sz]ation|\bVRM\b", 0.95),
    PatternSpec("age_reset", "oppose", "impact, hydrothermal, metamorphic or thermal reset", r"pseudotachylite|hydrothermal|metamorphic resetting|thermal resetting|reset(?:ting)? of (?:the )?magneti[sz]ation|locally remagnetized after", 0.9),
    PatternSpec("age_ambiguity", "oppose", "multiple or younger age ambiguity", r"older and younger ages|age spread|multi[- ]phase|multiple intrusive episodes|much younger.{0,100}overprints", 0.75),
    PatternSpec("primary_origin", "rescue", "explicit primary magnetization", r"primary (?:remanence|magneti[sz]ation|origin)|primary ChRM|believed to be primary|acquired (?:during|at) (?:cooling|emplacement|deposition)", 1.05),
    PatternSpec("field_test", "rescue", "positive field test", r"positive (?:fold|reversal|baked contact|contact|conglomerate) test|fold test.{0,100}(?:positive|passed)|reversal test.{0,100}(?:positive|passed)", 1.1),
    PatternSpec("independent_age", "rescue", "independent radiometric age", r"U[- ]Pb|baddeleyite|zircon|40Ar/39Ar|Ar[- ]Ar|radiometric age|well dated|independent age", 0.55),
    PatternSpec("contemporaneous_match", "rescue", "pole agrees with contemporaneous reference", r"(?:consistent|agrees?|matches?).{0,160}(?:contemporaneous|coeval|same age).{0,100}(?:pole|poles|APWP)|(?:pole|poles).{0,120}(?:consistent|agree|match).{0,120}(?:contemporaneous|coeval)", 0.95),
    PatternSpec("stable_component", "rescue", "stable characteristic component", r"characteristic remanent magneti[sz]ation|\bChRM\b|stable component|high[- ]temperature component", 0.45),
)

ATTRIBUTE_PATTERNS = (
    PatternSpec("component", "attribute", "magnetization component", r"(?:component|components)\s+[A-Z](?:\s+and\s+[A-Z])?|high[- ]temperature component|low[- ]temperature component|ChRM", 0.0),
    PatternSpec("age_mention", "attribute", "numerical geological age", r"\b\d{1,4}(?:\.\d+)?\s*(?:Ma|Myr|Ga|m\.y\.|million years)\b", 0.0),
    PatternSpec("rock_type", "attribute", "rock or unit type", r"basalt|lava|tuff|ignimbrite|sandstone|limestone|dolomite|shale|red bed|redbed|dyke|dike|sill|dolerite|diabase|granite|gabbro|gneiss|schist", 0.0),
)

ALL_PATTERNS = R3_PATTERNS + R5_PATTERNS + R7_PATTERNS + ATTRIBUTE_PATTERNS
COMPILED = [(spec, re.compile(spec.pattern, re.I | re.S)) for spec in ALL_PATTERNS]

STOP_TOKENS = {
    "formation", "group", "member", "rocks", "rock", "beds", "series", "suite", "complex",
    "combined", "mean", "data", "component", "components", "dykes", "dikes", "basalts",
    "granites", "volcanics", "intrusives", "sediments", "sandstones", "limestones",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def target_tokens(target: str) -> list[str]:
    tokens = [token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", target)]
    return list(dict.fromkeys(token for token in tokens if token not in STOP_TOKENS and len(token) >= 4))


def source_reliability(source: dict[str, Any], pages: list[dict[str, Any]]) -> float:
    if not pages:
        return 0.0
    flags = set(source.get("source_quality_flags") or [])
    value = 0.92
    if "glyph_or_cid_text" in flags:
        value -= 0.28
    if "short_text" in flags or "possibly_incomplete_extraction" in flags:
        value -= 0.18
    if "no_extractable_text" in flags:
        value = 0.1
    return max(0.1, min(1.0, value))


def excerpt(text: str, start: int, end: int, radius: int = 430) -> str:
    return clean(text[max(0, start - radius) : min(len(text), end + radius)])[:1200]


def is_reference_noise(page_text: str, position: int) -> bool:
    before = page_text[:position].casefold()
    last_reference = max(before.rfind("references"), before.rfind("bibliography"))
    return last_reference >= 0 and position - last_reference < 12000


def collect_raw_nodes(cache_row: dict[str, Any]) -> list[RawNode]:
    nodes: list[RawNode] = []
    seen: set[tuple[str, int, int]] = set()
    for page in cache_row.get("pages") or []:
        page_no = int(page.get("page") or 0)
        text = str(page.get("text") or "")
        for spec, pattern in COMPILED:
            count = 0
            for match in pattern.finditer(text):
                key = (spec.field, page_no, match.start() // 320)
                if key in seen:
                    continue
                seen.add(key)
                nodes.append(
                    RawNode(
                        field=spec.field,
                        polarity=spec.polarity,
                        label=spec.label,
                        matched_text=clean(match.group(0))[:260],
                        quote=excerpt(text, match.start(), match.end()),
                        page=page_no,
                        base_weight=spec.weight,
                        reference_noise=is_reference_noise(text, match.start()),
                    )
                )
                count += 1
                if count >= 5:
                    break
    return nodes


def page_target_anchors(cache_row: dict[str, Any], target: str) -> set[int]:
    target_norm = normalized(target)
    tokens = target_tokens(target)
    anchors: set[int] = set()
    for page in cache_row.get("pages") or []:
        text = normalized(str(page.get("text") or ""))
        token_hits = sum(token in text for token in tokens)
        if target_norm and target_norm in text:
            anchors.add(int(page.get("page") or 0))
        elif tokens and token_hits >= max(1, math.ceil(len(tokens) * 0.6)):
            anchors.add(int(page.get("page") or 0))
    return anchors


def binding_score(node: RawNode, target: str, target_pages: set[int], multi_target: bool) -> tuple[float, str]:
    quote_norm = normalized(node.quote)
    target_norm = normalized(target)
    tokens = target_tokens(target)
    hits = sum(token in quote_norm for token in tokens)
    if target_norm and target_norm in quote_norm:
        return 1.0, "exact_target"
    if tokens and hits >= max(1, math.ceil(len(tokens) * 0.6)):
        return 0.9, "target_token_majority"
    if tokens and hits:
        return 0.68, "partial_target_tokens"
    if node.page in target_pages:
        return (0.62 if multi_target else 0.78), "same_target_page"
    if not multi_target:
        return 0.64, "single_target_article"
    return 0.18, "unanchored_multi_target"


def bind_nodes(
    raw_nodes: list[RawNode],
    cache_row: dict[str, Any],
    target: str,
    multi_target: bool,
) -> list[EvidenceNode]:
    reliability = source_reliability(cache_row.get("source") or {}, cache_row.get("pages") or [])
    target_pages = page_target_anchors(cache_row, target)
    bound: list[EvidenceNode] = []
    for raw in raw_nodes:
        binding, anchor = binding_score(raw, target, target_pages, multi_target)
        noise_factor = 0.18 if raw.reference_noise else 1.0
        score = raw.base_weight * binding * reliability * noise_factor
        bound.append(
            EvidenceNode(
                **asdict(raw),
                binding=round(binding, 4),
                source_reliability=round(reliability, 4),
                score=round(score, 4),
                target_anchor=anchor,
            )
        )
    return sorted(bound, key=lambda node: node.score, reverse=True)


def top_nodes(nodes: Iterable[EvidenceNode], field: str, limit: int = 4) -> list[EvidenceNode]:
    return [node for node in nodes if node.field == field][:limit]


def polarity_nodes(nodes: Iterable[EvidenceNode], fields: set[str], polarity: str, limit: int = 6) -> list[EvidenceNode]:
    return [node for node in nodes if node.field in fields and node.polarity == polarity][:limit]


def unique_score(nodes: list[EvidenceNode], limit: int = 3) -> float:
    used: set[str] = set()
    total = 0.0
    for node in nodes:
        if node.field in used:
            continue
        used.add(node.field)
        total += node.score
        if len(used) >= limit:
            break
    return round(total, 4)


def compact_nodes(nodes: list[EvidenceNode], limit: int = 5) -> list[dict[str, Any]]:
    return [asdict(node) for node in nodes[:limit]]


def recover_fields(target: str, nodes: list[EvidenceNode]) -> dict[str, Any]:
    attributes: dict[str, list[dict[str, Any]]] = {}
    for field in (
        "rock_type", "component", "age_mention", "r3_method_strong", "r3_method_demag", "r3_carrier",
        "structural_attitude", "structural_correction", "paleohorizontal", "field_test",
        "younger_pole", "pole_age_mismatch", "remagnetization", "primary_origin", "independent_age",
    ):
        attributes[field] = compact_nodes(top_nodes(nodes, field, 3), 3)
    inferred_rock_type = "unclear"
    sample = target.casefold()
    for label, pattern in (
        ("volcanic", r"basalt|lava|tuff|ignimbrite|rhyolite|andesite|volcan"),
        ("sedimentary", r"sandstone|limestone|dolomite|shale|red bed|redbed|sediment"),
        ("intrusive", r"dyke|dike|sill|dolerite|diabase|granite|gabbro|pluton|intrus"),
        ("metamorphic", r"gneiss|schist|slate|phyllite|metamorph"),
    ):
        if re.search(pattern, sample):
            inferred_rock_type = label
            break
    return {
        "target_name": target,
        "inferred_rock_type": inferred_rock_type,
        "attributes": attributes,
        "provenance": "original paper page cache only",
    }


def judge_r3(
    nodes: list[EvidenceNode],
    multi_target: bool,
    identity_ambiguous: bool,
    source_available: bool,
) -> dict[str, Any]:
    strong = top_nodes(nodes, "r3_method_strong", 4)
    demag = top_nodes(nodes, "r3_method_demag", 3)
    carrier = top_nodes(nodes, "r3_carrier", 4)
    bridge = top_nodes(nodes, "r3_bridge", 4)
    method = strong[0] if strong else demag[0] if demag else None
    carrier_top = carrier[0] if carrier else None
    bridge_top = bridge[0] if bridge else None
    a_ok = bool(method and method.score >= (0.5 if method.field == "r3_method_strong" else 0.34))
    b_ok = bool(carrier_top and carrier_top.score >= 0.48)
    c_ok = bool(bridge_top and bridge_top.score >= 0.55)
    selected = [node for node in (method, carrier_top, bridge_top) if node]
    page_counts = Counter(node.page for node in selected)
    linked = bool(page_counts and max(page_counts.values()) >= 2)
    strongly_bound = bool(selected and all(node.binding >= 0.62 for node in selected))
    score = int(a_ok and b_ok and c_ok and (not multi_target or linked or strongly_bound))
    directional = top_nodes(nodes, "r3_directional_only", 2)
    manual = (
        not source_available
        or identity_ambiguous
        or (a_ok and b_ok and not c_ok)
        or (multi_target and score == 0 and bool(selected))
    )
    return {
        "score": score,
        "conditions": {"carrier_investigation": a_ok, "carrier_identity": b_ok, "remanence_bridge": c_ok},
        "evidence_family_linked": linked,
        "support": compact_nodes(selected, 4),
        "noise": compact_nodes(directional, 2),
        "manual_review": manual,
        "identity_ambiguous": identity_ambiguous,
        "reason": "R3=1 requires source-backed investigation, carrier identity, and a carrier-remanence bridge.",
    }


def astute_binary_judgement(
    nodes: list[EvidenceNode],
    fields: set[str],
    default: int,
    oppose_threshold: float,
    margin: float,
    min_override_binding: float,
    override_fields: set[str],
    identity_ambiguous: bool,
    source_available: bool,
) -> dict[str, Any]:
    oppose = polarity_nodes(nodes, fields, "oppose")
    rescue = polarity_nodes(nodes, fields, "rescue")
    oppose_score = unique_score(oppose)
    rescue_score = unique_score(rescue)
    anchored_oppose = [node for node in oppose if node.binding >= 0.62 and not node.reference_noise]
    anchored_rescue = [node for node in rescue if node.binding >= 0.62 and not node.reference_noise]
    override_oppose = [
        node
        for node in anchored_oppose
        if node.binding >= min_override_binding and node.field in override_fields
    ]
    conflict = bool(anchored_oppose and anchored_rescue and abs(oppose_score - rescue_score) < 0.55)
    filtered_noise = [node for node in oppose + rescue if node.binding < 0.5 or node.reference_noise]
    if (
        not identity_ambiguous
        and override_oppose
        and oppose_score >= oppose_threshold
        and oppose_score - rescue_score >= margin
    ):
        score = 0
        resolution = "target-bound external conflict overrides internal default"
    else:
        score = default
        resolution = "internal default retained after source-aware consolidation"
    manual = not source_available or identity_ambiguous or conflict or bool(filtered_noise and anchored_oppose)
    return {
        "score": score,
        "source_aware_consolidation": {
            "internal_candidate": default,
            "external_oppose_score": oppose_score,
            "external_rescue_score": rescue_score,
            "conflict": conflict,
            "identity_ambiguous": identity_ambiguous,
            "resolution": resolution,
            "support_or_rescue": compact_nodes(anchored_rescue, 4),
            "opposition": compact_nodes(anchored_oppose, 4),
            "override_eligible_opposition": compact_nodes(override_oppose, 4),
            "filtered_noise": compact_nodes(filtered_noise, 4),
        },
        "manual_review": manual,
    }


def load_targets(path: Path) -> pd.DataFrame:
    return pd.read_excel(
        path,
        usecols=["REFNO", "RESULTNO", "ROCKNAME"],
        dtype={"REFNO": str, "RESULTNO": str, "ROCKNAME": str},
    ).fillna("")


def load_page_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                cache[str(row["REFNO"]).strip()] = row
    return cache


def latest_page_cache() -> Path:
    candidates = sorted(
        ROOT.glob("outputs/**/r3_page_source_cache.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit("No r3_page_source_cache.jsonl found under outputs/")
    return candidates[0]


def predict_all(targets_path: Path, page_cache_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    targets = load_targets(targets_path)
    if limit is not None:
        targets = targets.head(limit).copy()
    cache = load_page_cache(page_cache_path)
    ref_counts = targets.groupby("REFNO")["RESULTNO"].nunique().to_dict()
    target_identity_counts = targets.groupby(["REFNO", "ROCKNAME"])["RESULTNO"].nunique().to_dict()
    raw_cache: dict[str, list[RawNode]] = {}
    rows: list[dict[str, Any]] = []
    r5_fields = {spec.field for spec in R5_PATTERNS}
    r7_fields = {spec.field for spec in R7_PATTERNS}
    for source_row in targets.to_dict(orient="records"):
        refno = clean(source_row["REFNO"])
        resultno = clean(source_row["RESULTNO"])
        target = clean(source_row["ROCKNAME"])
        cache_row = cache.get(refno, {"REFNO": refno, "source": {"source_quality_flags": ["missing"]}, "pages": []})
        if refno not in raw_cache:
            raw_cache[refno] = collect_raw_nodes(cache_row)
        multi_target = int(ref_counts.get(refno, 1)) > 1
        identity_ambiguous = int(target_identity_counts.get((refno, target), 1)) > 1
        nodes = bind_nodes(raw_cache[refno], cache_row, target, multi_target)
        source_available = bool(cache_row.get("pages"))
        r3 = judge_r3(nodes, multi_target, identity_ambiguous, source_available)
        r5 = astute_binary_judgement(
            nodes,
            r5_fields,
            1,
            1.15,
            0.35,
            0.9,
            {"structural_control_missing", "structural_attitude_unknown", "block_rotation"},
            identity_ambiguous,
            source_available,
        )
        r7 = astute_binary_judgement(
            nodes,
            r7_fields,
            1,
            1.25,
            0.55,
            0.9,
            {"younger_pole", "pole_age_mismatch", "remagnetization", "present_day_overprint", "age_reset"},
            identity_ambiguous,
            source_available,
        )
        fields = recover_fields(target, nodes)
        rows.append(
            {
                "REFNO": refno,
                "RESULTNO": resultno,
                "TARGET": target,
                "R3": r3["score"],
                "R5": r5["score"],
                "R7": r7["score"],
                "PAPER_FIELDS": fields,
                "R3_JUDGEMENT": r3,
                "R5_JUDGEMENT": r5,
                "R7_JUDGEMENT": r7,
                "SOURCE_AUDIT": {
                    "page_cache": str(page_cache_path),
                    "source": cache_row.get("source") or {},
                    "multi_target_refno": multi_target,
                    "identity_ambiguous": identity_ambiguous,
                    "runtime_columns": ["REFNO", "RESULTNO", "ROCKNAME"],
                    "forbidden_scoring_sources": ["PMAGRESULT", "ROCKUNIT", "R3", "R5", "R7"],
                },
            }
        )
    return rows


def f1_for(gold: list[int], pred: list[int], label: int) -> float:
    tp = sum(g == label and p == label for g, p in zip(gold, pred))
    fp = sum(g != label and p == label for g, p in zip(gold, pred))
    fn = sum(g == label and p != label for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate(rows: list[dict[str, Any]], labels_path: Path | None) -> dict[str, Any]:
    if labels_path is None:
        return {}
    labels = pd.read_excel(labels_path, usecols=["RESULTNO", "R3", "R5", "R7"], dtype=str).fillna("")
    lookup = labels.set_index("RESULTNO").to_dict(orient="index")
    results: dict[str, Any] = {}
    for indicator in ("R3", "R5", "R7"):
        gold: list[int] = []
        pred: list[int] = []
        reviewed: list[bool] = []
        for row in rows:
            value = clean(lookup.get(row["RESULTNO"], {}).get(indicator))
            if value not in {"0", "1", "0.0", "1.0"}:
                continue
            gold.append(int(float(value)))
            pred.append(int(row[indicator]))
            reviewed.append(bool(row[f"{indicator}_JUDGEMENT"].get("manual_review")))
        tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
        tn = sum(g == 0 and p == 0 for g, p in zip(gold, pred))
        fp = sum(g == 0 and p == 1 for g, p in zip(gold, pred))
        fn = sum(g == 1 and p == 0 for g, p in zip(gold, pred))
        accepted = [(g, p) for g, p, review in zip(gold, pred, reviewed) if not review]
        review_errors = sum(g != p and review for g, p, review in zip(gold, pred, reviewed))
        total_errors = fp + fn
        results[indicator] = {
            "rows": len(gold),
            "accuracy": (tp + tn) / len(gold) if gold else None,
            "macro_f1": (f1_for(gold, pred, 0) + f1_for(gold, pred, 1)) / 2 if gold else None,
            "f1_0": f1_for(gold, pred, 0),
            "f1_1": f1_for(gold, pred, 1),
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "pred_counts": dict(Counter(pred)),
            "gold_counts": dict(Counter(gold)),
            "manual_review_rows": sum(reviewed),
            "manual_review_rate": sum(reviewed) / len(reviewed) if reviewed else 0.0,
            "accepted_accuracy": sum(g == p for g, p in accepted) / len(accepted) if accepted else None,
            "error_capture_by_review": review_errors / total_errors if total_errors else 0.0,
        }
    return results


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paper_only_r357_evidence.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                "REFNO": row["REFNO"],
                "RESULTNO": row["RESULTNO"],
                "TARGET": row["TARGET"],
                "R3": row["R3"],
                "R5": row["R5"],
                "R7": row["R7"],
                "R3_MANUAL_REVIEW": row["R3_JUDGEMENT"]["manual_review"],
                "R5_MANUAL_REVIEW": row["R5_JUDGEMENT"]["manual_review"],
                "R7_MANUAL_REVIEW": row["R7_JUDGEMENT"]["manual_review"],
                "PAPER_FIELDS_JSON": json.dumps(row["PAPER_FIELDS"], ensure_ascii=False),
                "R3_RESULT_JSON": json.dumps(row["R3_JUDGEMENT"], ensure_ascii=False),
                "R5_RESULT_JSON": json.dumps(row["R5_JUDGEMENT"], ensure_ascii=False),
                "R7_RESULT_JSON": json.dumps(row["R7_JUDGEMENT"], ensure_ascii=False),
            }
        )
    pd.DataFrame(flat_rows).to_excel(output_dir / "paper_only_r357_predictions.xlsx", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Paper-Only R357 Exploration",
        "",
        f"Generated: {summary['created_at']}",
        "",
        "## Leakage Boundary",
        "",
        "- Scoring reads only REFNO, RESULTNO, ROCKNAME and original-paper page text.",
        "- PMAGRESULT and ROCKUNIT are not imported or read.",
        "- R3/R5/R7 labels are loaded only after predictions are frozen, for evaluation.",
        "",
        "## Metrics",
        "",
        "| Indicator | Accuracy | Macro-F1 | F1(0) | FP | FN | Review rate | Accepted accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for indicator, values in summary.get("evaluation", {}).items():
        accepted = values["accepted_accuracy"]
        lines.append(
            f"| {indicator} | {values['accuracy']:.4f} | {values['macro_f1']:.4f} | {values['f1_0']:.4f} | "
            f"{values['false_positive']} | {values['false_negative']} | {values['manual_review_rate']:.4f} | "
            f"{accepted:.4f} |" if accepted is not None else "n/a"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "The workflow recovers domain attributes from paper text, binds evidence to the current target, "
            "and performs Astute-RAG-inspired source-aware consolidation over internal criterion defaults, "
            "external opposition, rescue evidence, and filtered noise.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=ROOT / "data/targets.xlsx")
    parser.add_argument("--page-cache", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/targets.xlsx")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-evaluation", action="store_true")
    args = parser.parse_args()
    page_cache = args.page_cache or latest_page_cache()
    output_dir = args.output_dir or OUTPUT_ROOT / f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rows = predict_all(args.targets, page_cache, args.limit)
    evaluation = evaluate(rows, None if args.no_evaluation else args.labels)
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "method": "paper-only field recovery plus Astute-inspired source-aware conflict consolidation",
        "inputs": {"targets": str(args.targets), "page_cache": str(page_cache)},
        "runtime_input_columns": ["REFNO", "RESULTNO", "ROCKNAME"],
        "runtime_forbidden_sources": ["PMAGRESULT.xlsx", "ROCKUNIT.xlsx", "R3/R5/R7 labels"],
        "prediction_rows": len(rows),
        "evaluation_loaded_after_prediction": not args.no_evaluation,
        "evaluation": evaluation,
    }
    write_outputs(output_dir, rows, summary)
    print(json.dumps({"output_dir": str(output_dir), "evaluation": evaluation}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
