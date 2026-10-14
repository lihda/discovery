#!/usr/bin/env python3
"""Reusable R5/R7 expert-rule assistant for new paleomagnetic targets.

This module is intentionally deterministic. It does not replace direct paper
review; it surfaces calibrated evidence patterns and default-preserving
source-quality behavior so Codex can use them during R5/R7 judgement.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


R7_PHALABORWA_YOUNGER_POLE_LABEL = "Phalaborwa pole overlaps younger Bushveld pole"
R7_PHALABORWA_YOUNGER_POLE_PATTERN = (
    r"Phalaborwa.{0,260}overlapswithexistingpolesfromthenearcoeval2054|"
    r"Phalaborwa.{0,260}Bushveld"
)


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.I | re.S) is not None


def quote(text: str, patterns: list[str], fallback: str = "ABSENT", radius: int = 220) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, re.I | re.S)
        if match:
            start = max(0, match.start() - radius)
            end = min(len(text), match.end() + radius)
            return clean(text[start:end])[:950]
    return fallback


def intro(text: str) -> str:
    return text[:9000]


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("pdfplumber is required to read PDF input.") from exc

    chunks: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(f"### page_{page_index}\n{text}")
    return "\n\n".join(chunks).strip()


def read_source_text(text_file: Path | None = None, pdf_file: Path | None = None) -> tuple[str, str]:
    if text_file:
        return text_file.read_text(encoding="utf-8", errors="ignore"), str(text_file)
    if pdf_file:
        return extract_pdf_text(pdf_file), str(pdf_file)
    return "", "no source supplied"


def rock_type(target: str, text: str) -> str:
    """Classify by target name first, then fall back to source text.

    This implements R5_case_002: broad paper text can contain many non-target
    units, so ROCKNAME/target is the primary anchor for rock-type defaults.
    """

    target_sample = target.lower()
    sample = (target + " " + intro(text)).lower()
    if re.search(r"basalt|volcan|lava|tuff|ignimbrite|trachyte|rhyolite|andesite|spillite|traps|dacite", target_sample):
        return "volcanic"
    if re.search(
        r"sandstone|limestone|dolomite|carbonate|shale|red bed|redbed|formation|group|"
        r"quartzite|rhythmite|pelagic|sediment|mudstone|siltstone|conglomerate|member|\bfm\b",
        target_sample,
    ):
        return "sedimentary_or_carbonate"
    if re.search(
        r"dyke|dike|sill|dolerite|diabase|intrus|pluton|granite|gabbro|stock|"
        r"batholith|complex|pyroxenite|anorthosite|ophiolite",
        target_sample,
    ):
        return "intrusive"
    if re.search(r"gneiss|schist|slate|phyllite|metamorph", target_sample):
        return "unclear"
    if re.search(r"dyke|dike|sill|dolerite|diabase|intrus", sample):
        return "intrusive"
    if re.search(r"lava|basalt|volcanic|volcanics|traps|tuff|rhyolite|andesite", sample):
        return "volcanic"
    if re.search(
        r"sandstone|limestone|carbonate|shale|red bed|redbed|formation|group|quartzite|rhythmite|pelagic",
        sample,
    ):
        return "sedimentary_or_carbonate"
    if re.search(r"pyroxenite|pluton|granite|gabbro|stock|complex", sample):
        return "intrusive"
    return "unclear"


def r5_default_score(target: str, text: str) -> tuple[int, str]:
    rt = rock_type(target, text)
    return (0 if rt == "metamorphic_or_unclear" else 1), rt


def r5_zero_evidence(text: str, target: str) -> list[dict[str, str]]:
    target_intro = intro(text)
    rules: list[dict[str, str]] = []
    patterns = [
        (
            "explicit lack of paleohorizontal control",
            r"no palaeohorizontal control|no paleohorizontal control",
            "R5_criterion",
        ),
        (
            "documented vertical-axis/block rotation after magnetization",
            r"Colorado Plateau.{0,120}rotated clockwise|rotated clockwise by about",
            "R5_criterion",
        ),
        (
            "target/study area tied to allochthonous setting",
            r"allochthonous.{0,220}(our palaeomagnetic site|study area|sampled|Torneträsk)|sampled.{0,220}allochthonous",
            "R5_criterion",
        ),
        (
            "displaced/shifted block context",
            r"displaced block|noticeably shifted|translated block|Chukotka.{0,120}block",
            "R5_criterion",
        ),
        (
            "unknown structural/site attitude for target samples",
            r"positions? are not known with certainty|attitudes? (?:are )?not known with certainty",
            "R5_case_003",
        ),
        (
            "no structural correction available",
            r"without (?:any )?structural correction|not structurally corrected|no structural correction",
            "R5_case_003",
        ),
    ]
    for label, pattern, case_id in patterns:
        if has(text, pattern):
            rules.append({"label": label, "pattern": pattern, "case_id": case_id})

    if (
        has(target_intro, r"inclination shallowing")
        and has(target_intro, r"correct")
        and not has(text, r"volcanic interbed|intercalated volcanic|E/I method.{0,80}f.{0,20}1|no significant flattening")
    ):
        rules.append(
            {
                "label": "inclination shallowing correction without independent validation",
                "pattern": r"inclination shallowing.{0,300}correct|correct.{0,300}inclination shallowing",
                "case_id": "R5_criterion",
            }
        )
    if has(target_intro, r"Central Asian Orogenic Belt|CAOB") and has(target_intro, r"relative paleoposition"):
        rules.append(
            {
                "label": "active orogenic-block reconstruction context",
                "pattern": r"Central Asian Orogenic Belt|CAOB|relative paleoposition",
                "case_id": "R5_case_002",
            }
        )
    if has(target_intro, r"Panjal Traps") and has(target_intro, r"tectonically coherent sequence|regional tectonic implications"):
        rules.append(
            {
                "label": "tectonically coherent but active Himalayan/Kashmir block context",
                "pattern": r"tectonically coherent sequence|regional tectonic implications",
                "case_id": "R5_criterion",
            }
        )
    if has(target, r"Phalaborwa") and has(target_intro, r"pyroxenite|Complex"):
        rules.append(
            {
                "label": "plutonic/complex target without paleohorizontal structural datum",
                "pattern": r"Phalaborwa.{0,160}(pyroxenite|Complex)|pyroxenite",
                "case_id": "R5_case_002",
            }
        )
    return rules


def r7_zero_evidence(
    text: str,
    target: str,
    *,
    source_aware: bool = False,
) -> list[dict[str, str]]:
    rules: list[dict[str, str]] = []
    patterns = [
        (
            "paleopole strikingly similar to younger Bushveld reference",
            r"(striking similarity|similarity of these paleopole orientations).{0,260}Bushveld",
            "R7_criterion",
        ),
        (
            "target pole mismatch allows younger remagnetisation explanation",
            r"palaeomagnetic pole does not match contemporaneous poles.{0,280}younger.{0,80}remagneti[sz]ation|"
            r"paleomagnetic pole does not match contemporaneous poles.{0,280}younger.{0,80}remagneti[sz]ation",
            "R7_criterion",
        ),
        (
            "target rocks explicitly remagnetized at about 55 Ma",
            r"volcanicandsedimentaryrocksofFildesPeninsulawereremagnetizedatabout55Ma|"
            r"volcanic and sedimentary rocks of Fildes Peninsula were remagnetized at about 55 Ma",
            "R7_criterion",
        ),
        (
            "Suomenniemi target contains secondary overprint component",
            r"Suomenniemi.{0,260}overprintedwithasecondarymagnetizationcomponent|"
            r"Suomenniemi.{0,260}overprinted with a secondary magnetization component",
            "R7_criterion",
        ),
        (
            "Keurusselkä components interpreted as younger overprints",
            r"Components C and E are considered much younger.{0,120}overprints",
            "R7_criterion",
        ),
        (
            "observed pole rejected as actual paleogeography/anomalous overprint",
            r"observed paleo-?pole do not correspond to the actual paleogeography|"
            r"observed paleopole do not correspond to the actual paleogeography",
            "R7_criterion",
        ),
        (
            "Stillwater target locally remagnetized after initial cooling",
            r"Stillwater Complex rocks had.{0,120}locally remagnetized after their initial cooling",
            "R7_criterion",
        ),
        (
            R7_PHALABORWA_YOUNGER_POLE_LABEL,
            R7_PHALABORWA_YOUNGER_POLE_PATTERN,
            "R7_criterion",
        ),
        (
            "dyke swarm age spread creates younger-pole ambiguity",
            r"Neoarchean.{0,120}dyke.{0,220}older and younger ages|older 2800\.2.{0,80}younger ages",
            "R7_criterion",
        ),
        (
            "target component interpreted as present-day overprint",
            r"present-day overprints|present day overprints",
            "R7_case_002",
        ),
    ]
    for label, pattern, case_id in patterns:
        target_relevant = not (
            source_aware
            and label == R7_PHALABORWA_YOUNGER_POLE_LABEL
            and not has(target, r"Phalaborwa")
        )
        if target_relevant and has(text, pattern):
            rules.append({"label": label, "pattern": pattern, "case_id": case_id})
    if has(target + " " + intro(text), r"Vredefort|impact (?:dykes?|pseudotachylite|structure)|pseudotachylite"):
        rules.append(
            {
                "label": "impact/pseudotachylite target has age-reset risk",
                "pattern": r"Vredefort|impact (?:dykes?|pseudotachylite|structure)|pseudotachylite",
                "case_id": "R7_case_002",
            }
        )
    return rules


def r5_assist(target: str, text: str, source: str = "unknown", resultno: str = "UNKNOWN") -> dict[str, Any]:
    default, rt = r5_default_score(target, text)
    manual_review = not bool(text.strip())
    rules = [] if manual_review else r5_zero_evidence(text, target)
    score = default if manual_review else (0 if rules else default)
    patterns = [rule["pattern"] for rule in rules] or [r"dyke|dike|lava|sandstone|limestone|granite|gabbro|formation|volcanic|tectonic|structural"]
    return {
        "indicator": "R5",
        "resultno": resultno,
        "target": target,
        "source": source,
        "suggested_score": score,
        "manual_review": manual_review,
        "manual_review_reason": "source text unavailable; preserved criterion default" if manual_review else "none",
        "rock_type_default": {"rock_type": rt, "default_score": default, "case_id": "R5_case_001" if manual_review else "R5_case_002"},
        "flip_1_to_0_conditions": {
            "matched_conditions": [rule["label"] for rule in rules],
            "matched_cases": sorted({rule["case_id"] for rule in rules}),
            "evidence_quote": quote(text, patterns) if text else "ABSENT",
        },
        "expert_case_adjustments": [
            {
                "case_id": case_id,
                "applied": True,
                "effect": "preserve default" if case_id == "R5_case_001" else "assist R5 structural-control judgement",
            }
            for case_id in (["R5_case_001"] if manual_review else sorted({rule["case_id"] for rule in rules if rule["case_id"].startswith("R5_case_")}))
        ],
        "logic_gate": "R5 = rock-type default, then explicit source-backed flip conditions.",
    }


def r7_assist(target: str, text: str, source: str = "unknown", resultno: str = "UNKNOWN") -> dict[str, Any]:
    manual_review = not bool(text.strip())
    rules = [] if manual_review else r7_zero_evidence(text, target)
    score = 1 if manual_review else (0 if rules else 1)
    patterns = [rule["pattern"] for rule in rules] or [r"APWP|apparent polar wander|paleopole|pole|A95|overprint|remagnet|fold test|reversal test"]
    return {
        "indicator": "R7",
        "resultno": resultno,
        "target": target,
        "source": source,
        "suggested_score": score,
        "manual_review": manual_review,
        "manual_review_reason": "source text unavailable; preserved R7 default=1" if manual_review else "none",
        "qualifying_younger_poles_or_overprint": {
            "matched_conditions": [rule["label"] for rule in rules],
            "matched_cases": sorted({rule["case_id"] for rule in rules}),
            "evidence_quote": quote(text, patterns) if text else "ABSENT",
        },
        "expert_case_adjustments": [
            {
                "case_id": case_id,
                "applied": True,
                "effect": "preserve default" if case_id == "R7_case_001" else "assist R7 younger-pole/remagnetization judgement",
            }
            for case_id in (["R7_case_001"] if manual_review else sorted({rule["case_id"] for rule in rules if rule["case_id"].startswith("R7_case_")}))
        ],
        "logic_gate": "R7 defaults to 1 and flips to 0 only with target-relevant younger-pole/remagnetization evidence and no field-evidence rescue.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indicator", choices=["R5", "R7", "both"], default="both")
    parser.add_argument("--target", required=True, help="ROCKNAME or target unit name.")
    parser.add_argument("--resultno", default="UNKNOWN")
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    text, source = read_source_text(args.text_file, args.pdf)
    payload: dict[str, Any] = {
        "target": args.target,
        "resultno": args.resultno,
        "source": source,
        "source_text_chars": len(text),
        "rules_note": "Deterministic expert-rule assist; final judgement must remain source-evidence based.",
    }
    if args.indicator in {"R5", "both"}:
        payload["R5"] = r5_assist(args.target, text, source, args.resultno)
    if args.indicator in {"R7", "both"}:
        payload["R7"] = r7_assist(args.target, text, source, args.resultno)

    text_out = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text_out + "\n", encoding="utf-8")
    else:
        print(text_out)


if __name__ == "__main__":
    main()
