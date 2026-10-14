#!/usr/bin/env python3
"""Target-bound retrieval of R3 and R5 descriptions from paper page text.

The module deliberately depends only on the standard library.  Its scoring
inputs are a page-cache row, a target name, the article's multi-target state,
and an indicator name.  No result-table fields or labels are consulted.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence


__all__ = [
    "PaperTextRetriever",
    "best_r3_connected_chain",
    "canonical_target_identity",
    "find_target_anchor",
    "retrieve_descriptions",
    "retrieve_target_descriptions",
]


@dataclass(frozen=True)
class _PatternSpec:
    category: str
    subtype: str
    label: str
    expression: str
    confidence: float = 1.0


@dataclass(frozen=True)
class _TargetProfile:
    raw: str
    folded: str
    tokens: tuple[str, ...]
    specific_tokens: tuple[str, ...]
    generic_tokens: tuple[str, ...]
    specificity: float
    common_target: bool


@dataclass(frozen=True)
class _BasePage:
    number: int
    text: str
    source: str
    section_hints: tuple[str, ...]
    folded: str
    index_map: tuple[int, ...]
    reference_cutoff: int | None


@dataclass(frozen=True)
class _PageInfo(_BasePage):
    exact_positions: tuple[int, ...]
    token_positions: tuple[int, ...]
    anchor_kind: str


@dataclass(frozen=True)
class _RawPatternMatch:
    spec: _PatternSpec
    page_index: int
    matched_text: str
    reference_noise: bool
    section: str
    quote: str
    start: int
    end: int
    scope: _TextScope


@dataclass(frozen=True)
class _TextScope:
    scope_id: str
    kind: str
    text: str
    folded: str
    bounded: bool
    start: int
    end: int


_R3_NUMBER = r"[~\u2248]?[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
_R3_FIELD_VALUE = _R3_NUMBER + r"\s*(?:mT|T|kOe|Oe)\b"
_R3_TEMPERATURE_VALUE = (
    _R3_NUMBER + r"\s*(?:\u00b0\s*)?(?:C|K)\b"
)
_R3_TRANSITION_RESULT = (
    r"(?:observ(?:e|ed)|detect(?:s|ed)?|show(?:s|ed)?|reveal(?:s|ed)?|"
    r"exhibit(?:s|ed)?|display(?:s|ed)?)[^.!?;\n]{0,80}"
    r"(?:Verwey|Morin)\s+transition|"
    r"(?:Verwey|Morin)\s+transition[^.!?;\n]{0,80}"
    r"(?:(?:was|were|is|are)\s+)?(?:observed|detected|identified|seen|evident|present|absent)"
)


_R3_PATTERNS = (
    _PatternSpec(
        "method",
        "rock_magnetic_experiment",
        "IRM, coercivity, or back-field experiment",
        r"\b(?:IRM|SIRM)\b|isothermal remanent magneti[sz]ation|"
        r"saturation\s+(?:isothermal\s+)?reman(?:ence|ent magneti[sz]ation)|"
        r"(?:IRM|remanence)\s+acquisition\s+curve|back[- ]field|"
        r"coerciv(?:e|ity)|\bHc(?:r)?\b|\bBcr\b",
    ),
    _PatternSpec(
        "method",
        "rock_magnetic_experiment",
        "Lowrie or three-axis experiment",
        r"\bLowrie(?:[- ]Fuller)?\b|three[- ]axis|3[- ]axis",
    ),
    _PatternSpec(
        "method",
        "low_temperature_irm_experiment",
        "low-temperature IRM treatment or transition experiment",
        r"low[- ]temperature.{0,100}(?:IRM|isothermal\s+remanent\s+magneti[sz]ation)|"
        r"(?:IRM|isothermal\s+remanent\s+magneti[sz]ation).{0,100}low[- ]temperature|"
        + _R3_TRANSITION_RESULT,
    ),
    _PatternSpec(
        "method",
        "carrier_spectrum_experiment",
        "unblocking-temperature or coercivity spectrum",
        r"unblocking(?:[- ]temperature)?\s+(?:spectra|spectrum|distribution)|"
        r"coerciv(?:ity|e)\s+(?:spectra|spectrum|distribution)",
    ),
    _PatternSpec(
        "method",
        "thermal_magnetic_experiment",
        "thermomagnetic or Curie-temperature experiment",
        r"thermo[- ]?magnetic|"
        r"susceptibility.{0,90}temperature|temperature.{0,90}susceptibility|"
        r"(?:kappa|chi|[\u03ba\u03c7])[- ]?T\b|\bJ(?:s|rs)[- ]?\(?T\)?\b",
    ),
    _PatternSpec(
        "method",
        "domain_state_experiment",
        "hysteresis or domain-state experiment",
        r"hysteresis|Day\s+plot|Henkel\s+plot|\bFORC\b|domain\s+state|"
        r"single[- ]domain|pseudo[- ]single[- ]domain|multi[- ]domain|\b(?:SD|PSD|MD)\b",
    ),
    _PatternSpec(
        "method",
        "magnetic_fabric_experiment",
        "magnetic fabric or remanence anisotropy experiment",
        r"\bAMS\b|anisotropy\s+of\s+magnetic\s+susceptibility|"
        r"\bAIR\b|anisotropy\s+of\s+isothermal\s+remanence|"
        r"\bAAR\b|anisotropy\s+of\s+anhysteretic\s+remanence|magnetic\s+fabric",
        0.92,
    ),
    _PatternSpec(
        "method",
        "mineralogical_observation",
        "microscopy, petrography, or microprobe observation",
        r"reflected[- ]light|ore\s+microscop|thin\s+section|petrograph|\bSEM\b|\bTEM\b|"
        r"electron\s+microprobe|\bEPMA\b|energy[- ]dispersive\s+(?:spectroscopy|analysis)|"
        r"\bEDS\b|opaque\s+(?:phase|mineral)|X[- ]?ray\s+diffraction|\bXRD\b|"
        r"M[o\u00f6]ssbauer|magnetic\s+(?:extract|separate)|mineral\s+separation|"
        r"separat(?:e|ed|ion).{0,60}magnetic\s+minerals?|mineralogical\s+analysis",
        0.95,
    ),
    _PatternSpec(
        "method",
        "demagnetization",
        "demagnetization or unblocking experiment",
        r"unblocking\s+temperature|blocking\s+temperature|thermal\s+demagneti[sz]ation|AF\s+demagneti[sz]ation|alternating[- ]field\s+demagneti[sz]ation",
        0.68,
    ),
    _PatternSpec(
        "diagnostic",
        "saturation_behavior",
        "measured saturation or non-saturation behavior",
        r"(?:does|do|did)\s+not\s+saturate[^.!?;\n]{0,35}"
        r"(?:until|at|by|above|below|up\s+to)\s*"
        + _R3_FIELD_VALUE
        + r"|fail(?:s|ed)?\s+to\s+saturate[^.!?;\n]{0,35}"
        r"(?:until|at|by|above|below|up\s+to)\s*"
        + _R3_FIELD_VALUE
        + r"|(?:not|never)\s+(?:fully\s+)?saturat(?:e|ed)[^.!?;\n]{0,35}"
        r"(?:until|at|by|above|below|up\s+to)\s*"
        + _R3_FIELD_VALUE
        + r"|no\s+saturation[^.!?;\n]{0,35}(?:until|at|by|above|below|up\s+to)\s*"
        + _R3_FIELD_VALUE
        + r"|(?:saturat(?:es?|ed)|saturation\s+(?:is|was)\s+"
        r"(?:reached|achieved))[^.!?;\n]{0,35}(?:at|by|above|below)\s*"
        + _R3_FIELD_VALUE
        + r"|(?:reached|achieved)\s+saturation[^.!?;\n]{0,35}(?:at|by)\s*"
        + _R3_FIELD_VALUE,
        1.0,
    ),
    _PatternSpec(
        "diagnostic",
        "curie_temperature",
        "measured Curie point or Curie temperature result",
        r"Curie\s+(?:points?|temperatures?)[^.!?;\n]{0,35}"
        r"(?:is|was|are|were)(?:\s+estimated)?(?:\s+at)?\s*"
        + _R3_TEMPERATURE_VALUE
        + r"|Curie\s+(?:points?|temperatures?)[^.!?;\n]{0,20}"
        r"(?:at|near|about|approximately|estimated\s+at|=|of(?:\s+about)?|[<>])\s*"
        + _R3_TEMPERATURE_VALUE
        + r"|"
        r"(?:reveal(?:s|ed)?|show(?:s|ed)?|yield(?:s|ed)?|observ(?:e|ed)|"
        r"determin(?:e|ed))[^.!?;\n]{0,70}Curie\s+(?:points?|temperatures?)"
        r"[^.!?;\n]{0,25}"
        + _R3_TEMPERATURE_VALUE
        + r"|"
        + _R3_TEMPERATURE_VALUE
        + r"[^.!?;\n]{0,35}"
        r"Curie\s+(?:points?|temperatures?)",
        0.98,
    ),
    _PatternSpec(
        "diagnostic",
        "domain_state",
        "reported magnetic domain state or domain size",
        r"single[- ]domain|pseudo[- ]single[- ]domain|multi[- ]domain|"
        r"\b(?:SD|PSD|MD)\b[^.!?;\n]{0,25}(?:domain|grains?|particles?)|"
        r"domain\s+(?:state|size)[^.!?;\n]{0,35}"
        r"(?:is|was|are|were|indicat(?:e|ed))[^.!?;\n]{0,20}"
        r"(?:SD|PSD|MD|single[- ]domain|pseudo[- ]single[- ]domain|multi[- ]domain)|"
        r"\b(?:samples?|data|points?)\b[^.!?;\n]{0,70}"
        r"\b(?:fall|fell|lie|lay|plot|plotted|cluster(?:ed)?)\b[^.!?;\n]{0,45}"
        r"\b(?:SD|PSD|MD)\s+(?:region|field)\b",
        0.96,
    ),
    _PatternSpec(
        "diagnostic",
        "hysteresis_ratio_result",
        "reported hysteresis or coercivity diagnostic ratio",
        r"\b(?:Hcr|Bcr)\s*/\s*(?:Hc|Bc)\b\s*(?:=|values?\s*"
        r"(?:=|of|between|from|ranging\s+from))\s*"
        + _R3_NUMBER
        + r"(?:\s*(?:to|-|and)\s*"
        + _R3_NUMBER
        + r")?|"
        r"\b(?:Mrs|Jrs)\s*/\s*(?:Ms|Js)\b\s*(?:=|values?\s*"
        r"(?:=|of|between|from|ranging\s+from))\s*"
        + _R3_NUMBER
        + r"(?:\s*(?:to|-|and)\s*"
        + _R3_NUMBER
        + r")?|"
        r"(?:coercivity|unblocking)\s+(?:spectrum|spectra|distribution)",
        0.94,
    ),
    _PatternSpec(
        "diagnostic",
        "mineral_transition",
        "reported Verwey or Morin mineral transition",
        _R3_TRANSITION_RESULT,
        0.98,
    ),
    _PatternSpec(
        "carrier",
        "named_carrier",
        "named magnetic mineral or carrier",
        r"\b(?:titano[- ]?magnetite|titano[- ]?maghemite|magnetite|hematite|haematite|"
        r"maghemite|goethite|pyrrhotite|greigite|ilmenite|specularite|martite|"
        r"Fe[- ]Ti\s+oxide(?:s)?|iron\s+sulphide(?:s)?|iron\s+sulfide(?:s)?|"
        r"magnetic\s+carrier(?:s)?|remanence\s+carrier(?:s)?)\b",
    ),
    _PatternSpec(
        "bridge",
        "explicit_carrier_bridge",
        "carrier explicitly linked to remanence or a component",
        r"(?:\bcarrier(?:s)?\b|\bcarr(?:y|ies|ying)\b|\bcarried\s+by\b).{0,180}(?:\bChRM\b|characteristic\s+remanence|remanence|remanent\s+magneti[sz]ation|magnetic\s+component|magneti[sz]ation)|(?:\bChRM\b|characteristic\s+remanence|remanence|remanent\s+magneti[sz]ation|magnetic\s+component|magneti[sz]ation).{0,180}(?:\bcarrier(?:s)?\b|\bcarr(?:y|ies|ying)\b|\bcarried\s+by\b)",
        1.0,
    ),
    _PatternSpec(
        "bridge",
        "mineral_remanence_bridge",
        "named mineral explicitly linked to remanence",
        r"(?:magnetite|titano[- ]?magnetite|hematite|haematite|maghemite|pyrrhotite|greigite).{0,100}(?:is|are|was|were|as|likely|interpreted\s+as|identified\s+as)?\s*(?:the\s+)?(?:predominant|principal|main|probable|likely)?\s*(?:magnetic\s+)?carrier(?:s)?\s+(?:of|for)\s+(?:the\s+)?(?:remanence|magneti[sz]ation|\bChRM\b|component)|(?:remanence|magneti[sz]ation|\bChRM\b|component).{0,100}(?:is|are|was|were)?\s*(?:carried\s+by|hosted\s+by|resides?\s+in|attributed\s+to|due\s+to)\s+(?:fine[- ]grained\s+|coarse[- ]grained\s+)?(?:magnetite|titano[- ]?magnetite|hematite|haematite|maghemite|pyrrhotite|greigite)",
        1.0,
    ),
    _PatternSpec(
        "bridge",
        "origin_bridge",
        "mineral or remanence linked to primary or secondary origin",
        r"(?:primary|secondary|overprint|remagneti[sz](?:ed|ation)|\bCRM\b|\bTRM\b|\bDRM\b(?![-\u2010-\u2015]?\s*\d)|\bVRM\b|detrital|chemical|authigenic).{0,150}(?:magnetite|titano[- ]?magnetite|hematite|haematite|maghemite|pyrrhotite|greigite|remanence|magneti[sz]ation)|(?:magnetite|titano[- ]?magnetite|hematite|haematite|maghemite|pyrrhotite|greigite|remanence|magneti[sz]ation).{0,150}(?:primary|secondary|overprint|remagneti[sz](?:ed|ation)|\bCRM\b|\bTRM\b|\bDRM\b(?![-\u2010-\u2015]?\s*\d)|\bVRM\b|detrital|chemical|authigenic)",
        0.88,
    ),
    _PatternSpec(
        "bridge",
        "carrier_significance",
        "carrier interpretation linked to remanence origin, alteration, or reliability",
        r"(?:magnetic\s+carrier|remanence\s+carrier|magnetite|titano[- ]?magnetite|"
        r"hematite|haematite|maghemite|pyrrhotite|greigite).{0,180}"
        r"(?:indicat(?:e|es|ed)|suggest(?:s|ed)?|impli(?:es|ed)|support(?:s|ed)?|"
        r"consistent\s+with|significan(?:ce|t)|alter(?:ation|ed)|primary|secondary|"
        r"detrital|chemical|authigenic|overprint|remagneti[sz])|"
        r"(?:primary|secondary|detrital|chemical|authigenic|overprint|remagneti[sz](?:ed|ation)|"
        r"alter(?:ation|ed)).{0,180}(?:magnetic\s+carrier|remanence\s+carrier|"
        r"magnetite|titano[- ]?magnetite|hematite|haematite|maghemite|pyrrhotite|greigite)",
        0.94,
    ),
)


_R5_PATTERNS = (
    _PatternSpec(
        "risk",
        "tilting_not_excluded",
        "possible tilting could not be ruled out",
        r"possible\s+tilting\s+(?:could|can)\s+not\s+be\s+ruled\s+out|tilting\s+(?:could|can)\s+not\s+be\s+(?:excluded|ruled\s+out)",
        1.12,
    ),
    _PatternSpec(
        "risk",
        "missing_paleohorizontal_control",
        "explicit lack of paleohorizontal control",
        r"no\s+palaeohorizontal\s+control|no\s+paleohorizontal\s+control",
    ),
    _PatternSpec(
        "risk",
        "missing_structural_correction",
        "no structural correction",
        r"without\s+(?:any\s+)?structural\s+correction|not\s+structurally\s+corrected|no\s+structural\s+correction|uncorrected\s+for\s+(?:bedding|tilt)",
    ),
    _PatternSpec(
        "risk",
        "unknown_structural_attitude",
        "unknown site or bedding attitude",
        r"positions?\s+are\s+not\s+known\s+with\s+certainty|attitudes?\s+(?:are\s+)?not\s+known|bedding\s+(?:is|was)\s+unknown|unknown\s+(?:bedding|structural)\s+attitude",
        0.95,
    ),
    _PatternSpec(
        "risk",
        "rotation",
        "vertical-axis or block rotation",
        r"vertical[- ]axis\s+rotation|block\s+rotation|rotated\s+(?:clockwise|counter[- ]?clockwise)|(?:clockwise|counter[- ]?clockwise)\s+rotation",
        0.9,
    ),
    _PatternSpec(
        "risk",
        "mobile_setting",
        "allochthonous, translated, or displaced block",
        r"allochthonous|translated\s+block|displaced\s+block|noticeably\s+shifted|mobile\s+belt|rift\s+block",
        0.78,
    ),
    _PatternSpec(
        "risk",
        "inclination_correction",
        "inclination shallowing or flattening correction",
        r"inclination\s+shallowing|flattening\s+correction|corrected\s+inclination",
        0.72,
    ),
    _PatternSpec(
        "risk",
        "fold_timing",
        "syn-folding magnetization",
        r"syn[- ]?fold(?:ing)?|magneti[sz]ation\s+acquired\s+during\s+folding",
        0.78,
    ),
    _PatternSpec(
        "risk",
        "post_folding_secondary",
        "post-folding secondary remanence or component",
        r"post[- ]?fold(?:ing|ed)?.{0,140}(?:secondary|overprint|remanence|magneti[sz]ation|component)|"
        r"(?:secondary|overprint|remanence|magneti[sz]ation|component).{0,140}post[- ]?fold(?:ing|ed)?",
        1.08,
    ),
    _PatternSpec(
        "rescue",
        "no_tilting",
        "no tilting has occurred",
        r"no\s+(?:detectable|significant\s+)?tilting\s+(?:has\s+)?(?:occurred|taken\s+place)(?:\s+since\s+(?:the\s+)?(?:intrusion|emplacement|deposition))?",
        1.12,
    ),
    _PatternSpec(
        "rescue",
        "rotation_negated",
        "rotation is explicitly absent or unlikely",
        r"no\s+(?:evidence\s+(?:for|of)\s+)?(?:vertical[- ]axis\s+|block\s+)?rotation|(?:vertical[- ]axis\s+|block\s+)?rotation.{0,70}\bunlikely\b|argue(?:s|d)?\s+against.{0,70}\brotation\b",
        1.08,
    ),
    _PatternSpec(
        "rescue",
        "correction_not_needed",
        "structural correction is explicitly unnecessary or negligible",
        r"(?:(?:post[- ]?(?:kinematic|tectonic).{0,100}(?:intrusion|pluton|dyke|dike|sill)|"
        r"(?:intrusion|pluton|dyke|dike|sill).{0,100}post[- ]?(?:kinematic|tectonic)).{0,180}"
        r"(?:primary|cooling|thermoremanent|\bTRM\b|unroofing))|"
        r"(?:post[- ]?kinematic\s+remanence.{0,120}(?:unroofing|cooling))|"
        r"(?:undeformed|no\s+(?:discernible|significant|appreciable)\s+(?:tectonic\s+)?(?:disturbance|deformation))|"
        r"(?:requires?\s+no\s+structural\s+correction)|"
        r"(?:maximum\s+tilt(?:\s+correction)?\s+.{0,120}(?:less\s+than|<)\s*(?:[0-9](?:\.\d+)?|10)\s*(?:degrees?|[oO]\b|°|‡))|"
        r"(?:(?:tilt|dip)\s+.{0,25}(?:less\s+than|<)\s*(?:[0-9](?:\.\d+)?|10)\s*(?:degrees?|[oO]\b|°|‡))",
        1.08,
    ),
    _PatternSpec(
        "rescue",
        "structural_attitude",
        "bedding strike or dip reported",
        r"bedding.{0,90}(?:strike|dip)|strikes?\s+[^.;\n]{0,70}\s+dips?|dip(?:s|ping)?\s+(?:at|between|from|of)?\s*\d{1,2}(?:\.\d+)?\s*(?:degrees?|[oO]\b|\u00b0)",
        0.68,
    ),
    _PatternSpec(
        "rescue",
        "structural_correction",
        "tilt or structural correction",
        r"tilt[- ]corrected|tilt\s+correction|structural\s+correction|corrected\s+for\s+(?:geological\s+)?(?:bedding|dip)|restored\s+to\s+palaeohorizontal|restored\s+to\s+paleohorizontal",
    ),
    _PatternSpec(
        "rescue",
        "paleohorizontal",
        "flat or subhorizontal target",
        r"flat[- ]lying|sub[- ]?horizontal|nearly\s+horizontal|almost\s+horizontal|gently\s+dipping",
        0.82,
    ),
    _PatternSpec(
        "rescue",
        "field_test",
        "positive fold or contact test",
        r"positive\s+(?:fold|contact|baked\s+contact|conglomerate)\s+test|fold\s+test.{0,100}(?:positive|passed)|baked\s+contact\s+test",
        1.0,
    ),
)


_PATTERNS = {
    "R3": tuple((spec, re.compile(spec.expression, re.I | re.S)) for spec in _R3_PATTERNS),
    "R5": tuple((spec, re.compile(spec.expression, re.I | re.S)) for spec in _R5_PATTERNS),
}


_HARD_GENERIC_TARGET_TOKENS = {
    "and",
    "combined",
    "complex",
    "component",
    "components",
    "data",
    "formation",
    "group",
    "member",
    "mean",
    "rock",
    "rocks",
    "series",
    "suite",
    "unit",
    "units",
}

_SOFT_GENERIC_TARGET_TOKENS = {
    "basalt",
    "basalts",
    "bed",
    "beds",
    "diabase",
    "dike",
    "dikes",
    "dolerite",
    "dyke",
    "dykes",
    "gabbro",
    "granite",
    "hematite",
    "lava",
    "lavas",
    "limestone",
    "magnetite",
    "magnetites",
    "quartz",
    "sandstone",
    "sediment",
    "sediments",
    "sill",
    "volcanics",
}

_SPECIAL_TRANSLITERATION = {
    "\u00d0": "D",
    "\u00d8": "O",
    "\u00de": "Th",
    "\u00df": "ss",
    "\u00e6": "ae",
    "\u00f0": "d",
    "\u00f8": "o",
    "\u00fe": "th",
    "\u0141": "L",
    "\u0142": "l",
    "\u0152": "OE",
    "\u0153": "oe",
}

_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:\d+(?:\.\d+)*[ \t]+)?"
    r"(?:references|bibliography|literature[ \t]+cited|works[ \t]+cited)"
    r"[ \t]*:?[ \t]*(?=\r?$|\n)"
)

_SECTION_RE = re.compile(
    r"(?im)(?:^|\n)[ \t]*(?:\d+(?:\.\d+)*[ \t]+)?"
    r"(summary|abstract|introduction|geology|geological setting|sampling(?: and measurements)?|"
    r"laboratory procedure|methods?|rock magnetism|magnetic mineralogy|demagnetization|"
    r"results(?: and discussion)?|discussion|interpretation|conclusions?|acknowledg(?:e)?ments?|"
    r"references|bibliography|literature cited|works cited)"
    r"[ \t]*(?:[.:][ \t]*|\r?\n)",
)
_DOI_RE = re.compile(
    r"\b(?:https?\s*:\s*/\s*/\s*(?:dx\s*\.\s*)?doi\s*\.\s*org\s*/\s*|"
    r"doi\s*:\s*)10\s*\.\s*\d{4,9}\s*/",
    re.I,
)
_BIBLIOGRAPHY_AUTHOR_YEAR_RE = re.compile(
    r"\b[A-Z][A-Za-z'\u2019-]+,\s*(?:[A-Z](?:\.-?[A-Z])?\.?\s*){1,4}"
    r"(?:,\s*[A-Z][A-Za-z'\u2019-]+,\s*(?:[A-Z]\.?\s*){1,3})*.{0,45}"
    r"\b(?:19|20)\d{2}[a-z]?\s*\.",
    re.S,
)
_BIBLIOGRAPHY_JOURNAL_RE = re.compile(
    r"\b(?:J\.|Journal|Geophys(?:ical)?|Earth|Tectono|Palaeo|Paleo|Precambrian|"
    r"Phys\.|Nature|Science|Bull\.|Rev\.|Res\.)\b.{0,100}"
    r"\b\d{1,4}\s*(?:\([A-Z0-9-]+\))?\s*[, :]\s*\d{1,5}",
    re.I | re.S,
)


def _fold_with_map(value: Any) -> tuple[str, tuple[int, ...]]:
    """Return NFKD/ASCII-folded text and an index map back to the input."""

    text = "" if value is None else str(value)
    output: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text):
        expanded = _SPECIAL_TRANSLITERATION.get(character, character)
        expanded = unicodedata.normalize("NFKD", expanded)
        expanded = "".join(item for item in expanded if not unicodedata.combining(item))
        expanded = expanded.encode("ascii", "ignore").decode("ascii").casefold()
        for item in expanded:
            if item.isalnum():
                output.append(item)
                positions.append(index)
            elif output and output[-1] != " ":
                output.append(" ")
                positions.append(index)
        if not expanded and character.isspace() and output and output[-1] != " ":
            output.append(" ")
            positions.append(index)

    while output and output[-1] == " ":
        output.pop()
        positions.pop()
    return "".join(output), tuple(positions)


def fold_text_with_position_map(value: Any) -> tuple[str, tuple[int, ...]]:
    """Return canonical folded text plus folded-to-original character positions."""

    return _fold_with_map(value)


def _fold(value: Any) -> str:
    return _fold_with_map(value)[0]


def canonical_target_identity(value: Any) -> str:
    """Return a conservative identity key for article-level target grouping."""

    return _fold(value)


def _unique_tokens(folded: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"[a-z0-9]+", folded)))


def _target_profile(target: Any) -> _TargetProfile:
    raw = "" if target is None else str(target)
    folded = _fold(raw)
    tokens = _unique_tokens(folded)
    weights: list[float] = []
    specific: list[str] = []
    generic: list[str] = []
    for token in tokens:
        if token in _HARD_GENERIC_TARGET_TOKENS or len(token) <= 2:
            weights.append(0.08)
            generic.append(token)
        elif token in _SOFT_GENERIC_TARGET_TOKENS:
            weights.append(0.35)
            generic.append(token)
        else:
            weights.append(1.0)
            specific.append(token)
    specificity = sum(weights) / len(weights) if weights else 0.0
    return _TargetProfile(
        raw=raw,
        folded=folded,
        tokens=tokens,
        specific_tokens=tuple(specific),
        generic_tokens=tuple(generic),
        specificity=round(specificity, 4),
        common_target=bool(tokens) and specificity < 0.45,
    )


def _safe_page_number(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _original_positions(
    pattern: re.Pattern[str],
    folded_text: str,
    index_map: Sequence[int],
) -> list[int]:
    positions: list[int] = []
    for match in pattern.finditer(folded_text):
        if not index_map:
            continue
        start = index_map[min(match.start(), len(index_map) - 1)]
        last = max(match.start(), match.end() - 1)
        end = index_map[min(last, len(index_map) - 1)]
        positions.append((start + end) // 2)
    return positions


def _phrase_pattern(phrase: str) -> re.Pattern[str] | None:
    if not phrase:
        return None
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])")


def _token_pattern(token: str) -> re.Pattern[str]:
    variants = {token}
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    elif token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    expression = "|".join(re.escape(value) for value in sorted(variants, key=len, reverse=True))
    return re.compile(r"(?<![a-z0-9])(?:" + expression + r")(?![a-z0-9])")


def _reference_cutoffs(pages: Sequence[Mapping[str, Any]]) -> list[int | None]:
    cutoffs: list[int | None] = []
    references_started = False
    for page in pages:
        text = str(page.get("text") or "")
        if references_started:
            cutoffs.append(0)
            continue
        match = _REFERENCE_HEADING_RE.search(text)
        if match:
            cutoffs.append(match.start())
            references_started = True
            continue
        hints = tuple(str(item) for item in (page.get("section_hints") or ()))
        if any(_fold(item) in {"references", "bibliography", "literature cited", "works cited"} for item in hints):
            cutoffs.append(0)
            references_started = True
        else:
            cutoffs.append(None)
    return cutoffs


def _bibliography_entry_noise(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 240) : min(len(text), end + 280)]
    if _DOI_RE.search(window):
        return True
    return bool(
        _BIBLIOGRAPHY_AUTHOR_YEAR_RE.search(window)
        and _BIBLIOGRAPHY_JOURNAL_RE.search(window)
    )


def _prepare_base_pages(cache_row: Mapping[str, Any]) -> list[_BasePage]:
    raw_pages = cache_row.get("pages") or []
    if not isinstance(raw_pages, Sequence) or isinstance(raw_pages, (str, bytes)):
        return []
    pages = [page for page in raw_pages if isinstance(page, Mapping)]
    cutoffs = _reference_cutoffs(pages)
    prepared: list[_BasePage] = []
    for offset, (page, reference_cutoff) in enumerate(zip(pages, cutoffs), start=1):
        text = str(page.get("text") or "")
        folded, index_map = _fold_with_map(text)
        prepared.append(
            _BasePage(
                number=_safe_page_number(page.get("page"), offset),
                text=text,
                source=str(page.get("source") or ""),
                section_hints=tuple(str(item) for item in (page.get("section_hints") or ())),
                folded=folded,
                index_map=index_map,
                reference_cutoff=reference_cutoff,
            )
        )
    return prepared


def _bind_pages(
    base_pages: Sequence[_BasePage],
    profile: _TargetProfile,
) -> list[_PageInfo]:
    phrase_pattern = _phrase_pattern(profile.folded)
    prepared: list[_PageInfo] = []
    for page in base_pages:
        exact_positions = (
            _original_positions(phrase_pattern, page.folded, page.index_map)
            if phrase_pattern
            else []
        )

        token_position_groups: list[list[int]] = []
        binding_tokens = profile.specific_tokens or profile.generic_tokens
        for token in binding_tokens:
            token_pattern = _token_pattern(token)
            token_position_groups.append(
                _original_positions(token_pattern, page.folded, page.index_map)
            )
        token_hits = sum(bool(group) for group in token_position_groups)
        token_required = max(1, math.ceil(len(binding_tokens) * 0.6))
        token_positions = [item for group in token_position_groups for item in group]
        if exact_positions:
            anchor_kind = "generic_exact" if profile.common_target else "exact"
        elif profile.specific_tokens and token_hits >= token_required:
            anchor_kind = "token"
        elif profile.common_target and token_hits >= token_required:
            anchor_kind = "generic_token"
        else:
            anchor_kind = "none"

        prepared.append(
            _PageInfo(
                number=page.number,
                text=page.text,
                source=page.source,
                section_hints=page.section_hints,
                folded=page.folded,
                index_map=page.index_map,
                reference_cutoff=page.reference_cutoff,
                exact_positions=tuple(exact_positions),
                token_positions=tuple(token_positions),
                anchor_kind=anchor_kind,
            )
        )
    return prepared


def _prepare_pages(cache_row: Mapping[str, Any], profile: _TargetProfile) -> list[_PageInfo]:
    return _bind_pages(_prepare_base_pages(cache_row), profile)


def _anchor_state(profile: _TargetProfile, pages: Sequence[_PageInfo]) -> dict[str, Any]:
    exact_pages = sorted(page.number for page in pages if page.anchor_kind in {"exact", "generic_exact"})
    token_pages = sorted(page.number for page in pages if page.anchor_kind == "token")
    generic_pages = sorted(page.number for page in pages if page.anchor_kind == "generic_token")
    anchor_pages = sorted(set(exact_pages + token_pages + generic_pages))
    if any(page.anchor_kind == "exact" for page in pages):
        status = "exact"
        strength = "strong"
    elif token_pages:
        status = "token"
        strength = "strong" if len(profile.specific_tokens) > 1 else "moderate"
    elif exact_pages or generic_pages:
        status = "generic_only"
        strength = "weak"
    else:
        status = "missing"
        strength = "missing"
    return {
        "found": bool(anchor_pages),
        "status": status,
        "strength": strength,
        "pages": anchor_pages,
        "exact_pages": exact_pages,
        "token_pages": token_pages,
        "generic_pages": generic_pages,
        "specific_tokens": list(profile.specific_tokens),
        "generic_tokens": list(profile.generic_tokens),
        "specificity": profile.specificity,
        "common_target": profile.common_target,
        "normalization": "Unicode NFKD with ASCII fallback",
    }


def find_target_anchor(cache_row: Mapping[str, Any], target: Any) -> dict[str, Any]:
    """Return the article-level anchor state for ``target``."""

    return PaperTextRetriever(cache_row).find_target_anchor(target)


def _source_reliability(cache_row: Mapping[str, Any], pages: Sequence[_BasePage]) -> float:
    if not pages:
        return 0.0
    source = cache_row.get("source") or {}
    flags = set(source.get("source_quality_flags") or ()) if isinstance(source, Mapping) else set()
    score = 0.94
    if "glyph_or_cid_text" in flags:
        score -= 0.28
    if "short_text" in flags or "possibly_incomplete_extraction" in flags:
        score -= 0.18
    if "no_extractable_text" in flags:
        score = 0.08
    return round(max(0.08, min(1.0, score)), 4)


def _clean_excerpt(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _bounded_claim_text(
    value: str,
    matched_text: str,
    max_chars: int = 1200,
) -> str:
    """Limit a claim while retaining the exact pattern match it promotes."""

    claim = _clean_excerpt(value)
    raw_match = _clean_excerpt(matched_text)
    if len(raw_match) > max_chars:
        raise ValueError("matched text exceeds the claim-text limit")
    if raw_match not in claim:
        context_chars = max_chars - len(raw_match) - (1 if claim and raw_match else 0)
        context = claim[:max(0, context_chars)].rstrip()
        return " ".join(part for part in (context, raw_match) if part)
    if len(claim) <= max_chars:
        return claim

    match_start = claim.index(raw_match)
    match_end = match_start + len(raw_match)
    left = match_start - (max_chars - len(raw_match)) // 2
    left = max(0, min(left, len(claim) - max_chars))
    left = min(left, match_start)
    left = max(left, match_end - max_chars)
    return claim[left : left + max_chars]


def _quote(text: str, start: int, end: int, radius: int = 300) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    excerpt = _clean_excerpt(text[left:right])
    return excerpt[:1000]


def _section_at(page: _BasePage, position: int, reference_noise: bool) -> str:
    if reference_noise:
        return "References"
    candidates: list[tuple[int, str]] = []
    for match in _SECTION_RE.finditer(page.text[:position]):
        candidates.append((match.start(1), _clean_excerpt(match.group(1)).title()))
    for hint in page.section_hints:
        hint_text = _clean_excerpt(hint)
        if not hint_text:
            continue
        hint_position = page.text.casefold().rfind(hint_text.casefold(), 0, position)
        if hint_position >= 0:
            candidates.append((hint_position, hint_text))
    return max(candidates, default=(-1, "Unknown"), key=lambda item: item[0])[1]


_SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?;][\"')\]]?\s+|\n\s*\n)")
_SENTENCE_ABBREVIATION_RE = re.compile(
    r"(?:\b(?:fig|figs|eq|eqs|e\.g|i\.e|cf|ca|approx|dr|mr|mrs|ms|vs|"
    r"no|nos|vol|pp|p|ref|refs|fm|inc)|\bet\s+al)\.$",
    re.I,
)
_SENTENCE_INITIALS_RE = re.compile(
    r"(?:(?:\b[A-Z]\.){2,4}|,\s*[A-Z]\.)$"
)
_DISCOURSE_MAX_FOLLOWING_SENTENCES = 2
_DISCOURSE_MAX_CHARS = 650
_TARGET_SECTION_MAX_FOLLOWING_SENTENCES = 8
_TARGET_SECTION_MAX_CHARS = 2400
_DISCOURSE_TARGET_CONTEXT_RE = re.compile(
    r"\b(?:samples?|specimens?|sites?|rock[- ]magnetic|magnetic\s+mineralogy|"
    r"remanence|magneti[sz]ation|measure(?:d|ment|ments)?|analy[sz](?:e|ed|is)|"
    r"investigat(?:e|ed|ion)|stud(?:y|ied)|IRM|SIRM|hysteresis|FORC|"
    r"thermomagnetic|susceptibility|microscop(?:y|ic)|XRD|saturat(?:e|ed|ion)|"
    r"Curie|coerciv(?:e|ity)|domain\s+(?:state|size)|Verwey|Morin)\b",
    re.I,
)
_HEADING_CONNECTORS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


@lru_cache(maxsize=4096)
def _sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(text):
        punctuation = text[boundary.start() : boundary.start() + 1]
        if punctuation == ".":
            prefix = text[max(0, boundary.start() - 16) : boundary.start() + 1]
            if _SENTENCE_ABBREVIATION_RE.search(prefix) or _SENTENCE_INITIALS_RE.search(
                prefix
            ):
                continue
        end = boundary.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return tuple(spans)


def _sentence_span_index(
    spans: Sequence[tuple[int, int]],
    position: int,
) -> int | None:
    return next(
        (
            index
            for index, (sentence_start, sentence_end) in enumerate(spans)
            if sentence_start <= position < sentence_end
        ),
        None,
    )


def _sentence_is_reliably_closed(text: str, span: tuple[int, int]) -> bool:
    start, end = span
    stripped = text[start:end].rstrip()
    if not stripped:
        return False
    if re.search(r"[.!?;][\"')\]]?$", stripped):
        return True
    return bool(re.search(r"\n\s*\n\s*$", text[start:end]))


_MATERIAL_RESULT_QUALIFIER_RE = re.compile(
    r"(?:\s*[,;:/-]\s*|\s+|\s*\(\s*)"
    r"(?:"
    r"(?:[NSEW]\s+)?combined|"
    r"[NSEW]\s+directions?|"
    r"(?:normal|reverse|reversed|[ABNSEW]|MT)\s+(?:components?|comp\.?)|"
    r"[AB]\s+magneti[sz]ation"
    r")\s*\)?\s*$",
    re.I,
)
_MULTI_MATERIAL_IDENTITY_RE = re.compile(r"\b(?:and|plus)\b|[&,/]", re.I)


def _has_material_result_qualifier(profile: _TargetProfile) -> bool:
    return bool(_MATERIAL_RESULT_QUALIFIER_RE.search(profile.raw.strip()))


def _material_alias_raw(profile: _TargetProfile) -> str | None:
    match = _MATERIAL_RESULT_QUALIFIER_RE.search(profile.raw.strip())
    if not match:
        return None
    material = profile.raw[: match.start()].strip(" \t,;:/-")
    if not material or "," in material or _MULTI_MATERIAL_IDENTITY_RE.search(material):
        return None
    return material


def _material_alias_candidate(profile: _TargetProfile) -> _TargetProfile | None:
    material = _material_alias_raw(profile)
    if not material:
        return None
    candidate = _target_profile(material)
    if len(candidate.specific_tokens) < 2 or candidate.common_target:
        return None
    return candidate


def _material_alias_profile(
    profile: _TargetProfile,
    competing_profiles: Sequence[_TargetProfile],
) -> _TargetProfile | None:
    candidate = _material_alias_candidate(profile)
    if candidate is None:
        return None
    for competitor in competing_profiles:
        competitor_material = _material_alias_raw(competitor)
        if candidate.folded == competitor.folded or (
            competitor_material and candidate.folded == _fold(competitor_material)
        ):
            return None
    return candidate


def _discourse_anchor_tokens(profile: _TargetProfile) -> tuple[str, ...]:
    return tuple(
        token for token in profile.tokens if token not in _HARD_GENERIC_TARGET_TOKENS
    )


def _has_compound_discourse_identity(tokens: Sequence[str]) -> bool:
    return len(tokens) >= 2 and any(
        token not in _SOFT_GENERIC_TARGET_TOKENS for token in tokens
    )


def _contains_local_anchor_tokens(
    folded: str,
    tokens: Sequence[str],
    max_window_tokens: int = 8,
) -> bool:
    if not _has_compound_discourse_identity(tokens):
        return False
    words = re.findall(r"[a-z0-9]+", folded)
    required = set(tokens)
    for index in range(len(words)):
        if words[index] not in required:
            continue
        if required.issubset(set(words[index : index + max_window_tokens])):
            return True
    return False


def _discourse_anchor_kind(
    folded: str,
    profile: _TargetProfile,
    anchor_tokens: Sequence[str],
) -> str | None:
    phrase_pattern = _phrase_pattern(profile.folded)
    if phrase_pattern and phrase_pattern.search(folded):
        return "exact"
    if _has_material_result_qualifier(profile):
        return None
    if _contains_local_anchor_tokens(folded, anchor_tokens):
        return "token_variant"
    return None


def _discourse_profile_present(folded: str, profile: _TargetProfile) -> bool:
    anchor_tokens = _discourse_anchor_tokens(profile)
    if _discourse_anchor_kind(folded, profile, anchor_tokens) is not None:
        return True
    if _has_compound_discourse_identity(anchor_tokens):
        return False
    return _folded_text_contains_target(folded, profile)


@lru_cache(maxsize=4096)
def _line_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return tuple(spans)


def _line_is_heading_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120 or "\t" in stripped:
        return False
    candidate = stripped[:-1].rstrip() if stripped.endswith(":") else stripped
    if not candidate or candidate[-1:] in ".!?;":
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", candidate)
    if not words or len(words) > 12:
        return False
    if _SECTION_RE.search("\n" + candidate + "\n"):
        return True
    content_words = [word for word in words if word.casefold() not in _HEADING_CONNECTORS]
    return bool(content_words) and (
        candidate.isupper()
        or all(word.isupper() or word[0].isupper() for word in content_words)
    )


def _line_is_target_heading(
    line: str,
    profile: _TargetProfile,
    anchor_tokens: Sequence[str],
) -> bool:
    stripped = line.strip()
    if (
        not stripped
        or len(stripped) > 120
        or "\t" in stripped
        or stripped.rstrip(":").endswith((".", "!", "?", ";"))
        or len(re.findall(r"[A-Za-z0-9]+", stripped)) > 12
    ):
        return False
    return _discourse_anchor_kind(_fold(stripped.rstrip(":")), profile, anchor_tokens) is not None


def _has_intervening_heading(
    text: str,
    start: int,
    end: int,
    profiles: Sequence[_TargetProfile],
) -> bool:
    if end <= start:
        return False
    profile_tokens = [(_profile, _discourse_anchor_tokens(_profile)) for _profile in profiles]
    for line_start, line_end in _line_spans(text):
        if line_start < start or line_end > end:
            continue
        line = text[line_start:line_end]
        if _line_is_heading_like(line) or any(
            _line_is_target_heading(line, profile, tokens)
            for profile, tokens in profile_tokens
        ):
            return True
    return False


def _sentence_at(
    page: _PageInfo,
    start: int,
    end: int,
) -> str:
    spans = _sentence_spans(page.text)
    if not spans:
        return ""
    bounded_start = max(0, min(start, len(page.text) - 1))
    bounded_end = max(bounded_start + 1, min(end, len(page.text)))
    first = next(
        (index for index, (left, right) in enumerate(spans) if left <= bounded_start < right),
        0,
    )
    last_position = max(bounded_start, bounded_end - 1)
    last = next(
        (index for index, (left, right) in enumerate(spans) if left <= last_position < right),
        first,
    )
    return page.text[spans[first][0] : spans[last][1]]


def _folded_text_contains_target(folded: str, profile: _TargetProfile) -> bool:
    phrase_pattern = _phrase_pattern(profile.folded)
    if phrase_pattern and phrase_pattern.search(folded):
        return True
    binding_tokens = profile.specific_tokens or profile.generic_tokens
    if not binding_tokens:
        return False
    hits = sum(bool(_token_pattern(token).search(folded)) for token in binding_tokens)
    return hits >= max(1, math.ceil(len(binding_tokens) * 0.6))


def _scope_at(page: _BasePage, start: int, end: int) -> _TextScope:
    line_start = page.text.rfind("\n", 0, start) + 1
    line_end_position = page.text.find("\n", end)
    line_end = len(page.text) if line_end_position < 0 else line_end_position
    line = page.text[line_start:line_end]
    if "\t" in line:
        row_index = page.text.count("\n", 0, line_start) + 1
        return _TextScope(
            scope_id=f"page:{page.number}:table_row:{row_index}",
            kind="table_row",
            text=line,
            folded=_fold(line),
            bounded=True,
            start=line_start,
            end=line_end,
        )

    preceding = list(re.finditer(r"\n\s*\n", page.text[:start]))
    paragraph_start = preceding[-1].end() if preceding else 0
    following = re.search(r"\n\s*\n", page.text[end:])
    paragraph_end = end + following.start() if following else len(page.text)
    paragraph_index = len(preceding) + 1
    return _TextScope(
        scope_id=f"page:{page.number}:paragraph:{paragraph_index}",
        kind="paragraph",
        text=page.text[paragraph_start:paragraph_end],
        folded=_fold(page.text[paragraph_start:paragraph_end]),
        bounded=bool(preceding or following),
        start=paragraph_start,
        end=paragraph_end,
    )


def _scope_target_binding(
    scope: _TextScope,
    profile: _TargetProfile,
    competing_profiles: Sequence[_TargetProfile],
) -> tuple[_TextScope, bool, tuple[str, ...]]:
    target_present = _folded_text_contains_target(scope.folded, profile)
    competitors = tuple(
        competitor.folded
        for competitor in competing_profiles
        if competitor.folded and _folded_text_contains_target(scope.folded, competitor)
    )
    # A page-long extraction block is not a reliable paragraph.  Table rows are
    # naturally bounded; prose scopes must remain local to the evidence hit.
    scope_is_bounded = scope.bounded and (
        scope.kind == "table_row" or len(scope.text) <= 1600
    )
    return scope, bool(target_present and scope_is_bounded and not competitors), competitors


def _target_forward_discourse_scope(
    page: _BasePage,
    start: int,
    end: int,
    profile: _TargetProfile,
    competing_profiles: Sequence[_TargetProfile],
) -> tuple[_TextScope | None, bool, tuple[str, ...]]:
    """Bind evidence to a short, exclusive target episode without using a whole page."""

    spans = _sentence_spans(page.text)
    material_profile = _material_alias_profile(profile, competing_profiles)
    anchor_profiles = tuple(
        candidate for candidate in (profile, material_profile) if candidate is not None
    )
    anchor_specs = tuple(
        (candidate, _discourse_anchor_tokens(candidate)) for candidate in anchor_profiles
    )
    if not spans or not page.text or not any(
        not candidate.common_target or _has_compound_discourse_identity(tokens)
        for candidate, tokens in anchor_specs
    ):
        return None, False, ()
    evidence_position = max(0, min(start, len(page.text) - 1))
    evidence_index = _sentence_span_index(spans, evidence_position)
    if (
        evidence_index is None
        or not _sentence_is_reliably_closed(page.text, spans[evidence_index])
    ):
        return None, False, ()

    candidates: list[tuple[int, int, bool]] = []
    earliest = max(0, evidence_index - _DISCOURSE_MAX_FOLLOWING_SENTENCES)
    for anchor_index in range(evidence_index, earliest - 1, -1):
        anchor_start, anchor_end = spans[anchor_index]
        anchor_text = page.text[anchor_start:anchor_end]
        anchor_folded = _fold(anchor_text)
        if not any(
            _discourse_anchor_kind(anchor_folded, candidate, tokens) is not None
            for candidate, tokens in anchor_specs
        ):
            continue
        first_line_end = page.text.find("\n", anchor_start, anchor_end)
        if first_line_end >= 0 and any(
            _line_is_target_heading(
                page.text[anchor_start : first_line_end + 1],
                candidate,
                tokens,
            )
            for candidate, tokens in anchor_specs
        ):
            continue
        if not _sentence_is_reliably_closed(page.text, spans[anchor_index]):
            continue
        candidates.append((anchor_start, anchor_end, False))

    evidence_end = spans[evidence_index][1]
    for line_start, line_end in _line_spans(page.text):
        if line_end > start or start - line_start > _DISCOURSE_MAX_CHARS:
            continue
        line = page.text[line_start:line_end]
        if not any(
            _line_is_target_heading(line, candidate, tokens)
            for candidate, tokens in anchor_specs
        ):
            continue
        body = page.text[line_end:evidence_end]
        body_spans = _sentence_spans(body)
        body_evidence_index = _sentence_span_index(body_spans, start - line_end)
        if (
            body_evidence_index is None
            or body_evidence_index >= _TARGET_SECTION_MAX_FOLLOWING_SENTENCES
        ):
            continue
        if not _sentence_is_reliably_closed(body, body_spans[body_evidence_index]):
            continue
        candidates.append((line_start, line_end, True))

    for anchor_start, anchor_end, heading_anchor in sorted(
        set(candidates),
        key=lambda candidate: (candidate[0], candidate[2]),
        reverse=True,
    ):
        block = page.text[anchor_start:evidence_end]
        between = page.text[anchor_end:start]
        maximum_chars = (
            _TARGET_SECTION_MAX_CHARS if heading_anchor else _DISCOURSE_MAX_CHARS
        )
        if len(block) > maximum_chars:
            continue
        if "\t" in block or (
            not heading_anchor and re.search(r"\n\s*\n", between)
        ):
            continue
        if _SECTION_RE.search(between):
            continue
        if _has_intervening_heading(
            page.text,
            anchor_end,
            start,
            (*anchor_profiles, *competing_profiles),
        ):
            continue
        if not heading_anchor and not _DISCOURSE_TARGET_CONTEXT_RE.search(block):
            continue
        if page.reference_cutoff is not None and (
            anchor_start >= page.reference_cutoff
            or anchor_start < page.reference_cutoff < evidence_end
        ):
            continue
        folded = _fold(block)
        competitors = tuple(
            competitor.folded
            for competitor in competing_profiles
            if competitor.folded
            and any(
                _discourse_profile_present(folded, variant)
                for variant in (
                    competitor,
                    _material_alias_candidate(competitor),
                )
                if variant is not None
            )
        )
        scope = _TextScope(
            scope_id=(
                f"page:{page.number}:target_section:{anchor_start}"
                if heading_anchor
                else f"page:{page.number}:discourse:{anchor_start}"
            ),
            kind="target_section" if heading_anchor else "discourse",
            text=block,
            folded=folded,
            bounded=True,
            start=anchor_start,
            end=evidence_end,
        )
        return scope, not competitors, competitors
    return None, False, ()


def _nearest_page_distance(page_number: int, anchor_pages: Iterable[int]) -> int | None:
    distances = [abs(page_number - anchor_page) for anchor_page in anchor_pages]
    return min(distances) if distances else None


def _binding(
    page: _PageInfo,
    start: int,
    end: int,
    profile: _TargetProfile,
    anchor_pages: Sequence[int],
    multi_target: bool,
    competing_profiles: Sequence[_TargetProfile],
    scope: _TextScope,
    allow_structured_scope: bool,
    allow_discourse_scope: bool,
) -> tuple[float, str, int | None, bool, _TextScope, tuple[str, ...]]:
    center = (start + end) // 2
    positions = page.exact_positions + page.token_positions
    character_distance = min((abs(center - item) for item in positions), default=None)
    sentence_text = _sentence_at(page, start, end) if scope.kind != "table_row" else ""
    sentence = _fold(sentence_text)
    sentence_bound = bool(sentence) and len(sentence_text) <= _DISCOURSE_MAX_CHARS
    sentence_bound = sentence_bound and _folded_text_contains_target(sentence, profile)
    sentence_competitors = tuple(
        competitor.folded
        for competitor in competing_profiles
        if competitor.folded and _folded_text_contains_target(sentence, competitor)
    )
    if sentence_competitors:
        sentence_bound = False
    scope, scope_bound, scope_competitors = _scope_target_binding(
        scope,
        profile,
        competing_profiles,
    )
    discourse_scope: _TextScope | None = None
    discourse_bound = False
    discourse_competitors: tuple[str, ...] = ()
    if (
        allow_discourse_scope
        and multi_target
        and competing_profiles
        and not scope_bound
    ):
        discourse_scope, discourse_bound, discourse_competitors = (
            _target_forward_discourse_scope(
                page,
                start,
                end,
                profile,
                competing_profiles,
            )
        )

    if sentence_bound:
        selected_scope = scope
        selected_competitors = scope_competitors
        if not scope_bound and discourse_bound and discourse_scope is not None:
            selected_scope = discourse_scope
            selected_competitors = discourse_competitors
        if profile.common_target:
            return (
                0.62,
                "generic_target_same_sentence",
                character_distance,
                True,
                selected_scope,
                selected_competitors,
            )
        return (
            1.0,
            "target_same_sentence",
            character_distance,
            True,
            selected_scope,
            selected_competitors,
        )
    if (
        allow_structured_scope
        and scope.kind == "table_row"
        and scope_bound
        and (not multi_target or bool(competing_profiles))
    ):
        value = 0.84 if not profile.common_target else 0.58
        return (
            value,
            "target_same_table_row",
            character_distance,
            True,
            scope,
            scope_competitors,
        )
    if allow_structured_scope and multi_target and competing_profiles and scope_bound:
        value = 0.8 if not profile.common_target else 0.55
        return (
            value,
            "target_same_paragraph",
            character_distance,
            True,
            scope,
            scope_competitors,
        )
    if discourse_bound and discourse_scope is not None:
        if discourse_scope.kind == "target_section":
            value = 0.78 if not profile.common_target else 0.53
            binding_kind = "target_same_section"
        else:
            value = 0.76 if not profile.common_target else 0.52
            binding_kind = "target_same_discourse"
        return (
            value,
            binding_kind,
            character_distance,
            True,
            discourse_scope,
            discourse_competitors,
        )
    if character_distance is not None and character_distance <= 240:
        value = 0.72 if multi_target else 0.84
        if profile.common_target:
            value *= 0.72
        return value, "near_target", character_distance, False, scope, scope_competitors
    if character_distance is not None and character_distance <= 600:
        value = 0.57 if multi_target else 0.74
        if profile.common_target:
            value *= 0.7
        return value, "target_context", character_distance, False, scope, scope_competitors
    if page.number in anchor_pages:
        value = 0.34 if multi_target else 0.66
        if profile.common_target:
            value *= 0.68
        return value, "same_target_page", character_distance, False, scope, scope_competitors

    page_distance = _nearest_page_distance(page.number, anchor_pages)
    if page_distance == 1:
        value = 0.22 if multi_target else 0.52
        if profile.common_target:
            value *= 0.68
        return value, "adjacent_target_page", None, False, scope, scope_competitors
    if not multi_target and anchor_pages:
        return 0.4, "single_target_article", None, False, scope, scope_competitors
    if not multi_target and not anchor_pages:
        return 0.26, "unanchored_single_target_article", None, False, scope, scope_competitors
    return 0.05, "unanchored_multi_target", None, False, scope, scope_competitors


def _false_carried_out_bridge(matched_text: str) -> bool:
    folded = _fold(matched_text)
    if "carried out" not in folded:
        return False
    valid_link = any(
        phrase in folded
        for phrase in ("carrier", "carried by", "carries", "carrying")
    )
    return not valid_link


def _negated_carrier_mention(
    page_text: str,
    spec: _PatternSpec,
    start: int,
    end: int,
) -> bool:
    """Reject carrier names that occur in an explicit absence statement."""

    if spec.category != "carrier":
        return False

    before = _fold(page_text[max(0, start - 140):start])
    after = _fold(page_text[end:min(len(page_text), end + 140)])
    separator = r"[.;:!?]|\b(?:but|however|whereas)\b"
    before_clause = re.split(separator, before)[-1]
    after_clause = re.split(separator, after)[0]
    absence = r"(?:identified|detected|observed|found|present)"

    leading_absence = bool(
        re.search(r"\bno(?:\s+[a-z0-9]+){0,3}\s*$", before_clause)
        or re.search(r"\bneither\b[^.;:!?]{0,100}$", before_clause)
    )
    if leading_absence and re.search(rf"\b{absence}\b", after_clause):
        return True

    if re.search(
        rf"^[^.;:!?]{{0,80}}\bnot(?:\s+[a-z0-9]+){{0,2}}\s+{absence}\b",
        after_clause,
    ):
        return True
    return bool(
        re.search(
            rf"\b(?:not|never)(?:\s+[a-z0-9]+){{0,2}}\s+{absence}\b[^.;:!?]{{0,80}}$",
            before_clause,
        )
    )


_DIAGNOSTIC_PURPOSE_RE = re.compile(
    r"\b(?:to|in\s+order\s+to|for\s+the\s+purpose\s+of)\s+"
    r"(?:determine|measure|estimate|assess|obtain|investigate|characteri[sz]e|"
    r"identify|evaluate)\b",
    re.I,
)
_DIAGNOSTIC_NON_EXECUTION_RE = re.compile(
    r"\b(?:is|are|was|were|could|can)\s+not\s+(?:be\s+)?"
    r"(?:measured|determined|estimated|calculated|reported|observed|detected)\b|"
    r"\b(?:not\s+possible|unable|failed)\s+to\s+"
    r"(?:measure|determine|estimate|calculate|observe|detect)\b",
    re.I,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[;]|\b(?:but|however|whereas)\b", re.I)


def _non_result_diagnostic(
    page_text: str,
    spec: _PatternSpec,
    start: int,
    end: int,
) -> bool:
    """Reject diagnostic vocabulary used only as a purpose or non-execution statement."""

    if spec.category != "diagnostic":
        return False
    spans = _sentence_spans(page_text)
    sentence_index = _sentence_span_index(spans, start)
    if sentence_index is None:
        return True
    sentence_start, sentence_end = spans[sentence_index]
    sentence = page_text[sentence_start:sentence_end]
    relative_start = max(0, start - sentence_start)
    relative_end = min(len(sentence), max(relative_start, end - sentence_start))

    left = 0
    for boundary in _CLAUSE_BOUNDARY_RE.finditer(sentence, 0, relative_start):
        left = boundary.end()
    following = _CLAUSE_BOUNDARY_RE.search(sentence, relative_end)
    right = following.start() if following else len(sentence)
    clause = sentence[left:right]
    match_offset = max(0, relative_start - left)
    purpose = _DIAGNOSTIC_PURPOSE_RE.search(clause)
    return bool(
        (purpose and purpose.start() <= match_offset)
        or _DIAGNOSTIC_NON_EXECUTION_RE.search(clause)
    )


def _negation_collision(
    page_text: str,
    spec: _PatternSpec,
    start: int,
    end: int,
) -> bool:
    """Suppress a positive/negative lexical hit contradicted by its clause."""

    before = _fold(page_text[max(0, start - 55):start])
    if spec.category == "rescue" and spec.subtype == "structural_correction":
        if re.search(r"(?:no|without(?: any)?|not)\s*$", before):
            return True
    if spec.category == "rescue" and spec.subtype == "field_test":
        context = _fold(page_text[max(0, start - 110):min(len(page_text), end + 110)])
        if re.search(
            r"(?:not possible|unable|failed|unsuccessful|negative).{0,55}(?:fold|contact|baked contact|conglomerate) test|"
            r"(?:fold|contact|baked contact|conglomerate) test.{0,55}(?:not performed|failed|unsuccessful|negative)",
            context,
        ):
            return True
    if spec.category == "risk" and spec.subtype == "missing_structural_correction":
        context = _fold(page_text[max(0, start - 340):min(len(page_text), end + 380)])
        correction_not_needed = bool(
            re.search(
                r"(?:(?:post (?:kinematic|tectonic).{0,100}(?:intrusion|pluton|dyke|dike|sill)|"
                r"(?:intrusion|pluton|dyke|dike|sill).{0,100}post (?:kinematic|tectonic)).{0,180}"
                r"(?:primary|cooling|thermoremanent|trm|unroofing))|"
                r"post kinematic remanence.{0,120}(?:unroofing|cooling)|"
                r"undeformed|no (?:discernible|significant|appreciable) (?:tectonic )?(?:disturbance|deformation)|"
                r"requires? no structural correction|"
                r"(?:maximum )?tilt(?: correction)?.{0,120}(?:less than|<) ?(?:[0-9](?: [0-9]+)?|10)(?: (?:degrees?|o))?|"
                r"(?:tilt|dip).{0,30}(?:less than|<) ?(?:[0-9](?: [0-9]+)?|10)(?: (?:degrees?|o))?",
                context,
            )
        )
        figure_projection = bool(
            re.search(
                r"(?:figure|fig ).{0,100}(?:orthogonal|vector endpoint|zijderveld).{0,140}uncorrected for tilt|"
                r"(?:orthogonal projection|vector endpoint|zijderveld).{0,120}uncorrected for tilt|"
                r"(?:figure|fig ) ?[0-9a-z]*.{0,240}uncorrected for tilt|"
                r"uncorrected for tilt.{0,240}(?:figure|fig )",
                context,
            )
        )
        if correction_not_needed or figure_projection:
            return True
    if spec.category == "risk" and spec.subtype == "rotation":
        context = _fold(page_text[max(0, start - 90):min(len(page_text), end + 100)])
        return bool(
            re.search(r"\bno (?:evidence (?:for|of) )?(?:vertical axis |block )?rotation\b", context)
            or re.search(r"\b(?:vertical axis |block )?rotation.{0,70}\bunlikely\b", context)
            or re.search(r"\bargue(?:s|d)? against.{0,70}\brotation\b", context)
        )
    return False


def _scan_pattern_matches(
    pages: Sequence[_BasePage],
    indicator: str,
) -> tuple[_RawPatternMatch, ...]:
    matches: list[_RawPatternMatch] = []
    seen: set[tuple[str, str, int, int]] = set()
    for page_index, page in enumerate(pages):
        sentence_spans = _sentence_spans(page.text)
        for spec, pattern in _PATTERNS[indicator]:
            count = 0
            for match in pattern.finditer(page.text):
                matched_text = _clean_excerpt(match.group(0))[:320]
                match_scope = _scope_at(page, match.start(), match.end())
                if indicator == "R3" and match.end() > match.start():
                    first_sentence = _sentence_span_index(sentence_spans, match.start())
                    last_sentence = _sentence_span_index(
                        sentence_spans,
                        match.end() - 1,
                    )
                    if first_sentence is None or first_sentence != last_sentence:
                        continue
                    terminal = _scope_at(page, match.end() - 1, match.end())
                    if terminal.scope_id != match_scope.scope_id:
                        continue
                if spec.category == "bridge" and _false_carried_out_bridge(matched_text):
                    continue
                if _negated_carrier_mention(page.text, spec, match.start(), match.end()):
                    continue
                if _non_result_diagnostic(page.text, spec, match.start(), match.end()):
                    continue
                if _negation_collision(page.text, spec, match.start(), match.end()):
                    continue
                dedupe_key = (spec.category, spec.subtype, page.number, match.start() // 70)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                reference_noise = bool(
                    (page.reference_cutoff is not None and match.start() >= page.reference_cutoff)
                    or _bibliography_entry_noise(page.text, match.start(), match.end())
                )
                matches.append(
                    _RawPatternMatch(
                        spec=spec,
                        page_index=page_index,
                        matched_text=matched_text,
                        reference_noise=reference_noise,
                        section=_section_at(page, match.start(), reference_noise),
                        quote=_quote(page.text, match.start(), match.end()),
                        start=match.start(),
                        end=match.end(),
                        scope=match_scope,
                    )
                )
                count += 1
                if count >= 8:
                    break
    return tuple(matches)


def _rank_pattern_matches(
    matches: Sequence[_RawPatternMatch],
    pages: Sequence[_PageInfo],
    profile: _TargetProfile,
    competing_profiles: Sequence[_TargetProfile],
    multi_target: bool,
    indicator: str,
    reliability: float,
    allow_structured_scope: bool,
    allow_discourse_scope: bool,
) -> list[dict[str, Any]]:
    anchor_pages = [page.number for page in pages if page.anchor_kind != "none"]
    hits: list[dict[str, Any]] = []
    for raw in matches:
        page = pages[raw.page_index]
        binding, target_anchor, distance, local_target, scope, scope_competitors = _binding(
            page,
            raw.start,
            raw.end,
            profile,
            anchor_pages,
            multi_target,
            competing_profiles,
            raw.scope,
            allow_structured_scope,
            allow_discourse_scope,
        )
        noise_factor = 0.05 if raw.reference_noise else 1.0
        confidence_factor = 0.72 + 0.28 * raw.spec.confidence
        relevance = binding * reliability * confidence_factor * noise_factor
        if multi_target and not local_target:
            claim_text = raw.matched_text
        elif target_anchor in {"target_same_paragraph", "target_same_table_row"}:
            claim_text = _clean_excerpt(scope.text)
        elif scope.kind == "table_row":
            claim_text = _clean_excerpt(scope.text)
        else:
            claim_text = _clean_excerpt(_sentence_at(page, raw.start, raw.end))
        claim_text = _bounded_claim_text(claim_text, raw.matched_text)
        hits.append(
            {
                "indicator": indicator,
                "category": raw.spec.category,
                "subtype": raw.spec.subtype,
                "label": raw.spec.label,
                "matched_text": raw.matched_text,
                "claim_text": claim_text,
                "claim_text_sha256": hashlib.sha256(
                    claim_text.encode("utf-8")
                ).hexdigest(),
                "target_relevance": round(relevance, 4),
                "binding": round(binding, 4),
                "target_anchor": target_anchor,
                "local_target": local_target,
                "target_distance_chars": distance,
                "scope_id": scope.scope_id,
                "scope_kind": scope.kind,
                "scope_bounded": scope.bounded,
                "scope_target_exclusive": not scope_competitors,
                "scope_competing_targets": list(scope_competitors),
                "reference_noise": raw.reference_noise,
                "page": page.number,
                "section": raw.section,
                "quote": raw.quote,
                "start": raw.start,
                "end": raw.end,
                "source": page.source,
            }
        )
    return sorted(
        hits,
        key=lambda hit: (
            hit["reference_noise"],
            -hit["target_relevance"],
            not hit["local_target"],
            hit["page"],
            hit["start"],
        ),
    )


def best_r3_connected_chain(hits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the best method/carrier/bridge chain, preferring page locality."""

    categories = {
        category: [
            dict(hit)
            for hit in hits
            if hit.get("category") == category
            and not hit.get("reference_noise")
            and float(hit.get("target_relevance") or 0.0) >= 0.12
        ][:12]
        for category in ("method", "carrier", "bridge")
    }
    missing = [category for category, members in categories.items() if not members]
    if missing:
        return {
            "complete": False,
            "connected": False,
            "connection": "incomplete",
            "missing": missing,
            "page_span": None,
            "pages": [],
            "score": 0.0,
            "hits": {},
        }

    best: tuple[tuple[int, float, int], tuple[dict[str, Any], ...]] | None = None
    for combination in itertools.product(
        categories["method"], categories["carrier"], categories["bridge"]
    ):
        pages = [int(hit["page"]) for hit in combination]
        page_span = max(pages) - min(pages)
        locality_tier = 3 if page_span == 0 else 2 if page_span == 1 else 1 if page_span == 2 else 0
        mean_relevance = sum(float(hit["target_relevance"]) for hit in combination) / 3.0
        local_target_bonus = 0.025 * sum(bool(hit.get("local_target")) for hit in combination)
        chain_score = mean_relevance + local_target_bonus
        rank = (locality_tier, chain_score, -page_span)
        if best is None or rank > best[0]:
            best = (rank, combination)

    assert best is not None
    combination = best[1]
    pages = sorted({int(hit["page"]) for hit in combination})
    page_span = max(pages) - min(pages)
    connection = "same_page" if page_span == 0 else "adjacent_pages" if page_span == 1 else "disconnected"
    selected = dict(zip(("method", "carrier", "bridge"), combination))
    return {
        "complete": True,
        "connected": page_span <= 1,
        "connection": connection,
        "missing": [],
        "page_span": page_span,
        "pages": pages,
        "score": round(best[0][1], 4),
        "hits": selected,
    }


class PaperTextRetriever:
    """Reusable retriever for all targets belonging to one paper.

    Page normalization is performed once during construction. Pattern scans
    are lazy and cached separately for R3 and R5. Target-specific calls only
    locate anchors, bind cached matches, and rank the resulting evidence.
    """

    def __init__(self, cache_row: Mapping[str, Any]) -> None:
        if not isinstance(cache_row, Mapping):
            raise TypeError("cache_row must be a mapping")
        self._base_pages = tuple(_prepare_base_pages(cache_row))
        self._reliability = _source_reliability(cache_row, self._base_pages)
        self._pattern_cache: dict[str, tuple[_RawPatternMatch, ...]] = {}
        self._target_page_cache: dict[_TargetProfile, tuple[_PageInfo, ...]] = {}
        self._indicator_scan_counts = {indicator: 0 for indicator in _PATTERNS}

    def _pages_for(self, profile: _TargetProfile) -> tuple[_PageInfo, ...]:
        pages = self._target_page_cache.get(profile)
        if pages is None:
            pages = tuple(_bind_pages(self._base_pages, profile))
            self._target_page_cache[profile] = pages
        return pages

    def _matches_for(self, indicator: str) -> tuple[_RawPatternMatch, ...]:
        matches = self._pattern_cache.get(indicator)
        if matches is None:
            matches = _scan_pattern_matches(self._base_pages, indicator)
            self._pattern_cache[indicator] = matches
            self._indicator_scan_counts[indicator] += 1
        return matches

    def cache_info(self) -> dict[str, Any]:
        """Return cache diagnostics without exposing mutable cached objects."""

        return {
            "page_count": len(self._base_pages),
            "cached_indicators": sorted(self._pattern_cache),
            "indicator_scan_counts": dict(self._indicator_scan_counts),
            "raw_match_counts": {
                indicator: len(matches)
                for indicator, matches in sorted(self._pattern_cache.items())
            },
            "cached_target_profiles": len(self._target_page_cache),
        }

    def find_target_anchor(self, target: Any) -> dict[str, Any]:
        """Return the article-level anchor state for one target."""

        profile = _target_profile(target)
        return _anchor_state(profile, self._pages_for(profile))

    def retrieve(
        self,
        target: Any,
        multi_target: bool = False,
        indicator: str = "R3",
        *,
        competing_targets: Sequence[Any] = (),
        allow_structured_scope: bool = True,
        allow_discourse_scope: bool = False,
        limit: int = 30,
        per_category_limit: int = 10,
    ) -> dict[str, Any]:
        """Retrieve and rank cached paper matches for one target."""

        indicator_name = str(indicator).strip().upper()
        if indicator_name not in _PATTERNS:
            raise ValueError("indicator must be 'R3' or 'R5'")
        if limit < 0 or per_category_limit < 0:
            raise ValueError("limits must be non-negative")
        if not isinstance(allow_structured_scope, bool) or not isinstance(
            allow_discourse_scope,
            bool,
        ):
            raise TypeError("scope policy flags must be booleans")
        if allow_discourse_scope and not allow_structured_scope:
            raise ValueError("discourse scope requires structured scope")
        if isinstance(competing_targets, (str, bytes)):
            raise TypeError("competing_targets must be a sequence of target names")

        profile = _target_profile(target)
        competing_profiles = tuple(
            competitor_profile
            for competitor in competing_targets
            if (competitor_profile := _target_profile(competitor)).folded
            and competitor_profile.folded != profile.folded
        )
        if not allow_structured_scope:
            scope_binding_status = "disabled_by_policy"
        elif multi_target and not competing_profiles:
            scope_binding_status = "disabled_without_competing_target_universe"
        elif multi_target:
            scope_binding_status = "enabled_with_competing_target_exclusion"
        else:
            scope_binding_status = "not_required_single_target"
        pages = self._pages_for(profile)
        anchor = _anchor_state(profile, pages)
        all_hits = _rank_pattern_matches(
            self._matches_for(indicator_name),
            pages,
            profile,
            competing_profiles,
            bool(multi_target),
            indicator_name,
            self._reliability,
            allow_structured_scope,
            allow_discourse_scope,
        )
        category_names = (
            ("method", "diagnostic", "carrier", "bridge", "verdict")
            if indicator_name == "R3"
            else ("risk", "rescue")
        )
        categories = {
            category: [hit for hit in all_hits if hit["category"] == category][
                :per_category_limit
            ]
            for category in category_names
        }
        result = {
            "target": "" if target is None else str(target),
            "indicator": indicator_name,
            "multi_target": bool(multi_target),
            "scope_binding": {
                "status": scope_binding_status,
                "competing_targets": [profile.folded for profile in competing_profiles],
                "policy": (
                    "same sentence, exclusive bounded paragraph/table row, or an exact-target "
                    "forward discourse window with sibling exclusion"
                ),
                "structured_scope_enabled": allow_structured_scope,
                "discourse_scope_enabled": allow_discourse_scope,
            },
            "source_available": bool(pages),
            "target_anchor": anchor,
            "hits": all_hits[:limit],
            "categories": categories,
            "provenance": "original paper page cache only",
        }
        if indicator_name == "R3":
            result["best_chain"] = best_r3_connected_chain(all_hits)
        return result


def retrieve_descriptions(
    cache_row: Mapping[str, Any],
    target: Any,
    multi_target: bool = False,
    indicator: str = "R3",
    *,
    competing_targets: Sequence[Any] = (),
    allow_structured_scope: bool = True,
    allow_discourse_scope: bool = False,
    limit: int = 30,
    per_category_limit: int = 10,
) -> dict[str, Any]:
    """Retrieve target-relevant R3 or R5 descriptions from a page-cache row.

    ``hits`` is globally ordered by target relevance. ``categories`` retains
    that order within the indicator's semantic families.  Every hit contains
    page, section, quote, match offsets, and reference-noise provenance.
    """

    return PaperTextRetriever(cache_row).retrieve(
        target,
        multi_target,
        indicator,
        competing_targets=competing_targets,
        allow_structured_scope=allow_structured_scope,
        allow_discourse_scope=allow_discourse_scope,
        limit=limit,
        per_category_limit=per_category_limit,
    )


def retrieve_target_descriptions(
    cache_row: Mapping[str, Any],
    target: Any,
    multi_target: bool = False,
    indicator: str = "R3",
    *,
    competing_targets: Sequence[Any] = (),
    allow_structured_scope: bool = True,
    allow_discourse_scope: bool = False,
    limit: int = 30,
    per_category_limit: int = 10,
) -> dict[str, Any]:
    """Compatibility alias with a name that emphasizes target binding."""

    return retrieve_descriptions(
        cache_row,
        target,
        multi_target,
        indicator,
        competing_targets=competing_targets,
        allow_structured_scope=allow_structured_scope,
        allow_discourse_scope=allow_discourse_scope,
        limit=limit,
        per_category_limit=per_category_limit,
    )
