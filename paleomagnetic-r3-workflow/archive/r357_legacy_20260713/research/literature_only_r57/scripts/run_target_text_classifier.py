#!/usr/bin/env python3
"""Evaluate target-centered literature-only R5/R7 classification.

Training may use historical R5/R7 labels, but prediction features are built
only from REFNO, RESULTNO, ROCKNAME and original-paper page text. PMAGRESULT
and ROCKUNIT are never read. Splits are grouped by REFNO to prevent the same
paper from appearing in both training and test data.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "research/literature_only_r57/outputs"
sys.path.insert(0, str(ROOT / "scripts"))
import r5_r7_expert_rules as legacy_rules  # noqa: E402


R5_PATTERNS = {
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

R7_PATTERNS = {
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

STOPWORDS = {
    "the", "and", "that", "with", "from", "this", "were", "was", "are", "for", "have", "has", "had",
    "into", "their", "these", "those", "which", "than", "then", "also", "between", "during", "after",
    "before", "been", "being", "using", "used", "show", "shown", "result", "results", "data", "study",
    "figure", "table", "paper", "sample", "samples", "site", "sites", "rock", "rocks", "formation",
    "group", "complex", "component", "components", "magnetic", "palaeomagnetic", "paleomagnetic",
}

GENERIC_TARGET = {
    "formation", "group", "member", "suite", "complex", "rocks", "rock", "beds", "series", "combined",
    "component", "components", "dykes", "dikes", "basalts", "granites", "volcanics", "intrusives",
    "sediments", "sandstones", "limestones",
}


@dataclass
class TextRecord:
    refno: str
    resultno: str
    target: str
    split: str
    gold_r5: int
    gold_r7: int
    text: str
    source_available: bool
    exact_anchor: bool
    target_pages: int
    pattern_flags: dict[str, int]
    legacy_r5: int
    legacy_r7: int


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def stable_split(refno: str) -> str:
    value = int(hashlib.sha256(refno.encode("utf-8")).hexdigest()[:12], 16) / float(16**12 - 1)
    if value < 0.20:
        return "test"
    if value < 0.35:
        return "dev"
    return "train"


def target_tokens(target: str) -> list[str]:
    tokens = [token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", target)]
    return list(dict.fromkeys(token for token in tokens if token not in GENERIC_TARGET and len(token) >= 4))


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def anchored_pages(cache_row: dict[str, Any], target: str) -> tuple[list[dict[str, Any]], bool]:
    target_norm = normalized(target)
    tokens = target_tokens(target)
    selected: list[dict[str, Any]] = []
    exact = False
    for page in cache_row.get("pages") or []:
        text = str(page.get("text") or "")
        norm = normalized(text)
        hits = sum(token in norm for token in tokens)
        exact_here = bool(target_norm and target_norm in norm)
        majority = bool(tokens and hits >= max(1, math.ceil(len(tokens) * 0.6)))
        if exact_here or majority:
            selected.append(page)
            exact = exact or exact_here
    return selected, exact


def windows(text: str, patterns: dict[str, str], tokens: list[str], radius: int = 430, limit: int = 14) -> list[str]:
    found: list[tuple[int, str]] = []
    seen: set[int] = set()
    for pattern in patterns.values():
        for match in re.finditer(pattern, text, flags=re.I | re.S):
            bucket = match.start() // 350
            if bucket in seen:
                continue
            seen.add(bucket)
            found.append((match.start(), clean(text[max(0, match.start() - radius) : match.end() + radius])))
    for token in tokens:
        for match in re.finditer(re.escape(token), text, flags=re.I):
            bucket = match.start() // 350
            if bucket in seen:
                continue
            seen.add(bucket)
            found.append((match.start(), clean(text[max(0, match.start() - radius) : match.end() + radius])))
            if len(found) >= limit * 2:
                break
    found.sort(key=lambda item: item[0])
    return [value[:1100] for _, value in found[:limit]]


def build_document(cache_row: dict[str, Any], target: str) -> tuple[str, bool, int, dict[str, int]]:
    pages = cache_row.get("pages") or []
    if not pages:
        return f"target {target}", False, 0, {}
    anchored, exact = anchored_pages(cache_row, target)
    multi_target = len(anchored) > 0
    selected = anchored[:5] if anchored else pages[:2]
    patterns = {**R5_PATTERNS, **R7_PATTERNS}
    tokens = target_tokens(target)
    chunks = [f"target {target}", clean(str(pages[0].get("text") or ""))[:1200]]
    for page in selected:
        text = str(page.get("text") or "")
        chunks.extend(windows(text, patterns, tokens, limit=10 if multi_target else 7))
    document = "\n".join(dict.fromkeys(chunk for chunk in chunks if chunk))[:14000]
    flags = {name: int(re.search(pattern, document, flags=re.I | re.S) is not None) for name, pattern in patterns.items()}
    flags["exact_target_anchor"] = int(exact)
    flags["target_page_count_ge2"] = int(len(anchored) >= 2)
    flags["source_missing"] = 0
    return document, exact, len(anchored), flags


def token_features(text: str, target: str, flags: dict[str, int]) -> set[str]:
    words = [word for word in re.findall(r"[a-z][a-z0-9'-]{2,}", text.casefold()) if word not in STOPWORDS]
    features = {f"w:{word}" for word in words}
    features.update(f"b:{left}_{right}" for left, right in zip(words, words[1:]) if left != right)
    features.update(f"flag:{name}" for name, value in flags.items() if value)
    target_sample = target.casefold()
    for name, pattern in (
        ("volcanic", r"basalt|lava|tuff|ignimbrite|rhyolite|andesite|volcan"),
        ("sedimentary", r"sandstone|limestone|dolomite|shale|red bed|redbed|sediment"),
        ("intrusive", r"dyke|dike|sill|dolerite|diabase|granite|gabbro|pluton|intrus"),
        ("metamorphic", r"gneiss|schist|slate|phyllite|metamorph"),
    ):
        if re.search(pattern, target_sample):
            features.add(f"target_type:{name}")
    return features


class BernoulliRiskModel:
    def __init__(self, min_df: int = 3, max_features: int = 8000, alpha: float = 1.0):
        self.min_df = min_df
        self.max_features = max_features
        self.alpha = alpha
        self.base = 0.0
        self.weights: dict[str, float] = {}

    def fit(self, docs: list[set[str]], labels: list[int]) -> None:
        # Risk class is label 0; positive score means stronger R=0 evidence.
        n0 = sum(label == 0 for label in labels)
        n1 = sum(label == 1 for label in labels)
        df0: Counter[str] = Counter()
        df1: Counter[str] = Counter()
        for doc, label in zip(docs, labels):
            (df0 if label == 0 else df1).update(doc)
        candidates = []
        for feature in set(df0) | set(df1):
            total = df0[feature] + df1[feature]
            if total < self.min_df or total > 0.96 * len(labels):
                continue
            p0 = (df0[feature] + self.alpha) / (n0 + 2 * self.alpha)
            p1 = (df1[feature] + self.alpha) / (n1 + 2 * self.alpha)
            weight = math.log(p0 / (1 - p0)) - math.log(p1 / (1 - p1))
            candidates.append((abs(weight), feature, weight, p0, p1))
        candidates.sort(reverse=True)
        selected = candidates[: self.max_features]
        self.weights = {feature: weight for _, feature, weight, _, _ in selected}
        prior0 = (n0 + self.alpha) / (len(labels) + 2 * self.alpha)
        prior1 = 1.0 - prior0
        self.base = math.log(prior0 / prior1)
        for _, _, _, p0, p1 in selected:
            self.base += math.log((1 - p0) / (1 - p1))

    def score(self, doc: set[str]) -> float:
        return self.base + sum(self.weights.get(feature, 0.0) for feature in doc)

    def top_features(self, limit: int = 30) -> dict[str, list[tuple[str, float]]]:
        ranked = sorted(self.weights.items(), key=lambda item: item[1], reverse=True)
        return {"risk_r0": ranked[:limit], "support_r1": ranked[-limit:][::-1]}


class NumericLogisticRiskModel:
    def __init__(self, epochs: int = 1800, learning_rate: float = 0.04, l2: float = 0.02):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.bias = 0.0

    def fit(self, x: list[list[float]], labels: list[int]) -> None:
        matrix = np.asarray(x, dtype=float)
        target = np.asarray([1.0 if label == 0 else 0.0 for label in labels], dtype=float)
        self.mean = matrix.mean(axis=0)
        self.std = matrix.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        matrix = np.clip((matrix - self.mean) / self.std, -8.0, 8.0)
        target = np.clip(target, 0.03, 0.97)
        target_logit = np.log(target / (1.0 - target))
        design = np.concatenate([matrix, np.ones((matrix.shape[0], 1), dtype=float)], axis=1)
        regularizer = self.l2 * np.eye(design.shape[1], dtype=float)
        regularizer[-1, -1] = 0.0
        with np.errstate(all="ignore"):
            solution = np.linalg.pinv(design.T @ design + regularizer) @ design.T @ target_logit
        self.weights = np.nan_to_num(solution[:-1], nan=0.0, posinf=8.0, neginf=-8.0)
        self.bias = float(np.nan_to_num(solution[-1], nan=0.0, posinf=8.0, neginf=-8.0))

    def score(self, values: list[float]) -> float:
        assert self.mean is not None and self.std is not None and self.weights is not None
        row = np.clip((np.asarray(values, dtype=float) - self.mean) / self.std, -8.0, 8.0)
        return float(row @ self.weights + self.bias)


def target_type(target: str) -> str:
    sample = target.casefold()
    for name, pattern in (
        ("volcanic", r"basalt|lava|tuff|ignimbrite|rhyolite|andesite|volcan"),
        ("sedimentary", r"sandstone|limestone|dolomite|shale|red bed|redbed|sediment"),
        ("intrusive", r"dyke|dike|sill|dolerite|diabase|granite|gabbro|pluton|intrus"),
        ("metamorphic", r"gneiss|schist|slate|phyllite|metamorph"),
    ):
        if re.search(pattern, sample):
            return name
    return "unclear"


def concept_feature_names(indicator: str) -> list[str]:
    patterns = R5_PATTERNS if indicator == "R5" else R7_PATTERNS
    return [
        "source_available", "exact_anchor", "target_pages_ge1", "target_pages_ge2", "target_pages_ge4",
        "type_volcanic", "type_sedimentary", "type_intrusive", "type_metamorphic", "type_unclear",
        *[f"pattern_{name}" for name in patterns],
    ]


def concept_features(record: TextRecord, indicator: str) -> list[float]:
    kind = target_type(record.target)
    patterns = R5_PATTERNS if indicator == "R5" else R7_PATTERNS
    return [
        float(record.source_available), float(record.exact_anchor), float(record.target_pages >= 1),
        float(record.target_pages >= 2), float(record.target_pages >= 4),
        *[float(kind == name) for name in ("volcanic", "sedimentary", "intrusive", "metamorphic", "unclear")],
        *[float(record.pattern_flags.get(name, 0)) for name in patterns],
    ]


def choose_threshold(scores: list[float], gold: list[int], min_zero_precision: float | None = None) -> dict[str, float]:
    candidates = sorted(set(scores))
    if len(candidates) > 800:
        candidates = [candidates[int(i * (len(candidates) - 1) / 799)] for i in range(800)]
    candidates = [max(scores) + 1.0] + candidates
    best = {"threshold": candidates[0], "accuracy": 0.0, "zero_precision": 1.0, "zero_recall": 0.0}
    for threshold in candidates:
        pred = [0 if score >= threshold else 1 for score in scores]
        correct = sum(g == p for g, p in zip(gold, pred))
        predicted_zero = sum(p == 0 for p in pred)
        true_zero = sum(g == 0 and p == 0 for g, p in zip(gold, pred))
        actual_zero = sum(g == 0 for g in gold)
        precision = true_zero / predicted_zero if predicted_zero else 1.0
        recall = true_zero / actual_zero if actual_zero else 0.0
        accuracy = correct / len(gold)
        if min_zero_precision is not None and precision < min_zero_precision:
            continue
        candidate_key = (accuracy, recall, threshold)
        best_key = (best["accuracy"], best["zero_recall"], best["threshold"])
        if candidate_key > best_key:
            best = {"threshold": threshold, "accuracy": accuracy, "zero_precision": precision, "zero_recall": recall}
    return best


def metric(gold: list[int], pred: list[int]) -> dict[str, Any]:
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
    tn = sum(g == 0 and p == 0 for g, p in zip(gold, pred))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, pred))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, pred))
    def f1(label: int) -> float:
        a = sum(g == label and p == label for g, p in zip(gold, pred))
        b = sum(g != label and p == label for g, p in zip(gold, pred))
        c = sum(g == label and p != label for g, p in zip(gold, pred))
        return 2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0
    return {
        "rows": len(gold), "accuracy": (tp + tn) / len(gold), "macro_f1": (f1(0) + f1(1)) / 2,
        "f1_0": f1(0), "f1_1": f1(1), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "pred_zero": sum(p == 0 for p in pred),
    }


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                cache[clean(row.get("REFNO"))] = row
    return cache


def build_records(targets: Path, page_cache: Path) -> list[TextRecord]:
    frame = pd.read_excel(targets, usecols=["REFNO", "RESULTNO", "ROCKNAME", "R5", "R7"], dtype=str).fillna("")
    cache = load_cache(page_cache)
    records = []
    for row in frame.to_dict(orient="records"):
        refno = clean(row["REFNO"]); target = clean(row["ROCKNAME"])
        cache_row = cache.get(refno, {"pages": []})
        document, exact, target_page_count, flags = build_document(cache_row, target)
        source_text = "\n".join(str(page.get("text") or "") for page in cache_row.get("pages") or [])
        r5_hits = legacy_rules.r5_zero_evidence(source_text, target) if source_text else []
        r7_hits = legacy_rules.r7_zero_evidence(source_text, target) if source_text else []
        records.append(TextRecord(
            refno=refno, resultno=clean(row["RESULTNO"]), target=target, split=stable_split(refno),
            gold_r5=int(float(row["R5"])), gold_r7=int(float(row["R7"])), text=document,
            source_available=bool(cache_row.get("pages")), exact_anchor=exact, target_pages=target_page_count,
            pattern_flags=flags, legacy_r5=0 if r5_hits else 1, legacy_r7=0 if r7_hits else 1,
        ))
    return records


def run_indicator(records: list[TextRecord], indicator: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = [record for record in records if record.split == "train"]
    dev = [record for record in records if record.split == "dev"]
    test = [record for record in records if record.split == "test"]
    model = BernoulliRiskModel()
    train_docs = [token_features(record.text, record.target, record.pattern_flags) for record in train]
    train_gold = [getattr(record, f"gold_{indicator.lower()}") for record in train]
    model.fit(train_docs, train_gold)
    concept_model = NumericLogisticRiskModel()
    concept_model.fit([concept_features(record, indicator) for record in train], train_gold)

    dev_scores = [model.score(token_features(record.text, record.target, record.pattern_flags)) for record in dev]
    dev_concept_scores = [concept_model.score(concept_features(record, indicator)) for record in dev]
    dev_gold = [getattr(record, f"gold_{indicator.lower()}") for record in dev]
    accuracy_threshold = choose_threshold(dev_scores, dev_gold)
    precision_threshold = choose_threshold(dev_scores, dev_gold, min_zero_precision=0.75)
    concept_accuracy_threshold = choose_threshold(dev_concept_scores, dev_gold)
    concept_precision_threshold = choose_threshold(dev_concept_scores, dev_gold, min_zero_precision=0.75)

    def evaluate_partition(partition: list[TextRecord]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        scores = [model.score(token_features(record.text, record.target, record.pattern_flags)) for record in partition]
        concept_scores = [concept_model.score(concept_features(record, indicator)) for record in partition]
        gold = [getattr(record, f"gold_{indicator.lower()}") for record in partition]
        legacy = [getattr(record, f"legacy_{indicator.lower()}") for record in partition]
        pred_accuracy = [0 if score >= accuracy_threshold["threshold"] else 1 for score in scores]
        pred_precision = [0 if score >= precision_threshold["threshold"] else 1 for score in scores]
        pred_concept_accuracy = [0 if score >= concept_accuracy_threshold["threshold"] else 1 for score in concept_scores]
        pred_concept_precision = [0 if score >= concept_precision_threshold["threshold"] else 1 for score in concept_scores]
        rows = []
        for record, score, concept_score, g, pa, pp, pca, pcp, old in zip(
            partition, scores, concept_scores, gold, pred_accuracy, pred_precision,
            pred_concept_accuracy, pred_concept_precision, legacy,
        ):
            rows.append({
                "REFNO": record.refno, "RESULTNO": record.resultno, "TARGET": record.target, "GOLD": g,
                "risk_score": score, "concept_risk_score": concept_score,
                "pred_accuracy": pa, "pred_precision": pp,
                "pred_concept_accuracy": pca, "pred_concept_precision": pcp, "legacy_rule": old,
                "source_available": record.source_available, "exact_anchor": record.exact_anchor,
                "target_pages": record.target_pages,
            })
        return {
            "all_one": metric(gold, [1] * len(gold)),
            "legacy_rule": metric(gold, legacy),
            "text_calibrated_accuracy": metric(gold, pred_accuracy),
            "text_calibrated_precision": metric(gold, pred_precision),
            "concept_calibrated_accuracy": metric(gold, pred_concept_accuracy),
            "concept_calibrated_precision": metric(gold, pred_concept_precision),
        }, rows

    test_metrics, test_rows = evaluate_partition(test)
    all_metrics, _ = evaluate_partition(records)
    summary = {
        "train_rows": len(train), "dev_rows": len(dev), "test_rows": len(test),
        "accuracy_threshold": accuracy_threshold, "precision_threshold": precision_threshold,
        "concept_accuracy_threshold": concept_accuracy_threshold,
        "concept_precision_threshold": concept_precision_threshold,
        "test": test_metrics, "all_with_dev_calibration": all_metrics,
        "top_features": model.top_features(),
        "concept_coefficients": sorted(
            zip(concept_feature_names(indicator), concept_model.weights.tolist()),
            key=lambda item: abs(item[1]), reverse=True,
        ) if concept_model.weights is not None else [],
    }
    return summary, test_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=ROOT / "data/targets.xlsx")
    parser.add_argument("--page-cache", type=Path, default=ROOT / "outputs/r357_targets_07081817/r3_page_source_cache.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or OUTPUT_ROOT / f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)
    records = build_records(args.targets, args.page_cache)
    with (output / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({
                "REFNO": record.refno, "RESULTNO": record.resultno, "TARGET": record.target,
                "split": record.split, "gold_R5": record.gold_r5, "gold_R7": record.gold_r7,
                "text": record.text, "source_available": record.source_available,
                "exact_anchor": record.exact_anchor, "target_pages": record.target_pages,
                "pattern_flags": record.pattern_flags,
            }, ensure_ascii=False) + "\n")
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "method": "target-centered paper text plus REFNO-grouped Bernoulli relevance calibration",
        "runtime_sources": ["REFNO", "RESULTNO", "ROCKNAME", "original paper page text"],
        "forbidden_runtime_sources": ["PMAGRESULT", "ROCKUNIT"],
        "embedding_backend": "offline lexical model; Qwen endpoint returned HTTP 502",
        "indicators": {},
    }
    case_rows = []
    for indicator in ("R5", "R7"):
        indicator_summary, rows = run_indicator(records, indicator)
        summary["indicators"][indicator] = indicator_summary
        for row in rows:
            case_rows.append({"indicator": indicator, **row})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "test_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0].keys()))
        writer.writeheader(); writer.writerows(case_rows)
    lines = [
        "# Literature-Only R5/R7 Target-Centered Experiment", "", f"Generated: {summary['created_at']}", "",
        "Scoring uses only target identifiers and original-paper text. The same REFNO never appears in both train and test.", "",
        "| Indicator | Method | Test accuracy | Macro-F1 | F1(0) | FP | FN | Pred 0 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for indicator, values in summary["indicators"].items():
        for method, m in values["test"].items():
            lines.append(f"| {indicator} | {method} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['f1_0']:.4f} | {m['fp']} | {m['fn']} | {m['pred_zero']} |")
    lines.extend(["", "Thresholds are calibrated only on the dev REFNO split; test metrics are held out."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "indicators": {k: v["test"] for k, v in summary["indicators"].items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
