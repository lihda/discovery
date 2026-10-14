#!/usr/bin/env python3
"""Audit R3 source-evidence conflicts against historical training labels.

``freeze`` derives target-bound scientific-evidence candidates without a label
source. ``evaluate`` verifies the frozen candidate and all source hashes before
opening fixed-training R3 cells. The report separates empirical conflict flags
from a mathematical error lower bound that applies only to classifiers
restricted to the exact frozen representation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts import experiment_r3_component_manifest as component
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import experiment_r3_component_manifest as component  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDENTITY = ROOT / "tmp/r3_distilled_training_preflight/identity.xlsx"
DEFAULT_PAGE_CACHE = ROOT / "tmp/mineru_cache_all_hard_risk.jsonl"
DEFAULT_PDF_DIR = ROOT / "data/raw_pdfs"
DEFAULT_TAXONOMY = ROOT / "data/calibration/r3_rockmag_text_distillation_v1.json"
DEFAULT_POSITIVE_KB = ROOT / "data/calibration/r3_positive_rockmag_expert_kb_v2.json"
DEFAULT_CRITERION = ROOT / "data/knowledge/r357_criterion_spec_v2.json"
DEFAULT_COMPONENT_MANIFEST = ROOT / "tmp/r3_component_manifest/component_manifest.jsonl"
DEFAULT_COMPONENT_RECEIPT = ROOT / "tmp/r3_component_manifest/freeze_receipt.json"
DEFAULT_TARGETS = ROOT / "data/targets.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_source_label_conflict_audit"

OWN_STUDY_RE = re.compile(
    r"\b(?:we|we\s+have)\s+(?:measured|performed|conducted|carried\s+out|"
    r"investigated|analysed|analyzed|used|applied|selected|collected|studied)\b|"
    r"\b(?:our|this|the\s+present)\s+(?:study|measurements?|experiments?|results?|samples?)\b|"
    r"\b(?:samples?|specimens?)\b.{0,100}\b(?:were|was|have\s+been)\s+"
    r"(?:measured|subjected|analysed|analyzed|investigated|heated|demagneti[sz]ed|tested)\b|"
    r"\b(?:measurements?|experiments?|analyses|tests?)\s+(?:were|was)\s+"
    r"(?:made|performed|conducted|carried\s+out)\b|"
    r"\b(?:results?|curves?|measurements?)\s+(?:show|showed|indicate|indicated|reveal|revealed)\b",
    re.I | re.S,
)
SIGNIFICANCE_RE = re.compile(
    r"(?:carrier|carried|carries|responsible\s+for|resides?\s+in).{0,180}"
    r"(?:remanence|magneti[sz]ation|ChRM|component|overprint)|"
    r"(?:remanence|magneti[sz]ation|ChRM|component|overprint).{0,180}"
    r"(?:carrier|carried|carries|responsible\s+for|resides?\s+in)|"
    r"(?:primary|secondary|overprint|remagneti[sz]|CRM|TRM|DRM|VRM|authigenic|"
    r"alteration).{0,180}(?:carrier|magnetic\s+mineral|magnetite|hematite|haematite|"
    r"maghemite|pyrrhotite|greigite)|"
    r"(?:carrier|magnetic\s+mineral|magnetite|hematite|haematite|maghemite|"
    r"pyrrhotite|greigite).{0,180}(?:primary|secondary|overprint|remagneti[sz]|"
    r"CRM|TRM|DRM|VRM|authigenic|alteration)",
    re.I | re.S,
)
TOKEN_RE = re.compile(r"[a-z][a-z0-9'-]{1,}", re.I)


def clean(value: Any) -> str:
    return component.clean(value)


def file_sha256(path: Path) -> str:
    return component.file_sha256(path)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    component.atomic_json(path, value)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    component.atomic_jsonl(path, rows)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def concept_categories(concepts: Iterable[str]) -> dict[str, set[str]]:
    output = {"method": set(), "diagnostic": set(), "carrier": set()}
    for concept in concepts:
        prefix, separator, name = str(concept).partition(":")
        if separator and prefix in output:
            output[prefix].add(name)
    return output


def own_study_scope(scope: Mapping[str, Any], text: str, direct_target: bool) -> bool:
    if OWN_STUDY_RE.search(text):
        return True
    if direct_target and scope.get("kind") in {"table_row", "figure_or_table_caption"}:
        return True
    return False


def bottom_k_shingle_sketch(text: str, *, size: int = 64) -> list[str]:
    tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
    width = 5 if len(tokens) >= 5 else 1
    shingles = {
        " ".join(tokens[index : index + width])
        for index in range(max(0, len(tokens) - width + 1))
    }
    hashes = sorted(
        hashlib.sha256(shingle.encode("utf-8")).hexdigest()[:16]
        for shingle in shingles
    )
    return hashes[:size]


def sketch_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _quote(scope: Mapping[str, Any], text: str) -> dict[str, Any]:
    normalized = component.normalized_text(text)
    return {
        "page": clean(scope.get("page")),
        "kind": clean(scope.get("kind")),
        "scope_id": clean(scope.get("scope_id")),
        "text_sha256": text_sha256(normalized),
        "text": normalized[:650],
    }


def classify_target_candidate(
    paper: Mapping[str, Any],
    target: Mapping[str, Any],
    scope_text: Mapping[str, str],
) -> dict[str, Any]:
    scope_meta = {
        str(scope["scope_id"]): scope for scope in paper.get("source_scopes", [])
    }
    direct_ids = set(target["scope_links"].get("direct_scope_ids", []))
    bridge_ids = set(target["scope_links"].get("bridged_scope_ids", []))
    linked_ids = direct_ids | bridge_ids
    high_proxy_scopes: list[dict[str, Any]] = []
    high_criterion_scopes: list[dict[str, Any]] = []
    own_method = False
    own_companion = False
    own_carrier = False
    significance_any = False
    all_concepts: set[str] = set()
    evidence_text_parts: list[str] = []

    for scope_id in sorted(linked_ids):
        meta = scope_meta.get(scope_id)
        text = scope_text.get(scope_id, "")
        if not meta or not text:
            continue
        concepts = set(meta.get("concepts", []))
        categories = concept_categories(concepts)
        direct = scope_id in direct_ids
        target_count = len(meta.get("target_hits", {}))
        exclusive = target_count == 1 or int(paper.get("target_count", 0)) == 1
        own = own_study_scope(meta, text, direct)
        method = bool(categories["method"])
        diagnostic = bool(categories["diagnostic"])
        carrier = bool(categories["carrier"])
        companion = diagnostic or carrier
        significance = bool(SIGNIFICANCE_RE.search(text))
        all_concepts.update(concepts)
        if own and method:
            own_method = True
        if own and companion:
            own_companion = True
        if own and carrier:
            own_carrier = True
        significance_any = significance_any or significance
        if concepts or significance:
            evidence_text_parts.append(text)
        if direct and exclusive and own and method and companion:
            high_proxy_scopes.append({"meta": meta, "text": text})
            if carrier and significance:
                high_criterion_scopes.append({"meta": meta, "text": text})

    proxy_status = (
        "HIGH_CONFIDENCE_SAME_SCOPE"
        if high_proxy_scopes
        else "MEDIUM_CONFIDENCE_LINKED"
        if own_method and own_companion and linked_ids
        else "NO_FROZEN_CANDIDATE"
    )
    criterion_status = (
        "HIGH_CONFIDENCE_COMPLETE_CHAIN"
        if high_criterion_scopes
        else "MEDIUM_CONFIDENCE_DISTRIBUTED_CHAIN"
        if own_method and own_carrier and significance_any
        else "INCOMPLETE_OR_ABSENT_CHAIN"
    )
    categories = concept_categories(all_concepts)
    concept_signature = {
        name: sorted(values) for name, values in categories.items()
    }
    evidence_text = component.normalized_text("\n".join(evidence_text_parts))
    structure = target.get("features", {})
    representation_payload = {
        "canonical_target": target["canonical_target"],
        "proxy_status": proxy_status,
        "criterion_status": criterion_status,
        "concept_signature": concept_signature,
        "direct_scope_count": structure.get("direct_scope_count", 0),
        "linked_scope_count": structure.get("linked_scope_count", 0),
        "sample_id_count": structure.get("sample_id_count", 0),
        "component_count": structure.get("component_count", 0),
        "evidence_text_sha256": text_sha256(evidence_text),
    }
    semantic_payload = {
        "proxy_status": proxy_status,
        "criterion_status": criterion_status,
        "concept_signature": concept_signature,
        "has_samples": bool(target.get("sample_ids")),
        "has_components": bool(target.get("components")),
    }
    quote_scopes = high_criterion_scopes + [
        item for item in high_proxy_scopes if item not in high_criterion_scopes
    ]
    if not quote_scopes:
        quote_scopes = [
            {"meta": scope_meta[scope_id], "text": scope_text[scope_id]}
            for scope_id in sorted(linked_ids)
            if scope_id in scope_meta and scope_id in scope_text
            and scope_meta[scope_id].get("concepts")
        ][:2]
    quotes = [_quote(item["meta"], item["text"]) for item in quote_scopes[:3]]
    return {
        "refno": paper["refno"],
        "target_id": target["target_id"],
        "canonical_target": target["canonical_target"],
        "rocknames": target["rocknames"],
        "resultnos": target["resultnos"],
        "proxy_support_status": proxy_status,
        "criterion_support_status": criterion_status,
        "concept_signature": concept_signature,
        "own_study_flags": {
            "method": own_method,
            "diagnostic_or_carrier": own_companion,
            "carrier": own_carrier,
            "remanence_significance": significance_any,
        },
        "scope_counts": {
            "direct": len(direct_ids),
            "bridged": len(bridge_ids),
            "high_proxy": len(high_proxy_scopes),
            "high_criterion": len(high_criterion_scopes),
        },
        "representation_fingerprint": text_sha256(
            json.dumps(representation_payload, sort_keys=True, separators=(",", ":"))
        ),
        "semantic_signature": text_sha256(
            json.dumps(semantic_payload, sort_keys=True, separators=(",", ":"))
        ),
        "evidence_text_sha256": text_sha256(evidence_text),
        "evidence_sketch": bottom_k_shingle_sketch(evidence_text),
        "review_quotes": quotes,
    }


def _load_component_inputs(
    identity_path: Path,
    page_cache_path: Path,
    pdf_dir: Path,
    taxonomy_path: Path,
    positive_kb_path: Path,
    component_manifest_path: Path,
    component_receipt_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    receipt, manifests = component.verify_frozen_manifest(
        component_receipt_path,
        component_manifest_path,
        identity_path,
        page_cache_path,
        pdf_dir,
        taxonomy_path,
        positive_kb_path,
    )
    pages = component.load_page_cache(
        page_cache_path, {str(paper["refno"]) for paper in manifests}
    )
    return receipt, manifests, pages


def freeze_candidates(
    identity_path: Path,
    page_cache_path: Path,
    pdf_dir: Path,
    taxonomy_path: Path,
    positive_kb_path: Path,
    criterion_path: Path,
    component_manifest_path: Path,
    component_receipt_path: Path,
    candidates_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Freeze evidence candidates. There is intentionally no label argument."""

    input_paths = {
        "identity": identity_path.resolve(),
        "page_cache": page_cache_path.resolve(),
        "taxonomy": taxonomy_path.resolve(),
        "positive_kb": positive_kb_path.resolve(),
        "criterion": criterion_path.resolve(),
        "component_manifest": component_manifest_path.resolve(),
        "component_receipt": component_receipt_path.resolve(),
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise ValueError(f"missing {name}: {path}")
    criterion = json.loads(input_paths["criterion"].read_text(encoding="utf-8"))
    if criterion.get("spec_version") != "r357_criterion_spec_v2":
        raise ValueError("unexpected criterion specification")
    component_receipt, manifests, pages = _load_component_inputs(
        input_paths["identity"], input_paths["page_cache"], pdf_dir.resolve(),
        input_paths["taxonomy"], input_paths["positive_kb"],
        input_paths["component_manifest"], input_paths["component_receipt"],
    )
    candidates: list[dict[str, Any]] = []
    for paper in manifests:
        scope_text = {
            str(scope["scope_id"]): str(scope["text"])
            for scope in component.extract_scopes(pages[str(paper["refno"])])
        }
        candidates.extend(
            classify_target_candidate(paper, target, scope_text)
            for target in paper["targets"]
        )
    candidates.sort(key=lambda row: (row["refno"], row["canonical_target"], row["target_id"]))
    atomic_jsonl(candidates_path.resolve(), candidates)
    counts = Counter(row["proxy_support_status"] for row in candidates)
    criterion_counts = Counter(row["criterion_support_status"] for row in candidates)
    receipt = {
        "schema_version": "r3_source_label_conflict_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "labels_opened": False,
        "candidate_artifact": {
            "path": str(candidates_path.resolve()),
            "sha256": file_sha256(candidates_path.resolve()),
            "target_candidates": len(candidates),
            "identity_rows": sum(len(row["resultnos"]) for row in candidates),
        },
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in input_paths.items()
        },
        "pdf_dir": {
            "path": str(pdf_dir.resolve()),
            "membership_sha256": component_receipt["membership"]["pdf_membership_sha256"],
        },
        "candidate_counts": {
            "proxy_support_status": dict(sorted(counts.items())),
            "criterion_support_status": dict(sorted(criterion_counts.items())),
        },
        "definitions": {
            "high_confidence_proxy": (
                "exclusive target-bound same source scope with an own-study performed "
                "method and a carrier or diagnostic concept"
            ),
            "high_confidence_criterion": (
                "high-confidence proxy plus a named carrier and an explicit connection "
                "to remanence significance in the same source scope"
            ),
            "medium_confidence": (
                "own-study method and companion evidence distributed across frozen "
                "target/sample/component-linked scopes"
            ),
            "no_frozen_candidate": (
                "the frozen extractor recovered no qualifying candidate; this is not "
                "a scientific assertion that evidence is absent from the paper"
            ),
        },
        "criterion_provenance": {
            "spec_version": criterion["spec_version"],
            "source_sha256": criterion.get("source_artifact", {}).get("sha256"),
            "r3_logic": (
                "eligible carrier investigation plus carrier identification plus "
                "comment on remanence significance"
            ),
        },
        "guardrails": {
            "allowed_inputs": list(input_paths),
            "forbidden_sources_not_read": [
                "held-out labels",
                "source metadata tables",
                "historical ROCKMAG cells",
                "prior predictions",
                "prior evaluations",
            ],
            "candidate_contains_labels": False,
            "candidate_contains_predictions": False,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
            "component_script_sha256": file_sha256(Path(inspect.getfile(component)).resolve()),
        },
    }
    atomic_json(receipt_path.resolve(), receipt)
    return receipt


def load_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("resultnos"):
                raise ValueError(f"invalid candidate row {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("candidate artifact is empty")
    return rows


def verify_candidate_freeze(
    receipt_path: Path,
    candidates_path: Path,
    input_paths: Mapping[str, Path],
    pdf_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "FROZEN_BEFORE_LABEL_ACCESS":
        raise ValueError("candidate receipt is not frozen before label access")
    if receipt.get("labels_opened") is not False:
        raise ValueError("candidate receipt reports label access")
    if file_sha256(candidates_path) != receipt.get("candidate_artifact", {}).get("sha256"):
        raise ValueError("candidate artifact hash mismatch")
    for name, path in input_paths.items():
        declared = receipt.get("inputs", {}).get(name, {}).get("sha256")
        if not path.is_file() or file_sha256(path) != declared:
            raise ValueError(f"source input changed after candidate freeze: {name}")
    if file_sha256(Path(__file__).resolve()) != receipt.get("script", {}).get("sha256"):
        raise ValueError("audit script changed after candidate freeze")
    if file_sha256(Path(inspect.getfile(component)).resolve()) != receipt.get("script", {}).get("component_script_sha256"):
        raise ValueError("component extraction script changed after candidate freeze")
    identity = component.load_training_identity(input_paths["identity"])
    selected, membership = component.select_pdf_identity(identity, pdf_dir.resolve())
    if membership["pdf_membership_sha256"] != receipt.get("pdf_dir", {}).get("membership_sha256"):
        raise ValueError("PDF-only membership changed after candidate freeze")
    candidates = load_candidates(candidates_path)
    candidate_resultnos = {
        resultno for row in candidates for resultno in row["resultnos"]
    }
    if candidate_resultnos != {row["RESULTNO"] for row in selected}:
        raise ValueError("candidate membership differs from frozen PDF-only identity")
    return receipt, candidates


def _expanded_rows(
    candidates: Sequence[Mapping[str, Any]], labels: Mapping[str, int]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        for resultno in candidate["resultnos"]:
            output.append({
                "candidate": candidate,
                "resultno": resultno,
                "label": int(labels[resultno]),
            })
    return output


def exact_representation_bound(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["candidate"]["representation_fingerprint"]].append(row)
    unavoidable = 0
    mixed: list[dict[str, Any]] = []
    for fingerprint, members in groups.items():
        counts = Counter(int(member["label"]) for member in members)
        if not counts[0] or not counts[1]:
            continue
        group_errors = min(counts[0], counts[1])
        unavoidable += group_errors
        first = members[0]["candidate"]
        mixed.append({
            "representation_fingerprint": fingerprint,
            "refno": first["refno"],
            "canonical_target": first["canonical_target"],
            "rocknames": first["rocknames"],
            "label_counts": {"0": counts[0], "1": counts[1]},
            "minimum_errors_in_group": group_errors,
            "resultnos_by_label": {
                "0": [member["resultno"] for member in members if member["label"] == 0][:10],
                "1": [member["resultno"] for member in members if member["label"] == 1][:10],
            },
            "proxy_support_status": first["proxy_support_status"],
            "criterion_support_status": first["criterion_support_status"],
            "review_quotes": first["review_quotes"][:1],
        })
    mixed.sort(key=lambda item: (-item["minimum_errors_in_group"], item["refno"], item["canonical_target"]))
    total = len(rows)
    return {
        "scope": (
            "deterministic full-coverage classifiers restricted to the exact frozen "
            "representation fingerprint"
        ),
        "mixed_representation_groups": len(mixed),
        "rows_in_mixed_groups": sum(sum(item["label_counts"].values()) for item in mixed),
        "minimum_unavoidable_errors": unavoidable,
        "error_rate_lower_bound": round(unavoidable / total, 8) if total else 0.0,
        "accuracy_upper_bound": round(1.0 - unavoidable / total, 8) if total else 1.0,
        "review_groups": mixed[:30],
        "limitation": (
            "This is not a mathematical ceiling for all source-only systems. A richer "
            "source representation or a newly recovered component locator may separate "
            "rows that this frozen representation treats as identical."
        ),
    }


def semantic_signature_diagnostic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[row["candidate"]["semantic_signature"]].append(int(row["label"]))
    errors = sum(
        min(counts[0], counts[1])
        for values in groups.values()
        if (counts := Counter(values))[0] and counts[1]
    )
    mixed = sum(len(set(values)) > 1 for values in groups.values())
    return {
        "semantic_signature_groups": len(groups),
        "mixed_label_groups": mixed,
        "majority_error_count": errors,
        "majority_accuracy": round(1.0 - errors / len(rows), 8),
        "interpretation": (
            "Coarse evidence-signature diagnostic only. It is not an unavoidable-error "
            "bound because target text and other source details are intentionally omitted."
        ),
    }


def near_duplicate_opposite_pairs(
    candidates: Sequence[Mapping[str, Any]], labels: Mapping[str, int]
) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["proxy_support_status"] == "NO_FROZEN_CANDIDATE":
            continue
        buckets[candidate["semantic_signature"]].append(candidate)
    pairs: list[dict[str, Any]] = []
    for members in buckets.values():
        if len(members) > 180:
            members = members[:180]
        for left_index, left in enumerate(members):
            left_labels = {labels[resultno] for resultno in left["resultnos"]}
            if len(left_labels) != 1:
                continue
            for right in members[left_index + 1 :]:
                if left["refno"] == right["refno"] and left["target_id"] == right["target_id"]:
                    continue
                right_labels = {labels[resultno] for resultno in right["resultnos"]}
                if len(right_labels) != 1 or left_labels == right_labels:
                    continue
                overlap = sketch_overlap(left["evidence_sketch"], right["evidence_sketch"])
                if overlap < 0.55:
                    continue
                pairs.append({
                    "sketch_overlap": round(overlap, 8),
                    "semantic_signature": left["semantic_signature"],
                    "left": {
                        "refno": left["refno"],
                        "resultnos": left["resultnos"][:5],
                        "rocknames": left["rocknames"],
                        "label": next(iter(left_labels)),
                        "review_quotes": left["review_quotes"][:1],
                    },
                    "right": {
                        "refno": right["refno"],
                        "resultnos": right["resultnos"][:5],
                        "rocknames": right["rocknames"],
                        "label": next(iter(right_labels)),
                        "review_quotes": right["review_quotes"][:1],
                    },
                })
    pairs.sort(key=lambda item: -item["sketch_overlap"])
    return pairs[:40]


def _example(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    return {
        "refno": candidate["refno"],
        "resultno": row["resultno"],
        "rocknames": candidate["rocknames"],
        "gold_r3": row["label"],
        "proxy_support_status": candidate["proxy_support_status"],
        "criterion_support_status": candidate["criterion_support_status"],
        "concept_signature": candidate["concept_signature"],
        "review_quotes": candidate["review_quotes"][:2],
    }


def audit_statistics(
    candidates: Sequence[Mapping[str, Any]], labels: Mapping[str, int]
) -> dict[str, Any]:
    rows = _expanded_rows(candidates, labels)
    high_proxy_zero = [
        row for row in rows
        if row["label"] == 0
        and row["candidate"]["proxy_support_status"] == "HIGH_CONFIDENCE_SAME_SCOPE"
    ]
    high_criterion_zero = [
        row for row in rows
        if row["label"] == 0
        and row["candidate"]["criterion_support_status"] == "HIGH_CONFIDENCE_COMPLETE_CHAIN"
    ]
    positive_no_candidate = [
        row for row in rows
        if row["label"] == 1
        and row["candidate"]["proxy_support_status"] == "NO_FROZEN_CANDIDATE"
    ]
    positive_no_high = [
        row for row in rows
        if row["label"] == 1
        and row["candidate"]["proxy_support_status"] != "HIGH_CONFIDENCE_SAME_SCOPE"
    ]
    positive_incomplete_criterion = [
        row for row in rows
        if row["label"] == 1
        and row["candidate"]["criterion_support_status"] != "HIGH_CONFIDENCE_COMPLETE_CHAIN"
    ]
    total = len(rows)
    label_counts = Counter(row["label"] for row in rows)

    def flag_record(values: Sequence[Mapping[str, Any]], denominator: int) -> dict[str, Any]:
        return {
            "rows": len(values),
            "denominator_rows": denominator,
            "rate": round(len(values) / denominator, 8) if denominator else 0.0,
        }

    empirical = {
        "high_confidence_proxy_support_but_gold_zero": flag_record(high_proxy_zero, total),
        "high_confidence_complete_criterion_chain_but_gold_zero": flag_record(high_criterion_zero, total),
        "gold_one_but_no_frozen_proxy_candidate": flag_record(positive_no_candidate, label_counts[1]),
        "gold_one_but_no_high_confidence_proxy": flag_record(positive_no_high, label_counts[1]),
        "gold_one_but_incomplete_frozen_criterion_chain": flag_record(positive_incomplete_criterion, label_counts[1]),
        "interpretation": {
            "support_gold_zero": (
                "Empirical conflict candidates requiring manual confirmation of target "
                "binding and own-study scope; not automatically proven label errors."
            ),
            "gold_one_no_candidate": (
                "Frozen retrieval/extraction misses or possible criterion-practice mismatch; "
                "absence of a recovered candidate is not evidence that the paper lacks evidence."
            ),
        },
    }
    exact_bound = exact_representation_bound(rows)
    near_pairs = near_duplicate_opposite_pairs(candidates, labels)
    maximum_errors_at_ninety = total - math.ceil(0.90 * total)
    return {
        "population": {
            "rows": total,
            "paper_groups": len({row["candidate"]["refno"] for row in rows}),
            "target_candidates": len(candidates),
            "label_counts": {"0": label_counts[0], "1": label_counts[1]},
        },
        "empirical_conflict_flags": empirical,
        "criterion_logic_audit": {
            "potential_direct_logical_conflicts": len(high_criterion_zero),
            "potential_direct_logical_conflict_rate": round(len(high_criterion_zero) / total, 8),
            "gold_one_without_complete_frozen_chain": len(positive_incomplete_criterion),
            "assessment": (
                "A manually confirmed complete source chain with gold R3=0 conflicts with "
                "the uploaded criterion's positive condition. Gold R3=1 without a recovered "
                "complete chain is unresolved rather than a logical contradiction because "
                "the extractor may have missed evidence and historical practice may use a "
                "weaker ROCKMAG-style proxy."
            ),
        },
        "representation_specific_mathematical_bound": exact_bound,
        "ninety_percent_target_context": {
            "target_accuracy": 0.90,
            "maximum_errors_at_or_above_target": maximum_errors_at_ninety,
            "exact_representation_minimum_errors": exact_bound[
                "minimum_unavoidable_errors"
            ],
            "remaining_error_budget_after_exact_bound": (
                maximum_errors_at_ninety
                - exact_bound["minimum_unavoidable_errors"]
            ),
            "exact_representation_conflicts_alone_prevent_target": bool(
                exact_bound["minimum_unavoidable_errors"]
                > maximum_errors_at_ninety
            ),
            "dominant_empirical_observation": (
                f"{len(positive_no_candidate)} of {label_counts[1]} gold-positive "
                "rows have no candidate under the conservative frozen extractor"
            ),
            "interpretation": (
                "The exact mixed-label lower bound does not explain failure to reach "
                "90%. The much larger no-candidate count identifies retrieval and target/"
                "component attribution as the observed bottleneck, but it remains an "
                "extractor-recall statistic rather than an all-source impossibility bound."
            ),
        },
        "coarse_semantic_signature_diagnostic": semantic_signature_diagnostic(rows),
        "near_duplicate_opposite_label_pairs": {
            "pair_count_retained": len(near_pairs),
            "method": (
                "same frozen semantic signature and bottom-k five-token shingle sketch "
                "overlap >= 0.55; review heuristic, not a mathematical equivalence relation"
            ),
            "pairs": near_pairs,
        },
        "review_examples": {
            "high_proxy_support_gold_zero": [_example(row) for row in high_proxy_zero[:25]],
            "complete_criterion_chain_gold_zero": [_example(row) for row in high_criterion_zero[:25]],
            "gold_one_no_frozen_candidate": [_example(row) for row in positive_no_candidate[:25]],
        },
    }


def _escape_md(value: Any) -> str:
    return component.normalized_text(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    stats = report["audit"]
    empirical = stats["empirical_conflict_flags"]
    bound = stats["representation_specific_mathematical_bound"]
    target_context = stats["ninety_percent_target_context"]
    lines = [
        "# R3 Source Evidence vs Historical Label Conflict Audit",
        "",
        "## Boundary",
        "",
        f"- Candidate SHA-256: `{report['freeze_boundary']['candidate_sha256']}`",
        "- Candidates were frozen before training R3 cells were opened.",
        "- Runtime evidence inputs were identity fields, original-paper text, frozen taxonomy/positive KB, criterion spec, and the frozen component manifest.",
        "- PMAGRESULT, ROCKMAG source cells, held-out labels, old predictions, and old evaluations were not read.",
        "",
        "## Empirical Flags",
        "",
        "| Audit class | Rows | Denominator | Rate |",
        "|---|---:|---:|---:|",
    ]
    for name, item in empirical.items():
        if not isinstance(item, Mapping) or "rows" not in item:
            continue
        lines.append(
            f"| `{name}` | {item['rows']} | {item['denominator_rows']} | {item['rate']:.4%} |"
        )
    lines.extend([
        "",
        "These are empirical audit flags, not unavoidable errors. A support/gold-zero row still requires manual confirmation; a gold-one/no-candidate row may be an extraction miss.",
        "",
        "## Representation-Specific Bound",
        "",
        f"- Mixed exact-representation groups: {bound['mixed_representation_groups']}",
        f"- Minimum errors for deterministic classifiers restricted to this exact frozen representation: {bound['minimum_unavoidable_errors']} / {stats['population']['rows']} ({bound['error_rate_lower_bound']:.4%})",
        f"- Corresponding representation-specific accuracy ceiling: {bound['accuracy_upper_bound']:.4%}",
        "- This is not a ceiling for all source-only systems. Richer source parsing or an original-text component locator can split currently identical rows.",
        "",
        "## Relation To The 90% Target",
        "",
        f"- Maximum errors at 90% accuracy: {target_context['maximum_errors_at_or_above_target']}",
        f"- Exact frozen-representation minimum errors: {target_context['exact_representation_minimum_errors']}",
        f"- Remaining error budget after that bound: {target_context['remaining_error_budget_after_exact_bound']}",
        f"- Exact representation conflicts alone prevent 90%: {str(target_context['exact_representation_conflicts_alone_prevent_target']).lower()}",
        f"- {target_context['dominant_empirical_observation']}.",
        "",
        target_context["interpretation"],
        "",
        "## Criterion Logic",
        "",
        stats["criterion_logic_audit"]["assessment"],
        "",
        "## Review Examples",
        "",
        "| Class | REFNO | RESULTNO | Target | Gold | Evidence excerpt |",
        "|---|---:|---:|---|---:|---|",
    ])
    for class_name, examples in stats["review_examples"].items():
        for example in examples[:10]:
            quote = example.get("review_quotes", [{}])[0].get("text", "") if example.get("review_quotes") else ""
            lines.append(
                f"| `{class_name}` | {_escape_md(example['refno'])} | {_escape_md(example['resultno'])} | "
                f"{_escape_md(', '.join(example['rocknames']))} | {example['gold_r3']} | {_escape_md(quote)[:420]} |"
            )
    lines.extend([
        "",
        "## Near-Duplicate Opposite Labels",
        "",
        f"Retained review pairs: {stats['near_duplicate_opposite_label_pairs']['pair_count_retained']}. These are heuristic review pairs, not mathematical duplicates.",
        "",
    ])
    return "\n".join(lines) + "\n"


def evaluate_candidates(
    receipt_path: Path,
    candidates_path: Path,
    identity_path: Path,
    page_cache_path: Path,
    pdf_dir: Path,
    taxonomy_path: Path,
    positive_kb_path: Path,
    criterion_path: Path,
    component_manifest_path: Path,
    component_receipt_path: Path,
    targets_path: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    input_paths = {
        "identity": identity_path.resolve(),
        "page_cache": page_cache_path.resolve(),
        "taxonomy": taxonomy_path.resolve(),
        "positive_kb": positive_kb_path.resolve(),
        "criterion": criterion_path.resolve(),
        "component_manifest": component_manifest_path.resolve(),
        "component_receipt": component_receipt_path.resolve(),
    }
    # Complete hash/membership verification precedes the only label-source call.
    receipt, candidates = verify_candidate_freeze(
        receipt_path.resolve(), candidates_path.resolve(), input_paths, pdf_dir.resolve()
    )
    permitted = {resultno for row in candidates for resultno in row["resultnos"]}
    labels = component.load_training_labels(targets_path.resolve(), permitted)
    audit = audit_statistics(candidates, labels)
    report = {
        "schema_version": "r3_source_label_conflict_ceiling_audit_v1",
        "status": "complete",
        "scope": "fixed PDF-only training partition",
        "freeze_boundary": {
            "candidate_frozen_before_label_access": True,
            "candidate_sha256": receipt["candidate_artifact"]["sha256"],
            "all_source_hashes_verified_before_label_access": True,
            "label_file_hash_omitted_to_avoid_heldout_byte_access": True,
        },
        "guardrails": {
            "forbidden_sources_not_read": receipt["guardrails"]["forbidden_sources_not_read"],
            "empirical_flags_are_not_mathematical_errors": True,
            "representation_bound_not_general_source_only_bound": True,
            "manual_review_required_for_logical_conflict_claim": True,
            "formal_scorer_modified": False,
        },
        "candidate_definitions": receipt["definitions"],
        "criterion_provenance": receipt["criterion_provenance"],
        "audit": audit,
        "artifacts": {
            "candidate_artifact": receipt["candidate_artifact"],
            "candidate_receipt": {
                "path": str(receipt_path.resolve()),
                "sha256": file_sha256(receipt_path.resolve()),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "output_json": str(output_json.resolve()),
            "output_markdown": str(output_markdown.resolve()),
        },
    }
    atomic_json(output_json.resolve(), report)
    atomic_text(output_markdown.resolve(), render_markdown(report))
    return report


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--page-cache", type=Path, default=DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=DEFAULT_POSITIVE_KB)
    parser.add_argument("--criterion", type=Path, default=DEFAULT_CRITERION)
    parser.add_argument("--component-manifest", type=Path, default=DEFAULT_COMPONENT_MANIFEST)
    parser.add_argument("--component-receipt", type=Path, default=DEFAULT_COMPONENT_RECEIPT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_OUTPUT_DIR / "source_evidence_candidates.jsonl")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_OUTPUT_DIR / "candidate_freeze_receipt.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    add_common(freeze_parser)
    evaluate_parser = subparsers.add_parser("evaluate")
    add_common(evaluate_parser)
    evaluate_parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    evaluate_parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "audit.json")
    evaluate_parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_DIR / "audit.md")
    args = parser.parse_args(argv)
    if args.phase == "freeze":
        receipt = freeze_candidates(
            args.identity, args.page_cache, args.pdf_dir, args.taxonomy,
            args.positive_kb, args.criterion, args.component_manifest,
            args.component_receipt, args.candidates, args.receipt,
        )
        print(json.dumps({
            "status": receipt["status"],
            "candidate_artifact": receipt["candidate_artifact"],
            "candidate_counts": receipt["candidate_counts"],
        }, ensure_ascii=False, indent=2))
        return 0
    report = evaluate_candidates(
        args.receipt, args.candidates, args.identity, args.page_cache, args.pdf_dir,
        args.taxonomy, args.positive_kb, args.criterion, args.component_manifest,
        args.component_receipt, args.targets, args.output_json, args.output_markdown,
    )
    print(json.dumps({
        "population": report["audit"]["population"],
        "empirical_conflict_flags": report["audit"]["empirical_conflict_flags"],
        "representation_bound": report["audit"]["representation_specific_mathematical_bound"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
