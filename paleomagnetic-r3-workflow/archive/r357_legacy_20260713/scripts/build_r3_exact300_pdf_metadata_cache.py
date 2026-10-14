#!/usr/bin/env python3
"""Freeze label-free PDF metadata features for the exact300 auxiliary pool.

The PDF filename and frozen identity keys are used only to align a local PDF
with its paper rows.  The output cache contains numeric features only; title,
author, subject, keywords, path, filename, REFNO, RESULTNO, and R3 are never
represented.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/pdf_metadata_cache_v1"
DEFAULT_CACHE = OUTPUT_DIR / "features.npz"
DEFAULT_MANIFEST = OUTPUT_DIR / "features.manifest.json"
PDFINFO_FIELDS = {
    "Creator",
    "Producer",
    "CreationDate",
    "ModDate",
    "Custom Metadata",
    "Metadata Stream",
    "Tagged",
    "UserProperties",
    "Suspects",
    "JavaScript",
    "Pages",
    "Encrypted",
    "Page size",
    "Page rot",
    "File size",
    "Optimized",
    "PDF version",
}
FORBIDDEN_FEATURE_RE = re.compile(
    r"(?i)(?:refno|resultno|rockname|label|answer|target|filename|filepath|path|title|author|subject|keyword)"
)

FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("acrobat_capture", re.compile(r"acrobat.*capture|paper\s*capture", re.I)),
    ("acrobat_import", re.compile(r"acrobat.*import", re.I)),
    ("acrobat_distiller", re.compile(r"distiller", re.I)),
    ("adobe_pdf_library", re.compile(r"adobe\s+pdf\s+library|pdf\s+library", re.I)),
    ("adobe_indesign", re.compile(r"indesign", re.I)),
    ("adobe_illustrator", re.compile(r"illustrator", re.I)),
    ("adobe_photoshop", re.compile(r"photoshop", re.I)),
    ("adobe_framemaker", re.compile(r"framemaker", re.I)),
    ("adobe_pagemaker", re.compile(r"pagemaker", re.I)),
    ("tex", re.compile(r"pdftex|luatex|xetex|latex|tex\.output", re.I)),
    ("ghostscript", re.compile(r"ghostscript|ps2pdf", re.I)),
    ("quartz", re.compile(r"quartz|mac\s*os\s*x", re.I)),
    ("microsoft_office", re.compile(r"microsoft|word|powerpoint|excel", re.I)),
    ("libreoffice", re.compile(r"libreoffice|openoffice", re.I)),
    ("itext", re.compile(r"\bitext\b", re.I)),
    ("cairo", re.compile(r"\bcairo\b", re.I)),
    ("abbyy", re.compile(r"abbyy", re.I)),
    ("imagemagick", re.compile(r"imagemagick", re.I)),
    ("reportlab", re.compile(r"reportlab", re.I)),
    ("pdflib", re.compile(r"pdflib", re.I)),
    ("adobe_other", re.compile(r"adobe|acrobat", re.I)),
)
FAMILY_VALUES = tuple(name for name, _pattern in FAMILY_PATTERNS) + (
    "other_present",
    "missing",
)
TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("adobe", re.compile(r"\badobe\b", re.I)),
    ("acrobat", re.compile(r"\bacrobat\b", re.I)),
    ("capture", re.compile(r"\bcapture\b", re.I)),
    ("import", re.compile(r"\bimport\b", re.I)),
    ("distiller", re.compile(r"\bdistiller\b", re.I)),
    ("pdf_library", re.compile(r"pdf\s+library", re.I)),
    ("indesign", re.compile(r"\bindesign\b", re.I)),
    ("illustrator", re.compile(r"\billustrator\b", re.I)),
    ("framemaker", re.compile(r"\bframemaker\b", re.I)),
    ("pagemaker", re.compile(r"\bpagemaker\b", re.I)),
    ("tex", re.compile(r"pdftex|luatex|xetex|latex", re.I)),
    ("ghostscript", re.compile(r"ghostscript|ps2pdf", re.I)),
    ("quartz", re.compile(r"\bquartz\b", re.I)),
    ("microsoft", re.compile(r"\bmicrosoft\b", re.I)),
    ("office", re.compile(r"\boffice\b|\bword\b|powerpoint|excel", re.I)),
    ("libreoffice", re.compile(r"libreoffice|openoffice", re.I)),
    ("itext", re.compile(r"\bitext\b", re.I)),
    ("cairo", re.compile(r"\bcairo\b", re.I)),
    ("abbyy", re.compile(r"\babbyy\b", re.I)),
    ("scanner", re.compile(r"scanner|scan\s*manager|scanjet", re.I)),
    ("imagemagick", re.compile(r"imagemagick", re.I)),
    ("reportlab", re.compile(r"reportlab", re.I)),
    ("pdflib", re.compile(r"pdflib", re.I)),
)
TOKEN_VALUES = tuple(name for name, _pattern in TOKEN_PATTERNS)

BASE_FEATURE_NAMES = (
    "pdfinfo_ok",
    "creator_present",
    "producer_present",
    "creator_producer_same_family",
    "creator_producer_both_missing",
    "creation_year_present",
    "creation_year_scaled",
    "creation_year_before_1990",
    "creation_year_1990s",
    "creation_year_2000s",
    "creation_year_2010s",
    "creation_year_2020s",
    "mod_year_present",
    "mod_year_scaled",
    "mod_year_before_1990",
    "mod_year_1990s",
    "mod_year_2000s",
    "mod_year_2010s",
    "mod_year_2020s",
    "date_pair_present",
    "date_lag_signed_scaled",
    "date_lag_abs_log",
    "date_lag_negative",
    "date_lag_same_year",
    "custom_metadata_yes",
    "custom_metadata_known",
    "metadata_stream_yes",
    "metadata_stream_known",
    "tagged_yes",
    "tagged_known",
    "optimized_yes",
    "optimized_known",
    "suspects_yes",
    "suspects_known",
    "encrypted_yes",
    "encrypted_known",
    "user_properties_yes",
    "user_properties_known",
    "javascript_yes",
    "javascript_known",
    "pages_present",
    "pages_log",
    "single_page",
    "pages_at_least_20",
    "pages_at_least_50",
    "file_size_present",
    "file_size_log",
    "bytes_per_page_log",
    "reported_size_present",
    "reported_size_mismatch",
    "page_size_present",
    "page_width_log",
    "page_height_log",
    "page_area_log",
    "page_aspect_log",
    "page_portrait",
    "page_landscape",
    "page_near_square",
    "page_rotation_present",
    "page_rotation_0",
    "page_rotation_90",
    "page_rotation_180",
    "page_rotation_270",
    "pdf_version_present",
    "pdf_version_scaled",
    "pdf_version_1_0_to_1_3",
    "pdf_version_1_4",
    "pdf_version_1_5",
    "pdf_version_1_6",
    "pdf_version_1_7",
    "pdf_version_2_or_later",
    "scan_pipeline_hint",
    "born_digital_hint",
)


def feature_names() -> list[str]:
    names = list(BASE_FEATURE_NAMES)
    names.extend(f"creator_family__{name}" for name in FAMILY_VALUES)
    names.extend(f"producer_family__{name}" for name in FAMILY_VALUES)
    names.extend(f"creator_token__{name}" for name in TOKEN_VALUES)
    names.extend(f"producer_token__{name}" for name in TOKEN_VALUES)
    if len(names) != len(set(names)) or any(FORBIDDEN_FEATURE_RE.search(name) for name in names):
        raise AssertionError("PDF metadata feature schema contains a forbidden or duplicate name")
    return names


def parse_pdfinfo_output(output: str) -> dict[str, str]:
    """Parse only explicitly allowed fields from pdfinfo output."""

    values: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        key = key.strip()
        if separator and key in PDFINFO_FIELDS:
            values[key] = value.strip()
    return values


def normalized_family(value: str) -> str:
    if not value.strip():
        return "missing"
    for name, pattern in FAMILY_PATTERNS:
        if pattern.search(value):
            return name
    return "other_present"


def normalized_tokens(value: str) -> set[str]:
    return {name for name, pattern in TOKEN_PATTERNS if pattern.search(value)}


def parse_bool(value: str) -> tuple[float, float]:
    normalized = value.strip().casefold()
    if normalized.startswith("yes"):
        return 1.0, 1.0
    if normalized.startswith("no"):
        return 0.0, 1.0
    return 0.0, 0.0


def parse_float(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_year(value: str) -> int | None:
    match = re.search(r"(?<!\d)(18\d{2}|19\d{2}|20\d{2}|21\d{2})(?!\d)", value)
    if not match:
        return None
    year = int(match.group(1))
    return year if 1800 <= year <= 2199 else None


def parse_page_size(value: str) -> tuple[float | None, float | None]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*pts", value, re.I)
    if not match:
        return None, None
    width, height = float(match.group(1)), float(match.group(2))
    if width <= 0.0 or height <= 0.0:
        return None, None
    return width, height


def year_features(year: int | None) -> list[float]:
    if year is None:
        return [0.0] * 7
    return [
        1.0,
        float(np.clip((year - 2000.0) / 30.0, -4.0, 4.0)),
        float(year < 1990),
        float(1990 <= year < 2000),
        float(2000 <= year < 2010),
        float(2010 <= year < 2020),
        float(2020 <= year < 2030),
    ]


def metadata_vector(
    metadata: Mapping[str, str], *, stat_size: int, pdfinfo_ok: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    creator = metadata.get("Creator", "")
    producer = metadata.get("Producer", "")
    creator_family = normalized_family(creator)
    producer_family = normalized_family(producer)
    creator_tokens = normalized_tokens(creator)
    producer_tokens = normalized_tokens(producer)
    creation_year = parse_year(metadata.get("CreationDate", ""))
    mod_year = parse_year(metadata.get("ModDate", ""))
    values: list[float] = [
        float(pdfinfo_ok),
        float(bool(creator.strip())),
        float(bool(producer.strip())),
        float(
            creator_family == producer_family
            and creator_family not in {"missing", "other_present"}
        ),
        float(creator_family == producer_family == "missing"),
    ]
    values.extend(year_features(creation_year))
    values.extend(year_features(mod_year))
    if creation_year is not None and mod_year is not None:
        lag = mod_year - creation_year
        values.extend(
            [
                1.0,
                float(np.clip(lag, -50, 50) / 50.0),
                math.log1p(abs(float(lag))),
                float(lag < 0),
                float(lag == 0),
            ]
        )
    else:
        values.extend([0.0] * 5)
    for field in (
        "Custom Metadata",
        "Metadata Stream",
        "Tagged",
        "Optimized",
        "Suspects",
        "Encrypted",
        "UserProperties",
        "JavaScript",
    ):
        values.extend(parse_bool(metadata.get(field, "")))

    pages_value = parse_float(metadata.get("Pages", ""))
    pages = int(pages_value) if pages_value is not None and pages_value >= 1 else None
    values.extend(
        [
            float(pages is not None),
            math.log1p(float(pages or 0)),
            float(pages == 1),
            float(pages is not None and pages >= 20),
            float(pages is not None and pages >= 50),
        ]
    )
    size = max(0, int(stat_size))
    reported_size_value = parse_float(metadata.get("File size", ""))
    reported_size = int(reported_size_value) if reported_size_value is not None else None
    values.extend(
        [
            float(size > 0),
            math.log1p(float(size)),
            math.log1p(float(size) / max(1, pages or 1)),
            float(reported_size is not None),
            float(reported_size is not None and reported_size != size),
        ]
    )
    width, height = parse_page_size(metadata.get("Page size", ""))
    if width is not None and height is not None:
        aspect = width / height
        values.extend(
            [
                1.0,
                math.log1p(width),
                math.log1p(height),
                math.log1p(width * height),
                math.log(aspect),
                float(aspect < 0.95),
                float(aspect > 1.05),
                float(0.95 <= aspect <= 1.05),
            ]
        )
    else:
        values.extend([0.0] * 8)
    rotation_value = parse_float(metadata.get("Page rot", ""))
    rotation = int(round(rotation_value)) % 360 if rotation_value is not None else None
    values.extend(
        [
            float(rotation is not None),
            float(rotation == 0),
            float(rotation == 90),
            float(rotation == 180),
            float(rotation == 270),
        ]
    )
    version_value = parse_float(metadata.get("PDF version", ""))
    version = version_value if version_value is not None and 0.5 <= version_value <= 9 else None
    values.extend(
        [
            float(version is not None),
            float((version - 1.0) / 1.0) if version is not None else 0.0,
            float(version is not None and 1.0 <= version < 1.4),
            float(version is not None and 1.4 <= version < 1.5),
            float(version is not None and 1.5 <= version < 1.6),
            float(version is not None and 1.6 <= version < 1.7),
            float(version is not None and 1.7 <= version < 2.0),
            float(version is not None and version >= 2.0),
        ]
    )
    scan_families = {"acrobat_capture", "acrobat_import", "abbyy", "imagemagick"}
    digital_families = {
        "acrobat_distiller",
        "adobe_pdf_library",
        "adobe_indesign",
        "tex",
        "ghostscript",
        "quartz",
        "microsoft_office",
        "libreoffice",
        "itext",
        "cairo",
        "reportlab",
        "pdflib",
    }
    all_tokens = creator_tokens | producer_tokens
    values.extend(
        [
            float(
                creator_family in scan_families
                or producer_family in scan_families
                or "scanner" in all_tokens
            ),
            float(
                creator_family in digital_families
                or producer_family in digital_families
            ),
        ]
    )
    values.extend(float(creator_family == family) for family in FAMILY_VALUES)
    values.extend(float(producer_family == family) for family in FAMILY_VALUES)
    values.extend(float(token in creator_tokens) for token in TOKEN_VALUES)
    values.extend(float(token in producer_tokens) for token in TOKEN_VALUES)
    vector = np.asarray(values, dtype=np.float32)
    names = feature_names()
    if vector.shape != (len(names),) or not np.isfinite(vector).all():
        raise ValueError("invalid PDF metadata feature vector")
    return vector, {
        "creator_family": creator_family,
        "producer_family": producer_family,
        "creation_year": creation_year,
        "mod_year": mod_year,
        "pages": pages,
        "pdf_version": version,
        "pdfinfo_ok": bool(pdfinfo_ok),
    }


def run_pdfinfo(path: Path) -> tuple[dict[str, str], bool, str]:
    try:
        completed = subprocess.run(
            ["pdfinfo", "-isodates", str(path)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, False, type(exc).__name__
    return (
        parse_pdfinfo_output(completed.stdout),
        completed.returncode == 0,
        completed.stderr.strip()[:500],
    )


def corpus_sha256(pdf_hashes: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(pdf_hashes):
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def build_features(
    records: Sequence[Mapping[str, str]],
    pdf_dir: Path,
    extractor: Callable[[Path], tuple[dict[str, str], bool, str]] = run_pdfinfo,
) -> tuple[np.ndarray, dict[str, Any]]:
    refnos = sorted({str(record["REFNO"]) for record in records})
    vectors: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, Any]] = {}
    pdf_hashes: list[str] = []
    failures: Counter[str] = Counter()
    for index, refno in enumerate(refnos, start=1):
        path = pdf_dir / f"{refno}.pdf"
        if not path.is_file():
            raise ValueError("frozen PDF identity is missing a local source PDF")
        metadata, ok, error = extractor(path)
        vector, summary = metadata_vector(metadata, stat_size=path.stat().st_size, pdfinfo_ok=ok)
        vectors[refno] = vector
        summaries[refno] = summary
        pdf_hashes.append(base.file_sha256(path))
        if not ok:
            failures[error or "nonzero_exit"] += 1
        if index % 100 == 0 or index == len(refnos):
            print(f"pdf metadata: {index}/{len(refnos)} papers", flush=True)
    matrix = np.vstack([vectors[str(record["REFNO"])] for record in records]).astype(
        np.float32, copy=False
    )
    creator_counts = Counter(item["creator_family"] for item in summaries.values())
    producer_counts = Counter(item["producer_family"] for item in summaries.values())
    creation_years = [item["creation_year"] for item in summaries.values() if item["creation_year"] is not None]
    mod_years = [item["mod_year"] for item in summaries.values() if item["mod_year"] is not None]
    audit = {
        "paper_count": len(refnos),
        "row_count": len(records),
        "paper_features_broadcast_to_result_rows": True,
        "pdfinfo_success_count": sum(bool(item["pdfinfo_ok"]) for item in summaries.values()),
        "pdfinfo_failure_count": sum(not bool(item["pdfinfo_ok"]) for item in summaries.values()),
        "failure_types": dict(sorted(failures.items())),
        "creator_family_counts": dict(sorted(creator_counts.items())),
        "producer_family_counts": dict(sorted(producer_counts.items())),
        "creation_year_coverage": len(creation_years),
        "creation_year_min": min(creation_years) if creation_years else None,
        "creation_year_max": max(creation_years) if creation_years else None,
        "mod_year_coverage": len(mod_years),
        "mod_year_min": min(mod_years) if mod_years else None,
        "mod_year_max": max(mod_years) if mod_years else None,
        "corpus_content_multiset_sha256": corpus_sha256(pdf_hashes),
    }
    return matrix, audit


def validate_arrays(arrays: Mapping[str, np.ndarray], expected_rows: int) -> dict[str, Any]:
    if set(arrays) != {"pdf_metadata"}:
        raise ValueError("PDF metadata cache arrays differ from the frozen schema")
    matrix = np.asarray(arrays["pdf_metadata"])
    if matrix.shape != (expected_rows, len(feature_names())):
        raise ValueError(f"unexpected PDF metadata matrix shape: {matrix.shape}")
    if matrix.dtype != np.float32:
        raise ValueError("PDF metadata matrix is not float32")
    if not np.isfinite(matrix).all():
        raise ValueError("PDF metadata matrix contains non-finite values")
    return {
        "pdf_metadata": {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "sha256": auxiliary.sha256_array(matrix),
            "finite": True,
        }
    }


def write_cache(path: Path, arrays: Mapping[str, np.ndarray], expected_rows: int) -> None:
    validate_arrays(arrays, expected_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, pdf_metadata=np.asarray(arrays["pdf_metadata"], dtype=np.float32))
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    exact_identity = args.exact_identity.resolve()
    full_identity = args.full_identity.resolve()
    fold_manifest = args.fold_manifest.resolve()
    pdf_dir = args.pdf_dir.resolve()
    cache_path = args.cache.resolve()
    manifest_path = args.manifest.resolve()
    if base.file_sha256(exact_identity) != auxiliary.EXACT_IDENTITY_SHA256:
        raise ValueError("exact300 identity hash mismatch")
    if base.file_sha256(full_identity) != auxiliary.FULL_IDENTITY_SHA256:
        raise ValueError("full auxiliary identity hash mismatch")
    if base.file_sha256(fold_manifest) != auxiliary.FOLD_MANIFEST_SHA256:
        raise ValueError("canonical fold manifest hash mismatch")
    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, exact_coverage = base.load_pdf_only_identity(exact_identity, pdf_dir)
    auxiliary.validate_membership(full_records, exact_records)
    auxiliary.load_fold_manifest(fold_manifest, exact_records)
    matrix, feature_audit = build_features(full_records, pdf_dir)
    arrays = {"pdf_metadata": matrix}
    matrix_audit = validate_arrays(arrays, len(full_records))
    write_cache(cache_path, arrays, len(full_records))
    manifest = {
        "schema_version": "r3_exact300_pdf_metadata_cache_v1",
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "scope": {
            "exact_evaluation": exact_coverage,
            "auxiliary_candidate_pool": full_coverage,
            "exact_alignment_sha256": auxiliary.sha256_records(exact_records),
            "full_alignment_sha256": auxiliary.sha256_records(full_records),
        },
        "runtime_boundary": {
            "labels_read": False,
            "targets_workbook_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifiers_used_only_for_pdf_alignment": True,
            "cache_contains_identifiers": False,
            "cache_contains_paths_or_filenames": False,
            "content_metadata_title_author_subject_keywords_ignored": True,
        },
        "allowed_sources": [
            "embedded PDF Creator and Producer normalized to fixed family/token features",
            "embedded PDF creation/modification years and lag",
            "pdfinfo structural flags, page geometry, page count, PDF version",
            "local PDF byte size",
        ],
        "inputs": {
            "exact_identity": {"path": str(exact_identity), "sha256": base.file_sha256(exact_identity)},
            "full_identity": {"path": str(full_identity), "sha256": base.file_sha256(full_identity)},
            "fold_manifest": {"path": str(fold_manifest), "sha256": base.file_sha256(fold_manifest)},
            "pdf_corpus": {
                "directory_used_only_for_alignment": True,
                "paper_count": feature_audit["paper_count"],
                "content_multiset_sha256": feature_audit["corpus_content_multiset_sha256"],
            },
        },
        "feature_names": feature_names(),
        "feature_audit": feature_audit,
        "matrices": matrix_audit,
        "output": {
            "path": str(cache_path),
            "sha256": base.file_sha256(cache_path),
            "contains_labels": False,
            "contains_identifiers": False,
            "contains_paths_or_filenames": False,
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.file_sha256(Path(__file__).resolve()),
        },
    }
    base.atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "cache": manifest["output"],
                "manifest": {"path": str(manifest_path), "sha256": base.file_sha256(manifest_path)},
                "feature_audit": feature_audit,
                "matrices": matrix_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
