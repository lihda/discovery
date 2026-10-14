#!/usr/bin/env python3
"""Fuse nonempty page-cache records for all labelled R3 papers with PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as base  # noqa: E402


IDENTITY = ROOT / "outputs/r357_strict_eval_07111758/inputs/identity_manifest.xlsx"
PRIMARY_CACHE = ROOT / "tmp/r3_proxy_v2/full_cache_holdout_mineru.jsonl"
FALLBACK_CACHES = (
    ROOT / "tmp/mineru_cache_all_hard_risk.jsonl",
    ROOT / "tmp/r3_page_fusion_quality_sample.jsonl",
    ROOT / "tmp/mineru_cache_zero_all.jsonl",
    ROOT / "tmp/mineru_cache_p0.jsonl",
    ROOT / "tmp/mineru_cache_3244.jsonl",
    ROOT / "outputs/r357_strict_eval_07111758/inputs/page_cache.jsonl",
)
OUTPUT = ROOT / "tmp/r3_all_labelled_pdf_page_cache_v1/page_cache.jsonl"
AUDIT = ROOT / "tmp/r3_all_labelled_pdf_page_cache_v1/audit.json"


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_text_length(record: Mapping[str, Any]) -> int:
    return sum(
        len(clean(page.get("text")))
        for page in record.get("pages", [])
        if isinstance(page, Mapping)
    )


def load_identity_refnos(path: Path) -> set[str]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.active.iter_rows(values_only=True)
        header = tuple(clean(value) for value in next(rows, ()))
        if header != ("REFNO", "RESULTNO", "ROCKNAME"):
            raise ValueError("identity manifest must contain exactly three columns")
        refnos = {clean(row[0]) for row in rows if clean(row[0])}
    finally:
        workbook.close()
    return refnos


def load_cache(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            refno = clean(record.get("REFNO"))
            if refno not in wanted:
                continue
            if refno in records:
                raise ValueError(f"duplicate REFNO {refno} in {path}:{line_number}")
            records[refno] = record
    return records


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--primary-cache", type=Path, default=PRIMARY_CACHE)
    parser.add_argument(
        "--fallback-cache",
        type=Path,
        action="append",
        dest="fallback_caches",
        help="Fallback cache in priority order; repeatable.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    args = parser.parse_args(argv)

    identity = args.identity.resolve()
    pdf_dir = args.pdf_dir.resolve()
    primary = args.primary_cache.resolve()
    fallbacks = tuple(
        path.resolve()
        for path in (args.fallback_caches or list(FALLBACK_CACHES))
    )
    output = args.output.resolve()
    audit_path = args.audit.resolve()

    labelled_refnos = load_identity_refnos(identity)
    wanted = labelled_refnos & base.local_pdf_refnos(pdf_dir)
    cache_paths = (primary, *fallbacks)
    caches = [load_cache(path, wanted) for path in cache_paths]

    selected: dict[str, dict[str, Any]] = {}
    selections: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for refno in sorted(wanted, key=lambda value: (int(value), value)):
        source_index = next(
            (
                index
                for index, records in enumerate(caches)
                if refno in records and record_text_length(records[refno]) > 0
            ),
            None,
        )
        if source_index is None:
            unresolved.append(refno)
            continue
        record = caches[source_index][refno]
        selected[refno] = record
        selections.append(
            {
                "REFNO": refno,
                "selected_cache_rank": source_index,
                "selected_cache": str(cache_paths[source_index]),
                "text_characters": record_text_length(record),
                "record_sha256": hashlib.sha256(
                    json.dumps(
                        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for refno in sorted(selected, key=lambda value: (int(value), value)):
            handle.write(json.dumps(selected[refno], ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(output)

    source_counts: dict[str, int] = {}
    for selection in selections:
        key = str(selection["selected_cache_rank"])
        source_counts[key] = source_counts.get(key, 0) + 1
    audit = {
        "schema_version": "r3_all_labelled_pdf_page_cache_fusion_v1",
        "selection_rule": "first nonempty complete record in declared cache priority",
        "identity": {"path": str(identity), "sha256": file_sha256(identity)},
        "pdf_dir": str(pdf_dir),
        "labelled_papers": len(labelled_refnos),
        "pdf_backed_papers": len(wanted),
        "selected_papers": len(selected),
        "unparseable_pdf_papers": len(unresolved),
        "unparseable_refnos": unresolved,
        "cache_priority": [
            {"rank": index, "path": str(path), "sha256": file_sha256(path)}
            for index, path in enumerate(cache_paths)
        ],
        "source_counts_by_rank": source_counts,
        "selections": selections,
        "output": {"path": str(output), "sha256": file_sha256(output)},
    }
    atomic_json(audit_path, audit)
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": audit["output"]["sha256"],
                "pdf_backed_papers": len(wanted),
                "selected_papers": len(selected),
                "unparseable_pdf_papers": len(unresolved),
                "source_counts_by_rank": source_counts,
                "audit": str(audit_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
