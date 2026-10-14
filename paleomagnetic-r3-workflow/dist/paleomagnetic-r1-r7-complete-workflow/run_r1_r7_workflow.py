#!/usr/bin/env python3
"""Public entrypoint for the complete paleomagnetic R1-R7 workflow.

Inputs and scoring profiles are declared in a versioned JSON configuration.
The current adapter accepts the project's existing Excel/PDF/text layout. A
future input format can be normalized in this file without changing the R1-R7
scoring modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the shell bootstrap
    raise SystemExit(
        "Missing Python dependencies. Run ./run_r1_r7_workflow.sh or install requirements.txt."
    ) from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config/workflow.current.json"
RUNTIME_MANIFEST = ROOT / "config/public_runtime_manifest.json"
REQUIRED_TARGET_COLUMNS = {"RESULTNO", "ROCKUNITNO", "REFNO", "ROCKNAME"}
REQUIRED_PMAG_COLUMNS = {
    "RESULTNO",
    "ROCKUNITNO",
    "HIMAGAGE",
    "LOMAGAGE",
    "B",
    "N",
    "KD",
    "ED95",
    "DEMAGCODE",
    "TESTS",
}
REQUIRED_ROCKUNIT_COLUMNS = {
    "ROCKUNITNO",
    "REFNO",
    "ROCKNAME",
    "HIGHAGE",
    "LOWAGE",
}
PATH_INPUTS = ("targets", "pmagresult", "rockunit", "parsed_dir", "pdf_dir")
R3_SOURCE_MODES = {"fresh", "paper_only", "retained", "supervised_stable"}
R5_PROFILES = {"paper_only", "semantic", "metadata_calibrated", "source_augmented"}
R7_PROFILES = {"source_text", "astute_source_aware"}
R3_RUNTIME_PATH_DEFAULTS = {
    "r3_stable_model": "data/models/r3_stable_production_v1.joblib",
    "r3_stable_manifest": "data/models/r3_stable_production_v1.manifest.json",
    "r3_model_snapshot": (
        "data/models/bge-base-en-v1.5/"
        "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    ),
}


class ConfigurationError(ValueError):
    """Raised when the public workflow configuration is invalid."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{path} must contain a JSON object.")
    if payload.get("schema_version") != 1:
        raise ConfigurationError("Only workflow configuration schema_version=1 is supported.")
    if payload.get("input_adapter") != "legacy_excel_v1":
        raise ConfigurationError(
            "Only input_adapter=legacy_excel_v1 is currently implemented. "
            "Normalize a new source format to the canonical columns before scoring."
        )

    inputs = payload.get("inputs")
    profiles = payload.get("profiles")
    runtime = payload.get("runtime", {})
    output = payload.get("output", {})
    if not isinstance(inputs, dict) or not isinstance(profiles, dict):
        raise ConfigurationError("Configuration requires object sections: inputs and profiles.")
    missing = [name for name in PATH_INPUTS if not inputs.get(name)]
    if missing:
        raise ConfigurationError(f"Configuration inputs are missing: {', '.join(missing)}")

    config = {
        "schema_version": 1,
        "input_adapter": "legacy_excel_v1",
        "inputs": {name: resolve_path(inputs[name]) for name in PATH_INPUTS},
        "profiles": dict(profiles),
        "runtime": dict(runtime),
        "output": dict(output),
    }
    for optional in ("r3_extracts", "r3_page_cache"):
        if inputs.get(optional):
            config["inputs"][optional] = resolve_path(inputs[optional])

    r3_mode = str(profiles.get("r3_source_mode", "supervised_stable"))
    r5_profile = str(profiles.get("r5_profile", "paper_only"))
    r7_profile = str(profiles.get("r7_profile", "source_text"))
    if r3_mode not in R3_SOURCE_MODES:
        raise ConfigurationError(f"Unsupported r3_source_mode: {r3_mode}")
    if r5_profile not in R5_PROFILES:
        raise ConfigurationError(f"Unsupported r5_profile: {r5_profile}")
    if r7_profile not in R7_PROFILES:
        raise ConfigurationError(f"Unsupported r7_profile: {r7_profile}")
    if r3_mode == "retained" and "r3_extracts" not in config["inputs"]:
        raise ConfigurationError("retained R3 mode requires inputs.r3_extracts.")

    config["profiles"].update(
        {
            "r3_source_mode": r3_mode,
            "r5_profile": r5_profile,
            "r7_profile": r7_profile,
        }
    )
    workers = int(runtime.get("r3_workers", 8))
    if workers < 1:
        raise ConfigurationError("runtime.r3_workers must be at least 1.")
    config["runtime"]["r3_workers"] = workers
    config["runtime"].setdefault("r3_pdfplumber_mode", "fallback")
    config["runtime"].setdefault("r3_pdftotext_mode", "never")
    for name, default in R3_RUNTIME_PATH_DEFAULTS.items():
        config["runtime"][name] = resolve_path(runtime.get(name, default))
    config["runtime"].setdefault("r3_model_device", "cpu")
    config["output"]["root"] = resolve_path(output.get("root", "outputs/r1_r7_runs"))
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    for name in PATH_INPUTS:
        value = getattr(args, name, None)
        if value is not None:
            config["inputs"][name] = value.expanduser().resolve()
    if args.r3_page_cache is not None:
        config["inputs"]["r3_page_cache"] = args.r3_page_cache.expanduser().resolve()
    if args.output_root is not None:
        config["output"]["root"] = args.output_root.expanduser().resolve()
    for name in ("r3_source_mode", "r5_profile", "r7_profile"):
        value = getattr(args, name, None)
        if value is not None:
            config["profiles"][name] = value
    if args.r3_workers is not None:
        config["runtime"]["r3_workers"] = args.r3_workers
    for name in R3_RUNTIME_PATH_DEFAULTS:
        value = getattr(args, name, None)
        if value is not None:
            config["runtime"][name] = value.expanduser().resolve()
    if args.r3_model_device is not None:
        config["runtime"]["r3_model_device"] = args.r3_model_device


def validate_resolved_config(config: dict[str, Any]) -> None:
    profiles = config["profiles"]
    inputs = config["inputs"]
    runtime = config["runtime"]
    if profiles["r3_source_mode"] == "retained" and "r3_extracts" not in inputs:
        raise ConfigurationError("retained R3 mode requires inputs.r3_extracts.")
    if int(runtime["r3_workers"]) < 1:
        raise ConfigurationError("runtime.r3_workers must be at least 1.")
    for name in ("r3_pdfplumber_mode", "r3_pdftotext_mode"):
        if runtime[name] not in {"never", "fallback", "always"}:
            raise ConfigurationError(f"runtime.{name} must be never, fallback, or always.")
    if runtime["r3_model_device"] not in {"cpu", "mps"}:
        raise ConfigurationError("runtime.r3_model_device must be cpu or mps.")


def load_sheet(path: Path, required: set[str], key: str) -> pd.DataFrame:
    try:
        frame = pd.read_excel(path, dtype=str).fillna("")
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Could not read {path}: {exc}") from exc
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ConfigurationError(f"{path} is missing columns: {', '.join(missing)}")
    for column in required:
        frame[column] = frame[column].astype(str).str.strip()
    if frame[key].eq("").any():
        raise ConfigurationError(f"{path} contains empty {key} values.")
    if frame[key].duplicated().any():
        examples = ", ".join(frame.loc[frame[key].duplicated(), key].head(10))
        raise ConfigurationError(f"{path} contains duplicate {key} values: {examples}")
    return frame


def validate_runtime_manifest(path: Path = RUNTIME_MANIFEST) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not load public runtime manifest {path}: {exc}") from exc
    declared = [
        *payload.get("entrypoints", []),
        *payload.get("code", []),
        *payload.get("fixed_runtime_assets", []),
    ]
    if not declared or not all(isinstance(item, str) and item for item in declared):
        raise ConfigurationError(f"{path} has an invalid or empty runtime file declaration.")
    missing = [item for item in declared if not (ROOT / item).is_file()]
    if missing:
        raise ConfigurationError("Public runtime files are missing: " + ", ".join(missing))
    return {
        "manifest": str(path),
        "manifest_sha256": sha256(path),
        "declared_files": len(declared),
        "status": "ok",
    }


def validate_r3_stable_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    model = runtime["r3_stable_model"]
    manifest_path = runtime["r3_stable_manifest"]
    snapshot = runtime["r3_model_snapshot"]
    missing = [str(path) for path in (model, manifest_path, snapshot) if not path.exists()]
    if missing:
        raise ConfigurationError("R3 stable runtime artifact(s) missing: " + ", ".join(missing))
    if not model.is_file() or not manifest_path.is_file() or not snapshot.is_dir():
        raise ConfigurationError("R3 stable model/manifest must be files and snapshot must be a directory.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not load R3 stable manifest {manifest_path}: {exc}") from exc
    expected_model_hash = str((manifest.get("model") or {}).get("sha256") or "")
    actual_model_hash = sha256(model)
    if actual_model_hash != expected_model_hash:
        raise ConfigurationError("R3 stable model hash does not match its manifest.")

    snapshot_spec = manifest.get("bge_model_snapshot") or {}
    declared_files = snapshot_spec.get("files") or []
    if not declared_files or not all(isinstance(item, dict) for item in declared_files):
        raise ConfigurationError("R3 stable manifest has no valid BGE snapshot file list.")
    actual_files = []
    for item in declared_files:
        name = str(item.get("name") or "")
        relative = Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts:
            raise ConfigurationError(f"Invalid BGE snapshot file declaration: {name!r}")
        current = snapshot / relative
        if not current.is_file():
            raise ConfigurationError(f"BGE snapshot runtime file is missing: {current}")
        actual_files.append(
            {
                "name": name,
                "size_bytes": current.stat().st_size,
                "sha256": sha256(current),
            }
        )
    snapshot_hash = hashlib.sha256(
        json.dumps(
            actual_files,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if snapshot_hash != str(snapshot_spec.get("snapshot_sha256") or ""):
        raise ConfigurationError("R3 stable BGE snapshot hash does not match its manifest.")
    training = manifest.get("training") or {}
    return {
        "status": "ok",
        "model": str(model),
        "model_sha256": actual_model_hash,
        "model_manifest": str(manifest_path),
        "model_snapshot": str(snapshot),
        "model_snapshot_sha256": snapshot_hash,
        "device": runtime["r3_model_device"],
        "internal_grouped_oof": {
            "correct": int(training.get("grouped_oof_correct") or 0),
            "rows": int(training.get("grouped_oof_rows") or 0),
            "accuracy": float(training.get("grouped_oof_accuracy") or 0.0),
            "external_validation": False,
        },
    }


def input_preflight(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    runtime = config["runtime"]
    missing_paths = [f"{name}={inputs[name]}" for name in PATH_INPUTS if not inputs[name].exists()]
    for optional in ("r3_extracts", "r3_page_cache"):
        if optional in inputs and not inputs[optional].exists():
            missing_paths.append(f"{optional}={inputs[optional]}")
    if missing_paths:
        raise ConfigurationError("Required input paths do not exist: " + "; ".join(missing_paths))
    if not inputs["parsed_dir"].is_dir() or not inputs["pdf_dir"].is_dir():
        raise ConfigurationError("inputs.parsed_dir and inputs.pdf_dir must be directories.")

    targets = load_sheet(inputs["targets"], REQUIRED_TARGET_COLUMNS, "RESULTNO")
    pmag = load_sheet(inputs["pmagresult"], REQUIRED_PMAG_COLUMNS, "RESULTNO")
    rockunit = load_sheet(inputs["rockunit"], REQUIRED_ROCKUNIT_COLUMNS, "ROCKUNITNO")

    source = pmag[["RESULTNO", "ROCKUNITNO"]].merge(
        rockunit[["ROCKUNITNO", "REFNO", "ROCKNAME"]],
        on="ROCKUNITNO",
        how="left",
        validate="many_to_one",
    )
    aligned = targets[["RESULTNO", "ROCKUNITNO", "REFNO", "ROCKNAME"]].merge(
        source,
        on="RESULTNO",
        how="left",
        suffixes=("_target", "_source"),
        validate="one_to_one",
    )
    mismatch = aligned["ROCKUNITNO_target"].ne(aligned["ROCKUNITNO_source"])
    mismatch |= aligned["REFNO_target"].ne(aligned["REFNO_source"])
    mismatch |= aligned["ROCKNAME_target"].ne(aligned["ROCKNAME_source"])
    if mismatch.any():
        examples = ", ".join(aligned.loc[mismatch, "RESULTNO"].head(10))
        raise ConfigurationError(f"Target/source identity mismatch for RESULTNO values: {examples}")

    target_refnos = set(targets["REFNO"])
    pdf_refnos = {path.stem for path in inputs["pdf_dir"].glob("*.pdf")}
    parsed_refnos = {
        path.stem
        for pattern in ("*.txt", "*.md", "*.markdown")
        for path in inputs["parsed_dir"].glob(pattern)
    }
    source_refnos = pdf_refnos | parsed_refnos
    covered_refnos = target_refnos & source_refnos
    label_columns = sorted(set(targets.columns) & {f"R{index}" for index in range(1, 8)})
    warnings: list[str] = []
    uncovered = target_refnos - source_refnos
    if uncovered:
        warnings.append(
            f"{len(uncovered)} target REFNO values have no local PDF or parsed text; "
            "text-derived indicators may return UNKNOWN/default values."
        )
    if label_columns:
        warnings.append(
            "Target labels are present and are used only for post-scoring evaluation: "
            + ", ".join(label_columns)
        )

    return {
        "status": "ok",
        "input_adapter": config["input_adapter"],
        "runtime": validate_runtime_manifest(),
        "r3_stable_runtime": (
            validate_r3_stable_runtime(runtime)
            if config["profiles"]["r3_source_mode"] == "supervised_stable"
            else {"status": "not_selected"}
        ),
        "targets": {
            "rows": int(len(targets)),
            "unique_refnos": int(targets["REFNO"].nunique()),
            "label_columns": label_columns,
        },
        "source_tables": {
            "pmagresult_rows": int(len(pmag)),
            "rockunit_rows": int(len(rockunit)),
        },
        "paper_sources": {
            "pdf_files": len(pdf_refnos),
            "parsed_files": len(parsed_refnos),
            "covered_target_refnos": len(covered_refnos),
            "uncovered_target_refnos": len(uncovered),
        },
        "warnings": warnings,
    }


def validate_timestamp(value: str) -> str:
    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("timestamp must use MMDDHHMM, for example 07221430")
    return value


def core_command(config: dict[str, Any], run_dir: Path, timestamp: str) -> list[str]:
    inputs = config["inputs"]
    profiles = config["profiles"]
    runtime = config["runtime"]
    command = [
        sys.executable,
        str(ROOT / "scripts/run_r357_latest_full_workflow.py"),
        "--targets",
        str(inputs["targets"]),
        "--parsed-dir",
        str(inputs["parsed_dir"]),
        "--pdf-dir",
        str(inputs["pdf_dir"]),
        "--rockunit",
        str(inputs["rockunit"]),
        "--pmagresult",
        str(inputs["pmagresult"]),
        "--r3-source-mode",
        profiles["r3_source_mode"],
        "--r5-profile",
        profiles["r5_profile"],
        "--r7-profile",
        profiles["r7_profile"],
        "--r3-workers",
        str(runtime["r3_workers"]),
        "--r3-pdfplumber-mode",
        str(runtime["r3_pdfplumber_mode"]),
        "--r3-pdftotext-mode",
        str(runtime["r3_pdftotext_mode"]),
        "--output-dir",
        str(run_dir / "work"),
        "--final-output-dir",
        str(run_dir / "results"),
        "--timestamp",
        timestamp,
    ]
    if profiles.get("r3_policy"):
        command.extend(["--r3-policy", str(profiles["r3_policy"])])
    if inputs.get("r3_extracts"):
        command.extend(["--r3-extracts", str(inputs["r3_extracts"])])
    if inputs.get("r3_page_cache"):
        command.extend(["--r3-page-cache", str(inputs["r3_page_cache"])])
    if profiles["r3_source_mode"] == "supervised_stable":
        command.extend(
            [
                "--r3-stable-model",
                str(runtime["r3_stable_model"]),
                "--r3-stable-manifest",
                str(runtime["r3_stable_manifest"]),
                "--r3-model-snapshot",
                str(runtime["r3_model_snapshot"]),
                "--r3-model-device",
                str(runtime["r3_model_device"]),
            ]
        )
    return command


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        section: {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }
        if isinstance(values, dict)
        else values
        for section, values in config.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without scoring.")
    parser.add_argument("--timestamp", type=validate_timestamp, default=None)
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument("--pmagresult", type=Path, default=None)
    parser.add_argument("--rockunit", type=Path, default=None)
    parser.add_argument("--parsed-dir", type=Path, default=None)
    parser.add_argument("--pdf-dir", type=Path, default=None)
    parser.add_argument("--r3-page-cache", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--r3-source-mode", choices=sorted(R3_SOURCE_MODES), default=None)
    parser.add_argument("--r5-profile", choices=sorted(R5_PROFILES), default=None)
    parser.add_argument("--r7-profile", choices=sorted(R7_PROFILES), default=None)
    parser.add_argument("--r3-workers", type=int, default=None)
    parser.add_argument("--r3-stable-model", type=Path, default=None)
    parser.add_argument("--r3-stable-manifest", type=Path, default=None)
    parser.add_argument("--r3-model-snapshot", type=Path, default=None)
    parser.add_argument("--r3-model-device", choices=["cpu", "mps"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = args.config.expanduser().resolve()
        config = load_config(config_path)
        apply_overrides(config, args)
        validate_resolved_config(config)
        preflight = input_preflight(config)
    except ConfigurationError as exc:
        raise SystemExit(f"Input preflight failed: {exc}") from exc

    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.check_only:
        print("Input preflight passed; no scoring was run.")
        return 0

    timestamp = args.timestamp or datetime.now().strftime("%m%d%H%M")
    run_dir = config["output"]["root"] / f"r1_r7_{timestamp}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Run directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    command = core_command(config, run_dir, timestamp)
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "workflow": "paleomagnetic_r1_r7",
        "status": "running",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "timestamp": timestamp,
        "configuration_file": str(config_path),
        "configuration_sha256": sha256(config_path),
        "configuration_resolved": serializable_config(config),
        "preflight": preflight,
        "command": command,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("$ " + " ".join(command))
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        manifest["status"] = "failed"
        manifest["exit_code"] = exc.returncode
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise

    core_summary_path = run_dir / "work/r357_latest_full_summary.json"
    core_summary = json.loads(core_summary_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "results": core_summary.get("deliverables", core_summary["total_review_html"]),
            "indicator_results": core_summary["indicators"],
            "core_summary": str(core_summary_path),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Run manifest: {manifest_path}")
    print(f"Combined workbook: {manifest['results']['workbook']}")
    print(f"Detailed HTML: {manifest['results']['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
