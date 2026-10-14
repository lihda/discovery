#!/usr/bin/env python3
"""Create a final user-facing workbook for an R-index indicator."""

from __future__ import annotations

import argparse
from pathlib import Path

from indicator_workflow import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TARGETS,
    default_predictions,
    normalize_indicator,
    run_create_final_workbook,
    validate_timestamp,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indicator", type=normalize_indicator, required=True, help="R3, R5, or R7.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Working prediction workbook. Defaults to outputs/<indicator>_working_predictions.xlsx.",
    )
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", type=validate_timestamp, default=None)
    args = parser.parse_args()

    run_create_final_workbook(
        args.indicator,
        args.predictions or default_predictions(args.indicator),
        args.targets,
        args.output,
        args.output_dir,
        args.timestamp,
    )


if __name__ == "__main__":
    main()
