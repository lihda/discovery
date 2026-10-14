#!/usr/bin/env python3
"""Inject final indicator result data into the reusable HTML review template."""

from __future__ import annotations

import argparse
from pathlib import Path

from indicator_workflow import DEFAULT_TARGETS, DEFAULT_TEMPLATE, normalize_indicator, run_create_results_page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indicator", type=normalize_indicator, required=True, help="R3, R5, or R7.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--standard", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    run_create_results_page(args.indicator, args.input, args.output, args.template, args.title, args.standard)


if __name__ == "__main__":
    main()
