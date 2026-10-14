#!/usr/bin/env python3
"""Compatibility entry point for the stable grouped R3 OOF experiment."""

from __future__ import annotations

from pathlib import Path
import runpy


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/experiment_r3_grouped_oof_text_model.py"
)
runpy.run_path(str(SCRIPT), run_name="__main__")
