#!/usr/bin/env python3
"""Run one scoring script with a fail-closed filesystem read boundary.

This is an internal helper for ``run_r357_strict_knowledge_workflow.py``.
Non-runtime files may be read only when explicitly allowlisted, except for
Python/native executable code. All filesystem writes are confined to a newly
created scoring output directory. The worker also blocks network connections
and child-process creation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Sequence


EXECUTABLE_CODE_SUFFIXES = {".py", ".pyc", ".pyd", ".so", ".dylib"}


class StrictBoundaryViolation(PermissionError):
    """Raised when scoring attempts an undeclared external operation."""


def _resolved(path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> Path:
    return Path(os.fsdecode(path)).expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _open_access(mode: Any, flags: Any) -> tuple[bool, bool]:
    if isinstance(mode, str):
        writing = any(marker in mode for marker in ("w", "a", "x", "+"))
        reading = "r" in mode or "+" in mode or not writing
        return reading, writing
    numeric_flags = flags if isinstance(flags, int) else 0
    writing = bool(
        numeric_flags
        & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)
    )
    reading = not bool(numeric_flags & os.O_WRONLY) or bool(numeric_flags & os.O_RDWR)
    return reading, writing


def install_boundary(
    allowed_reads: Sequence[Path],
    write_root: Path,
    code_roots: Sequence[Path] = (),
) -> None:
    allowed = {path.resolve(strict=True) for path in allowed_reads}
    write_root = write_root.resolve(strict=True)
    resolved_code_roots = [path.resolve(strict=True) for path in code_roots]
    runtime_roots = {
        Path(sys.prefix).resolve(strict=False),
        Path(sys.base_prefix).resolve(strict=False),
    }
    virtual_system_roots = {
        Path("/dev").resolve(strict=False),
        # Standard-library modules such as mimetypes read static OS config.
        Path("/etc").resolve(strict=False),
        Path("/proc").resolve(strict=False),
        Path("/sys").resolve(strict=False),
    }

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and args:
            raw_path = args[0]
            if isinstance(raw_path, int):
                return
            try:
                path = _resolved(raw_path)
            except (OSError, TypeError, ValueError):
                return
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            reading, writing = _open_access(mode, flags)
            if writing and not _is_within(path, write_root):
                raise StrictBoundaryViolation(
                    f"Strict scoring boundary blocked write outside output directory: {path}"
                )
            if (
                reading
                and path not in allowed
                and not _is_within(path, write_root)
                and not any(_is_within(path, root) for root in runtime_roots)
                and not any(_is_within(path, root) for root in virtual_system_roots)
                and not (
                    path.suffix.casefold() in EXECUTABLE_CODE_SUFFIXES
                    and any(_is_within(path, root) for root in resolved_code_roots)
                )
            ):
                raise StrictBoundaryViolation(
                    f"Strict scoring boundary blocked undeclared data read: {path}"
                )
        elif event == "socket.connect":
            raise StrictBoundaryViolation(
                "Strict scoring boundary blocked a network connection"
            )
        elif event == "subprocess.Popen":
            raise StrictBoundaryViolation(
                "Strict scoring boundary blocked child-process creation"
            )

    sys.addaudithook(audit)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--allow-read", type=Path, action="append", default=[])
    parser.add_argument("--write-root", type=Path, required=True)
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments for --script; place them after --.",
    )
    args = parser.parse_args(argv)
    if args.script_args[:1] == ["--"]:
        args.script_args = args.script_args[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script = args.script.resolve(strict=True)
    if not args.write_root.is_dir():
        raise SystemExit(f"Strict write root is not a directory: {args.write_root}")
    missing = [path for path in args.allow_read if not path.is_file()]
    if missing:
        raise SystemExit(
            "Allowed scoring input(s) do not exist: "
            + ", ".join(str(path) for path in missing)
        )

    install_boundary(
        args.allow_read,
        args.write_root,
        code_roots=[script.parent, Path(__file__).resolve().parent],
    )
    sys.argv = [str(script), *args.script_args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code in (None, 0):
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
