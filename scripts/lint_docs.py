#!/usr/bin/env python3
"""Reject pre-canonical terminology in active docs and source comments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FORBIDDEN_PATTERNS = (
    ("TIR-H", re.compile(r"\bTIR-H\b")),
    ("TIR-T", re.compile(r"\bTIR-T\b")),
    ("old TypeScript-style function decorator", re.compile(r"@ts\.function\b")),
    ("old IR inspection helper", re.compile(r"\bts\.inspect_ir\b")),
    ("old compile decorator", re.compile(r"@tessera\.compile\b")),
    ("old autodiff decorator", re.compile(r"@autodiff\b")),
)

PRODUCTION_READY = re.compile(r"\bProduction Ready\b")
PHASE_OR_STATUS = re.compile(r"\b(phase|status)\b", re.IGNORECASE)

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cmake",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hip",
    ".hpp",
    ".json",
    ".md",
    ".mlir",
    ".pdll",
    ".py",
    ".rst",
    ".sh",
    ".td",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

DEFAULT_PATHS = ("docs", "src", "README.md", "PROJECT_STRUCTURE.md")


def is_archive_path(path: Path) -> bool:
    return "archive" in path.parts


def is_text_file(path: Path) -> bool:
    if path.name in {"CMakeLists.txt"}:
        return True
    return path.suffix in TEXT_SUFFIXES


def iter_files(root: Path, scan_paths: tuple[str, ...]):
    for raw_path in scan_paths:
        path = root / raw_path
        if not path.exists():
            continue
        if path.is_file():
            if not is_archive_path(path.relative_to(root)) and is_text_file(path):
                yield path
            continue
        for candidate in path.rglob("*"):
            rel = candidate.relative_to(root)
            if candidate.is_file() and not is_archive_path(rel) and is_text_file(candidate):
                yield candidate


def production_ready_has_note(lines: list[str], index: int) -> bool:
    start = max(0, index - 2)
    end = min(len(lines), index + 3)
    context = "\n".join(lines[start:end])
    return bool(PHASE_OR_STATUS.search(context))


def lint_file(root: Path, path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    rel = path.relative_to(root)
    lines = text.splitlines()
    findings: list[str] = []

    for line_number, line in enumerate(lines, start=1):
        for label, pattern in FORBIDDEN_PATTERNS:
            if pattern.search(line):
                findings.append(f"{rel}:{line_number}: forbidden term/API: {label}")

        if PRODUCTION_READY.search(line) and not production_ready_has_note(lines, line_number - 1):
            findings.append(
                f"{rel}:{line_number}: 'Production Ready' needs a nearby phase/status note"
            )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    findings: list[str] = []
    for path in iter_files(root, tuple(args.paths)):
        findings.extend(lint_file(root, path))

    if findings:
        print("docs lint failed: forbidden pre-canonical terms found outside archive/")
        for finding in findings:
            print(finding)
        return 1

    print("docs lint passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
