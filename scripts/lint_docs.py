#!/usr/bin/env python3
"""Reject stale terminology and broken active-doc references."""

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

DEFAULT_PATHS = ("docs", "src", "README.md", "PROJECT_STRUCTURE.md", "examples/README.md")

MARKDOWN_LINK = re.compile(r"!?\[[^\]]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)
PATH_PREFIXES = (
    "benchmarks/",
    "docs/",
    "examples/",
    "python/",
    "scripts/",
    "src/",
    "tests/",
    "tools/",
)
PATH_TRAILING = ".,;:)]}"


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


def text_without_fenced_blocks(text: str) -> str:
    return FENCED_BLOCK.sub("", text)


def resolve_doc_link(root: Path, source: Path, target: str) -> Path | None:
    target = target.split("#", 1)[0]
    if not target:
        return None
    if re.match(r"^[a-z][a-z0-9+.-]*:", target):
        return None
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if target.startswith("/"):
        return root / target.lstrip("/")
    return (source.parent / target).resolve()


def normalize_inline_path(raw: str) -> str | None:
    candidate = raw.strip().strip(PATH_TRAILING)
    if not candidate.startswith(PATH_PREFIXES):
        return None
    if any(token in candidate for token in ("*", "...", "…", "{", "}", "$")):
        return None
    if " " in candidate:
        return None
    return candidate


def lint_markdown_links(root: Path, path: Path, text: str) -> list[str]:
    if path.suffix.lower() not in {".md", ".mdx"}:
        return []
    findings: list[str] = []
    rel = path.relative_to(root)
    text = text_without_fenced_blocks(text)
    for match in MARKDOWN_LINK.finditer(text):
        target = match.group(1)
        resolved = resolve_doc_link(root, path, target)
        if resolved is None:
            continue
        if not resolved.exists():
            line_number = text.count("\n", 0, match.start()) + 1
            findings.append(f"{rel}:{line_number}: broken markdown link: {target}")
    return findings


def lint_inline_paths(root: Path, path: Path, text: str) -> list[str]:
    findings: list[str] = []
    rel = path.relative_to(root)
    text = MARKDOWN_LINK.sub("", text_without_fenced_blocks(text))
    for match in INLINE_CODE.finditer(text):
        candidate = normalize_inline_path(match.group(1))
        if candidate is None:
            continue
        if not (root / candidate).exists():
            line_number = text.count("\n", 0, match.start()) + 1
            findings.append(f"{rel}:{line_number}: missing path reference: {candidate}")
    return findings


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

    if rel.parts and (rel.parts[0] == "docs" or rel.as_posix() in {"README.md", "examples/README.md"}):
        findings.extend(lint_markdown_links(root, path, text))
        findings.extend(lint_inline_paths(root, path, text))
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
        print("docs lint failed")
        for finding in findings:
            print(finding)
        return 1

    print("docs lint passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
