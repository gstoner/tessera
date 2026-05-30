#!/usr/bin/env python3
"""Reject stale terminology and broken active-doc references."""

from __future__ import annotations

import argparse
import functools
import os
import re
import subprocess
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

# Planning / audit / status docs describe *future* or hypothetical state and
# deliberately reference paths that do not exist yet (planned backends, roadmap
# files, out-of-scope features, illustrative placeholders).  Inline-path
# existence linting is wrong for them by design, so it is skipped for these
# subtrees.  Forbidden-term, "Production Ready", and broken-markdown-link checks
# still apply everywhere — only the "does this code path exist today" check is
# relaxed for roadmap content.
PLANNING_DOC_SUBTREES = ("audit", "architecture", "status")

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


def is_planning_doc(rel: Path) -> bool:
    """True for roadmap/audit/status docs that intentionally reference paths
    which do not exist yet — these are exempt from inline-path existence linting."""
    return (
        len(rel.parts) >= 2
        and rel.parts[0] == "docs"
        and rel.parts[1] in PLANNING_DOC_SUBTREES
    )


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


# A trailing location annotation on a path is a pointer, not part of the file
# name: ``foo.py:42``, ``foo.py:42+``, ``foo.py:42-99``, ``foo.py:42–99`` (en/em
# dash), ``foo.py:~42``. Strip it before the existence check (the ``path:line``
# convention is documented in CLAUDE.md).
_LINE_ANNOTATION = re.compile(r":~?\d+(?:\s*[-–—]\s*\d*)?\+?$")


def normalize_inline_path(raw: str) -> str | None:
    candidate = raw.strip().strip(PATH_TRAILING)
    candidate = _LINE_ANNOTATION.sub("", candidate)
    # pytest node ids (``test_x.py::TestY::test_z``) point at a test, not a file.
    if "::" in candidate:
        candidate = candidate.split("::", 1)[0]
    if not candidate.startswith(PATH_PREFIXES):
        return None
    if any(token in candidate for token in ("*", "...", "…", "{", "}", "$")):
        return None
    if " " in candidate:
        return None
    return candidate


@functools.lru_cache(maxsize=1)
def _git_tracked(root_str: str) -> frozenset[str] | None:
    """Exact-case set of git-tracked paths (posix-relative), or ``None`` when
    the tree is not a git checkout / git is unavailable.  This is the casing
    Linux CI checks out — authoritative even on a case-insensitive working tree
    whose on-disk filenames disagree with the index."""
    try:
        out = subprocess.run(
            ["git", "-C", root_str, "ls-files", "-z"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return frozenset(p for p in out.stdout.split("\0") if p)


def _exists_case_sensitive(root: Path, rel: str) -> bool:
    """True only if ``root/rel`` exists with *exact* casing.  ``Path.exists()``
    on macOS/Windows is case-insensitive, so a wrong-case reference
    (``CONFORMANCE.md`` for ``conformance.md``) passes locally but breaks on
    case-sensitive Linux CI.  Prefer git's tracked-path set (matches CI's
    checkout exactly); fall back to an ``os.listdir`` walk when git is absent."""
    rel = rel.strip("/")
    tracked = _git_tracked(str(root))
    if tracked is not None:
        if rel in tracked:
            return True
        prefix = rel + "/"
        return any(p.startswith(prefix) for p in tracked)
    cur = root
    for part in Path(rel).parts:
        if part in ("", "."):
            continue
        try:
            if part not in os.listdir(cur):
                return False
        except (OSError, NotADirectoryError):
            return False
        cur = cur / part
    return True


def _inline_path_exists(root: Path, candidate: str) -> bool:
    """A reference resolves if the path exists (exact case), or — for an
    extension-less module reference (``python/tessera/ops``) — ``<path>.py`` or
    ``<path>/__init__.py`` exists."""
    if _exists_case_sensitive(root, candidate):
        return True
    if not Path(candidate).suffix:
        return (_exists_case_sensitive(root, candidate + ".py")
                or _exists_case_sensitive(root, candidate + ".pyi")
                or _exists_case_sensitive(root, candidate + "/__init__.py"))
    return False


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
        if not _inline_path_exists(root, candidate):
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
        if not is_planning_doc(rel):
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
