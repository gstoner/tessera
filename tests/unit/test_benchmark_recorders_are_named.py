"""Every benchmark recorder is named by some other tracked file.

A recorder nothing names — no README row, no packet manifest, no test, no log
entry — is a declaration without a consumer (Decision #29): it reads in the tree
as a lane that exists, while nothing can say what it recorded or where. The
2026-09-17 review found 50 such files, 21 of them `benchmarks/nvidia/record_*`
whose products are cited by date in the NVIDIA queue but never by the recorder
that made them.

Those 50 were pruned on 2026-09-17 (owner-approved): 41 are now named by the
"Recorders and their outputs" table in `benchmarks/README.md`, by a
"Recorded by" line in their packet's README, or by the SuperBench README; the
nine with no tracked product were deleted. The ratchet now holds at an empty
floor: a new recorder that lands unnamed fails here, on the CPU-only lane.
Never add an entry to `KNOWN_UNNAMED` to make a new orphan pass.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMING_ROOTS = ("tests", "docs", "python", "scripts", "benchmarks", "tools", ".github", "CLAUDE.md", "AGENTS.md")

#: Orphans as of 2026-09-17, pruned the same day — empty by construction.
#: Never add an entry here to make a new orphan pass.
KNOWN_UNNAMED: frozenset[str] = frozenset()


def _recorders() -> list[Path]:
    return sorted(p for p in (ROOT / "benchmarks").rglob("*.py")
                  if "__pycache__" not in p.parts and p.name != "__init__.py" and len(p.stem) >= 6)


def _unnamed(recorders: list[Path]) -> set[str]:
    """Stems no other tracked file mentions as a whole token.

    One pass over the tracked files under the naming roots, tokenised on
    identifier characters, intersected with the stem set — a few seconds, where
    `git grep -o` over ~300 fixed strings cost two minutes on the doc tree. Two
    recorders can share a stem (`rocm/` and `x86/` both have
    `record_deltanet_backward_selectors.py`); a recorder is named only by a file
    outside that stem's own set, so siblings do not name each other.
    """
    import re

    paths_by_stem: dict[str, set[str]] = {}
    for p in recorders:
        paths_by_stem.setdefault(p.stem, set()).add(str(p.relative_to(ROOT)))
    tracked = subprocess.run(
        ["git", "ls-files", "--", *NAMING_ROOTS], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.split("\n")
    token = re.compile(r"[A-Za-z0-9_]+")
    named: set[str] = set()
    self_name = f"tests/unit/{Path(__file__).name}"
    for rel in tracked:
        if not rel or rel == self_name:
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for stem in set(token.findall(text)) & paths_by_stem.keys():
            if rel not in paths_by_stem[stem]:
                named.add(stem)
    return {path for stem, paths in paths_by_stem.items() if stem not in named for path in paths}


def test_every_recorder_is_named_or_frozen():
    unnamed = _unnamed(_recorders())
    new = sorted(unnamed - KNOWN_UNNAMED)
    assert not new, f"new recorder(s) nothing names — add a README row, a packet manifest or a test: {new}"
    healed = sorted(KNOWN_UNNAMED - unnamed)
    assert not healed, f"now named (or gone); remove from KNOWN_UNNAMED: {healed}"
