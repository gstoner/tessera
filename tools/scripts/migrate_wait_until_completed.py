#!/usr/bin/env python3
"""Migrate ``[cb commit]; [cb waitUntilCompleted];`` sites in
``apple_gpu_runtime.mm`` to the Pattern-4 ``commit_and_wait_with_timeout``
wrapper.

Audit follow-on for the waitUntilCompleted migration (2026-06-01).
Batch 4+ — the canonical patterns are uniform enough across the
remaining ~25 sites that a regex-based migration is safe + faster
than 25 hand-edits. The migration is idempotent — running again
on an already-migrated file is a no-op.

Patterns handled:

Pattern 1 (return-on-failure / bool function):

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return false;

→

    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "<op_name>")) return false;

Pattern 2 (return-on-failure / int function):

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return 0;

→

    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "<op_name>")) return 0;

Pattern 3 (capture-status / `bool ok` variant):

    [cb commit];
    [cb waitUntilCompleted];
    bool ok = (cb.status == MTLCommandBufferStatusCompleted);

→

    bool ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                           "<op_name>");

Pattern 4 (capture-status / `bool _pool_ok` variant, autoreleasepool body):

    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);

→

    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "<op_name>");

``op_name`` is derived from the enclosing function — the most recent
``dispatch_<name>(`` / ``tessera_apple_gpu_<name>(`` declaration above
the site. Used only as a diagnostic on timeout; not a behavior change.

Hard-skipped sites (these are correct as-is):

* The fallback path INSIDE ``commit_and_wait_with_timeout`` itself
  (around the helper's lazy event init). This block intentionally
  uses ``waitUntilCompleted`` because event creation just failed —
  the wrapper would recurse otherwise.
* ``[s->mtlcb waitUntilCompleted]`` in ``bmm_dev_f32_enc`` — encode
  session, uses ``s->mtlcb`` (not ``cb``); separate migration follow-on.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Source location.
_SRC = (Path(__file__).resolve().parent.parent.parent
        / "src" / "compiler" / "codegen"
        / "Tessera_Apple_Backend" / "runtime"
        / "apple_gpu_runtime.mm")


# Each pattern is anchored on the literal ``[cb commit];\n<whitespace>[cb
# waitUntilCompleted];`` pair so we never touch the wrapper's fallback
# (which uses ``[cb commit]; [cb waitUntilCompleted];`` but without the
# trailing status-check pattern).

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Pattern 1 — bool function, return false on failure.
    (re.compile(
        r"^(?P<indent>[ ]+)\[cb commit\];\n"
        r"(?P=indent)\[cb waitUntilCompleted\];\n"
        r"\s*\n?"
        r"(?P=indent)if \(cb\.status != MTLCommandBufferStatusCompleted\) "
        r"return false;",
        re.MULTILINE),
     'BOOL_FALSE'),
    # Pattern 2 — int function, return 0 on failure.
    (re.compile(
        r"^(?P<indent>[ ]+)\[cb commit\];\n"
        r"(?P=indent)\[cb waitUntilCompleted\];\n"
        r"\s*\n?"
        r"(?P=indent)if \(cb\.status != MTLCommandBufferStatusCompleted\) "
        r"return 0;",
        re.MULTILINE),
     'INT_ZERO'),
    # Pattern 3 — ``bool ok = (cb.status == ...)``.
    (re.compile(
        r"^(?P<indent>[ ]+)\[cb commit\];\n"
        r"(?P=indent)\[cb waitUntilCompleted\];\n"
        r"(?P=indent)bool ok = \(cb\.status == "
        r"MTLCommandBufferStatusCompleted\);",
        re.MULTILINE),
     'BOOL_OK'),
    # Pattern 4 — ``bool _pool_ok = (cb.status == ...)``.
    (re.compile(
        r"^(?P<indent>[ ]+)\[cb commit\];\n"
        r"(?P=indent)\[cb waitUntilCompleted\];\n"
        r"(?P=indent)bool _pool_ok = \(cb\.status == "
        r"MTLCommandBufferStatusCompleted\);",
        re.MULTILINE),
     'BOOL_POOL_OK'),
]


# Walk backward from a match to find the enclosing function name.
# Looks for ``dispatch_<name>(`` or ``tessera_apple_gpu_<name>(`` at the
# start of a line. Returns a short identifier suitable for the
# diagnostic name argument.
_FN_PROBE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_ <>:,\*&\n]*\b)?"
    r"(?P<name>(?:dispatch|tessera_apple_gpu)_[A-Za-z0-9_]+)\s*\(",
    re.MULTILINE)


def _fn_name_above(src: str, pos: int) -> str:
    """Find the most recent function declaration starting at column 0
    above ``pos``. Returns the function name or a sentinel."""
    # Scan backward in 8KB windows.
    start = max(0, pos - 8192)
    window = src[start:pos]
    last_match = None
    for m in _FN_PROBE.finditer(window):
        last_match = m
    if last_match is None:
        return "apple_gpu_unknown"
    name = last_match.group("name")
    # Strip the ``dispatch_`` / ``tessera_apple_gpu_`` prefix; both are
    # noise in a diagnostic that already says "[tessera_apple_gpu] ...".
    for prefix in ("tessera_apple_gpu_", "dispatch_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def _replacement(pattern_kind: str, indent: str, op_name: str) -> str:
    """Build the replacement text for one site."""
    # Long-running ops (SVD, decomposition, batched matrix-multiply,
    # flash_attn variants) get 60 s; others get 30 s. Heuristic — we
    # bump timeouts for the small set of known-slow ops; everything
    # else uses the default.
    long_running = (
        "svd" in op_name or "lu" in op_name
        or "flash_attn" in op_name or "matmul_softmax_tiled" in op_name
        or "ebm" in op_name or "convolution2d" in op_name)
    timeout_ms = 60000 if long_running else 30000

    if pattern_kind == 'BOOL_FALSE':
        return (
            f"{indent}// waitUntilCompleted migration (2026-06-01) "
            f"— Pattern 4 wrapper.\n"
            f"{indent}if (!commit_and_wait_with_timeout(ctx, cb, "
            f"{timeout_ms},\n"
            f"{indent}                                  "
            f'"{op_name}")) return false;')
    if pattern_kind == 'INT_ZERO':
        return (
            f"{indent}// waitUntilCompleted migration (2026-06-01) "
            f"— Pattern 4 wrapper.\n"
            f"{indent}if (!commit_and_wait_with_timeout(ctx, cb, "
            f"{timeout_ms},\n"
            f"{indent}                                  "
            f'"{op_name}")) return 0;')
    if pattern_kind == 'BOOL_OK':
        return (
            f"{indent}// waitUntilCompleted migration (2026-06-01) "
            f"— Pattern 4 wrapper.\n"
            f"{indent}bool ok = commit_and_wait_with_timeout(ctx, cb, "
            f"{timeout_ms},\n"
            f"{indent}                                       "
            f'"{op_name}");')
    if pattern_kind == 'BOOL_POOL_OK':
        return (
            f"{indent}// waitUntilCompleted migration (2026-06-01) "
            f"— Pattern 4 wrapper.\n"
            f"{indent}bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, "
            f"{timeout_ms},\n"
            f"{indent}                                             "
            f'"{op_name}");')
    raise ValueError(f"unknown pattern kind: {pattern_kind!r}")


def migrate(text: str) -> tuple[str, int]:
    """Apply all migration patterns. Returns ``(new_text, n_sites)``."""
    # We accumulate edits as (start, end, new_text) tuples, then apply
    # them in reverse order so indices don't shift.
    edits: list[tuple[int, int, str]] = []
    for pat, kind in _PATTERNS:
        for m in pat.finditer(text):
            indent = m.group("indent")
            op = _fn_name_above(text, m.start())
            new = _replacement(kind, indent, op)
            edits.append((m.start(), m.end(), new))
    # Sort by start descending so later edits don't disrupt earlier
    # indices.
    edits.sort(key=lambda e: e[0], reverse=True)
    out = text
    for s, e, n in edits:
        out = out[:s] + n + out[e:]
    return out, len(edits)


def main(argv: list[str]) -> int:
    dry_run = "--apply" not in argv
    text = _SRC.read_text()
    new_text, n = migrate(text)
    if n == 0:
        print("no migration sites found — file is already migrated")
        return 0
    if dry_run:
        # Show a brief preview of each migrated site (first 80 chars
        # of replacement, for quick spot-check).
        print(f"would migrate {n} site(s). Re-run with --apply to write.")
        return 0
    _SRC.write_text(new_text)
    print(f"migrated {n} site(s) in {_SRC}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
