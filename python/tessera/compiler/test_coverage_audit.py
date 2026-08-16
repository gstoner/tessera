"""Audit-D (2026-05-22) — Test coverage by op family.

The primitive coverage registry at ``primitive_coverage.py`` marks
all 432 entries as ``tests=complete``.  That's the registry's
weakest defended axis — the claim is a category-level rollup, not a
per-op proof.  This module surfaces the actual numbers:

  * Per-op reference count in ``tests/unit/*.py`` (Python).
  * Per-op reference count in ``tests/tessera-ir/**/*.mlir`` (lit).
  * Negative-test count per op (``pytest.raises`` for Python,
    ``expected-error`` for lit).
  * Dtype variant coverage (which of ``f32 / f16 / bf16 / fp8_e4m3``
    etc. each op is tested with).
  * "Thinly-tested" op set — ops with ``tests=complete`` but zero
    or one reference across the whole test surface.

The by-op section in ``docs/audit/generated/test_coverage.md``
surfaces:

  * Headline counts (ops with 0 refs, 1 ref, ≥2 refs, ≥10 refs).
  * Top 20 most-tested ops.
  * "Suspiciously thin" ops (0 or 1 test reference).
  * Per-op-family rollups so structural ops (transpose, reshape,
    cast, pack, unpack) aren't lumped with compute ops.

Drift gates at ``tests/unit/test_test_coverage_audit.py`` and the
fleet-wide generated-doc registry:

  * Total reference count floor (catches a regression where a
    sweep accidentally deletes tests).
  * Sentinel-op floors for high-traffic primitives (matmul,
    flash_attn, softmax) so a major rewrite doesn't silently
    drop their coverage.
  * Dashboard ↔ live data sync.
  * Canonical generated artifact drift via
    ``tests/unit/test_generated_docs_registry.py``.

Honest scope note: this audit measures **reference counts**, not
**numerical coverage quality**.  A single test that exercises ``matmul``
across 5 shapes × 3 dtypes counts as one reference but covers more
ground than 5 trivial happy-path tests.  Refining the audit toward
"numerical coverage quality" is a follow-up sprint.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_TESTS_UNIT = _REPO_ROOT / "tests" / "unit"
_TESTS_LIT = _REPO_ROOT / "tests" / "tessera-ir"


# ─────────────────────────────────────────────────────────────────────────
# Op-reference parsing
# ─────────────────────────────────────────────────────────────────────────


#: Family modules that route op calls through their own namespace
#: rather than ``tessera.ops``.  For each module ``M`` we count
#: ``tessera.M.X``, ``M.X``, and ``from tessera.M import X`` as a real
#: reference to op ``X``.  This catches the family-test pattern where
#: e.g. ``test_s10_optim.py`` calls ``optim.sgd(...)`` directly without
#: ever going through ``tessera.ops.sgd``.
_FAMILY_MODULES: tuple[str, ...] = (
    "optim",
    "losses",
    "rl",
    "quantization",
    "memory",
    "sharding",
    "control",
    "rng",
    "state",
    "data",
    "aot",
    "custom",
    "cache",
    "nn",
    "checkpoint",
    # Sub-modules used directly by tests (``from tessera.nn import
    # functional as F`` is common, and ``tessera.complex`` carries the
    # M7 visual-complex op surface).
    "functional",
    "complex",
    # Host-level solver primitives (P2 tranche) and the coalition-lattice
    # family.  Both route through their own namespace: ``solvers_ops``
    # exports ``tridiagonal_solve`` directly, and ``game`` exports the
    # ``game_``-prefixed registry ops under their short names
    # (``game_subset_zeta`` → ``tessera.game.subset_zeta``) — see
    # ``_MODULE_PREFIXED_FAMILIES``.
    "solvers_ops",
    "game",
)


#: Family modules whose registry op names carry the module name as a
#: prefix.  ``tessera.game`` exports ``subset_zeta``, but the primitive
#: registry (and the op catalog) name it ``game_subset_zeta``; without
#: this mapping every op in such a family scores zero references no
#: matter how thoroughly it is tested.
_MODULE_PREFIXED_FAMILIES: frozenset[str] = frozenset({"game"})


def _ops_namespace_patterns(op_name: str) -> tuple[re.Pattern, ...]:
    """Patterns that match a real call to a TSOL op.

    We deliberately use word-boundary anchors so ``matmul`` doesn't
    match ``batched_matmul``.  Examples that should match:

      tessera.ops.matmul(...)
      ts.ops.matmul(...)
      ops.matmul(...)
      "tessera.matmul"                     # MLIR-style string ref
      from tessera.ops import matmul       # import-form reference
      tessera.optim.sgd(...)               # family-module call
      optim.sgd(...)                       # short-form family call
      from tessera.optim import sgd        # family-module import
      losses.ppo_policy_loss(...)          # cross-family (rl in losses)

    Examples that should NOT match:

      tessera.ops.batched_matmul(...)
      __batched_matmul__
      # comment referencing matmul prose
    """
    escaped = re.escape(op_name)
    patterns: list[re.Pattern] = [
        re.compile(rf"\b(?:tessera|ts)\.ops\.{escaped}\b"),
        re.compile(rf"(?<![A-Za-z0-9_])ops\.{escaped}\b"),
        re.compile(rf'"tessera\.{escaped}"'),
        re.compile(
            rf"\bfrom tessera\.ops import [^\n]*\b{escaped}\b"
        ),
    ]
    # Per-family-module patterns: catches ``optim.sgd``,
    # ``tessera.optim.sgd``, ``from tessera.optim import sgd``, etc.
    for mod in _FAMILY_MODULES:
        patterns.append(
            re.compile(rf"\btessera\.{mod}\.{escaped}\b")
        )
        patterns.append(
            re.compile(rf"(?<![A-Za-z0-9_]){mod}\.{escaped}\b")
        )
        patterns.append(
            re.compile(
                rf"\bfrom tessera\.{mod} import [^\n]*\b{escaped}\b"
            )
        )
    return tuple(patterns)


# Negative-test markers in Python.  We match the block that surrounds
# an op reference: ``with pytest.raises(...)`` within ~20 lines of
# the reference counts as a negative test.
_PYTEST_RAISES_RE = re.compile(r"pytest\.raises\s*\(")

# Negative-test markers in lit fixtures.
_LIT_EXPECTED_ERROR_RE = re.compile(r"//\s*expected-error\b")

# Dtype literal patterns.  Used to bucket which dtypes a test
# exercises.  The set covers the canonical dtype names from
# ``python/tessera/dtype.py``.
_DTYPE_NAMES = (
    "f32", "f16", "bf16", "f64",
    "fp32", "fp16", "bf16", "fp64",
    "fp8_e4m3", "fp8_e5m2",
    "fp6_e2m3", "fp6_e3m2",
    "fp4_e2m1", "nvfp4",
    "int8", "int16", "int32", "int64",
)
_DTYPE_RE = re.compile(
    r'(?:dtype\s*=\s*)?["\']('
    + "|".join(re.escape(n) for n in _DTYPE_NAMES)
    + r')["\']'
)


@dataclass(frozen=True)
class OpTestCoverage:
    """One op's test-coverage snapshot."""

    op_name: str
    python_refs: int            # tests/unit/*.py reference count
    lit_refs: int               # tests/tessera-ir/**/*.mlir reference count
    negative_refs: int          # Python `pytest.raises` near op references
    dtype_variants: tuple[str, ...]  # dtypes exercised, sorted
    test_files: tuple[str, ...]      # files that reference the op (top 5)

    @property
    def total_refs(self) -> int:
        return self.python_refs + self.lit_refs

    @property
    def is_thinly_tested(self) -> bool:
        """True if the op has 0 or 1 references — suspicious for
        anything in `primitive_coverage` with ``tests=complete``."""
        return self.total_refs <= 1


# ─────────────────────────────────────────────────────────────────────────
# Scanning
# ─────────────────────────────────────────────────────────────────────────


def _scan_python_for_op(op_name: str) -> tuple[int, int, set[str], set[str]]:
    """Return ``(refs, negative_refs, dtypes_seen, files_touched)``.

    ``negative_refs`` counts pytest.raises blocks that appear within
    20 lines of an op reference (approximation — good enough for the
    audit dashboard, not a strict semantic check).
    """
    patterns = _ops_namespace_patterns(op_name)
    refs = 0
    neg = 0
    dtypes: set[str] = set()
    files: set[str] = set()

    for path in _TESTS_UNIT.rglob("*.py"):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        # Count positive references.
        path_refs = 0
        match_positions: list[int] = []
        for pat in patterns:
            for m in pat.finditer(text):
                path_refs += 1
                match_positions.append(m.start())
        if not path_refs:
            continue
        refs += path_refs
        files.add(path.relative_to(_REPO_ROOT).as_posix())
        # Look for pytest.raises within ±20 lines of any reference.
        lines = text.splitlines()
        line_offsets = [0]
        for ln in lines:
            line_offsets.append(line_offsets[-1] + len(ln) + 1)

        # ``line_offsets`` is sorted ascending; the line index for a
        # character offset is the rightmost slot whose offset is <= it.
        ref_lines = {bisect.bisect_right(line_offsets, p) - 1 for p in match_positions}
        for m in _PYTEST_RAISES_RE.finditer(text):
            raises_line = bisect.bisect_right(line_offsets, m.start()) - 1
            if any(abs(raises_line - rl) <= 20 for rl in ref_lines):
                neg += 1
        # Capture dtype literals appearing in the same file.
        for d in _DTYPE_RE.finditer(text):
            dtypes.add(d.group(1))
    return refs, neg, dtypes, files


def _scan_lit_for_op(op_name: str) -> int:
    """Count references in tests/tessera-ir/**/*.mlir."""
    refs = 0
    # MLIR-side ops are typically `tessera.<name>` or `tessera.queue.<name>`.
    # We accept either form.
    pat = re.compile(
        rf'"tessera(?:\.[a-z_]+)?\.{re.escape(op_name)}"'
        rf"|tessera(?:\.[a-z_]+)?\.{re.escape(op_name)}\b"
    )
    for path in _TESTS_LIT.rglob("*.mlir"):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        refs += len(pat.findall(text))
    return refs


# ─────────────────────────────────────────────────────────────────────────
# Collection
# ─────────────────────────────────────────────────────────────────────────


# Cache the per-op scan so repeated dashboard renders + drift-gate
# parametrizations don't redo the file-walk work.
_COVERAGE_CACHE: dict[str, OpTestCoverage] | None = None


def _all_op_names() -> tuple[str, ...]:
    """Return every op name in the primitive coverage registry."""
    from .primitive_coverage import all_primitive_coverages
    return tuple(sorted(all_primitive_coverages().keys()))


# Combined regexes that capture every op-reference form in one pass.
# Python's re module forbids duplicate named groups across alternatives,
# so we use anonymous groups and pick whichever fired.
_OP_NAME_RE = r"([a-zA-Z_][a-zA-Z0-9_]*)"
_TRAINING_OP_NAME_RE = r"(training\.[a-zA-Z_][a-zA-Z0-9_]*)"

# Each entry is (pattern, owning_module).  ``owning_module`` is the
# family module a dotted call belongs to (``"game"`` for
# ``game.subset_zeta(...)``), or ``None`` for the module-agnostic forms
# (``tessera.ops.*`` / ``ops.*`` / the MLIR-string form).  Built as a list
# of parts — rather than one f-string — so the group-index-to-module
# mapping below can be derived from the same source instead of hand-counted.
_PY_OP_REFERENCE_PARTS: tuple[tuple[str, str | None], ...] = (
    (rf'"tessera\.{_TRAINING_OP_NAME_RE}"', None),
    (rf"\b(?:tessera|ts)\.ops\.{_OP_NAME_RE}\b", None),
    (rf"(?<![A-Za-z0-9_])ops\.{_OP_NAME_RE}\b", None),
    (rf'"tessera\.{_OP_NAME_RE}"', None),
    # NOTE: the ``from tessera.<mod> import ...`` alternatives that used
    # to live here are owned by ``_import_binding_refs`` — see the
    # comment above ``_FAMILY_IMPORT_RE``.
) + tuple(
    part
    for mod in _FAMILY_MODULES
    for part in (
        (rf"\btessera\.{mod}\.{_OP_NAME_RE}\b", mod),
        (rf"(?<![A-Za-z0-9_]){mod}\.{_OP_NAME_RE}\b", mod),
    )
)

_PY_OP_REFERENCE_RE = re.compile(
    "|".join(pattern for pattern, _module in _PY_OP_REFERENCE_PARTS)
)

# 1-based group index -> owning module (or None), so a match's
# ``m.lastindex`` tells us which alternative fired without needing named
# groups (which ``re`` forbids duplicating across alternatives).  Every
# part above has exactly one capturing group, so group order == part order.
_PY_OP_REFERENCE_GROUP_MODULE: tuple[str | None, ...] = tuple(
    module for _pattern, module in _PY_OP_REFERENCE_PARTS
)


def _resolve_py_match(m: re.Match[str], registry: frozenset[str]) -> str | None:
    """Turn one ``_PY_OP_REFERENCE_RE`` match into a registry op name.

    Returns ``None`` when the match doesn't resolve to a registry op
    (e.g. a dotted call into a module-prefixed family whose short name
    isn't a registered op — a helper, not an op).
    """
    name = next((g for g in m.groups() if g is not None), None)
    if name is None:
        return None
    module = (
        _PY_OP_REFERENCE_GROUP_MODULE[m.lastindex - 1] if m.lastindex else None
    )
    if module in _MODULE_PREFIXED_FAMILIES:
        return _resolve_family_op(module, name, registry)
    return name

_LIT_OP_REFERENCE_RE = re.compile(
    rf'"tessera(?:\.[a-z_]+)?\.{_OP_NAME_RE}"'
    rf"|tessera(?:\.[a-z_]+)?\.{_OP_NAME_RE}\b"
)


# ── Import bindings ──────────────────────────────────────────────────
#
# ``from tessera.<mod> import ...`` is handled here rather than in
# ``_PY_OP_REFERENCE_RE`` so that ONE code path owns the import form.
# The combined regex could only match single-line imports (its
# ``[^\n]*?`` cannot cross a newline), which silently scored zero for
# every test using the parenthesized multi-line form — the style the
# coalition-lattice suite uses.
#
# Binding-awareness also lets us count the call sites.  A test that does
# ``from tessera.solvers_ops import tridiagonal_solve`` and then calls
# ``tridiagonal_solve(...)`` twelve times is not a one-reference op; the
# name is in file scope and every bare call is a real exercise of it.
_FAMILY_IMPORT_RE = re.compile(
    r"\bfrom\s+tessera\.(" + "|".join(
        re.escape(m) for m in ("ops",) + _FAMILY_MODULES
    ) + r")\s+import\s+(?:\(([^)]*)\)|([^\n(]*))",
    re.MULTILINE,
)

#: ``name`` / ``name as alias`` inside an import list, comments stripped.
_IMPORT_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s+as\s+([A-Za-z_][A-Za-z0-9_]*))?"
)


def _resolve_family_op(module: str, imported: str, registry: frozenset[str]) -> str | None:
    """Map an imported short name back to its registry op name.

    Returns ``None`` when the name is not a registry op — test files
    import plenty of helpers and fixtures alongside the ops.
    """
    if imported in registry:
        return imported
    if module in _MODULE_PREFIXED_FAMILIES:
        prefixed = f"{module}_{imported}"
        if prefixed in registry:
            return prefixed
    return None


def _import_binding_refs(
    text: str, registry: frozenset[str]
) -> dict[str, list[int]]:
    """Return ``{registry_op: [char offsets]}`` for imported ops.

    Each binding contributes the import statement itself plus every
    bare call site of the locally bound name in the same file.
    """
    found: dict[str, list[int]] = {}
    # Local names bound to each op.  A test file may import the same op
    # several times (module scope plus function-local imports); each
    # import is its own reference, but the call sites must be counted
    # ONCE per bound name, not once per import statement.
    locals_by_op: dict[str, set[str]] = {}
    for m in _FAMILY_IMPORT_RE.finditer(text):
        module = m.group(1)
        names_blob = m.group(2) if m.group(2) is not None else (m.group(3) or "")
        # Strip trailing comments (a lint-suppression pragma on the
        # import line is common in these suites).
        names_blob = re.sub(r"#[^\n]*", "", names_blob)
        if "*" in names_blob:
            continue
        for nm in _IMPORT_NAME_RE.finditer(names_blob):
            imported, alias = nm.group(1), nm.group(2)
            if imported == "as":
                continue
            op = _resolve_family_op(module, imported, registry)
            if op is None:
                continue
            found.setdefault(op, []).append(m.start())
            locals_by_op.setdefault(op, set()).add(alias or imported)

    # Bare call sites of each bound name, e.g. ``subset_zeta(v)``.  The
    # negative lookbehind for ``.`` avoids double-counting
    # ``mod.subset_zeta(...)``, which the combined regex owns.
    for op, names in locals_by_op.items():
        import_offsets = set(found.get(op, ()))
        seen: set[int] = set()
        for local in names:
            call_re = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(local)}\s*\(")
            for cm in call_re.finditer(text):
                if cm.start() in import_offsets or cm.start() in seen:
                    continue
                seen.add(cm.start())
                found[op].append(cm.start())
    return found


def _scan_all_files_vectorized() -> tuple[
    dict[str, dict[Path, list[int]]],  # py_refs_by_op[op][path] = [match_starts]
    dict[str, set[Path]],               # negative_refs_by_op[op] = {paths_with_nearby_raises}
    dict[str, set[str]],                # dtypes_by_op[op]
    dict[str, int],                     # lit_refs_by_op[op]
    dict[Path, list[int]],              # raises_by_path[path] = [line numbers]
]:
    """Walk every test file ONCE and collect per-op reference data.

    Vectorized scanner — replaces the old O(ops × files × patterns)
    nested loop with a single pass over each file using a combined
    regex that captures the op name as a named group.
    """
    py_refs_by_op: dict[str, dict[Path, list[int]]] = {}
    dtypes_by_op_per_path: dict[Path, set[str]] = {}
    lit_refs_by_op: dict[str, int] = {}
    raises_by_path: dict[Path, list[int]] = {}  # line numbers

    # Registry op names, needed to resolve import bindings back to
    # their canonical names (``subset_zeta`` → ``game_subset_zeta``).
    registry = frozenset(_all_op_names())

    # ── Python pass ───────────────────────────────────────────────
    for path in _TESTS_UNIT.rglob("*.py"):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        # Match all op references in one go.  ``_resolve_py_match`` maps
        # a dotted call into a module-prefixed family (e.g.
        # ``game.subset_zeta(...)``) back to its registry key
        # (``game_subset_zeta``) — without it, that reference is recorded
        # under a key no registry op has and is silently dropped by
        # ``collect_op_test_coverage``'s ``for op in all_ops`` loop.
        any_matches = False
        for m in _PY_OP_REFERENCE_RE.finditer(text):
            name = _resolve_py_match(m, registry)
            if name is None:
                continue
            any_matches = True
            py_refs_by_op.setdefault(name, {}).setdefault(path, []).append(
                m.start()
            )
        # Import bindings + their bare call sites (multi-line aware).
        for op, offsets in _import_binding_refs(text, registry).items():
            any_matches = True
            py_refs_by_op.setdefault(op, {}).setdefault(path, []).extend(offsets)
        if not any_matches:
            continue
        # Cache dtype literals per file.
        dtypes_by_op_per_path[path] = {
            m.group(1) for m in _DTYPE_RE.finditer(text)
        }
        # Cache pytest.raises positions per file (as character offsets).
        # Convert to line numbers lazily.
        line_offsets = [0]
        for ln in text.splitlines():
            line_offsets.append(line_offsets[-1] + len(ln) + 1)

        raises_by_path[path] = [
            bisect.bisect_right(line_offsets, m.start()) - 1
            for m in _PYTEST_RAISES_RE.finditer(text)
        ]
        # Also pre-compute line numbers of every op-ref match for
        # this file, op by op.
        for name, paths in py_refs_by_op.items():
            if path in paths:
                # Convert positions to lines once.
                paths[path] = [
                    bisect.bisect_right(line_offsets, p) - 1 for p in paths[path]
                ]

    # ── Compute negatives and dtypes per (op, path) ─────────────
    negatives: dict[str, set[Path]] = {}
    dtypes_by_op: dict[str, set[str]] = {}
    for op, paths in py_refs_by_op.items():
        for path, ref_lines in paths.items():
            # Negatives: pytest.raises within ±20 lines of any ref.
            for rl in raises_by_path.get(path, []):
                if any(abs(rl - r) <= 20 for r in ref_lines):
                    negatives.setdefault(op, set()).add(path)
                    break
            # Dtypes seen in the file (proxy — same dtype could be
            # used by a different op in the same file).
            dtypes_by_op.setdefault(op, set()).update(
                dtypes_by_op_per_path.get(path, set())
            )

    # ── Lit pass ──────────────────────────────────────────────────
    for path in _TESTS_LIT.rglob("*.mlir"):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for m in _LIT_OP_REFERENCE_RE.finditer(text):
            name = next((g for g in m.groups() if g is not None), None)
            if name is None:
                continue
            lit_refs_by_op[name] = lit_refs_by_op.get(name, 0) + 1

    return py_refs_by_op, negatives, dtypes_by_op, lit_refs_by_op, raises_by_path


def collect_op_test_coverage() -> tuple[OpTestCoverage, ...]:
    """Scan every test file once + bucket references per op.

    Cached after first call.  Use :func:`reset_cache` in tests if you
    want to re-scan after editing test files in the same process.

    Uses a vectorized file-walk: each test file is opened exactly once
    and a single combined regex extracts every op reference in one
    pass.  This is ~50× faster than the original O(ops × files)
    nested loop.
    """
    global _COVERAGE_CACHE
    if _COVERAGE_CACHE is not None:
        return tuple(_COVERAGE_CACHE.values())

    (
        py_refs_by_op, negatives, dtypes_by_op,
        lit_refs_by_op, _raises_by_path,
    ) = _scan_all_files_vectorized()

    # Only emit rows for ops in the primitive_coverage registry.
    all_ops = _all_op_names()
    cache: dict[str, OpTestCoverage] = {}
    for op in all_ops:
        paths = py_refs_by_op.get(op, {})
        py_count = sum(len(v) for v in paths.values())
        cache[op] = OpTestCoverage(
            op_name=op,
            python_refs=py_count,
            lit_refs=lit_refs_by_op.get(op, 0),
            negative_refs=len(negatives.get(op, ())),
            dtype_variants=tuple(sorted(dtypes_by_op.get(op, ()))),
            test_files=tuple(
                sorted(p.relative_to(_REPO_ROOT).as_posix() for p in paths)
            )[:5],
        )
    _COVERAGE_CACHE = cache
    return tuple(cache.values())


def reset_cache() -> None:
    """Invalidate the cached scan results (for in-process re-runs)."""
    global _COVERAGE_CACHE
    _COVERAGE_CACHE = None


def coverage_summary() -> dict[str, int]:
    """Return headline counts across the op surface."""
    rows = collect_op_test_coverage()
    return {
        "total_ops": len(rows),
        "zero_refs": sum(1 for r in rows if r.total_refs == 0),
        "thinly_tested": sum(1 for r in rows if r.is_thinly_tested),
        "well_tested": sum(1 for r in rows if r.total_refs >= 10),
        "with_negative_tests": sum(1 for r in rows if r.negative_refs > 0),
        "total_python_refs": sum(r.python_refs for r in rows),
        "total_lit_refs": sum(r.lit_refs for r in rows),
    }


def thinly_tested_ops() -> tuple[OpTestCoverage, ...]:
    """Ops with ≤1 test reference.  These are the highest-priority
    targets for follow-up test work."""
    return tuple(
        r for r in collect_op_test_coverage() if r.is_thinly_tested
    )


def top_tested_ops(n: int = 20) -> tuple[OpTestCoverage, ...]:
    """Top-N ops by total reference count."""
    rows = sorted(
        collect_op_test_coverage(),
        key=lambda r: (-r.total_refs, r.op_name),
    )
    return tuple(rows[:n])


# ─────────────────────────────────────────────────────────────────────────
# Dashboard render
# ─────────────────────────────────────────────────────────────────────────


#: Stable CSV column order for the test-coverage-by-op dashboard.
TEST_COVERAGE_CSV_COLUMNS: tuple[str, ...] = (
    "op", "python_refs", "lit_refs", "negative_refs", "total_refs",
    "is_thinly_tested", "dtype_variants",
)


def render_csv() -> str:
    """Render the canonical machine-readable test-coverage-by-op table.

    One row per op, sorted by op name.  Drift-gated artifact; the
    Markdown is the human companion.
    """
    import csv as _csv
    import io as _io

    rows = sorted(collect_op_test_coverage(), key=lambda r: r.op_name)
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(TEST_COVERAGE_CSV_COLUMNS)
    for r in rows:
        writer.writerow([
            r.op_name, r.python_refs, r.lit_refs, r.negative_refs,
            r.total_refs, "1" if r.is_thinly_tested else "0",
            " ".join(r.dtype_variants),
        ])
    return buf.getvalue()


def render_dashboard() -> str:
    """Render the test-coverage-by-op dashboard as Markdown."""
    rows = collect_op_test_coverage()
    summary = coverage_summary()
    thin = thinly_tested_ops()
    top = top_tested_ops()

    lines: list[str] = []
    lines.append("# Test Coverage by Op Family")
    lines.append("")
    lines.append(
        "Generated from "
        "`python/tessera/compiler/test_coverage_audit.py`.  "
        "Don't edit by hand — regenerate via "
        "`python -m tessera.compiler.generated_docs --write test_coverage`.  "
        "Drift gated by `tests/unit/test_generated_docs_registry.py` "
        "and `tests/unit/test_test_coverage_audit.py`."
    )
    lines.append("")
    lines.append(
        "**Honest scope note:** this audit measures *reference counts*, "
        "not numerical coverage quality.  A single test that exercises "
        "an op across 5 shapes × 3 dtypes counts as one reference but "
        "covers more ground than 5 happy-path tests.  Use the thin-"
        "coverage list as a starting point for triage, not a hard "
        "verdict."
    )
    lines.append("")

    # ── Headline ──
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- **{summary['total_ops']}** ops in "
        f"`primitive_coverage` registry."
    )
    lines.append(
        f"- **{summary['total_python_refs']}** total Python-test "
        f"references, **{summary['total_lit_refs']}** total lit-fixture "
        f"references."
    )
    lines.append(
        f"- **{summary['zero_refs']}** ops have **zero** references "
        f"in either test surface."
    )
    lines.append(
        f"- **{summary['thinly_tested']}** ops have ≤1 reference "
        f"(\"thinly tested\")."
    )
    lines.append(
        f"- **{summary['well_tested']}** ops have ≥10 references "
        f"(\"well tested\")."
    )
    lines.append(
        f"- **{summary['with_negative_tests']}** ops have at least one "
        f"associated `pytest.raises` negative test."
    )
    lines.append("")

    # ── Top tested ops ──
    lines.append("## Top 20 most-tested ops")
    lines.append("")
    lines.append("| Op | py refs | lit refs | total | neg | dtypes |")
    lines.append("|----|--------:|---------:|------:|----:|--------|")
    for r in top:
        dts = ", ".join(f"`{d}`" for d in r.dtype_variants[:4])
        if len(r.dtype_variants) > 4:
            dts += " …"
        lines.append(
            f"| `{r.op_name}` | {r.python_refs:>4} | {r.lit_refs:>4} "
            f"| {r.total_refs:>4} | {r.negative_refs:>3} | {dts} |"
        )
    lines.append("")

    # ── Thinly tested (the actionable section) ──
    lines.append("## Thinly-tested ops (≤1 reference)")
    lines.append("")
    lines.append(
        f"These **{len(thin)}** ops have at most one test reference "
        f"across the whole test surface.  Many will be legitimate — "
        f"variant aliases, structural ops, or category rollups that "
        f"inherit coverage from a parent family — but each one is a "
        f"candidate for explicit per-op test coverage."
    )
    lines.append("")
    lines.append("| Op | py refs | lit refs | total |")
    lines.append("|----|--------:|---------:|------:|")
    # Sort by name for stable rendering; show only the first 60 to
    # keep the dashboard compact (full list is one Python call away).
    for r in sorted(thin, key=lambda x: x.op_name)[:60]:
        lines.append(
            f"| `{r.op_name}` | {r.python_refs:>4} | "
            f"{r.lit_refs:>4} | {r.total_refs:>4} |"
        )
    if len(thin) > 60:
        lines.append("")
        lines.append(
            f"_({len(thin) - 60} additional thinly-tested ops omitted; "
            f"see `collect_op_test_coverage()` for the full list.)_"
        )
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_dashboard(path: Path | None = None) -> Path:
    if path is None:
        from . import generated_docs as gd

        gd.write(gd.get("test_coverage"))
        return gd.get("test_coverage").md_path
    target = path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_dashboard())
    return target


__all__ = [
    "OpTestCoverage",
    "collect_op_test_coverage",
    "coverage_summary",
    "reset_cache",
    "render_dashboard",
    "thinly_tested_ops",
    "top_tested_ops",
    "write_dashboard",
]
