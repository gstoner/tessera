"""Arch-2 (2026-05-22) — MLIR verifier coverage dashboard.

This module is the Python analogue to ``primitive_coverage.py`` but
for the MLIR-side verifier surface.  It enumerates every ODS op
declared across Tessera's ``.td`` files and classifies each by
verifier status:

  * ``real``         — ``hasVerifier = 1;`` + a non-trivial
                       ``OpName::verify()`` body in the matching .cpp.
  * ``trivial_stub`` — ``hasVerifier = 1;`` declared OR a
                       ``OpName::verify() { return success(); }``
                       stub.  Pre-V8 the Queue dialect had three of
                       these; the V8 sprint flipped them to ``real``.
  * ``absent``       — Neither hasVerifier nor a verify() impl.  Most
                       op classes are in this bucket today — fine
                       when the op's TD constraints are sufficient,
                       worth a sprint when behavior depends on
                       cross-operand structure.

Designed to be parser-driven rather than a hand-maintained list:
when a new ODS op lands, the dashboard picks it up automatically.
The drift gate is the inverse — when an op leaves the ``real``
bucket (e.g., someone deletes a ``verify()`` body), structural tests
catch it before it ships.

Two artifacts are emitted, both under ``docs/audit/generated/``:

  * ``verifier_coverage.csv`` — the **canonical, machine-readable**
    surface and the only thing the drift gate compares.  One row per
    op, stable column order, stable sort, no alignment padding.  This
    is what CI checks because a CSV diff is trivial and whitespace
    never drifts.
  * ``verifier_coverage.md`` — a **human-readable** summary rendered
    from the same data.  It is regenerated alongside the CSV but is
    *not* byte-compared by the drift gate (only its presence and a
    couple of canonical heading phrases are checked), so cosmetic
    formatting never reds CI on its own.

Regenerate both via::

    python -m tessera.compiler.audit verifier_coverage --write

and the same command with ``--check`` is the drift gate (it compares
the CSV).  Both are wired into ``scripts/check_generated_docs.sh`` so
``scripts/check_generated_docs.sh --write`` refreshes them at the end
of a sprint.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"

#: Canonical output locations.  The CSV is the drift-gated artifact;
#: the MD is the human-readable companion (not byte-gated).
CSV_PATH = _REPO_ROOT / "docs" / "audit" / "generated" / "verifier_coverage.csv"
MD_PATH = _REPO_ROOT / "docs" / "audit" / "generated" / "verifier_coverage.md"

#: Stable CSV column order.  Append-only — never reorder or rename, as
#: downstream tooling parses by header name.
CSV_COLUMNS: tuple[str, ...] = (
    "td_file",
    "op_class",
    "has_verifier_declared",
    "verify_impl_present",
    "impl_status",
)


# ─────────────────────────────────────────────────────────────────────────
# Locations of Tessera ODS sources + the .cpp files holding verify() bodies
# ─────────────────────────────────────────────────────────────────────────

# Maps a .td path → list of .cpp files where its ops' verify() bodies live.
# Manually maintained — when a new dialect lands, add its TD + impl files
# here.  The drift gate verifies the list stays consistent with the file
# system (no orphan TD files referenced).
_DIALECT_FILE_MAP: tuple[tuple[Path, tuple[Path, ...]], ...] = (
    (
        _SRC_ROOT / "compiler/ir/TesseraOps.td",
        (_SRC_ROOT / "compiler/ir/TesseraOps.cpp",),
    ),
    (
        _SRC_ROOT / "compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td",
        (
            _SRC_ROOT / "compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp",
            _SRC_ROOT / "compiler/tile_opt_fa4/lib/Dialect/Attn/AttnVerifiers.cpp",
        ),
    ),
    (
        _SRC_ROOT / "compiler/tile_opt_fa4/include/tessera/Dialect/Queue/Queue.td",
        (
            _SRC_ROOT / "compiler/tile_opt_fa4/lib/Dialect/Queue/QueueOps.cpp",
            _SRC_ROOT / "compiler/tile_opt_fa4/lib/Dialect/Queue/QueueVerifiers.cpp",
        ),
    ),
)


# Regex for `def OpName : Op<...>` or `def OpName : Tessera_XYZOp<...>`
# in TD files.  We accept op-class-template variants like Tessera_CollectiveOp
# and Tessera_UnaryPureOp by matching any `: <Identifier>` after `def`.
_OP_DEF_PATTERN = re.compile(
    r'^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)',
    re.MULTILINE,
)

# `class Name<...> : Op<...> { ... }` — op-template base classes that may
# themselves declare `hasVerifier = 1` for one-line `def X : Name<...>;` ops.
_CLASS_DEF_PATTERN = re.compile(
    r'^class\s+([A-Za-z_][A-Za-z0-9_]*)',
    re.MULTILINE,
)

# We classify an op def as an Op (subject to verification) vs a Type /
# Attr / Dialect / Enum definition by checking the second token after
# the colon.  All known Tessera op-template names contain "Op" — Op,
# Tessera_UnaryPureOp, Tessera_CollectiveOp, etc.  TypeDef/AttrDef/
# Enum templates do not.
def _looks_like_op_template(template: str) -> bool:
    return "Op" in template and "OpInterface" not in template


# Regex for `let hasVerifier = 1;` inside the op block.
_HAS_VERIFIER_RE = re.compile(r'\blet\s+hasVerifier\s*=\s*1\s*;')


# Regex for `LogicalResult OpName::verify()` (with optional whitespace
# and an optional ``mlir::`` prefix).  Captures the op name.
_VERIFY_IMPL_RE = re.compile(
    r'(?:mlir::)?LogicalResult\s+([A-Za-z_][A-Za-z0-9_]*)::verify\s*\(\s*\)',
)

# Regex for the trivial-stub body shape.  Matches both the free-function
# stub pattern (``LogicalResult verifyCreate(Operation*) { return success(); }``)
# and the OpClass::verify variant (``LogicalResult Foo::verify() { return success(); }``).
_TRIVIAL_STUB_RE = re.compile(
    r'LogicalResult\s+([A-Za-z_][A-Za-z0-9_]*)(?:::verify)?\s*\([^)]*\)\s*'
    r'(?:const\s*)?\{\s*return\s+success\(\)\s*;\s*\}',
)


# ─────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VerifierEntry:
    """One op's verifier status snapshot.

    Fields
    ------
    op_class
        The C++ class name (e.g., ``MatmulOp``, ``LayerNormOp``).
    td_file
        Relative path to the ODS .td file that declared the op.
    has_verifier_declared
        True iff the TD block contains ``let hasVerifier = 1;``.
    verify_impl_present
        True iff a matching ``OpName::verify()`` body exists in the
        impl .cpp files.
    impl_status
        One of ``"real"``, ``"trivial_stub"``, ``"absent"``,
        ``"no_verifier"``.  See module docstring for definitions.
    """

    op_class: str
    td_file: Path
    has_verifier_declared: bool
    verify_impl_present: bool
    impl_status: str


_VALID_STATUSES = frozenset({"real", "trivial_stub", "absent", "no_verifier"})


# ─────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────


def _extract_op_block(td_text: str, op_class: str) -> str:
    """Return the text of a `def OpName : ... { ... }` block.

    We approximate the block boundary by finding the opening brace
    after the def, then scanning forward through balanced braces.
    TD syntax is brace-balanced for op blocks in practice.

    The ``[^{{;]`` (note the ``;``) is load-bearing: a *one-line* def
    ``def X : Base<...>;`` has no block, so the scan must stop at its ``;``
    rather than greedily running into the **next** op's ``{ ... }`` block and
    mis-attributing that op's ``hasVerifier`` to the block-less def.
    """
    return _extract_block(td_text, op_class, "def")


def _extract_block(td_text: str, name: str, keyword: str) -> str:
    """Generic ``def``/``class`` block extractor (see ``_extract_op_block``).
    ``[<:]`` after the name handles both ``def X : Base<...> {`` and
    ``class X<args> : Op<...> {``; ``[^{{;]`` stops a block-less one-line
    ``def X : Base<...>;`` from bleeding into the next block."""
    pattern = re.compile(
        rf'{keyword}\s+{re.escape(name)}\s*[<:][^{{;]*\{{',
    )
    match = pattern.search(td_text)
    if not match:
        return ""
    start = match.end() - 1  # position of `{`
    depth = 0
    for i in range(start, len(td_text)):
        ch = td_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return td_text[start:i + 1]
    return ""


def _scan_td(td_path: Path) -> list[tuple[str, bool]]:
    """Return a list of ``(op_class, has_verifier_declared)`` for every
    ODS op in the given TD file.

    Filters out TypeDef / AttrDef / EnumAttr / Dialect / etc. by
    checking whether the template name "looks like an Op".  Type/Attr
    defs use ``TypeDef<...>``, ``AttrDef<...>``, ``EnumAttr<...>``
    which don't contain ``"Op"`` as a substring.
    """
    text = td_path.read_text()
    # Op-template base classes that declare `hasVerifier = 1` in their own
    # `class ... { ... }` block — one-line `def X : Base<...>;` ops inherit it
    # (e.g. Tessera_CliffordBinaryOp), so we must resolve through the base.
    base_has_verifier: dict[str, bool] = {}
    for m in _CLASS_DEF_PATTERN.finditer(text):
        cls = m.group(1)
        base_has_verifier[cls] = bool(
            _HAS_VERIFIER_RE.search(_extract_block(text, cls, "class")))
    ops: list[tuple[str, bool]] = []
    for match in _OP_DEF_PATTERN.finditer(text):
        op_name, template = match.group(1), match.group(2)
        if not _looks_like_op_template(template):
            continue
        block = _extract_op_block(text, op_name)
        has_verifier = (bool(_HAS_VERIFIER_RE.search(block))
                        or base_has_verifier.get(template, False))
        # Strip the "Tessera_" / "TPP_" / etc. prefix to get the C++
        # class name.  ODS uses the prefixed form in `def Tessera_FooOp`
        # but the C++ class is just `FooOp`.
        cpp_class = op_name
        for prefix in ("Tessera_", "TPP_", "Apple_"):
            if cpp_class.startswith(prefix):
                cpp_class = cpp_class[len(prefix):]
                break
        ops.append((cpp_class, has_verifier))
    return ops


def _scan_cpp_for_verify_impls(cpp_paths: tuple[Path, ...]) -> dict[str, str]:
    """Walk the listed .cpp files and return ``{op_class: impl_status}``
    for every ``OpName::verify()`` body found.

    ``impl_status`` is ``"trivial_stub"`` if the body matches the
    canonical ``return success();``-only shape; ``"real"`` otherwise.
    """
    impls: dict[str, str] = {}
    for path in cpp_paths:
        if not path.exists():
            continue
        text = path.read_text()
        # Collect all (op_class, body) pairs by walking ::verify decls.
        for match in _VERIFY_IMPL_RE.finditer(text):
            op_class = match.group(1)
            # Capture the body text (best-effort — scan forward for the
            # first opening brace, then match braces).
            body_start = text.find("{", match.end())
            if body_start == -1:
                continue
            depth = 0
            body_end = body_start
            for i in range(body_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        body_end = i + 1
                        break
            body = text[body_start:body_end]
            # Classify.  A trivial stub is exactly { return success(); }
            # (whitespace-tolerant).  Anything with more content is real.
            normalized = re.sub(r'\s+', ' ', body).strip()
            if normalized == "{ return success(); }":
                impls[op_class] = "trivial_stub"
            else:
                # If we previously classified as trivial_stub, upgrading
                # to real is fine (defensive against accidental dupes).
                impls.setdefault(op_class, "real")
                if impls[op_class] != "real":
                    impls[op_class] = "real"
    return impls


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────


def collect_verifier_coverage() -> tuple[VerifierEntry, ...]:
    """Walk every TD file in :data:`_DIALECT_FILE_MAP`, extract op
    definitions, and classify each by verifier status.

    Returns entries in TD-file-then-class-name order so the generated
    dashboard is stable across runs.
    """
    out: list[VerifierEntry] = []
    for td_path, cpp_paths in _DIALECT_FILE_MAP:
        if not td_path.exists():
            continue
        ops = _scan_td(td_path)
        impls = _scan_cpp_for_verify_impls(cpp_paths)
        for op_class, has_verifier in sorted(ops):
            impl_state = impls.get(op_class)
            if has_verifier and impl_state == "real":
                status = "real"
            elif has_verifier and impl_state == "trivial_stub":
                status = "trivial_stub"
            elif has_verifier and impl_state is None:
                status = "absent"
            elif not has_verifier and impl_state is not None:
                # Edge case: impl body exists without hasVerifier
                # declared.  ODS won't call it, so it's effectively
                # absent from the verifier surface.
                status = "absent"
            else:
                status = "no_verifier"
            out.append(VerifierEntry(
                op_class=op_class,
                td_file=td_path.relative_to(_REPO_ROOT),
                has_verifier_declared=has_verifier,
                verify_impl_present=impl_state is not None,
                impl_status=status,
            ))
    return tuple(out)


def coverage_summary() -> dict[str, int]:
    """Return ``{status: count}`` across the full op surface."""
    entries = collect_verifier_coverage()
    out: dict[str, int] = {status: 0 for status in _VALID_STATUSES}
    out["total"] = len(entries)
    for entry in entries:
        out[entry.impl_status] += 1
    return out


def render_csv() -> str:
    """Render the canonical, drift-gated CSV.

    One row per op, sorted by ``(td_file, op_class)`` so the output is
    byte-stable across runs and platforms.  This is the artifact CI
    compares — a CSV diff is trivial and there is no alignment padding
    to drift.
    """
    entries = collect_verifier_coverage()
    rows = sorted(entries, key=lambda e: (e.td_file.as_posix(), e.op_class))
    buf = io.StringIO()
    # Force '\n' line endings so the file is identical on every platform.
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for e in rows:
        writer.writerow([
            e.td_file.as_posix(),
            e.op_class,
            "1" if e.has_verifier_declared else "0",
            "1" if e.verify_impl_present else "0",
            e.impl_status,
        ])
    return buf.getvalue()


def render_dashboard() -> str:
    """Render the human-readable Markdown summary.

    This view is regenerated alongside the CSV but is **not** byte-gated
    — the drift gate compares :func:`render_csv` instead.  The headings
    here are pinned by the test so downstream docs can link to them.
    """
    entries = collect_verifier_coverage()
    summary = coverage_summary()
    lines: list[str] = []
    lines.append("# MLIR Verifier Coverage Dashboard")
    lines.append("")
    lines.append(
        "Human-readable view. The canonical machine-readable artifact is "
        "`verifier_coverage.csv` in this directory — that CSV is what the "
        "drift gate compares. Don't edit either by hand; run "
        "`python -m tessera.compiler.audit verifier_coverage --write` "
        "(or `scripts/check_generated_docs.sh --write`) to refresh both. "
        "Drift is gated by `tests/unit/test_verifier_coverage.py`."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count | Meaning |")
    lines.append("|--------|-------|---------|")
    lines.append(
        f"| `real` | {summary['real']} | "
        f"`hasVerifier = 1;` + substantive `verify()` body. |"
    )
    lines.append(
        f"| `trivial_stub` | {summary['trivial_stub']} | "
        f"`hasVerifier = 1;` + trivial `return success();` stub. |"
    )
    lines.append(
        f"| `absent` | {summary['absent']} | "
        f"`hasVerifier = 1;` but no `verify()` body (build error risk). |"
    )
    lines.append(
        f"| `no_verifier` | {summary['no_verifier']} | "
        f"No verifier declared.  TD constraints suffice — fine for many ops. |"
    )
    lines.append(f"| **Total** | {summary['total']} | |")
    lines.append("")

    # Group entries by TD file for readability.
    by_td: dict[Path, list[VerifierEntry]] = {}
    for entry in entries:
        by_td.setdefault(entry.td_file, []).append(entry)

    lines.append("## Per-dialect details")
    lines.append("")
    for td_file in sorted(by_td):
        lines.append(f"### `{td_file}`")
        lines.append("")
        lines.append("| Op | Status |")
        lines.append("|----|--------|")
        for entry in by_td[td_file]:
            lines.append(f"| `{entry.op_class}` | `{entry.impl_status}` |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_dashboard(csv_path: Path | None = None) -> tuple[Path, Path]:
    """Write both the canonical CSV and the human Markdown summary.

    Returns ``(csv_path, md_path)``.  The Markdown path is derived from
    the CSV path by swapping the suffix, so callers only override one.
    """
    target_csv = csv_path or CSV_PATH
    target_md = target_csv.with_suffix(".md")
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    target_csv.write_text(render_csv())
    target_md.write_text(render_dashboard())
    return target_csv, target_md


__all__ = [
    "VerifierEntry",
    "CSV_PATH",
    "MD_PATH",
    "CSV_COLUMNS",
    "collect_verifier_coverage",
    "coverage_summary",
    "render_csv",
    "render_dashboard",
    "write_dashboard",
]
