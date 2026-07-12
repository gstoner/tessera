"""Single source of truth for every generated audit dashboard.

Before this module the regeneration + drift-gate logic for the generated
docs under ``docs/audit/`` was scattered across **two** shell/Python
entry points (``scripts/check_generated_docs.sh`` and
``scripts/release_gate.py``) plus a dozen per-generator unit tests, each
with its own CLI flag convention (``--render`` / ``--write`` /
default-write / ``--surface=`` / ``--target=``).  That fragmentation is
exactly what let ``runtime_abi.md`` silently drift and red CI.

This module replaces all of that with one registry.  Each
:class:`GeneratedDoc` declares:

  * where the doc lives on disk (Markdown, and optionally a canonical
    CSV companion),
  * the **side-effect-free** render callables that produce its text,
  * whether it is drift-gated, and which artifact is the canonical
    (byte-compared) one.

**Canonical-artifact rule:** when a doc has a CSV companion, the CSV is
the canonical machine-readable artifact and the *only* thing the drift
gate byte-compares; the Markdown is regenerated beside it but not
byte-gated (cosmetic Markdown churn never reds CI).  Markdown-only docs
are byte-gated on the Markdown until they grow a CSV.

One CLI drives the fleet::

    python -m tessera.compiler.generated_docs --list
    python -m tessera.compiler.generated_docs --check [name ...]   # CI / pre-commit gate
    python -m tessera.compiler.generated_docs --write [name ...]   # sprint regen

``scripts/check_generated_docs.sh`` and ``scripts/release_gate.py`` both
delegate here, so adding/retiring a dashboard means editing this one
registry.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GEN = _REPO_ROOT / "docs" / "audit" / "generated"
_AUDIT = _REPO_ROOT / "docs" / "audit"


# ─────────────────────────────────────────────────────────────────────────
# Lazy render adapters.
#
# Each adapter imports its generator inside the call so importing this
# registry stays cheap (``--list`` / ``--check name`` shouldn't drag in
# the 190 KB primitive_coverage module unless that doc is touched) and so
# we never create import cycles with ``audit.py`` (which imports several
# of these generators itself).
# ─────────────────────────────────────────────────────────────────────────


def _r_contract_consumers() -> str:
    from . import contract_consumers
    return contract_consumers.render_markdown()


def _r_contract_consumers_csv() -> str:
    from . import contract_consumers
    return contract_consumers.render_csv()


def _r_support_table() -> str:
    from . import audit
    return audit.render_markdown()


def _r_support_table_csv() -> str:
    from . import audit
    return audit.render_csv()


def _r_compiler_progress() -> str:
    from . import compiler_progress
    return compiler_progress.render_markdown()


def _r_compiler_progress_csv() -> str:
    from . import compiler_progress
    return compiler_progress.render_csv()


def _r_single_gpu_closeout() -> str:
    from . import single_gpu_closeout
    return single_gpu_closeout.render_markdown()


def _r_single_gpu_closeout_csv() -> str:
    from . import single_gpu_closeout
    return single_gpu_closeout.render_csv()


def _r_conformance() -> str:
    from . import conformance_matrix
    return conformance_matrix.render_markdown()


def _r_conformance_csv() -> str:
    from . import conformance_matrix
    return conformance_matrix.render_csv()


def _r_e2e() -> str:
    from . import e2e_coverage
    return e2e_coverage.render_markdown()


def _r_s_series() -> str:
    from . import s_series_status
    return s_series_status.render_markdown()


def _r_accel_proof() -> str:
    from . import accelerator_proof
    return accelerator_proof.render_markdown()


def _r_runtime_abi_md() -> str:
    from . import runtime_abi_audit
    return runtime_abi_audit.render_dashboard()


def _r_runtime_abi_csv() -> str:
    from . import runtime_abi_audit
    return runtime_abi_audit.render_csv()


def _r_exec_matrix() -> str:
    from . import execution_matrix
    return execution_matrix.render_dashboard()


def _r_exec_matrix_csv() -> str:
    from . import execution_matrix
    return execution_matrix.render_csv()


def _r_verifier_md() -> str:
    from . import verifier_coverage
    return verifier_coverage.render_dashboard()


def _r_verifier_csv() -> str:
    from . import verifier_coverage
    return verifier_coverage.render_csv()


def _r_apple_target_map() -> str:
    from . import apple_target_map
    return apple_target_map.render_markdown()


def _r_apple_target_map_csv() -> str:
    from . import apple_target_map
    return apple_target_map.render_csv()


def _r_gpu_target_map_csv(target: str) -> Callable[[], str]:
    def render() -> str:
        from . import gpu_target_map
        return gpu_target_map.render_csv(target)
    return render


def _r_gpu_target_map(target: str) -> Callable[[], str]:
    def _render() -> str:
        from . import gpu_target_map
        return gpu_target_map.render_markdown(target)
    return _render


def _r_test_coverage_md() -> str:
    """Merged test-coverage doc: the per-op coverage table + the
    thinly-tested classification triage (formerly two docs)."""
    from . import coverage_classification, test_coverage_audit
    by_op = test_coverage_audit.render_dashboard().rstrip()
    classification = coverage_classification.render_classification_dashboard()
    # Demote the classification doc's leading H1 so headings stay nested
    # under one document.
    cls_lines = classification.splitlines()
    if cls_lines and cls_lines[0].startswith("# "):
        cls_lines[0] = "## " + cls_lines[0][2:]
    return by_op + "\n\n---\n\n" + "\n".join(cls_lines).rstrip() + "\n"


def _r_test_coverage_csv() -> str:
    """Merged machine-readable test-coverage table: per-op reference
    counts + the classification bucket/reason in one CSV."""
    import csv as _csv
    import io as _io

    from . import coverage_classification, test_coverage_audit

    cols = (
        "op", "python_refs", "lit_refs", "negative_refs", "total_refs",
        "is_thinly_tested", "dtype_variants", "bucket", "reason",
    )
    rows = sorted(
        test_coverage_audit.collect_op_test_coverage(), key=lambda r: r.op_name
    )
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for r in rows:
        cls = coverage_classification.classify_op(r.op_name, r)
        writer.writerow([
            r.op_name, r.python_refs, r.lit_refs, r.negative_refs,
            r.total_refs, "1" if r.is_thinly_tested else "0",
            " ".join(r.dtype_variants), cls.bucket, cls.reason,
        ])
    return buf.getvalue()


def _r_tsol() -> str:
    from . import tsol_coverage
    return tsol_coverage.render_dashboard()


def _r_tsol_csv() -> str:
    from . import tsol_coverage
    return tsol_coverage.render_csv()


def _r_effect_lattice() -> str:
    from . import effect_audit
    return effect_audit.render_dashboard()


def _r_effect_lattice_csv() -> str:
    from . import effect_audit
    return effect_audit.render_csv()


def _r_docs_freshness() -> str:
    from . import docs_manifest
    return docs_manifest.render_dashboard()


def _r_autodiff_ledger() -> str:
    from . import autodiff_ledger
    return autodiff_ledger.render_markdown()


def _r_autodiff_ledger_csv() -> str:
    from . import autodiff_ledger
    return autodiff_ledger.render_csv()


def _r_manifest_reconciliation() -> str:
    from . import manifest_runtime_reconciliation
    return manifest_runtime_reconciliation.render_markdown()


#: The five SurfaceEntry-based surfaces, in display order.
_SURFACES: tuple[str, ...] = ("examples", "benchmarks", "research", "tools", "tests")


def _surface_modules() -> dict[str, Any]:
    import importlib
    return {
        s: importlib.import_module(f"tessera.compiler.{s}_manifest")
        for s in _SURFACES
    }


def _r_surface_status_csv() -> str:
    """Combined machine-readable repo-surface status — one row per
    manifest entry across all five surfaces, with a ``surface`` column."""
    import csv as _csv
    import io as _io

    cols = (
        "surface", "directory", "entry_point", "status", "command",
        "extras_required", "reason", "notes",
    )
    mods = _surface_modules()
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for surface in _SURFACES:
        entries = sorted(mods[surface].all_entries(), key=lambda e: e.directory)
        for e in entries:
            writer.writerow([
                surface, e.directory, e.entry_point, e.status,
                e.command or "", " ".join(e.extras_required), e.reason, e.notes,
            ])
    return buf.getvalue()


def _r_surface_status_md() -> str:
    """Combined human-readable repo-surface status, consolidating the
    five former ``*_status.md`` docs + operator-benchmark coverage."""
    mods = _surface_modules()
    lines: list[str] = [
        "# Repo Surface Status (generated)",
        "",
        "Consolidated status of the repo's audited surfaces — examples / "
        "benchmarks / research / tools / tests (formerly five separate "
        "`*_status.md` docs) plus operator-benchmark coverage. The canonical "
        "machine-readable artifact is `surface_status.csv` in this directory. "
        "Regenerate via `scripts/check_generated_docs.sh --write`.",
        "",
        "## Aggregate",
        "",
        "| Surface | Entries | Status breakdown |",
        "|---|--:|---|",
    ]
    for surface in _SURFACES:
        entries = mods[surface].all_entries()
        counts = mods[surface].status_counts()
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()) if v)
        lines.append(f"| {surface} | {len(entries)} | {breakdown or '—'} |")
    lines.append("")
    for surface in _SURFACES:
        entries = sorted(mods[surface].all_entries(), key=lambda e: e.directory)
        lines.append(f"## {surface}")
        lines.append("")
        lines.append("| Directory | Status | Entry point | Reason |")
        lines.append("|---|---|---|---|")
        for e in entries:
            reason = e.reason.replace("|", "\\|") if e.reason else ""
            lines.append(
                f"| `{e.directory}` | {e.status} | `{e.entry_point}` | {reason} |"
            )
        lines.append("")
    # Operator-benchmark coverage folded in as an appendix section.
    from . import operator_benchmarks_coverage as _obc
    obc_lines = _obc.render_markdown().splitlines()
    # Drop the appendix's AUTO-GENERATED banner (HTML comments + blanks)
    # so only its content lands in the combined doc.
    while obc_lines and (
        obc_lines[0].lstrip().startswith("<!--") or not obc_lines[0].strip()
    ):
        obc_lines.pop(0)
    # Demote the appendix's leading H1 to an H2 so headings stay nested.
    if obc_lines and obc_lines[0].startswith("# "):
        obc_lines[0] = "## " + obc_lines[0][2:]
    lines.extend(obc_lines)
    return "\n".join(lines).rstrip() + "\n"


# ─────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GeneratedDoc:
    """One generated audit dashboard.

    Fields
    ------
    name
        Registry key + CLI selector (the stem of the on-disk file).
    group
        Coarse grouping for ``--list`` readability.
    md_path
        Human-readable Markdown output path.
    render_md
        Side-effect-free callable returning the full Markdown text.
    csv_path / render_csv
        Optional canonical machine-readable CSV.  When present, the CSV
        is the byte-gated artifact and the Markdown is a non-gated
        companion.
    gated
        False for inherently non-deterministic docs (e.g. date-stamped
        freshness) — they are regenerated by ``--write`` but skipped by
        ``--check``.
    also_gate_md
        When the canonical artifact is a CSV, the Markdown is normally not
        byte-gated (cosmetic churn).  Set this True for a doc whose Markdown
        carries a *semantic* field absent from the CSV (e.g. the toolchain pin
        string in ``op_target_conformance.md``) so the drift gate byte-compares
        the Markdown too — otherwise a CSV-invisible change (a pin bump) sails
        past ``--check`` and only a separate per-doc test catches it.
    """

    name: str
    group: str
    md_path: Path
    render_md: Callable[[], str]
    csv_path: Optional[Path] = None
    render_csv: Optional[Callable[[], str]] = None
    gated: bool = True
    also_gate_md: bool = False

    @property
    def canonical_path(self) -> Path:
        """The artifact the drift gate byte-compares."""
        return self.csv_path if self.csv_path is not None else self.md_path

    def render_canonical(self) -> str:
        if self.render_csv is not None:
            return self.render_csv()
        return self.render_md()


REGISTRY: tuple[GeneratedDoc, ...] = (
    # ── All-up compiler progress rollup ──
    GeneratedDoc(
        "compiler_progress", "compiler_progress",
        _GEN / "compiler_progress.md", _r_compiler_progress,
        csv_path=_GEN / "compiler_progress.csv",
        render_csv=_r_compiler_progress_csv,
    ),
    # ── Op / primitive coverage ──
    GeneratedDoc(
        "support_table", "op_coverage", _GEN / "support_table.md", _r_support_table,
        csv_path=_GEN / "support_table.csv", render_csv=_r_support_table_csv,
    ),
    GeneratedDoc(
        "single_gpu_closeout", "op_coverage",
        _GEN / "single_gpu_closeout.md", _r_single_gpu_closeout,
        csv_path=_GEN / "single_gpu_closeout.csv",
        render_csv=_r_single_gpu_closeout_csv,
    ),
    # ── Contract-pass plan meta-gap tracker (Phase 0) ──
    GeneratedDoc(
        "contract_consumers", "op_coverage",
        _GEN / "contract_consumers.md", _r_contract_consumers,
        csv_path=_GEN / "contract_consumers.csv",
        render_csv=_r_contract_consumers_csv,
    ),
    GeneratedDoc(
        "op_target_conformance", "op_coverage",
        _AUDIT / "op_target_conformance.md", _r_conformance,
        csv_path=_AUDIT / "op_target_conformance.csv", render_csv=_r_conformance_csv,
        # The MD carries the toolchain-pin string (e.g. "CUDA Toolkit 13.3") that
        # the CSV lacks — gate the MD too so a pin bump can't slip past --check
        # (it previously only reddened the separate test_op_target_conformance).
        also_gate_md=True,
    ),
    GeneratedDoc(
        "e2e_op_coverage", "op_coverage", _GEN / "e2e_op_coverage.md", _r_e2e,
    ),
    GeneratedDoc(
        "s_series_status", "op_coverage", _GEN / "s_series_status.md", _r_s_series,
    ),
    GeneratedDoc(
        "s_series_accelerator_proof", "op_coverage",
        _GEN / "s_series_accelerator_proof.md", _r_accel_proof,
    ),
    # ── Compiler-autodiff connection ledger (AUTODIFF_UNIFICATION_PLAN §3) ──
    GeneratedDoc(
        "autodiff_connection_ledger", "op_coverage",
        _GEN / "autodiff_connection_ledger.md", _r_autodiff_ledger,
        csv_path=_GEN / "autodiff_connection_ledger.csv",
        render_csv=_r_autodiff_ledger_csv,
    ),
    # ── Manifest-vs-runtime reconciliation (does the manifest lag the runtime?) ──
    GeneratedDoc(
        "manifest_runtime_reconciliation", "op_coverage",
        _GEN / "manifest_runtime_reconciliation.md", _r_manifest_reconciliation,
    ),
    # ── Runtime / ABI ──
    GeneratedDoc(
        "runtime_abi", "runtime", _GEN / "runtime_abi.md", _r_runtime_abi_md,
        csv_path=_GEN / "runtime_abi.csv", render_csv=_r_runtime_abi_csv,
    ),
    GeneratedDoc(
        "runtime_execution_matrix", "runtime",
        _GEN / "runtime_execution_matrix.md", _r_exec_matrix,
        csv_path=_GEN / "runtime_execution_matrix.csv", render_csv=_r_exec_matrix_csv,
    ),
    # ── Verifier ──
    GeneratedDoc(
        "verifier_coverage", "verifier", _GEN / "verifier_coverage.md", _r_verifier_md,
        csv_path=_GEN / "verifier_coverage.csv", render_csv=_r_verifier_csv,
    ),
    # ── Target maps (CSV-canonical 2026-06-11) ──
    GeneratedDoc(
        "apple_target_map", "target_map", _GEN / "apple_target_map.md",
        _r_apple_target_map,
        csv_path=_GEN / "apple_target_map.csv", render_csv=_r_apple_target_map_csv,
    ),
    GeneratedDoc(
        "nvidia_sm90_target_map", "target_map", _GEN / "nvidia_sm90_target_map.md",
        _r_gpu_target_map("nvidia_sm90"),
        csv_path=_GEN / "nvidia_sm90_target_map.csv",
        render_csv=_r_gpu_target_map_csv("nvidia_sm90"),
    ),
    GeneratedDoc(
        "rocm_target_map", "target_map", _GEN / "rocm_target_map.md",
        _r_gpu_target_map("rocm"),
        csv_path=_GEN / "rocm_target_map.csv",
        render_csv=_r_gpu_target_map_csv("rocm"),
    ),
    # ── Test coverage (per-op counts + classification triage, merged) ──
    GeneratedDoc(
        "test_coverage", "test_coverage", _GEN / "test_coverage.md",
        _r_test_coverage_md,
        csv_path=_GEN / "test_coverage.csv", render_csv=_r_test_coverage_csv,
    ),
    # ── Specialized ──
    GeneratedDoc(
        "tsol_coverage", "specialized", _GEN / "tsol_coverage.md", _r_tsol,
        csv_path=_GEN / "tsol_coverage.csv", render_csv=_r_tsol_csv,
    ),
    GeneratedDoc(
        "effect_lattice_audit", "specialized", _GEN / "effect_lattice_audit.md",
        _r_effect_lattice,
        csv_path=_GEN / "effect_lattice_audit.csv", render_csv=_r_effect_lattice_csv,
    ),
    GeneratedDoc(
        # Date-stamped → regenerated but not byte-gated.
        "docs_freshness", "specialized", _GEN / "docs_freshness.md",
        _r_docs_freshness, gated=False,
    ),
    # ── Repo surface status (examples / benchmarks / research / tools /
    #    tests + operator-benchmark coverage), consolidated into one doc ──
    GeneratedDoc(
        "surface_status", "surface", _GEN / "surface_status.md",
        _r_surface_status_md,
        csv_path=_GEN / "surface_status.csv", render_csv=_r_surface_status_csv,
    ),
)


_BY_NAME = {d.name: d for d in REGISTRY}


def get(name: str) -> GeneratedDoc:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown generated doc {name!r}; known: {sorted(_BY_NAME)}"
        ) from None


def _select(names: Sequence[str]) -> tuple[GeneratedDoc, ...]:
    if not names:
        return REGISTRY
    return tuple(get(n) for n in names)


# ─────────────────────────────────────────────────────────────────────────
# Operations
# ─────────────────────────────────────────────────────────────────────────


def write(doc: GeneratedDoc) -> list[Path]:
    """(Re)generate ``doc`` on disk.  Writes the CSV (if any) and the
    Markdown.  Returns the paths written."""
    written: list[Path] = []
    doc.md_path.parent.mkdir(parents=True, exist_ok=True)
    if doc.csv_path is not None and doc.render_csv is not None:
        doc.csv_path.parent.mkdir(parents=True, exist_ok=True)
        doc.csv_path.write_text(doc.render_csv())
        written.append(doc.csv_path)
    doc.md_path.write_text(doc.render_md())
    written.append(doc.md_path)
    return written


def check(doc: GeneratedDoc) -> Optional[str]:
    """Return ``None`` if ``doc`` is in sync, else a human-readable drift
    message.  Non-gated docs always return ``None``."""
    if not doc.gated:
        return None
    target = doc.canonical_path
    if not target.exists():
        return (
            f"{doc.name}: {target.relative_to(_REPO_ROOT)} does not exist — "
            f"run `python -m tessera.compiler.generated_docs --write {doc.name}`"
        )
    # Build the list of (artifact path, live text) pairs to byte-compare. The
    # canonical artifact (CSV when present) is always gated; a doc may *also* gate
    # its Markdown when the MD carries a semantic field the CSV lacks (see
    # `also_gate_md`).
    pairs = [(target, doc.render_canonical())]
    if doc.also_gate_md and doc.md_path != target:
        pairs.append((doc.md_path, doc.render_md()))
    for art, live in pairs:
        if not art.exists():
            return (
                f"{doc.name}: {art.relative_to(_REPO_ROOT)} does not exist — "
                f"run `python -m tessera.compiler.generated_docs --write {doc.name}`"
            )
        on_disk = art.read_text()
        if live == on_disk:
            continue
        live_lines, disk_lines = live.splitlines(), on_disk.splitlines()
        idx = next(
            (i for i, (a, b) in enumerate(zip(live_lines, disk_lines)) if a != b),
            min(len(live_lines), len(disk_lines)),
        )
        near = disk_lines[idx] if idx < len(disk_lines) else "<EOF>"
        return (
            f"{doc.name}: drift in {art.relative_to(_REPO_ROOT)} at line {idx + 1} "
            f"(on-disk: {near!r}) — regenerate with "
            f"`python -m tessera.compiler.generated_docs --write {doc.name}`"
        )
    return None


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def _cmd_list() -> int:
    width = max(len(d.name) for d in REGISTRY)
    group = None
    for d in sorted(REGISTRY, key=lambda x: (x.group, x.name)):
        if d.group != group:
            group = d.group
            print(f"\n[{group}]")
        fmt = "csv+md" if d.csv_path else "md"
        gate = "" if d.gated else "  (not gated)"
        print(f"  {d.name:<{width}}  {fmt:<6}{gate}")
    print(f"\n{len(REGISTRY)} generated docs.")
    return 0


def _cmd_write(names: Sequence[str]) -> int:
    for d in _select(names):
        paths = write(d)
        rel = ", ".join(str(p.relative_to(_REPO_ROOT)) for p in paths)
        print(f"wrote {d.name}: {rel}", file=sys.stderr)
    return 0


def _cmd_check(names: Sequence[str]) -> int:
    failures = [msg for d in _select(names) if (msg := check(d))]
    if not failures:
        print(f"ok: {len(_select(names))} generated doc(s) in sync")
        return 0
    for msg in failures:
        print(f"DRIFT: {msg}", file=sys.stderr)
    print(
        "\nRegenerate everything with "
        "`scripts/check_generated_docs.sh --write` "
        "(or `python -m tessera.compiler.generated_docs --write`) and "
        "commit the source + regenerated docs.",
        file=sys.stderr,
    )
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tessera.compiler.generated_docs",
        description="Unified registry, drift gate, and regenerator for "
                    "every generated audit dashboard.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="list registered docs")
    g.add_argument("--check", action="store_true",
                   help="drift gate: fail if any selected doc is stale")
    g.add_argument("--write", action="store_true",
                   help="regenerate selected docs (default: all)")
    p.add_argument("names", nargs="*", help="doc name(s); default = all")
    args = p.parse_args(argv)
    if args.list:
        return _cmd_list()
    if args.write:
        return _cmd_write(args.names)
    return _cmd_check(args.names)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "GeneratedDoc",
    "REGISTRY",
    "get",
    "write",
    "check",
    "main",
]
