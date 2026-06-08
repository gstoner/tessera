#!/usr/bin/env python3
"""Fast structural-sync pre-flight — the full-suite-only gates that bit twice.

Two invariants that live only in the full pytest job (not in the per-commit
Python-Quality / drift gates), so they used to surface ~10 min into CI:

  1. **op-catalog → PYTHON_API_SPEC** — every `op_catalog.OP_SPECS` entry must
     appear (by public or graph name) in `docs/spec/PYTHON_API_SPEC.md`. A new
     primitive that isn't documented trips
     `test_python_api_spec_lists_current_runtime_op_catalog`.
  2. **generated-markdown registry** — every `docs/audit/generated/*.md` must be
     in `tessera.compiler.generated_docs.REGISTRY`. A derived report dropped into
     that dir (instead of `docs/audit/` root) trips
     `test_no_unregistered_generated_markdown`.

Both are pure structural reads (<1s, no GPU/build), so the pre-push hook can run
them before every push. Mirrors the test assertions; CI remains the backstop.

    python3 scripts/check_spec_sync.py        # exit 0 in sync, 1 + reason on drift
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def _check_op_catalog_in_spec() -> list[str]:
    from tessera.compiler.op_catalog import OP_SPECS

    spec = (ROOT / "docs" / "spec" / "PYTHON_API_SPEC.md").read_text(encoding="utf-8")
    return [
        name for name in sorted(OP_SPECS)
        if name not in spec and OP_SPECS[name].graph_name not in spec
    ]


def _check_generated_markdown_registered() -> list[str]:
    from tessera.compiler import generated_docs as gd

    gen_dir = ROOT / "docs" / "audit" / "generated"
    registered = {d.md_path.resolve() for d in gd.REGISTRY}
    on_disk = {p.resolve() for p in gen_dir.glob("*.md")}
    return sorted(p.name for p in (on_disk - registered))


def _check_docs_dated() -> list[str]:
    """Every canonical-tree doc carries a ``last_updated:`` marker or is in the
    docs-freshness allow-list (mirrors test_docs_freshness). The allow-list lives
    in the test, so import it from there to stay single-source."""
    try:
        from tessera.compiler.docs_manifest import undated_docs
    except Exception:                                    # noqa: BLE001
        return []                                        # CI still gates
    sys.path.insert(0, str(ROOT / "tests" / "unit"))
    try:
        from test_docs_freshness import _KNOWN_UNDATED_DOCS  # type: ignore
    except Exception:                                    # noqa: BLE001
        return []
    undated = {e.path for e in undated_docs()}
    return sorted(undated - set(_KNOWN_UNDATED_DOCS))


def main() -> int:
    problems: list[str] = []

    missing_ops = _check_op_catalog_in_spec()
    if missing_ops:
        problems.append(
            "op-catalog ops missing from docs/spec/PYTHON_API_SPEC.md:\n"
            f"    {missing_ops}\n"
            "  Add a row for each (public or graph name) — see the matmul/loss "
            "tables in that doc.")

    orphans = _check_generated_markdown_registered()
    if orphans:
        problems.append(
            "generated Markdown not in the generated_docs REGISTRY:\n"
            f"    {orphans}\n"
            "  Either register it in tessera.compiler.generated_docs.REGISTRY, "
            "or — if it is a derived *report*, not a drift-gated dashboard — move "
            "it out of docs/audit/generated/ to docs/audit/ root (Decision #26).")

    undated = _check_docs_dated()
    if undated:
        problems.append(
            "docs missing a last_updated marker (not in the freshness allow-list):\n"
            f"    {undated}\n"
            "  Add YAML frontmatter `last_updated: YYYY-MM-DD` to the doc, or "
            "extend _KNOWN_UNDATED_DOCS in tests/unit/test_docs_freshness.py.")

    if problems:
        print("[check_spec_sync] FAIL\n", file=sys.stderr)
        for p in problems:
            print("  - " + p + "\n", file=sys.stderr)
        return 1

    print("[check_spec_sync] ok: op-catalog/spec + generated-markdown registry + "
          "docs-freshness in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
