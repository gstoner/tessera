"""AD-LAW-1 — generated dashboard for the autodiff law sweep.

Renders ``docs/audit/generated/autodiff_law_audit.{md,csv}`` from
``tessera.autodiff.laws.run_law_sweep()``. Registered in
``generated_docs.py`` and byte-gated, so the committed dashboard cannot
drift from what the sweep actually reports (Decision #26).

The gated CSV carries **statuses only, never floats** — residual magnitudes
vary at the last ulp across BLAS builds, and a byte-compared artifact must
be identical on every host. Numeric residuals live in the law tests'
output, not here.

Design authority: ``docs/audit/compiler/AUTODIFF_NEXTGEN_PLAN.md`` §4/§7
(AD-LAW-1). Statuses:

- ``pass`` / ``fail`` — the law was evaluated on the op's registered spec.
- ``rule_error`` — a rule or spec raised; visible, not skipped.
- ``no_spec`` — both modes exist but no input spec is registered yet:
  the explicit unswept-debt bucket (claim integrity — a sweep that only
  reports what it checked would read as complete when it is not).
- ``jvp_only`` / ``vjp_only`` — the adjoint law needs both modes; these
  count the single-mode population.

CLI:  python -m tessera.compiler.law_audit          # print markdown
"""

from __future__ import annotations

import io
from collections import Counter


_HEADER = (
    "# Autodiff law audit — AD-LAW-1\n"
    "\n"
    "> **Generated** by `python/tessera/compiler/law_audit.py` from\n"
    "> `tessera.autodiff.laws.run_law_sweep()`. Do not hand-edit; regenerate\n"
    "> via `scripts/check_generated_docs.sh --write`. Design authority:\n"
    "> [`AUTODIFF_NEXTGEN_PLAN.md`](../compiler/AUTODIFF_NEXTGEN_PLAN.md) §4.\n"
    "\n"
    "Law 3 (adjoint, `⟨Jv,u⟩ = ⟨v,Jᵀu⟩`) is complete for the transpose\n"
    "relationship between a paired JVP/VJP; it cannot certify the derivative\n"
    "itself (a matched-wrong pair passes). Law 1 (chain vs finite\n"
    "differences) is the derivative-correctness complement. `no_spec` rows\n"
    "are real unswept debt, not exclusions.\n"
)


def _rows():
    from tessera.autodiff.laws import run_law_sweep

    return run_law_sweep()


def render_csv() -> str:
    rows = _rows()
    out = io.StringIO()
    out.write("registry,op,law,status,probes\n")
    for r in sorted(rows, key=lambda r: (r.registry, r.op, r.law)):
        out.write(f"{r.registry},{r.op},{r.law},{r.status},{r.probes}\n")
    return out.getvalue()


def render_markdown() -> str:
    rows = _rows()
    out = io.StringIO()
    out.write(_HEADER)

    for registry in ("tensor", "geometric"):
        sub = [r for r in rows if r.registry == registry]
        if not sub:
            continue
        counts = Counter((r.law, r.status) for r in sub)
        out.write(f"\n## `{registry}` registry\n\n")
        out.write("| Law | Status | Ops |\n|---|---|---:|\n")
        for (law, status), n in sorted(counts.items()):
            out.write(f"| {law} | {status} | {n} |\n")

        flagged = [r for r in sub if r.status in ("fail", "rule_error")]
        if flagged:
            out.write("\n**Flagged (findings — triage before trusting the "
                      "affected rules):**\n\n")
            out.write("| Op | Law | Status | Detail |\n|---|---|---|---|\n")
            for r in sorted(flagged, key=lambda r: (r.op, r.law)):
                out.write(f"| `{r.op}` | {r.law} | {r.status} | {r.detail} |\n")

    checked = sorted({r.op for r in rows
                      if r.registry == "tensor" and r.status in ("pass", "fail")})
    out.write("\n## Checked tensor ops\n\n")
    out.write(", ".join(f"`{op}`" for op in checked) + "\n")
    return out.getvalue()


if __name__ == "__main__":
    print(render_markdown())
