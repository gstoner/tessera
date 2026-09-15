"""Domain proof ladder — one generated status row per mathematical domain.

The domain audit (``docs/audit/domain/DOMAIN_AUDIT.md``) requires support to be
reported **separately** for reference math, derivative rules, native execution
and measured selection, and it routes every remaining boundary to an owner in
the integrated compiler plan. Prose cannot keep four independent proof columns
current across six domains and four backends; this dashboard derives them:

* **registry** rows from ``primitive_coverage.all_primitive_coverages`` whose
  name or model family matches the domain (Decision #24 audit truth);
* **derivative** rows from the AD connection ledger (registered adjoints,
  tangents and device-verified backward launches);
* **native execution** rows from the runtime execution matrix (executable
  native rows per target), never counts copied into prose (Decision #26);
* the **plan owners** the domain routes to, checked against the integrated
  plan's routing index so a retired ID surfaces as ``missing``.

Matching is by explicit name prefixes and keywords per domain, listed in
``DOMAINS`` so the mapping is reviewable; a primitive that matches no domain is
simply not a domain primitive. This is an inventory of *evidence*, not a claim
that any domain is complete: an empty native column is the finding.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[3]
_PLAN = _ROOT / "docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md"


@dataclass(frozen=True)
class Domain:
    key: str
    label: str
    owners: tuple[str, ...]          # integrated-plan IDs the domain routes to
    prefixes: tuple[str, ...]        # primitive / ledger family name prefixes
    keywords: tuple[str, ...]        # execution-matrix compiler_path / executor keywords


DOMAINS: tuple[Domain, ...] = (
    Domain("geometric_algebra", "Geometric algebra / Clifford",
           ("W6.4", "AD-HIGHER-1"),
           ("clifford", "geometric_product", "rotor", "multivector", "wedge", "grade_"),
           ("clifford", "rotor", "geometric")),
    Domain("energy_based_models", "Energy-based models",
           ("W4-PRODUCT-1", "AD-SOLVER-IFT-1", "AD-RESIDUAL-EVAL-1"),
           ("ebm_", "langevin", "energy_", "ais_"),
           ("ebm", "langevin", "energy")),
    Domain("attention_persistent_state", "Attention / persistent state",
           ("W5.2", "AD-RESIDUAL-EVAL-1", "W2.4a"),
           ("flash_attn", "attention", "gqa", "mla_", "kv_cache", "paged_", "attn_"),
           ("attn", "attention", "kv", "paged", "mla")),
    Domain("field_pde_spectral", "Matrix/field calculus, PDE and spectral",
           ("MSW-9", "TSOL-POLICY-PHYS-1", "TSOL-PHYS-TAIL-1", "PDE-STENCIL-FOUNDATION-1"),
           ("fft", "spectral", "stencil", "halo", "pde_", "laplacian", "poisson", "helmholtz"),
           ("fft", "spectral", "stencil", "halo", "pde")),
    Domain("game_theory_structured", "Game theory / structured contractions",
           ("TSOL-PHYS-TAIL-1",),
           ("butterfly", "coalition", "shapley", "banzhaf", "game_"),
           ("butterfly", "coalition", "shapley")),
    Domain("domain_sharding_distributed", "Domain sharding and distributed execution",
           ("DIST-NATIVE-1", "TSOL-SHARD-1"),
           ("shard", "reshard", "all_to_all", "all_reduce", "reduce_scatter", "all_gather", "collective"),
           ("shard", "collective", "all_to_all", "mesh")),
)


def _matches(name: str, prefixes: Iterable[str]) -> bool:
    low = name.lower()
    return any(p in low for p in prefixes)


def _routing_index(plan_text: str) -> dict[str, str]:
    """{ID: relationship} from the plan's routing index (owner/successor/archive)."""
    routes: dict[str, str] = {}
    for line in plan_text.splitlines():
        m = re.fullmatch(r"\| ([A-Z][A-Za-z0-9.-]*) \| \[[^\]]+\]\([^)]+\) \| (owner|successor|archive) \|", line)
        if m:
            routes[m.group(1)] = m.group(2)
    return routes


def _registry_rows(domain: Domain) -> list[tuple[str, str, str]]:
    from .primitive_coverage import all_primitive_coverages
    rows = []
    for name, entry in sorted(all_primitive_coverages().items()):
        families = " ".join(getattr(entry, "model_families", ()) or ())
        if _matches(name, domain.prefixes) or _matches(families, domain.prefixes):
            rows.append((name, str(getattr(entry, "category", "")), str(getattr(entry, "status", ""))))
    return rows


def _ledger_rows(domain: Domain) -> list[dict[str, str]]:
    from . import autodiff_ledger
    reader = csv.DictReader(io.StringIO(autodiff_ledger.render_csv()))
    return [r for r in reader if _matches(r.get("family", ""), domain.prefixes)]


def _execution_rows(domain: Domain) -> list[dict[str, str]]:
    from . import execution_matrix
    reader = csv.DictReader(io.StringIO(execution_matrix.render_csv()))
    out = []
    for r in reader:
        hay = " ".join((r.get("compiler_path", ""), r.get("executor_id", "") or "", r.get("reason", "")))
        if _matches(hay, domain.keywords):
            out.append(r)
    return out


@dataclass(frozen=True)
class LadderRow:
    domain: str
    label: str
    registry_total: int
    registry_planned: int
    ledger_total: int
    ledger_adjoint: int
    ledger_device_verified: int
    native_targets: str            # "target=n_executable;..." sorted
    owners: str                    # "ID(owner);ID(successor);ID(missing)"


def collect_rows() -> list[LadderRow]:
    plan_text = _PLAN.read_text(encoding="utf-8") if _PLAN.is_file() else ""
    routes = _routing_index(plan_text)
    rows: list[LadderRow] = []
    for d in DOMAINS:
        reg = _registry_rows(d)
        led = _ledger_rows(d)
        exe = _execution_rows(d)
        per_target: dict[str, int] = {}
        for r in exe:
            if r.get("executable", "") in ("1", "True", "true") and r.get("execution_kind", "").startswith("native"):
                per_target[r["target"]] = per_target.get(r["target"], 0) + 1
        rows.append(LadderRow(
            domain=d.key, label=d.label,
            registry_total=len(reg),
            registry_planned=sum(1 for _, _, s in reg if s == "planned"),
            ledger_total=len(led),
            ledger_adjoint=sum(1 for r in led if r.get("ir_adjoint", "none") not in ("", "none")),
            ledger_device_verified=sum(1 for r in led if (r.get("bwd_device_verified_jit") or r.get("bwd_device_verified_abi"))),
            native_targets=";".join(f"{t}={n}" for t, n in sorted(per_target.items())),
            owners=";".join(f"{o}({routes.get(o, 'missing')})" for o in d.owners),
        ))
    return rows


CSV_COLUMNS = ("domain", "label", "registry_total", "registry_planned", "ledger_total",
               "ledger_adjoint", "ledger_device_verified", "native_targets", "owners")


def render_csv() -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for r in collect_rows():
        w.writerow([getattr(r, c) for c in CSV_COLUMNS])
    return buf.getvalue()


def render_markdown() -> str:
    rows = collect_rows()
    out = [
        "# Domain proof ladder",
        "",
        "<!-- Generated by `python -m tessera.compiler.generated_docs --write domain_proof_ladder`;",
        "     drift-gated by scripts/check_generated_docs.sh. Do not hand-edit. -->",
        "",
        "Per domain, the four proof columns the [domain audit](../domain/DOMAIN_AUDIT.md)",
        "requires to be reported separately, derived from the registries rather than",
        "prose: primitive-coverage rows (reference math; the registry vocabulary is",
        "`partial` / `planned`, never complete), AD-ledger rows (derivative",
        "rules; `adjoint` = registered IR adjoint, `device` = device-verified backward),",
        "executable native rows per target from the execution matrix (native execution),",
        "and the integrated-plan owners the domain routes to, each checked against the",
        "plan's routing index. `missing` in the owners column is drift: the domain audit",
        "cites an ID the plan no longer routes. An empty native column is a finding, not",
        "a formatting gap. Measured selection lives in the route ledgers and packets and",
        "is deliberately not summarized here.",
        "",
        "| Domain | Registry rows (of which still `planned`) | AD ledger rows (adjoint / device) | Native executable rows per target | Plan owners |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        native = r.native_targets.replace(";", ", ") or "—"
        owners = ", ".join(f"`{o}`" for o in r.owners.split(";"))
        out.append(f"| {r.label} | {r.registry_total} ({r.registry_planned}) | "
                   f"{r.ledger_total} ({r.ledger_adjoint} / {r.ledger_device_verified}) | {native} | {owners} |")
    out += ["", "## Matching rules", "",
            "| Domain | Name prefixes (registry, ledger) | Execution-matrix keywords |", "|---|---|---|"]
    for d in DOMAINS:
        out.append(f"| {d.label} | {', '.join(f'`{p}`' for p in d.prefixes)} | {', '.join(f'`{k}`' for k in d.keywords)} |")
    out.append("")
    return "\n".join(out)
