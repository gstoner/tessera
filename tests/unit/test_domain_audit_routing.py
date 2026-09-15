"""Domain audit ↔ integrated compiler plan routing gate.

The domain audit maps mathematical domains onto the shared compiler and routes
every remaining boundary to an owner in the integrated compiler plan. The plan
reorganizes (W-items become successors, cuts F0–F5 get new IDs), and prose in
the domain tree silently kept steering by retired IDs (found 2026-09-15: W3.5,
W3.6, W5.1, W6.3 were all `successor` rows by then). This gate makes that a
failure instead of a reading exercise:

* every plan-shaped ID a live domain document cites must be in the plan's
  routing index;
* an ID routed as ``successor`` or ``archive`` may be cited only on a line that
  also names its canonical destination, so the reader is routed, never parked;
* every plan task whose owner document lives under ``docs/audit/domain`` must be
  cited by the domain audit, so the plan and the audit agree on who owns what;
* the generated domain proof ladder exists and is referenced by the audit, so
  status comes from registries rather than prose.

Historical archives under ``docs/audit/domain/archive`` are provenance and are
deliberately out of scope.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md"
DOMAIN = ROOT / "docs/audit/domain"
AUDIT = DOMAIN / "DOMAIN_AUDIT.md"
LADDER = ROOT / "docs/audit/generated/domain_proof_ladder.md"

_spec = importlib.util.spec_from_file_location("plan_check", ROOT / "scripts/check_compiler_plan.py")
assert _spec and _spec.loader
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

# IDs the plan hands out: W-items (W3.6, W2.4a), dash-numbered programs
# (AD-HIGHER-1, TSOL-PHYS-TAIL-1, E2E-REAL-6F) and the few bare names.
_ID = re.compile(r"(?<![\w.-])(W\d+\.\d+[a-z]?|[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+|RIEMANNIAN-OT|DISPATCH-BREAKER)(?![\w.-])")
_LINK = re.compile(r"\[([^\]\n]+)\]\(([^\s)]+)\)")


def _live_domain_docs() -> list[Path]:
    return sorted(p for p in DOMAIN.glob("*.md"))


def _routes() -> dict[str, tuple[str, str]]:
    return checker.routes(PLAN.read_text(encoding="utf-8"))


def _cited(text: str) -> dict[str, list[str]]:
    """{ID: [lines citing it]} outside fenced code."""
    cited: dict[str, list[str]] = {}
    fenced = False
    for line in text.splitlines():
        if line.startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced:
            continue
        for m in _ID.finditer(line):
            cited.setdefault(m.group(1), []).append(line)
    return cited


def test_domain_docs_cite_only_routed_ids():
    routes = _routes()
    offenders = []
    for doc in _live_domain_docs():
        for ident, lines in _cited(doc.read_text(encoding="utf-8")).items():
            if ident not in routes:
                # Only plan-shaped IDs count; a doc may name its own sections
                # (e.g. "PGA/CGA") but those do not match the ID grammar above.
                if re.fullmatch(r"W\d+\.\d+[a-z]?", ident) or ident in checker.records(PLAN.read_text(encoding="utf-8")):
                    offenders.append(f"{doc.relative_to(ROOT)}: `{ident}` is not in the plan's routing index")
    assert not offenders, "\n".join(offenders)


def test_retired_ids_are_cited_with_their_destination():
    routes = _routes()
    offenders = []
    for doc in _live_domain_docs():
        for ident, lines in _cited(doc.read_text(encoding="utf-8")).items():
            dest, relation = routes.get(ident, ("", "owner"))
            if relation not in ("successor", "archive"):
                continue
            target = dest.lstrip("#")
            for line in lines:
                # The destination anchor is the slug of the successor's ID.
                if target and target not in checker.slug(line) and target.upper().replace("-", "") not in line.upper().replace("-", "").replace(".", ""):
                    offenders.append(f"{doc.relative_to(ROOT)}: `{ident}` is a {relation} of `{dest}` but the line does not route the reader there: {line.strip()[:120]}")
    assert not offenders, "\n".join(offenders)


def test_plan_tasks_owned_by_domain_docs_are_cited_by_the_audit():
    plan_text = PLAN.read_text(encoding="utf-8")
    audit_text = AUDIT.read_text(encoding="utf-8")
    missing = []
    for ident, fields in checker.records(plan_text).items():
        owner = fields.get("Owner", "")
        link = _LINK.search(owner)
        if link and "domain/" in link.group(2) and ident not in audit_text:
            missing.append(f"plan task {ident} is owned by {link.group(2)} but DOMAIN_AUDIT.md never cites {ident}")
    assert not missing, "\n".join(missing)


def test_domain_audit_points_at_the_generated_proof_ladder():
    assert LADDER.is_file(), "generated/domain_proof_ladder.md missing: run generated_docs --write domain_proof_ladder"
    assert "generated/domain_proof_ladder.md" in AUDIT.read_text(encoding="utf-8")
    body = LADDER.read_text(encoding="utf-8")
    assert "(missing)" not in body, "the ladder routes a domain to an ID the plan no longer carries"
