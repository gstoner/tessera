"""Host-free: Decision #29's op-level clause, gated at last.

#29 says a declared ODS op must have a named consumer or be deleted, and cites
`test_governance_declarations.py` as its drift gate. Measured 2026-09-19, that
file checks coverage *axes* and duplicate *dialect names* — and nothing mapped
an op to a consumer. Confirmed the expensive way: `tessera.scaled_matmul` was
committed with no consumer and all fourteen governance tests passed.

That absence also explains #29's own history. Every instance it cites —
`manifold` reaching no backend, `MultivectorSpec.grades`, nine `!tile.*` types,
`numeric_policy` with no carrier, `TilingInterface` — was found **by hand**,
which is what an ungated rule produces.

An op counts as consumed when something outside its own `.td` mentions either
its generated C++ class or its full `dialect.mnemonic` name: a pass, a
lowering, a verifier, a fixture. That is deliberately generous. The failure
this catches is a declaration nothing anywhere refers to, which reads in review
as a closed contract while carrying nothing; distinguishing a *good* consumer
from a nominal one is a judgement no scan should be trusted with.

Declarations held deliberately under Decision #29a stay in the waiver below,
which may only shrink.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: `def Foo_BarOp : Op<Foo_Dialect, "bar", [...]>` and its ODS variants.
_OP_DEF = re.compile(
    r"^def\s+(\w+)\s*:\s*(?:\w*Op|Op)\s*<\s*\n?\s*(\w+)\s*,\s*\"([^\"]+)\"", re.M)
_DIALECT_DEF = re.compile(r"^def\s+(\w+)\s*:\s*Dialect\s*\{(.*?)\n\}", re.M | re.S)
_DIALECT_NAME = re.compile(r'let\s+name\s*=\s*"([^"]+)"')

#: Only the shipped compiler. `examples/` and `research/` carry their own
#: dialects as worked demonstrations; holding them to the product's consumer
#: rule would either delete illustrations or pad the waiver with entries nobody
#: intends to close.
_SCOPES = ("src",)
_SEARCH_ROOTS = ("src", "python", "tools", "tests")


def _td_files() -> list[Path]:
    return sorted(
        p for scope in _SCOPES for p in (ROOT / scope).rglob("*.td")
        if "archive" not in p.parts and "build" not in str(p)
    )


def _declared_ops() -> list[tuple[Path, str, str, str]]:
    names: dict[str, str] = {}
    for td in _td_files():
        for m in _DIALECT_DEF.finditer(td.read_text(errors="ignore")):
            found = _DIALECT_NAME.search(m.group(2))
            if found:
                names[m.group(1)] = found.group(1)
    ops: list[tuple[Path, str, str, str]] = []
    for td in _td_files():
        for m in _OP_DEF.finditer(td.read_text(errors="ignore")):
            record, dialect, mnemonic = m.group(1), m.group(2), m.group(3)
            cpp = re.sub(r"^[A-Za-z0-9]+_", "", record)   # Tessera_MatmulOp -> MatmulOp
            ops.append((td, record, cpp, f"{names.get(dialect, '?')}.{mnemonic}"))
    return ops


#: This file names every waived op as a string, so a naive search would find
#: each one here and call it consumed -- adding an entry to the waiver would
#: silently remove the op from the gate. Caught by `test_unconsumed_waiver_only
#: _shrinks` on the first run, which is the only reason the gate is not hollow.
_SELF = Path(__file__).resolve()


def _has_consumer(td: Path, cpp_class: str, full_name: str) -> bool:
    # A colliding class name belongs to more than one dialect; see
    # `_AMBIGUOUS_CPP`. Fall back to the fully qualified mnemonic alone.
    needles = (full_name,) if cpp_class in _AMBIGUOUS_CPP else (cpp_class, full_name)
    for needle in needles:
        result = subprocess.run(
            ["grep", "-rlF", "--include=*.cpp", "--include=*.h", "--include=*.py",
             "--include=*.td", "--include=*.mlir", needle,
             *(str(ROOT / r) for r in _SEARCH_ROOTS)],
            capture_output=True, text=True)
        for hit in result.stdout.split():
            path = Path(hit).resolve()
            if path != td.resolve() and path != _SELF and "build" not in hit:
                return True
    return False


_OPS = _declared_ops()

#: ODS strips the record's dialect prefix to name the generated C++ class, so
#: `Cache_RingCreateOp` and `Tessera_RingCreateOp` BOTH generate `RingCreateOp`
#: (in different namespaces). Searching the bare class name therefore lets one
#: dialect's consumer vouch for another dialect's op. Measured 2026-09-20: 11
#: stripped names collide across 22 ops, and exactly one op -- `cache.ring.create`
#: -- was passing this gate solely on a hit belonging to `tessera.ring.create`.
#: For a colliding name the bare class is not evidence, so only the unambiguous
#: `dialect.mnemonic` counts. That fails closed, which is the right direction
#: for a governance gate: it can call a consumed op unconsumed, never the
#: reverse.
_AMBIGUOUS_CPP: frozenset[str] = frozenset(
    cpp for cpp in {c for _, _, c, _ in _OPS}
    if len({f for _, _, c2, f in _OPS if c2 == cpp}) > 1
)


def test_ods_scan_finds_ops_at_all() -> None:
    """A scan that silently matches nothing would pass every other test here."""
    assert len(_OPS) > 150, (
        f"only {len(_OPS)} ODS ops parsed; the regex has drifted from the ODS "
        f"spelling and this gate is now vacuous"
    )


@pytest.mark.parametrize(
    "td,record,cpp,full", _OPS,
    ids=[f"{full}" for _, _, _, full in _OPS])
def test_declared_op_has_a_consumer(td: Path, record: str, cpp: str, full: str) -> None:
    if _has_consumer(td, cpp, full):
        return
    assert full in _UNCONSUMED_ON_2026_09_20, (
        f"{full} ({record}, {td.relative_to(ROOT)}) is declared in ODS and "
        f"nothing outside its own .td refers to it — no pass, lowering, "
        f"verifier or fixture. Decision #29: give it a consumer or delete it. "
        f"If it is deliberate debt, Decision #29a requires it be marked AT THE "
        f"SITE with the queue item that owns the wiring, and only then added "
        f"here."
    )


def test_unconsumed_waiver_only_shrinks() -> None:
    """A stale waiver re-permits a name that has since been wired."""
    stale = []
    for name in sorted(_UNCONSUMED_ON_2026_09_20):
        match = [(td, c, f) for td, _, c, f in _OPS if f == name]
        if not match:
            stale.append(f"{name} (no longer declared)")
        elif _has_consumer(*match[0]):
            stale.append(f"{name} (now has a consumer)")
    assert not stale, (
        f"remove from _UNCONSUMED_ON_2026_09_20: {stale}")


#: Shrink-only. Seeded 2026-09-20 from the first scan that ever ran: 23 of 291
#: declared ops had no consumer anywhere, 19 of them under `src/`. They are
#: listed rather than deleted because #29a's exemption may apply to some, but
#: NONE of them currently satisfies its conditions — none is marked at its site
#: with an owning queue item. Each entry is either wiring that was never
#: finished or an op that should go; both answers need the person who knows
#: which, so the list starts full and empties as they are worked.
_UNCONSUMED_ON_2026_09_20: frozenset[str] = frozenset({
    "cache.kv.create",
    "cache.page.read",
    "cache.page.write",
    "cache.pt.create",
    # Added 2026-09-20 when the ambiguity fix above stopped
    # `tessera.ring.create`'s consumer from vouching for this op. Its two
    # siblings were already here; this completes the family rather than
    # recording new debt.
    "cache.ring.create",
    "cache.ring.pop",
    "cache.ring.push",
    "tessera_collective.pack_cast",
    "tessera_collective.qos.acquire",
    "tessera_collective.qos.release",
    "tessera_solver.ir_step",
    "tessera_solver.trsm",
    "trng.create_state",
    "tsl.root.brent",
    "tsl.root.newton",
    "tsl.solve_trig",
    "tss.cg",
    "tss.gmres",
    "tss.spmm",
    "tss.spmv",
})
