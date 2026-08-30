"""Decision #19 membership audit — does a Target IR op require its contract?

Decision #19 was amended 2026-08-30: the `tessera_<backend>` layer exists for
**contract carriage**, not for lit-testability (NVVM and ROCDL are MLIR
dialects too, and `mlir-opt` verifies them on any host). Its membership test is
that an op belongs when it carries a Tessera contract the upstream dialect
cannot express.

This audit measures whether that is true, and the distinction it draws is the
one that matters:

* **requires** — the contract attribute is non-optional. The op cannot exist
  without it.
* **optional-only** — the attribute is declared `OptionalAttr<...>`, usually
  inherited from the dialect's op base class, which hands the same bag to
  every op. The contract is *expressible* but not *enforced*: an op can be
  emitted with `arch`, `accum` and `dtype` all unset and still verify.

That second category is the finding. An optional contract **fails open** — it
is Decision #32's information loss (the attribute silently absent rather than
declared dropped) and Decision #21a's "a semantic key never defaults", both
violated by construction rather than by mistake. An op in that state satisfies
#19's membership test on paper and carries nothing in practice.

Why this is a ratchet rather than a report
------------------------------------------
#19's amendment invites Apple (~12 ops) and x86 (~8) to expand. Expansion is
the moment this regresses: the cheapest way to add an op is to derive it from
the base class and inherit the optional bag, which adds surface that looks
contract-carrying and is not. The test beside this module holds the
required-contract count from falling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

#: Target IR dialect definitions, one per backend.
_DIALECTS: tuple[tuple[str, str], ...] = (
    ("nvidia", "src/compiler/codegen/tessera_gpu_backend_NVIDIA/include/tessera/gpu/IR/TesseraNVIDIADialect.td"),
    ("rocm", "src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMOps.td"),
    ("x86", "src/compiler/codegen/tessera_x86_backend/include/TesseraX86/IR/TesseraX86Ops.td"),
    ("apple", "src/compiler/codegen/Tessera_Apple_Backend/include/Tessera/Target/Apple/TesseraAppleOps.td"),
)

#: Attribute-name fragments that denote a Tessera contract upstream dialects
#: have no field for — Decision #15a's attributes plus the arbiter metadata
#: added by the delegation contract.
_CONTRACT_TERMS: tuple[str, ...] = (
    "accum", "numeric_policy", "math_mode", "rounding", "layout", "distribution",
    "storage", "dtype", "arch", "feature", "provenance", "accuracy",
    "determinism", "covers", "tolerance", "manifold", "algebra",
    "tile_m", "tile_n", "tile_k", "warps", "staging", "epilogue", "schedule",
    "abi", "space", "order",
)


@dataclass(frozen=True)
class OpMembership:
    backend: str
    mnemonic: str
    required: tuple[str, ...]
    optional: tuple[str, ...]

    @property
    def verdict(self) -> str:
        if self.required:
            return "requires"
        if self.optional:
            return "optional-only"
        return "no-contract"


def _balanced_block(text: str, start: int) -> str:
    depth, i = 1, start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i]


def _base_bodies(text: str) -> dict[str, str]:
    """Op base classes contribute their attributes to every derived op.

    Matched up to the first colon rather than through a bracketed generic list:
    `class TesseraNVIDIA_Op<string mnemonic, list<Trait> traits = []> :` nests
    `<>`, so a `<[^>]*>` pattern stops inside `list<Trait>` and silently finds
    no base classes at all — which reports every inheriting op as carrying no
    contract.
    """
    return {
        m.group(1): _balanced_block(text, m.end())
        for m in re.finditer(r"^class\s+(\w+)[^:{]*:[^{]*\{", text, re.M)
    }


def _contract_names(text: str, *, optional: bool) -> set[str]:
    pattern = (
        r"OptionalAttr<[^>]*>\s*:\s*\$(\w+)" if optional
        else r"(?<!OptionalAttr<)\b[A-Za-z_][\w<>, ]*?:\s*\$(\w+)"
    )
    return {
        name for name in re.findall(pattern, text)
        if any(term in name for term in _CONTRACT_TERMS)
    }


def collect() -> tuple[OpMembership, ...]:
    rows: list[OpMembership] = []
    for backend, relative in _DIALECTS:
        path = _ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        bases = _base_bodies(text)
        for match in re.finditer(r"^def\s+(\w+)\s*:\s*([^{]*?)\{", text, re.M):
            head = match.group(2)
            mnemonic = re.search(r'"([a-z0-9_.]+)"', head)
            base = re.match(r"\s*(\w+)", head)
            if not mnemonic or not base:
                continue
            base_name = base.group(1)
            if "Attr" in base_name or "Type" in base_name:
                continue  # enum/type definitions are not operations
            body = _balanced_block(text, match.end()) + bases.get(base_name, "")
            optional = _contract_names(body, optional=True)
            everything = {
                n for n in re.findall(r"\$(\w+)", body)
                if any(t in n for t in _CONTRACT_TERMS)
            }
            rows.append(OpMembership(
                backend=backend,
                mnemonic=mnemonic.group(1),
                required=tuple(sorted(everything - optional)),
                optional=tuple(sorted(optional)),
            ))
    return tuple(rows)


def summary() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in collect():
        bucket = out.setdefault(
            row.backend, {"requires": 0, "optional-only": 0, "no-contract": 0})
        bucket[row.verdict] += 1
    return out


def render_markdown() -> str:
    rows = collect()
    per = summary()
    total = len(rows)
    requires = sum(b["requires"] for b in per.values())

    out = [
        "# Target IR Membership — does an op *require* its contract?",
        "",
        "**Generated. Do not hand-edit.** Regenerate with",
        "`python -m tessera.compiler.generated_docs --write`.",
        "",
        "Decision #19 (amended 2026-08-30) says the `tessera_<backend>` layer",
        "exists for **contract carriage**. This measures whether that holds.",
        "",
        "`optional-only` is the row to read. Those ops declare their contract",
        "attributes as `OptionalAttr`, almost always inherited from the",
        "dialect's op base class, which hands the same bag to every op. The",
        "contract is *expressible* but not *enforced* — the op verifies with",
        "`arch`, `accum` and `dtype` all unset. That **fails open**: it is",
        "Decision #32 information loss (silently absent rather than declared",
        "dropped) and Decision #21a (a semantic key never defaults), violated",
        "by construction rather than by mistake. Such an op satisfies #19's",
        "membership test on paper and carries nothing in practice.",
        "",
        f"**{requires} of {total} ops require the contract they carry.**",
        "",
        "| Backend | requires | optional-only | no contract |",
        "|---|---|---|---|",
    ]
    for backend, _ in _DIALECTS:
        b = per.get(backend)
        if b is None:
            continue
        out.append(
            f"| `{backend}` | {b['requires']} | {b['optional-only']} "
            f"| {b['no-contract']} |"
        )

    out += [
        "",
        "## What this means for the operator expansion",
        "",
        "#19's amendment invites Apple and x86 to grow their Target IR, and",
        "expansion is exactly when this regresses: the cheapest way to add an",
        "op is to derive it from the base class and inherit the optional bag,",
        "which adds surface that *looks* contract-carrying and is not.",
        "**A new op should declare its contract attributes as required.**",
        "",
        "Apple is the sharpest case — it requires nothing at all, which is the",
        "same gap as its missing machine primitives seen from the contract",
        "side: a dialect of dispatch containers has no contract to enforce.",
        "",
        "## Ops requiring no contract at all",
        "",
        "| Backend | Op |",
        "|---|---|",
    ]
    for row in rows:
        if row.verdict == "no-contract":
            out.append(f"| `{row.backend}` | `{row.mnemonic}` |")
    out.append("")
    return "\n".join(out)


def render_csv() -> str:
    lines = ["backend,mnemonic,verdict,required,optional"]
    for row in collect():
        lines.append(
            f"{row.backend},{row.mnemonic},{row.verdict},"
            f"{' '.join(row.required)},{' '.join(row.optional)}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    print(render_markdown())
