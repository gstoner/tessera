"""Contract-consumer audit axis (Phase 0 of the contract-pass plan).

The recurring meta-gap the plan attacks: a typed **contract** exists (handle
fields, cost scores, sharding specs, multimodal nodes) but **no compiler pass
consumes it as an obligation**. This module makes that gap *measurable*.

Each row pairs a contract with its consuming pass and a status that is **derived
by a live probe**, not hand-asserted:

  * ``live``     — a consumer exists in the tree right now (the probe imports it).
  * ``declared`` — the contract exists but its consumer is still planned.
  * ``none``     — neither.

Because status is probed, a workstream flipping its consumer on automatically
flips its row ``declared → live`` the next time the dashboard regenerates — the
drift gate (``scripts/check_generated_docs.sh``) then forces the committed doc to
match. That is the whole point: "we have the field" vs "a pass consumes it"
becomes a number nobody can fake.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Phase 0).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable


def _has(module: str, attr: str) -> Callable[[], bool]:
    """A probe: True iff ``module`` imports and exposes ``attr``."""

    def probe() -> bool:
        try:
            return hasattr(importlib.import_module(module), attr)
        except Exception:
            return False

    return probe


@dataclass(frozen=True)
class ContractConsumer:
    """One contract and the pass obligated to consume it."""

    workstream: str          # A..F (matches CONTRACT_PASS_PLAN.md)
    contract: str            # the typed object
    item: str                # the original audit list item (#1..#8)
    contract_site: str       # where the contract lives
    consumer: str            # the pass/fn that must read it
    oracle: str              # the verifier that proves consumption
    _probe: Callable[[], bool]
    notes: str = ""

    @property
    def status(self) -> str:
        return "live" if self._probe() else "declared"


# Registry — one row per workstream contract. Probes point at the *consumer*, so
# status reflects whether the contract is actually consumed today.
_CONSUMERS: tuple[ContractConsumer, ...] = (
    ContractConsumer(
        "A", "PagedKVState", "#1/#6",
        "tessera.cache.paged_kv.PagedKVState",
        "tessera.ops.paged_attention / flash_attn(kv_state=)",
        "evaluator.paged_kv_equivalence + paged_kv_native_equivalence "
        "(Metal + ROCm rungs)",
        _has("tessera", "ops"),  # refined below to check the op itself
        "Unifies contiguous/tiered/latent/quantized-tail KV; runs native on "
        "Metal and the compiled ROCm FA-2 lane.",
    ),
    ContractConsumer(
        "B", "SchedulePolicy / CacheHandoff", "#2",
        "tessera.compiler.phase_specialization",
        "phase_specialization.specialize / PhaseSpecializedProgram",
        "verify_phase_split (prefill ▸ decode ≡ forward)",
        _has("tessera.compiler.phase_specialization", "specialize"),
        "@jit(phase=,slo=) attaches the policy.",
    ),
    ContractConsumer(
        "C", "FusionCost / bytes_moved", "#3",
        "tessera.compiler.fusion.FusionCost",
        "fusion.select_attention_lowering (byte-scored selector)",
        "cost-monotonicity + feasibility invariant",
        _has("tessera.compiler.fusion", "select_attention_lowering"),
        "Today FusionCost gates; the selector ranks variants by total bytes.",
    ),
    ContractConsumer(
        "D", "NumericPolicy.scale / quant_axis", "#4",
        "tessera.compiler.primitive_coverage.NumericPolicy",
        "smoothquant.migrate_activation_scale (producer pass)",
        "W8A8 parity vs fp16 + anti-fallback (direct-consume kernel fired)",
        _has("tessera.compiler.smoothquant", "migrate_activation_scale"),
        "Backend already direct-consumes W8A8/int4; the producer pass is the gap.",
    ),
    ContractConsumer(
        "E", "TPSpec (col/row/seq parallel)", "#5",
        "tessera.compiler.tensor_parallel",
        "tensor_parallel.rewrite_linear (auto nn.Linear → parallel)",
        "cross-rank gradient equivalence (sharded grad ≡ single-rank)",
        _has("tessera.compiler.tensor_parallel", "rewrite_linear"),
        "AdjointCollectiveInsertionPass exists; auto-rewrite + numeric grad test are the gap.",
    ),
    ContractConsumer(
        "F", "ModelWalk", "#7/#8",
        "tessera.compiler.model_walk",
        "model_walk.partition_walks (named vision_prefill/text_decode/image_gen)",
        "per-walk parity vs full forward",
        _has("tessera.compiler.model_walk", "partition_walks"),
        "MiniMax-M3 graph nodes exist; named walks + native audio/coord ops are the gap.",
    ),
)


def _refine_a_probe() -> ContractConsumer:
    """A's probe should check the consumer op exists, not just that ops imports."""

    def probe() -> bool:
        try:
            from .. import ops
            return hasattr(ops, "paged_attention")
        except Exception:
            return False

    a = _CONSUMERS[0]
    return ContractConsumer(a.workstream, a.contract, a.item, a.contract_site,
                            a.consumer, a.oracle, probe, a.notes)


CONSUMERS: tuple[ContractConsumer, ...] = (_refine_a_probe(),) + _CONSUMERS[1:]


def status_counts() -> dict[str, int]:
    counts = {"live": 0, "declared": 0, "none": 0}
    for c in CONSUMERS:
        counts[c.status] += 1
    return counts


_COLUMNS = ("workstream", "item", "contract", "status", "contract_site",
            "consumer", "oracle", "notes")


def render_csv() -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(_COLUMNS)
    for c in sorted(CONSUMERS, key=lambda x: x.workstream):
        w.writerow([c.workstream, c.item, c.contract, c.status, c.contract_site,
                    c.consumer, c.oracle, c.notes])
    return buf.getvalue()


def render_markdown() -> str:
    counts = status_counts()
    lines = [
        "<!-- AUTO-GENERATED by tessera.compiler.contract_consumers — do not edit. -->",
        "<!-- Regenerate: scripts/check_generated_docs.sh --write contract_consumers -->",
        "",
        "# Contract Consumers (generated)",
        "",
        "The contract-pass plan's meta-gap tracker: each typed contract paired with "
        "the compiler pass obligated to consume it. `status` is **probed live** — "
        "`live` when the consumer exists in the tree, `declared` when only the "
        "contract does. A workstream landing its consumer flips its row "
        "automatically. Canonical artifact: `contract_consumers.csv`. See "
        "`docs/audit/roadmap/CONTRACT_PASS_PLAN.md`.",
        "",
        f"**Status:** live={counts['live']} · declared={counts['declared']} · "
        f"none={counts['none']} (of {len(CONSUMERS)})",
        "",
        "| WS | Item | Contract | Status | Consumer (pass) | Oracle |",
        "|----|------|----------|--------|-----------------|--------|",
    ]
    for c in sorted(CONSUMERS, key=lambda x: x.workstream):
        lines.append(
            f"| {c.workstream} | {c.item} | `{c.contract}` | **{c.status}** | "
            f"{c.consumer} | {c.oracle} |"
        )
    lines += [
        "",
        "## Detail",
        "",
    ]
    for c in sorted(CONSUMERS, key=lambda x: x.workstream):
        lines += [
            f"### {c.workstream} — {c.contract} ({c.status})",
            "",
            f"- **Item:** {c.item}",
            f"- **Contract site:** `{c.contract_site}`",
            f"- **Consumer:** {c.consumer}",
            f"- **Oracle:** {c.oracle}",
            f"- **Notes:** {c.notes}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ContractConsumer",
    "CONSUMERS",
    "status_counts",
    "render_markdown",
    "render_csv",
]
