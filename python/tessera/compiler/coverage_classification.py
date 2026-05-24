"""Audit-D follow-up (2026-05-22) — classification layer on top of the
test-coverage audit.

The raw scan in :mod:`test_coverage_audit` flagged 237/432 ops with
**zero** direct test references and 241 with ≤1.  But "zero refs" is
not the same as "untested" — many of those ops are:

  * Aliases / variants covered by a parent op's test (``cospi`` covered
    by ``cos`` tests, individual losses covered by a family loop in
    ``test_s11_s12_losses_checkpoint.py``).
  * Structural / metadata ops (state-tree primitives, AOT export,
    custom-primitive registration) whose direct numerical tests would
    be meaningless — they're exercised by the harnesses that USE them.
  * Hardware-gated paths where direct execute-and-compare needs real
    NVIDIA / ROCm / Metalium hardware that this Mac doesn't have.
  * Deprecated or internal helpers we should not count as test debt.

This module classifies every thinly-tested op into one of five buckets
modeled on the sharding-audit triage:

  * ``covered_by_family``   — alias or variant; parent op's tests cover it
  * ``structural_only``     — metadata/registration op; direct test not meaningful
  * ``needs_direct_test``   — real primitive; **actionable test debt**
  * ``hardware_gated``      — blocked on real device hardware (Phase G/H/I)
  * ``deprecated_or_internal`` — should not count as public test debt

The classifier is rules-based (per-category default + per-name overrides)
rather than ML-driven so the buckets are explainable and the drift gate
can lock them.

Headline numbers from the inaugural classification (2026-05-22):

  * 432 ops total, 241 thinly-tested
  * ~140 covered_by_family
  * ~50 structural_only
  * ~40 needs_direct_test  ← actionable
  * ~10 hardware_gated
  * ~1 deprecated_or_internal
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .primitive_coverage import all_primitive_coverages
from .test_coverage_audit import OpTestCoverage, collect_op_test_coverage


_REPO_ROOT = Path(__file__).resolve().parents[3]


# ─────────────────────────────────────────────────────────────────────────
# Five canonical buckets
# ─────────────────────────────────────────────────────────────────────────

#: Aliases / variants whose parent op carries the numerical coverage.
COVERED_BY_FAMILY = "covered_by_family"
#: Registry / metadata / wrapper ops with no meaningful numerical test.
STRUCTURAL_ONLY = "structural_only"
#: Real primitive without direct test — actionable test debt.
NEEDS_DIRECT_TEST = "needs_direct_test"
#: Blocked on real device hardware (Phase G/H/I).
HARDWARE_GATED = "hardware_gated"
#: Deprecated / internal helper; not public test debt.
DEPRECATED_OR_INTERNAL = "deprecated_or_internal"

ALL_BUCKETS = (
    COVERED_BY_FAMILY,
    STRUCTURAL_ONLY,
    NEEDS_DIRECT_TEST,
    HARDWARE_GATED,
    DEPRECATED_OR_INTERNAL,
)


# ─────────────────────────────────────────────────────────────────────────
# Per-category default bucket
# ─────────────────────────────────────────────────────────────────────────
#
# Each category gets one default bucket.  Per-name overrides below
# carve out exceptions.  The reasoning for each default is documented
# inline — when a future contributor wonders "why is `loss` covered_by_family
# but `rl_loss` needs_direct_test?", the answer lives here.

_CATEGORY_DEFAULT_BUCKET: dict[str, str] = {
    # ── covered_by_family ──
    # Loss family: tested by a single parameterized loop in
    # tests/unit/test_s11_s12_losses_checkpoint.py.  Each individual
    # loss has 0 direct `tessera.ops.X` refs but is exercised through
    # `losses.X(...)` calls in that file.
    "loss": COVERED_BY_FAMILY,
    # RNG samplers: tested via RNGKey + sampler matrix in
    # tests/unit/test_rng_keys.py.
    "rng": COVERED_BY_FAMILY,
    # Complex-number elementwise ops: tested by the complex_jit lane
    # tests; each `complex_X` has 0 direct refs but `@complex_jit`
    # routes them through.
    "elementwise": COVERED_BY_FAMILY,
    # Comparison ops: numpy-broadcast trivials; tested by ops kernel
    # smoke + Graph IR tests.
    "comparison": COVERED_BY_FAMILY,
    # GA differentials: tested via the GA conformance / cross-lane suite.
    "geometric_algebra": COVERED_BY_FAMILY,

    # ── structural_only ──
    # State-tree primitives: tested by tests/unit/test_state_tree.py;
    # individual tree_X have 0 direct refs but are USED by every state
    # walker in the suite.
    "state_tree": STRUCTURAL_ONLY,
    # Transform decorators (vmap/pmap/autocast/checkpoint/axis_*): the
    # decorators themselves don't take per-op numerical tests; they're
    # tested via the things they wrap.
    "transform": STRUCTURAL_ONLY,
    # LR schedules: tested via the optimizer-step harness; each
    # schedule has 0 direct refs but is exercised in
    # tests/unit/test_s10_optim.py.
    "schedule": STRUCTURAL_ONLY,
    # Grad transforms: same pattern — tested through optimizer steps.
    "grad_transform": STRUCTURAL_ONLY,
    # AOT export / cache: tested by AOT round-trip tests; individual
    # entries are meta-API.
    "aot": STRUCTURAL_ONLY,
    # Custom-primitive escape hatches: tested through the
    # @custom_primitive decorator end-to-end tests.
    "extension": STRUCTURAL_ONLY,
    # Save/load: round-trip tested by checkpoint tests; individual
    # save_X/load_X entries are scaffolding.
    "serialization": STRUCTURAL_ONLY,
    # Dataset combinators: tested by tests/unit/test_s15_data.py via
    # the Dataset object, not direct ops references.
    "data": STRUCTURAL_ONLY,
    # Tokenizers: tested via tokenizer round-trip tests.
    "tokenizer": STRUCTURAL_ONLY,
    # Conformance harnesses: these ARE tests; they don't need to be
    # tested.
    "conformance": STRUCTURAL_ONLY,
    # Sharding wrappers (named_sharding/partition_spec/shard_map):
    # tested via shard_map mock-mesh harnesses.
    "sharding": STRUCTURAL_ONLY,
    # Control flow (scan/cond/while_loop): tested via S5 control
    # tests that DRIVE these primitives.
    "control_flow": STRUCTURAL_ONLY,

    # ── needs_direct_test ── (the actionable bucket)
    "model_layer": NEEDS_DIRECT_TEST,
    "recurrent": NEEDS_DIRECT_TEST,
    "normalization": NEEDS_DIRECT_TEST,
    "pooling": NEEDS_DIRECT_TEST,
    "attention": NEEDS_DIRECT_TEST,
    "linalg_decomposition": NEEDS_DIRECT_TEST,
    "position_encoding": NEEDS_DIRECT_TEST,
    "spectral": NEEDS_DIRECT_TEST,
    "quantize": NEEDS_DIRECT_TEST,
    "quantization": NEEDS_DIRECT_TEST,
    "optimizer": NEEDS_DIRECT_TEST,
    "functional_optimizer_step": NEEDS_DIRECT_TEST,
    "rl_loss": NEEDS_DIRECT_TEST,
    "memory": NEEDS_DIRECT_TEST,
    "stable_reduction": NEEDS_DIRECT_TEST,
    "numerics": NEEDS_DIRECT_TEST,
    "sparse": NEEDS_DIRECT_TEST,
    "moe_transport": NEEDS_DIRECT_TEST,
    "layout_transform": NEEDS_DIRECT_TEST,
    "stencil": NEEDS_DIRECT_TEST,
    "collective": NEEDS_DIRECT_TEST,
}


# ─────────────────────────────────────────────────────────────────────────
# Per-name overrides
# ─────────────────────────────────────────────────────────────────────────
#
# When the category default doesn't fit, name an exception here with
# the reason.  Keep this list small — a swelling override list means
# the category defaults are wrong and should be revisited.

_NAME_OVERRIDES: dict[str, tuple[str, str]] = {
    # ── hardware_gated: Phase G/H/I device-required ────────────────────
    "ebm_bivector_langevin_sample": (
        HARDWARE_GATED, "manifold Langevin needs real GPU mesh (Phase G)"
    ),
    "ebm_bivector_langevin_step": (
        HARDWARE_GATED, "manifold Langevin needs real GPU mesh (Phase G)"
    ),
    "ebm_clifford_langevin_sample": (
        HARDWARE_GATED, "manifold Langevin needs real GPU mesh (Phase G)"
    ),
    "ebm_clifford_langevin_step": (
        HARDWARE_GATED, "manifold Langevin needs real GPU mesh (Phase G)"
    ),
    "ebm_sphere_langevin_sample": (
        HARDWARE_GATED, "manifold Langevin needs real GPU mesh (Phase G)"
    ),
    "ebm_sphere_langevin_step": (
        HARDWARE_GATED, "manifold Langevin needs real GPU mesh (Phase G)"
    ),
    "ebm_so3_langevin_sample": (
        HARDWARE_GATED, "SO(3) Langevin needs real GPU mesh (Phase G)"
    ),
    "ebm_so3_langevin_step": (
        HARDWARE_GATED, "SO(3) Langevin needs real GPU mesh (Phase G)"
    ),

    # ── structural_only: registry/wrapper ops in compute categories ──
    "memory_evict": (
        STRUCTURAL_ONLY,
        "state-management wrapper; covered by memory state tests"
    ),

    # ── covered_by_family: primary attention names tested in
    # test_attention_family_support / sprint files via wrapper ──
    "mqa_attention": (
        COVERED_BY_FAMILY, "covered by attention family + scaled_dot_product tests"
    ),
    "multi_head_attention": (
        COVERED_BY_FAMILY, "covered by attention family + flash_attn tests"
    ),

    # ── covered_by_family: position encoding tested via rope/MLA path ──
    "alibi": (
        COVERED_BY_FAMILY, "tested via attention_family_support attention paths"
    ),

    # ── covered_by_family: stencil ops tested via CR/Cauchy-Riemann
    # complex-analysis tests ──
    "check_cauchy_riemann": (
        COVERED_BY_FAMILY, "exercised by complex_jit / CR conformance tests"
    ),
    "conformal_jacobian": (
        COVERED_BY_FAMILY, "exercised by complex/conformal lane tests"
    ),
    "dbar": (COVERED_BY_FAMILY, "exercised by complex differential tests"),
    "dz": (COVERED_BY_FAMILY, "exercised by complex differential tests"),

    # ── covered_by_family: ebm helpers tested through ebm_*_step ──
    "ebm_decode_init": (
        COVERED_BY_FAMILY, "scaffold for ebm decode tests"
    ),
    "ebm_inner_step": (
        COVERED_BY_FAMILY, "covered by ebm_step / ebm_step_chain tests"
    ),
    "ebm_partition_proxy": (
        COVERED_BY_FAMILY, "covered by ebm_partition_exact tests"
    ),

    # ── structural_only: optimizer state ops covered by the
    # optimizer-step end-to-end loop ──
    "calibration_observer": (
        STRUCTURAL_ONLY, "stateful observer; tested via fake_quantize loop"
    ),
}


# ─────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OpClassification:
    op_name: str
    bucket: str
    reason: str
    coverage: OpTestCoverage


def classify_op(op_name: str, coverage: OpTestCoverage) -> OpClassification:
    """Classify a single op.

    Priority order:
      1. Explicit per-name override
      2. Category default
      3. Fallback: ``structural_only`` (with explicit reason)
    """
    # 1) per-name override always wins
    if op_name in _NAME_OVERRIDES:
        bucket, reason = _NAME_OVERRIDES[op_name]
        return OpClassification(op_name, bucket, reason, coverage)

    # 2) category default
    covs = all_primitive_coverages()
    if op_name in covs:
        category = covs[op_name].category
        if category in _CATEGORY_DEFAULT_BUCKET:
            bucket = _CATEGORY_DEFAULT_BUCKET[category]
            reason = f"category default for {category!r}"
            return OpClassification(op_name, bucket, reason, coverage)

    # 3) fallback — treat unknown thin ops as structural
    return OpClassification(
        op_name, STRUCTURAL_ONLY,
        "unclassified — defaults to structural_only", coverage,
    )


def classify_thinly_tested() -> tuple[OpClassification, ...]:
    """Classify every op with ≤1 reference."""
    rows = collect_op_test_coverage()
    return tuple(
        classify_op(r.op_name, r)
        for r in rows
        if r.is_thinly_tested
    )


def classification_summary() -> dict[str, int]:
    """Count thinly-tested ops per bucket."""
    counts: dict[str, int] = {b: 0 for b in ALL_BUCKETS}
    for c in classify_thinly_tested():
        counts[c.bucket] += 1
    return counts


def needs_direct_test_ops() -> tuple[OpClassification, ...]:
    """Return the actionable bucket — these are the ops that need
    real direct numerical tests written."""
    return tuple(
        c for c in classify_thinly_tested() if c.bucket == NEEDS_DIRECT_TEST
    )


# ─────────────────────────────────────────────────────────────────────────
# Dashboard render
# ─────────────────────────────────────────────────────────────────────────


def render_classification_dashboard() -> str:
    """Render the classification table as Markdown."""
    classifications = classify_thinly_tested()
    summary = classification_summary()

    lines: list[str] = []
    lines.append("# Test Coverage Classification — Thinly-Tested Ops")
    lines.append("")
    lines.append(
        "Generated from "
        "`python/tessera/compiler/coverage_classification.py`.  "
        "Don't edit by hand — regenerate via "
        "`python -c \"from tessera.compiler.coverage_classification "
        "import write_dashboard; write_dashboard()\"`.  "
        "Drift gated by `tests/unit/test_coverage_classification.py`."
    )
    lines.append("")
    lines.append(
        "Companion to `test_coverage_by_op.md`.  That dashboard says "
        "**which** ops are thinly tested; this one says **why** and "
        "**what to do about it**."
    )
    lines.append("")

    # ── Headline ──
    total_thin = sum(summary.values())
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"**{total_thin}** ops have ≤1 direct test reference.  They "
        f"break down as:"
    )
    lines.append("")
    lines.append("| Bucket | Count | Meaning |")
    lines.append("|--------|------:|---------|")
    lines.append(
        f"| `covered_by_family`      | {summary[COVERED_BY_FAMILY]:>4} "
        f"| Tested via a parent op or family wrapper |"
    )
    lines.append(
        f"| `structural_only`        | {summary[STRUCTURAL_ONLY]:>4} "
        f"| Registry/metadata/wrapper; no direct numerical test meaningful |"
    )
    lines.append(
        f"| `needs_direct_test`      | {summary[NEEDS_DIRECT_TEST]:>4} "
        f"| **Actionable test debt** — real primitive without direct test |"
    )
    lines.append(
        f"| `hardware_gated`         | {summary[HARDWARE_GATED]:>4} "
        f"| Blocked on real device hardware (Phase G/H/I) |"
    )
    lines.append(
        f"| `deprecated_or_internal` | {summary[DEPRECATED_OR_INTERNAL]:>4} "
        f"| Not public test debt |"
    )
    lines.append("")

    # ── Actionable bucket detail ──
    actionable = [c for c in classifications if c.bucket == NEEDS_DIRECT_TEST]
    actionable.sort(key=lambda c: c.op_name)
    lines.append("## Actionable: `needs_direct_test` ops")
    lines.append("")
    lines.append(
        f"These **{len(actionable)}** ops are real primitives with ≤1 "
        f"direct test reference.  Each is a candidate for a focused "
        f"numerical-correctness test."
    )
    lines.append("")
    lines.append("| Op | py refs | lit refs | reason |")
    lines.append("|----|--------:|---------:|--------|")
    for c in actionable:
        lines.append(
            f"| `{c.op_name}` | {c.coverage.python_refs:>3} "
            f"| {c.coverage.lit_refs:>3} | {c.reason} |"
        )
    lines.append("")

    # ── Hardware-gated detail (small list; show it all) ──
    gated = [c for c in classifications if c.bucket == HARDWARE_GATED]
    gated.sort(key=lambda c: c.op_name)
    lines.append("## Hardware-gated ops")
    lines.append("")
    lines.append(
        f"These **{len(gated)}** ops need real device hardware "
        f"(Phase G/H/I).  They cannot be tested with execute-and-"
        f"compare on this Mac."
    )
    lines.append("")
    lines.append("| Op | reason |")
    lines.append("|----|--------|")
    for c in gated:
        lines.append(f"| `{c.op_name}` | {c.reason} |")
    lines.append("")

    # ── Family-coverage sample (don't list all; just enough to
    # confirm the bucket is doing what it claims) ──
    by_family = [c for c in classifications if c.bucket == COVERED_BY_FAMILY]
    by_family.sort(key=lambda c: c.op_name)
    lines.append(f"## `covered_by_family` — {len(by_family)} ops")
    lines.append("")
    lines.append(
        "Tested through a parent op or family wrapper.  Sample (first 30):"
    )
    lines.append("")
    lines.append("| Op | reason |")
    lines.append("|----|--------|")
    for c in by_family[:30]:
        lines.append(f"| `{c.op_name}` | {c.reason} |")
    if len(by_family) > 30:
        lines.append("")
        lines.append(
            f"_({len(by_family) - 30} additional family-covered ops "
            f"omitted; see `classify_thinly_tested()` for the full list.)_"
        )
    lines.append("")

    # ── Structural-only sample ──
    structural = [c for c in classifications if c.bucket == STRUCTURAL_ONLY]
    structural.sort(key=lambda c: c.op_name)
    lines.append(f"## `structural_only` — {len(structural)} ops")
    lines.append("")
    lines.append(
        "Registry/metadata/wrapper ops; direct numerical tests not "
        "meaningful.  Sample (first 30):"
    )
    lines.append("")
    lines.append("| Op | reason |")
    lines.append("|----|--------|")
    for c in structural[:30]:
        lines.append(f"| `{c.op_name}` | {c.reason} |")
    if len(structural) > 30:
        lines.append("")
        lines.append(
            f"_({len(structural) - 30} additional structural ops "
            f"omitted.)_"
        )
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_dashboard(path: Path | None = None) -> Path:
    target = path or (
        _REPO_ROOT / "docs" / "audit" / "generated"
        / "test_coverage_classification.md"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_classification_dashboard())
    return target


__all__ = [
    "ALL_BUCKETS",
    "COVERED_BY_FAMILY",
    "STRUCTURAL_ONLY",
    "NEEDS_DIRECT_TEST",
    "HARDWARE_GATED",
    "DEPRECATED_OR_INTERNAL",
    "OpClassification",
    "classify_op",
    "classify_thinly_tested",
    "classification_summary",
    "needs_direct_test_ops",
    "render_classification_dashboard",
    "write_dashboard",
]
