"""Apple GPU pipeline pass-order matrix.

The ``tessera-lower-to-apple_gpu-runtime`` pipeline composes 11 lowering
passes that *must* run in a specific order. CORE-COMPILER-1 replaces seven
fusion pass shells with one declarative fusion pass; that pass still has to
fire before the per-op lowerings that could steal pieces of a fused chain.

The contract surface is documented in
``docs/audit/compiler/COMPILER_AUDIT.md`` § "Coverage matrix —
pass-order matrices".  This file pins:

  1. The exact canonical order in ``Passes.cpp``.
  2. The dependency contract (declarative fusion table → per-op).
  3. The pipeline alias name (other tooling calls it by name).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APPLE_PASSES_CPP = (
    REPO_ROOT / "src" / "compiler" / "codegen"
    / "Tessera_Apple_Backend" / "lib" / "Target" / "Apple" / "Passes.cpp"
)


# Canonical order of the 11 lowering passes in tessera-lower-to-apple_gpu-
# runtime.  This list is the source of truth — any reorder requires an
# explicit edit *and* the docstring on Passes.cpp must update to explain
# why.  Comments before each pass capture the ordering contract:
APPLE_GPU_CANONICAL_ORDER = [
    # One table owns all 2/3/4-op fusion rows and uses pattern benefit to
    # preserve longest-chain-first matching.
    "createLowerDeclarativeFusionsToAppleGPUPass",
    # ── Per-op matmul ────────────────────────────────────────────────
    "createLowerMatmulToAppleGPUPass",
    # ── Per-op attention family (each owns a distinct op name) ───────
    "createLowerRopeToAppleGPUPass",
    "createLowerFlashAttnToAppleGPUPass",
    "createLowerLinearAttnToAppleGPUPass",
    "createLowerAttnLocalWindow2DToAppleGPUPass",           # Sub-2
    # ── Per-op element-wise ──────────────────────────────────────────
    "createLowerSoftmaxToAppleGPUPass",
    "createLowerGeluToAppleGPUPass",
    # ── MPSGraph Tier-1 lane (2026-05-29): activations / SwiGLU gate /
    #    last-axis row ops. Each owns a distinct op name, so order among
    #    them is immaterial; they run after the fusions. ───────────────
    "createLowerUnaryToAppleGPUPass",
    "createLowerSiluMulToAppleGPUPass",
    "createLowerRowOpToAppleGPUPass",
]


# ─────────────────────────────────────────────────────────────────────────────
# Canonical order locked at source level
# ─────────────────────────────────────────────────────────────────────────────


def _extract_apple_gpu_runtime_passes() -> list[str]:
    """Return the list of `createLowerXxxPass` function names appearing
    in the `gAppleGPURuntimePipeline` body, in source order."""
    src = APPLE_PASSES_CPP.read_text()
    # Locate the gAppleGPURuntimePipeline lambda body.  Bounded by its
    # opening `[](OpPassManager &pm) {` and the closing `});`.
    anchor = src.find("gAppleGPURuntimePipeline")
    assert anchor != -1, "gAppleGPURuntimePipeline declaration missing"
    body_start = src.find("[](OpPassManager &pm)", anchor)
    assert body_start != -1
    body_end = src.find("});", body_start)
    body = src[body_start:body_end]
    # Each pm.addPass(createXxxPass()) appears on its own line.
    return re.findall(r"pm\.addPass\((\w+)\(\)\);", body)


def test_apple_gpu_runtime_pipeline_alias_is_documented() -> None:
    """The alias name itself is a public contract — Python tooling
    (matmul_pipeline.py dispatch, validate.sh, release_gate.py) calls
    it by name."""
    src = APPLE_PASSES_CPP.read_text()
    assert '"tessera-lower-to-apple_gpu-runtime"' in src


def test_apple_gpu_pipeline_has_exactly_eleven_passes() -> None:
    """Lock the count.  Adding another pass must update this file +
    APPLE_GPU_CANONICAL_ORDER + the architecture doc — a deliberate
    three-step change rather than a silent slip-in."""
    passes = _extract_apple_gpu_runtime_passes()
    assert len(passes) == len(APPLE_GPU_CANONICAL_ORDER), (
        f"expected {len(APPLE_GPU_CANONICAL_ORDER)} passes in apple_gpu-runtime "
        f"pipeline, found {len(passes)}: {passes}"
    )


def test_apple_gpu_canonical_order_matches_source() -> None:
    """The full sequence must byte-match APPLE_GPU_CANONICAL_ORDER."""
    passes = _extract_apple_gpu_runtime_passes()
    assert passes == APPLE_GPU_CANONICAL_ORDER, (
        "Apple GPU runtime pipeline source order drifted!\n"
        f"  expected: {APPLE_GPU_CANONICAL_ORDER}\n"
        f"  found   : {passes}\n"
        "If this is intentional, update APPLE_GPU_CANONICAL_ORDER and "
        "the 'longest-fusion-first' comments in Passes.cpp."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dependency-pair contracts — these are the bug classes pass-order
# matrices exist to catch.  Each pair says: pass A must run before pass
# B, or B will silently steal ops that A would have fused.
# ─────────────────────────────────────────────────────────────────────────────


# (declarative_fusion_pass, competing_per_op_pass, rationale)
APPLE_GPU_FUSION_DEPENDENCIES: list[tuple[str, str, str]] = [
    ("createLowerDeclarativeFusionsToAppleGPUPass",
     "createLowerMatmulToAppleGPUPass",
     "declarative fusion rows own their inner matmuls"),
    ("createLowerDeclarativeFusionsToAppleGPUPass",
     "createLowerSoftmaxToAppleGPUPass",
     "per-op softmax steals fusion candidate"),
    ("createLowerDeclarativeFusionsToAppleGPUPass",
     "createLowerGeluToAppleGPUPass",
     "per-op gelu steals fusion candidate"),
    ("createLowerDeclarativeFusionsToAppleGPUPass",
     "createLowerAttnLocalWindow2DToAppleGPUPass",
     "fusion table precedes the per-op 2D-window route"),
]


@pytest.mark.parametrize(
    "longer_pass,shorter_pass,rationale",
    APPLE_GPU_FUSION_DEPENDENCIES,
    ids=lambda p: p.replace("createLower", "").replace("ToAppleGPUPass", "")
                  if isinstance(p, str) else p,
)
def test_fusion_dependency_pair_locked_in_order(
    longer_pass: str, shorter_pass: str, rationale: str,
) -> None:
    """Each (longer_or_fused, shorter_or_per_op) pair must appear with
    longer_pass first in APPLE_GPU_CANONICAL_ORDER."""
    assert longer_pass in APPLE_GPU_CANONICAL_ORDER, (
        f"{longer_pass} not in canonical order — "
        f"add it before testing dependencies"
    )
    assert shorter_pass in APPLE_GPU_CANONICAL_ORDER, (
        f"{shorter_pass} not in canonical order — "
        f"add it before testing dependencies"
    )
    longer_idx = APPLE_GPU_CANONICAL_ORDER.index(longer_pass)
    shorter_idx = APPLE_GPU_CANONICAL_ORDER.index(shorter_pass)
    assert longer_idx < shorter_idx, (
        f"FUSION ORDERING VIOLATION:\n"
        f"  {longer_pass} (idx {longer_idx}) must precede\n"
        f"  {shorter_pass} (idx {shorter_idx})\n"
        f"  rationale: {rationale}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Group-level contracts — softer assertions about pass grouping.
# ─────────────────────────────────────────────────────────────────────────────


def test_all_fusion_passes_precede_all_per_op_passes() -> None:
    """The "longest fusion first" macro-contract: every fusion pass
    must appear before every per-op pass that owns a fusion target."""
    fusion_passes = [p for p in APPLE_GPU_CANONICAL_ORDER if "Fusion" in p]
    per_op_competing = [
        "createLowerMatmulToAppleGPUPass",
        "createLowerSoftmaxToAppleGPUPass",
        "createLowerGeluToAppleGPUPass",
    ]
    for f in fusion_passes:
        for op in per_op_competing:
            if op not in APPLE_GPU_CANONICAL_ORDER:
                continue
            assert (APPLE_GPU_CANONICAL_ORDER.index(f)
                    < APPLE_GPU_CANONICAL_ORDER.index(op)), (
                f"{f} must run before {op} per the longest-fusion-first contract"
            )


def test_passes_cpp_documents_longest_fusion_first_invariant() -> None:
    """The architectural invariant must be documented in source so a
    future reader knows *why* the order matters.  The phrase 'longest'
    is retained in the pipeline comment after collapsing seven shells."""
    text = APPLE_PASSES_CPP.read_text().lower()
    assert text.count("longest") >= 2, (
        f"expected >= 2 'longest' mentions documenting the fusion-first "
        f"contract; found {text.count('longest')}"
    )


def test_attn_local_window_2d_is_grouped_with_per_op_attention() -> None:
    """Sub-2's attn_local_window_2d lowering belongs with the per-op
    attention family (rope, flash_attn, linear_attn), not with the
    fusion passes.  This grouping is a structural sanity check that a
    future refactor won't move it into the fusion block by accident."""
    order = APPLE_GPU_CANONICAL_ORDER
    awl_idx = order.index("createLowerAttnLocalWindow2DToAppleGPUPass")
    rope_idx = order.index("createLowerRopeToAppleGPUPass")
    flash_idx = order.index("createLowerFlashAttnToAppleGPUPass")
    linear_idx = order.index("createLowerLinearAttnToAppleGPUPass")
    softmax_idx = order.index("createLowerSoftmaxToAppleGPUPass")
    # The attention family is contiguous; awl is inside it.
    assert rope_idx < awl_idx < softmax_idx
    assert flash_idx < awl_idx
    assert linear_idx < awl_idx


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive enumeration — for ALL pairs (i, j) with i < j in the
# canonical order, assert there is no documented dependency requiring
# the reverse.  This is the "no missing dependency" check.
# ─────────────────────────────────────────────────────────────────────────────


def test_no_documented_dependency_is_in_reverse_order() -> None:
    """A safety check: if any documented dependency pair appears in
    reverse order in APPLE_GPU_CANONICAL_ORDER, the source has a real
    bug.  Re-runs all documented pairs as a single batch assertion so a
    future engineer adding a new dependency can drop it into
    APPLE_GPU_FUSION_DEPENDENCIES without worrying about ordering."""
    violations = []
    for longer, shorter, rationale in APPLE_GPU_FUSION_DEPENDENCIES:
        l_idx = APPLE_GPU_CANONICAL_ORDER.index(longer)
        s_idx = APPLE_GPU_CANONICAL_ORDER.index(shorter)
        if l_idx >= s_idx:
            violations.append((longer, shorter, rationale))
    assert not violations, (
        f"{len(violations)} fusion ordering violations:\n"
        + "\n".join(f"  {l} must precede {s} ({r})" for l, s, r in violations)
    )
