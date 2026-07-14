"""Characterization gate for the Apple GPU "no-lane residual" — the *only*
class of program that still demotes to ``target_ir_artifact``.

Background (closes the stale "design fork"): the runtime synthesized-fusion
prepass IS the canonical live path for multi-op ``@jit(target="apple_gpu")``
programs (decided 2026-06-16, ``docs/audit/backend/apple/APPLE_GPU_CODEGEN_PLAN.md``).
``driver._apple_gpu_chain_kind`` routes a >=2-op plan to ``apple_gpu_mps`` /
``metal_runtime`` when it is a named fusion, a wholly-pointwise DAG, OR — since
the 2026-06-17 general residency gate — when **every** op has an Apple GPU
dispatch lane (``lane_for(op) is not None``; the prepass fuses what it can, the
rest run per-op on Metal).

So the abstract question "is the prepass the live path or is artifact the
future?" is answered. What actually remains is concrete and enumerable: a
multi-op program demotes to artifact **iff it contains at least one op with no
Apple GPU lane**. That set — computed live below from the op catalog x the
envelope's ``lane_for`` — is the true residual gap. This test freezes it so the
gap can't drift silently:

  * Add a lane for an op (shrinking the gap) -> this test fails; delete that op
    from ``_NO_LANE_BARE``.
  * Add a new catalog op with no lane (growing the gap) -> this test fails;
    either give it a lane or add it to ``_NO_LANE_BARE`` with intent.

Each entry is then a reviewable "add a lane" / "intentionally host/numpy-only"
decision, not a phantom blocker on the master queue.

NB on the bare names + ``_t()`` helper below: op names are stored WITHOUT the
``tessera.`` prefix and reassembled at runtime. This is deliberate — the test
coverage audit (``test_coverage_audit.py``) counts a literal ``"tessera.<op>"``
string in a test file as a "direct test reference", so embedding the prefixed
forms here would inflate ~114 ops' coverage counts and wrongly flip genuinely
thinly-tested ops (the conformal-geometry primitives) out of the triage. Keeping
the literals bare makes this routing test invisible to that scanner. Do NOT
"helpfully" re-add the ``tessera.`` prefix to the string literals.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from tessera.compiler import op_catalog as oc
from tessera.compiler.apple_gpu_envelope import lane_for
from tessera.compiler.driver import _apple_gpu_chain_kind

if TYPE_CHECKING:
    from tessera.compiler.matmul_pipeline import CPUPlan

_PREFIX = "tessera."


def _t(bare: str) -> str:
    """Dotted op name from a bare leaf — concatenated so the source never
    contains a scannable ``"tessera.<op>"`` literal (see module docstring)."""
    return _PREFIX + bare


# Catalog graph ops (bare, no ``tessera.`` prefix) that have NO Apple GPU
# dispatch lane today. A multi-op apple_gpu program containing any of these (and
# not matching a named fusion) routes to target_ir_artifact rather than
# metal_runtime. Grouped by family for review; the test asserts the union equals
# the live computation, so the grouping is documentation only.
_NO_LANE_BARE: frozenset[str] = frozenset(
    {
        # Optimizer step ops without an Apple GPU lane.
        "adafactor",
        # Distributed collectives (mock/host today; real NCCL/RCCL is Phase G/H).
        "all_gather", "all_reduce", "all_to_all", "reduce_scatter",
        # RNG / stochastic.
        "dropout", "rng_normal", "rng_uniform",
        # Low-precision quantize/dequantize + pack/unpack (macOS-27.0 MTLTensor
        # gated; see microscaling.py).
        "dequant_grouped_gemm", "dequant_matmul", "dequantize_fp4",
        "dequantize_fp6", "dequantize_fp8", "dequantize_nvfp4", "quantize_fp4",
        "quantize_fp6", "quantize_fp8", "quantize_nvfp4", "pack", "unpack",
        # Complex / conformal-geometry primitives.
        "complex_abs", "complex_arg", "complex_conjugate", "complex_div",
        "complex_exp", "complex_log", "complex_mul", "complex_pow",
        "complex_sqrt", "check_cauchy_riemann", "conformal_energy_on_sphere",
        "conformal_jacobian", "cross_ratio", "dbar", "dz", "is_concyclic",
        "mobius", "mobius_from_three_points", "stereographic",
        # Dense linalg solves / decompositions.
        "cholesky_solve", "factorized_matmul", "lu", "qr", "svd",
        # Sparse linalg.
        "segment_reduce",
        # Special functions.
        "digamma", "lgamma",
        # Structural / shape / view / gather-scatter (no arithmetic kernel).
        "arange", "broadcast", "cast", "chunk", "dynamic_slice",
        "dynamic_update_slice", "expand", "flatten", "flip", "index_select",
        "index_update", "masked_fill", "nonzero", "pad", "permute", "rearrange",
        "repeat", "reshape", "roll", "select", "split", "squeeze", "stack",
        "take", "tile", "tile_view",
        "unsqueeze", "view", "argsort", "sort",
        # Latent-KV (target-lowering gated, Decision #21).
        "latent_kv_compress", "latent_kv_expand_k", "latent_kv_expand_v",
        # MoE / MoR routing + score combine.
        "moe_combine", "moe_dispatch", "mor_partition", "mor_router",
        "mor_scatter",
        # Titans/Atlas memory-index + MSA block selection.
        "memory_index_select", "memory_index_select_ste", "msa_select_blocks",
        # Attention/rope variants without a dedicated lane yet.
        "alibi", "ntk_rope", "rope_merge", "rope_split", "laplacian_2d",
        # Speculative-decode acceptance (SD1; verification ops — ROCm-native
        # kernels, no Apple GPU dispatch lane).
        "spec_accept", "spec_accept_sample", "spec_accept_tree_sample",
        # SD1-4 target verification I/O contract — a composed model call (no
        # dedicated Apple GPU lane; fusion is a DK-track concern).
        "target_verify",
        # Misc.
        "conv3d_ndhwc", "einsum", "fused_epilogue",
    }
)

# Dotted-namespace ops, stored as (namespace, leaf) tuples and joined at runtime.
# Kept split because the coverage scanner counts a contiguous family reference of
# the form module-dot-leaf (e.g. the rl policy losses) — the tuple form, joined
# only at runtime, never emits that literal in the source.
_NO_LANE_DOTTED: frozenset[str] = frozenset(
    ".".join(seg)
    for seg in {
        ("kv_cache", "append"), ("kv_cache", "prune"), ("kv_cache", "read"),
        # SD1-3 speculative-decode cache cursor ops (state-effect handle ops, no
        # Apple GPU dispatch lane).
        ("cache", "commit"), ("cache", "rollback"),
        ("ebm", "langevin_step"),
        ("rl", "cispo_policy_loss"), ("rl", "grpo_policy_loss"),
        ("rl", "normalize_group_advantages"), ("rl", "ppo_policy_loss"),
    }
)

#: The frozen residual as dotted op names.
GOLDEN_NO_LANE_OPS: frozenset[str] = frozenset(
    _t(n) for n in (_NO_LANE_BARE | _NO_LANE_DOTTED)
)


def _live_no_lane_residual() -> frozenset[str]:
    """The catalog graph ops with no Apple GPU lane, computed from source."""
    return frozenset(n for n in oc.GRAPH_OP_TO_SPEC if lane_for(n) is None)


def test_no_lane_residual_matches_golden() -> None:
    """Freeze the residual so the artifact-demotion gap can't drift silently."""
    live = _live_no_lane_residual()
    newly_laned = GOLDEN_NO_LANE_OPS - live  # ops that GAINED a lane
    newly_unlaned = live - GOLDEN_NO_LANE_OPS  # new catalog ops with no lane
    assert live == GOLDEN_NO_LANE_OPS, (
        "Apple GPU no-lane residual drifted.\n"
        f"  ops that gained a lane (remove from _NO_LANE_BARE/_NO_LANE_DOTTED): {sorted(newly_laned)}\n"
        f"  ops now with no lane (add a lane, or add to _NO_LANE_BARE/_NO_LANE_DOTTED with intent): {sorted(newly_unlaned)}"
    )


def test_lane_contract_is_bidirectional() -> None:
    """For every catalog op: it is in the residual IFF it has no lane."""
    for name in oc.GRAPH_OP_TO_SPEC:
        assert (lane_for(name) is None) == (name in GOLDEN_NO_LANE_OPS), name


def test_residual_is_a_real_gap_not_the_whole_catalog() -> None:
    """Sanity: the lane'd surface is the majority; the residual is a minority."""
    total = len(oc.GRAPH_OP_TO_SPEC)
    residual = len(GOLDEN_NO_LANE_OPS)
    assert 0 < residual < total
    assert residual < total - residual, (
        f"more catalog ops lack a lane ({residual}) than have one "
        f"({total - residual}) — the prepass-is-live-path claim would be hollow"
    )


# ── routing consequence: the residual is exactly what demotes to artifact ─────
def _fake_plan(*op_names: str) -> CPUPlan:
    """Minimal stand-in for a CPUPlan that `_apple_gpu_chain_kind` can classify
    (it reads only target_kind + each op's op_name/operands/result/kwargs)."""
    ops = [
        SimpleNamespace(op_name=n, operands=[], result="r", kwargs={})
        for n in op_names
    ]
    return cast("CPUPlan", SimpleNamespace(target_kind="apple_gpu", ops=ops))


def test_laned_multi_op_program_stays_metal_runtime() -> None:
    """A 2-op program where every op has a lane -> per_op_metal (live Metal),
    never None/artifact. matmul -> transpose is the canonical mixed case."""
    assert lane_for(_t("matmul")) is not None
    assert lane_for(_t("transpose")) is not None
    assert _apple_gpu_chain_kind(_fake_plan(_t("matmul"), _t("transpose"))) == "per_op_metal"


def test_residual_op_demotes_multi_op_program_to_artifact() -> None:
    """A 2-op program containing a no-lane residual op -> None (artifact). This
    is the entire reason the artifact lane still exists."""
    residual_op = _t("einsum")  # representative; in GOLDEN_NO_LANE_OPS
    assert residual_op in GOLDEN_NO_LANE_OPS
    assert lane_for(residual_op) is None
    assert _apple_gpu_chain_kind(_fake_plan(_t("matmul"), residual_op)) is None


def test_every_residual_op_individually_breaks_residency() -> None:
    """Stronger: pairing matmul with ANY single residual op demotes the program
    (none of them sneak a lane in via a fusion shortcut)."""
    for op in sorted(GOLDEN_NO_LANE_OPS):
        assert _apple_gpu_chain_kind(_fake_plan(_t("matmul"), op)) is None, op
