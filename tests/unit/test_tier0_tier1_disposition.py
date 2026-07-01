"""P0 of the S-series closure plan (docs/audit/roadmap/S_SERIES_GAP_CLOSURE_PLAN.md):
pin the honest `backend_kernel` disposition for the Tier-0 structural ops and the
Tier-1 transport ops, so future edits can't silently regress them.

Tier-0 sub-classes (plan §2 / §6):
  - 0-view  (pure metadata, zero data movement) → `not_applicable` — a device
    never runs a compute kernel for a reshape/squeeze/broadcast.
  - 0-move  (materialized data movement) → stays `partial`; earns a real
    memory-movement lane (plan §6.C). NOT not_applicable.
  - 0-reduce (indexed reduction) → stays `partial`; earns a real atomic kernel.

Tier-1 transport (collective + moe_transport) → stays `partial`: genuine
single-rank reference + mock-mesh execution, mesh-gated on real multi-GPU
hardware (Phase H). NOT not_applicable (a real, non-universal gap — the PR #132
distinction) and NOT complete (no real transport proof on a single-GPU box).
"""

from __future__ import annotations

import pytest

from tessera.compiler.primitive_coverage import coverage_for

# 0-view — genuinely zero-FLOP, zero-movement metadata.
TIER0_PURE_VIEW = (
    "reshape", "view", "squeeze", "unsqueeze", "flatten", "expand",
    "broadcast", "tile_view",
)

# 0-move / 0-reduce + transpose/permute — must stay open (a real lane is owed).
TIER0_MOVEMENT_STAYS_OPEN = (
    "transpose", "permute", "cat", "stack", "split", "chunk", "pad", "roll",
    "flip", "slice", "take", "index_select", "gather", "scatter", "scatter_add",
    "repeat", "tile",
)

TIER1_TRANSPORT = (
    "all_gather", "all_reduce", "all_to_all", "reduce_scatter",
    "moe_dispatch", "moe_combine",
)


@pytest.mark.parametrize("op", TIER0_PURE_VIEW)
def test_tier0_pure_view_backend_kernel_is_not_applicable(op):
    e = coverage_for(op)
    assert e.contract_status["backend_kernel"] == "not_applicable", (
        f"{op} is a pure-view metadata op (zero FLOP, zero data movement) — its "
        f"backend_kernel must be not_applicable, not "
        f"{e.contract_status['backend_kernel']!r}"
    )


def test_host_control_surfaces_have_no_backend_kernel_contract():
    from tessera.compiler.primitive_coverage import coverage_for

    for op in (
        "aot_export", "dataset_map", "custom_primitive", "scan",
        "karras_sigma_schedule", "chained_schedule", "save_state",
        "tree_map", "tokenizer_bpe", "vmap", "tiny_attention_conformance",
        "clip_grad_norm", "ema_update", "grad_scaler_step",
        "memory_write", "memory_evict",
    ):
        assert coverage_for(op).contract_status["backend_kernel"] == "not_applicable", op


def test_rng_key_bookkeeping_is_not_a_backend_kernel_contract():
    for op in ("rng_key", "rng_split", "rng_fold_in", "rng_clone"):
        assert coverage_for(op).contract_status["backend_kernel"] == "not_applicable", op


def test_tensor_rng_and_quantization_tails_stay_backend_owned_work():
    for op in (
        "rng_bernoulli", "rng_randint", "rng_gamma", "rng_poisson",
        "quantize_int8", "dequantize_int8", "quantize_int4", "dequantize_int4",
        "fake_quantize", "contrastive_loss", "ctc_loss",
    ):
        assert coverage_for(op).contract_status["backend_kernel"] != "not_applicable", op


@pytest.mark.parametrize("op", TIER0_MOVEMENT_STAYS_OPEN)
def test_tier0_movement_ops_stay_open(op):
    bk = coverage_for(op).contract_status["backend_kernel"]
    assert bk != "not_applicable", (
        f"{op} materializes data movement / reduction — it owes a real "
        f"memory-movement or atomic lane (plan §6.C), so backend_kernel must "
        f"NOT be not_applicable (that would hide a real gap — the PR #132 "
        f"mistake). Got {bk!r}."
    )


@pytest.mark.parametrize("op", TIER1_TRANSPORT)
def test_tier1_transport_stays_mesh_gated_partial(op):
    bk = coverage_for(op).contract_status["backend_kernel"]
    assert bk == "partial", (
        f"{op} is mesh-gated transport — it has real single-rank/mock-mesh "
        f"execution but proves on real multi-GPU hardware (Phase H). "
        f"backend_kernel must stay 'partial': never 'not_applicable' (it is a "
        f"real, non-universal gap) and never 'complete' without a real "
        f"multi-accelerator proof. Got {bk!r}."
    )
