"""Track-R (ReplaySSM) Phase 2 — route-selection contract tests.

Locks: (1) the flush rule ``count + 2*spec_window + n_new > capacity`` lives in
exactly one place (``compiler.ssm_replay``) and ``SSMStateHandle`` delegates to
it — no divergence; (2) the kernel taxonomy mirrors the reference
implementation's named kernels with honest statuses.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache import SSMStateHandle
from tessera.compiler import ssm_replay as R


# ── The flush rule ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "count,cap,spec,n,expect_flush",
    [
        (0, 8, 0, 1, False),     # 0+0+1 <= 8
        (7, 8, 0, 1, False),     # 7+0+1 == 8, not >
        (8, 8, 0, 1, True),      # 8+0+1 > 8
        (4, 8, 2, 1, True),      # 4+4+1 = 9 > 8
        (3, 8, 2, 1, False),     # 3+4+1 = 8, not >
        (0, 8, 0, 9, True),      # block append exceeds capacity
    ],
)
def test_should_flush_rule(count, cap, spec, n, expect_flush):
    assert R.should_flush(count, cap, spec, n) is expect_flush
    expected_route = R.ROUTE_STATE_AND_OUTPUT if expect_flush else R.ROUTE_OUTPUT_ONLY
    assert R.select_route(count, cap, spec, n) == expected_route


def test_routes_are_distinct():
    routes = {R.ROUTE_SUMMARY, R.ROUTE_OUTPUT_ONLY, R.ROUTE_STATE_AND_OUTPUT, R.ROUTE_SPEC}
    assert len(routes) == 4
    assert R.REPLAY_ROUTES == (R.ROUTE_OUTPUT_ONLY, R.ROUTE_STATE_AND_OUTPUT)


# ── Single source of truth: handle delegates to the contract ────────────

def test_handle_delegates_to_contract():
    rng = np.random.default_rng(0)
    for _ in range(50):
        cap = int(rng.integers(1, 16))
        spec = int(rng.integers(0, 4))
        count = int(rng.integers(0, cap + 1))
        n = int(rng.integers(1, 5))
        h = SSMStateHandle(batch=1, num_channels=2, state_dim=2,
                           a=np.array([-1.0, -1.0]), capacity=cap, spec_window=spec)
        for _ in range(count):
            h.append(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)))
        assert h.should_flush(n) == R.should_flush(count, cap, spec, n)
        assert h.route_for(n) == R.select_route(count, cap, spec, n)


# ── Kernel taxonomy ─────────────────────────────────────────────────────

def test_mamba2_kernel_taxonomy():
    # Reference lane exists today; Metal lane is planned.
    ref_oo = R.kernel_for("mamba2", R.ROUTE_OUTPUT_ONLY, "reference")
    assert ref_oo is not None and ref_oo.status == "reference"
    for route in (R.ROUTE_OUTPUT_ONLY, R.ROUTE_STATE_AND_OUTPUT, R.ROUTE_SPEC):
        k = R.kernel_for("mamba2", route, "metal")
        assert k is not None and k.status == "planned"
    # The vLLM-mirrored symbol names are pinned.
    assert (R.kernel_for("mamba2", R.ROUTE_OUTPUT_ONLY, "metal").symbol
            == "selective_state_update_replayssm_output_only")
    assert (R.kernel_for("mamba2", R.ROUTE_STATE_AND_OUTPUT, "metal").symbol
            == "selective_state_update_replayssm_state_and_output")
    assert (R.kernel_for("mamba2", R.ROUTE_SPEC, "metal").symbol
            == "selective_state_update_replayssm_spec")


def test_gdn_kernel_taxonomy():
    assert (R.kernel_for("gdn", R.ROUTE_OUTPUT_ONLY, "metal").symbol
            == "fused_recurrent_gated_delta_rule_replayssm")
    assert (R.kernel_for("gdn", R.ROUTE_SPEC, "metal").symbol
            == "gdn_replayssm_spec_decode")
    assert R.kernel_for("gdn", R.ROUTE_OUTPUT_ONLY, "metal").status == "planned"


def test_shipped_fused_block_kernels():
    """The single-dispatch block decode kernels ship now (status 'fused')."""
    mamba = R.kernel_for("mamba2", R.ROUTE_BLOCK, "metal_fused")
    gdn = R.kernel_for("gdn", R.ROUTE_BLOCK, "metal_fused")
    assert mamba is not None and mamba.status == "fused"
    assert mamba.symbol == "tessera_apple_gpu_ssm_block_decode_f32"
    assert gdn is not None and gdn.status == "fused"
    assert gdn.symbol == "tessera_apple_gpu_gated_delta_rule_decode_f32"


def test_no_runtime_op_claimed_for_replay_decode():
    """The handle-side decode kernels (C ABI symbols) must NOT be in the Apple
    GPU runtime *envelope* (which is graph-IR op dispatch) — they are
    handle-side, not metal_runtime graph ops (Decision #27)."""
    from tessera.compiler.apple_gpu_envelope import _APPLE_GPU_RUNTIME_OPS
    handle_kernels = {k.symbol for k in R.REPLAYSSM_KERNELS.values()
                      if k.backend.startswith("metal")}
    assert handle_kernels.isdisjoint(_APPLE_GPU_RUNTIME_OPS)
    # The prefill scan stays runtime-registered as before.
    assert "tessera.selective_ssm" in _APPLE_GPU_RUNTIME_OPS


def test_kernel_for_unknown_returns_none():
    assert R.kernel_for("mamba2", R.ROUTE_OUTPUT_ONLY, "tpu") is None
    assert R.kernel_for("nope", R.ROUTE_SPEC, "metal") is None
