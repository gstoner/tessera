"""Track-R (ReplaySSM) Phase 2 — the route-selection contract.

ReplaySSM decode is a *route-over-a-state-ABI* decision, not a kernel fork.
For each decode step the compiler picks one of two algebraically-equivalent
routes over the same `SSMStateHandle` (checkpoint state + replay ring buffer):

* ``output_only``      — reconstruct ``y_t`` from ``S0`` + buffer; write no state;
* ``state_and_output`` — materialize + write the new checkpoint state (*flush*),
  done only when the buffer is near-full.

This module is the **single source of truth** for that decision (the
``--replayssm-route`` flag as a contract): the flush rule lives in exactly one
place, :func:`select_route`, which ``SSMStateHandle.should_flush`` /
``route_for`` delegate to so the handle and the compiler can never diverge.

It also pins the **kernel taxonomy** (:data:`REPLAYSSM_KERNELS`) mirroring the
reference implementation's named kernels — Mamba-2
``selective_state_update_replayssm_{output_only,state_and_output,spec}`` and
Gated DeltaNet ``fused_recurrent_gated_delta_rule_replayssm`` /
``gdn_replayssm_spec_decode``.  Each entry carries an honest **status**: the
host numpy ``reference`` lane exists today (``SSMStateHandle``); the fused
Metal/CUDA/ROCm decode kernels that realize the state-traffic halving are
``planned`` (Track-R Phase 5 / Phase G–H). The Apple resident Mamba-2 route
ships both output-only replay and a deterministic checkpoint fold. Nothing here
registers a runtime op — a replay kernel only joins a backend runtime envelope
once it exists.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Routes ──────────────────────────────────────────────────────────────

#: Baseline: eager summary recurrence, full state write every token (no replay).
ROUTE_SUMMARY = "summary"
#: Reconstruct output from checkpoint + buffer; no state write (most steps).
ROUTE_OUTPUT_ONLY = "output_only"
#: Materialize + write the checkpoint state (flush) — buffer near-full.
ROUTE_STATE_AND_OUTPUT = "state_and_output"

#: The two replay routes a decode step chooses between (summary is the
#: non-replay baseline and is never *selected* by the flush rule).
REPLAY_ROUTES = (ROUTE_OUTPUT_ONLY, ROUTE_STATE_AND_OUTPUT)


def should_flush(count: int, capacity: int, spec_window: int = 0, n_new: int = 1) -> bool:
    """ReplaySSM flush rule: ``count + 2*spec_window + n_new > capacity``.

    Reserves room for ``2*spec_window`` so a speculative draft burst never
    truncates the buffer mid-stream.  Single source of truth — the handle and
    any backend lowering must consult this, not re-derive it.
    """
    return int(count) + 2 * int(spec_window) + int(n_new) > int(capacity)


def select_route(
    count: int, capacity: int, spec_window: int = 0, n_new: int = 1
) -> str:
    """Pick the decode route for appending ``n_new`` tokens to a buffer that
    currently holds ``count`` live replay tokens.

    Returns :data:`ROUTE_STATE_AND_OUTPUT` when the flush rule fires, else
    :data:`ROUTE_OUTPUT_ONLY`.
    """
    return (
        ROUTE_STATE_AND_OUTPUT
        if should_flush(count, capacity, spec_window, n_new)
        else ROUTE_OUTPUT_ONLY
    )


# ── Kernel taxonomy (mirrors the ReplaySSM reference implementation) ────

@dataclass(frozen=True)
class ReplayKernel:
    """A named replay kernel for one (family, route, backend).

    ``status`` is honest about reality:
      * ``reference`` — the host numpy ``SSMStateHandle`` lane (exists today);
      * ``fused``     — a real on-device kernel that ships now (the Metal block
        decode kernels — one dispatch for a whole block);
      * ``planned``   — a fused on-device kernel not yet built (e.g. the
        vLLM-named per-token spec kernels, or CUDA/ROCm — Phase G/H).
    """

    family: str            # "mamba2" | "gdn"
    route: str             # ROUTE_OUTPUT_ONLY | ROUTE_STATE_AND_OUTPUT | "spec"
    backend: str           # "reference" | "metal" | "cuda" | "rocm"
    symbol: str            # kernel/function name
    status: str            # "reference" | "planned"


#: The speculative-decode route (a spec burst over the ring buffer); rollback is
#: a cursor move (``SSMStateHandle.rollback`` / ``speculative.advance_ssm``).
ROUTE_SPEC = "spec"

#: Block decode — a whole block of tokens in ONE dispatch (prefill /
#: speculative verification / benchmark; the dispatch-overhead fix).
ROUTE_BLOCK = "block"

_MAMBA2_KERNELS = (
    ReplayKernel("mamba2", ROUTE_OUTPUT_ONLY, "reference",
                 "SSMStateHandle.read_output", "reference"),
    ReplayKernel("mamba2", ROUTE_STATE_AND_OUTPUT, "reference",
                 "SSMStateHandle.flush", "reference"),
    ReplayKernel("mamba2", ROUTE_SPEC, "reference",
                 "SSMStateHandle.rollback", "reference"),
    ReplayKernel("mamba2", ROUTE_OUTPUT_ONLY, "metal",
                 "selective_state_update_replayssm_output_only", "planned"),
    ReplayKernel("mamba2", ROUTE_STATE_AND_OUTPUT, "metal",
                 "selective_state_update_replayssm_state_and_output", "planned"),
    ReplayKernel("mamba2", ROUTE_SPEC, "metal",
                 "selective_state_update_replayssm_spec", "planned"),
    # Shipped: single-dispatch per-token and block decode kernels (Metal).
    ReplayKernel("mamba2", ROUTE_OUTPUT_ONLY, "metal_fused",
                 "tessera_apple_gpu_ssm_replay_decode_f32", "fused"),
    ReplayKernel("mamba2", ROUTE_BLOCK, "metal_fused",
                 "tessera_apple_gpu_ssm_block_decode_f32", "fused"),
    ReplayKernel("mamba2", ROUTE_BLOCK, "metal_fused_f16",
                 "tessera_apple_gpu_ssm_block_decode_f16", "fused"),
    ReplayKernel("mamba2", ROUTE_BLOCK, "metal_resident_ring",
                 "tessera_apple_gpu_ssm_replay_decode_dev_f32_enc", "fused"),
    ReplayKernel("mamba2", ROUTE_STATE_AND_OUTPUT, "metal_resident_ring",
                 "tessera_apple_gpu_ssm_replay_flush_dev_f32_enc", "fused"),
)

_GDN_KERNELS = (
    ReplayKernel("gdn", ROUTE_OUTPUT_ONLY, "reference",
                 "DeltaNetStateHandle.read_output", "reference"),
    ReplayKernel("gdn", ROUTE_STATE_AND_OUTPUT, "reference",
                 "DeltaNetStateHandle.flush", "reference"),
    ReplayKernel("gdn", ROUTE_SPEC, "reference",
                 "DeltaNetStateHandle.rollback", "reference"),
    ReplayKernel("gdn", ROUTE_OUTPUT_ONLY, "metal",
                 "fused_recurrent_gated_delta_rule_replayssm", "planned"),
    ReplayKernel("gdn", ROUTE_SPEC, "metal",
                 "gdn_replayssm_spec_decode", "planned"),
    # Shipped: single-dispatch block decode from checkpoint (Metal).
    ReplayKernel("gdn", ROUTE_BLOCK, "metal_fused",
                 "tessera_apple_gpu_gated_delta_rule_decode_f32", "fused"),
    ReplayKernel("gdn", ROUTE_BLOCK, "metal_fused_f16",
                 "tessera_apple_gpu_gated_delta_rule_decode_f16", "fused"),
    # Beyond the register envelope — (d_k,d_v) state in threadgroup memory.
    ReplayKernel("gdn", ROUTE_BLOCK, "metal_fused_big",
                 "tessera_apple_gpu_gated_delta_rule_decode_big_f32", "fused"),
)

#: All known replay kernels, keyed by ``(family, route, backend)``.
REPLAYSSM_KERNELS: dict[tuple[str, str, str], ReplayKernel] = {
    (k.family, k.route, k.backend): k for k in (_MAMBA2_KERNELS + _GDN_KERNELS)
}


def kernel_for(family: str, route: str, backend: str) -> ReplayKernel | None:
    """Look up the replay kernel for ``(family, route, backend)`` or ``None``."""
    return REPLAYSSM_KERNELS.get((family, route, backend))


def select_apple_serving_route(
    batch: int, num_channels: int, state_dim: int, tokens: int, *,
    dtype: str = "f32", timing_domain: str = "end_to_end",
    device: str | None = None,
) -> str:
    """Exact-row Apple serving choice from the retained paired corpus."""
    if min(batch, num_channels, state_dim, tokens) <= 0:
        raise ValueError("ReplaySSM serving geometry must be positive")
    from .apple_route_selector import production_route_for
    return production_route_for(
        op="resident_replay",
        shape=f"{batch}x{num_channels}x{state_dim}_t{tokens}",
        dtype=dtype, incumbent_route="fused_block", device=device,
        timing_domain=timing_domain)


__all__ = [
    "ROUTE_SUMMARY",
    "ROUTE_OUTPUT_ONLY",
    "ROUTE_STATE_AND_OUTPUT",
    "ROUTE_SPEC",
    "ROUTE_BLOCK",
    "REPLAY_ROUTES",
    "should_flush",
    "select_route",
    "ReplayKernel",
    "REPLAYSSM_KERNELS",
    "kernel_for",
    "select_apple_serving_route",
]
