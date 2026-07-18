"""Track-R (ReplaySSM) Phase 5 — fused single-dispatch SSM decode kernel.

``tessera_apple_gpu_ssm_replay_decode_f32`` reconstructs the output-only decode
step in one Metal dispatch, keeping the checkpoint state ``S0`` resident and
reading only the small replay inputs (the state-traffic-halving kernel).  On
Darwin it runs the MSL kernel; on non-Metal hosts the C ABI symbol runs its own
host reference — either way these tests validate numerically against the eager
``selective_ssm`` reference (f32 precision) and the numpy reference handle.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera import runtime as rt
from tessera.cache import SSMStateHandle


def _decode(handle, dt, x, Bp, Cp, gate=None):
    B, S, D = x.shape
    y = np.zeros((B, S, D))
    for t in range(S):
        gt = None if gate is None else gate[:, t, :]
        y[:, t, :] = handle.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :], gate_t=gt)
    return y


def _inputs(rng, B, S, D, N):
    x = rng.standard_normal((B, S, D))
    a = -np.abs(rng.standard_normal(D))
    Bp = rng.standard_normal((B, S, N))
    Cp = rng.standard_normal((B, S, N))
    dt = np.abs(rng.standard_normal((B, S, D))) * 0.5
    return x, a, Bp, Cp, dt


def test_fused_factory_wires_decode_fn():
    h = rt.apple_gpu_fused_ssm_state_handle(1, 4, 3, -np.ones(4), capacity=8)
    assert isinstance(h, SSMStateHandle)
    assert h.backend == "apple_gpu_fused"
    assert h.decode_fn is not None


@pytest.mark.hardware_apple_gpu
def test_fused_decode_reports_native_gpu_and_matches_eager():
    """A callable ABI is insufficient: this asserts the actual MSL dispatch."""
    rng = np.random.default_rng(37)
    B, S, D, N = 1, 6, 4, 3
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt))
    h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=16)
    out = _decode(h, dt, x, Bp, Cp)

    assert h.last_decode_execution == "native_gpu"
    assert np.max(np.abs(out - eager)) < 5e-4


def test_fused_decode_forced_missing_binding_is_reference_and_correct(monkeypatch):
    """A missing fused binding must retain reference provenance."""
    monkeypatch.setattr(rt, "apple_gpu_ssm_decode_callable", lambda: lambda *_: None)
    rng = np.random.default_rng(38)
    B, S, D, N = 1, 6, 4, 3
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt))
    h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=16)
    out = _decode(h, dt, x, Bp, Cp)

    assert h.last_decode_execution == "reference_cpu"
    assert np.max(np.abs(out - eager)) < 5e-4


@pytest.mark.parametrize("B,S,D,N,cap,spec", [
    (1, 20, 4, 3, 64, 0),
    (2, 24, 8, 4, 6, 2),
    (3, 18, 6, 5, 9, 1),
])
def test_fused_decode_matches_eager_f32(B, S, D, N, cap, spec):
    rng = np.random.default_rng(B * 17 + S + D + N)
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    y_eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt))
    h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=cap, spec_window=spec)
    y = _decode(h, dt, x, Bp, Cp)
    assert np.max(np.abs(y - y_eager)) < 5e-4


def test_fused_matches_reference_handle():
    rng = np.random.default_rng(2)
    B, S, D, N = 2, 16, 6, 4
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    ref = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a, capacity=5)
    fused = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=5)
    y_ref = _decode(ref, dt, x, Bp, Cp)
    y_fused = _decode(fused, dt, x, Bp, Cp)
    assert np.max(np.abs(y_ref - y_fused)) < 5e-4


def test_fused_applies_output_gate():
    rng = np.random.default_rng(8)
    B, S, D, N = 1, 12, 5, 3
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    gate = np.abs(rng.standard_normal((B, S, D)))
    y_eager = np.asarray(tessera.ops.selective_ssm(x, a, Bp, Cp, dt, gate=gate))
    h = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=4, spec_window=1)
    y = _decode(h, dt, x, Bp, Cp, gate=gate)
    assert np.max(np.abs(y - y_eager)) < 5e-4


def test_fused_speculative_rollback_still_exact():
    """The fused decode path must compose with speculative rollback (Phase 3):
    appending drafts then rolling back leaves identical continuations."""
    from tessera.speculative import advance_ssm
    rng = np.random.default_rng(4)
    B, S, D, N = 1, 8, 4, 3
    x, a, Bp, Cp, dt = _inputs(rng, B, S, D, N)
    base = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=64, spec_window=4)
    for t in range(3):
        base.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
    spec = base.clone()
    for _ in range(3):  # 3 drafts, all rejected
        spec.step(rng.standard_normal((B, D)), rng.standard_normal((B, D)),
                  rng.standard_normal((B, N)), rng.standard_normal((B, N)))
    advance_ssm(spec, 0, num_drafts=3)
    for t in range(3, S):
        yb = base.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
        ys = spec.step(dt[:, t, :], x[:, t, :], Bp[:, t, :], Cp[:, t, :])
        assert np.max(np.abs(yb - ys)) < 1e-5


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("compute_dtype,tolerance", [("fp32", 5e-4), ("fp16", 3e-2)])
def test_block_decode_reports_native_resources_and_matches_reference(
        compute_dtype, tolerance):
    from tessera._apple_gpu_dispatch import (
        clear_dispatch_telemetry,
        read_dispatch_telemetry,
        set_dispatch_telemetry_enabled,
    )
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    rng = np.random.default_rng(1907)
    B, T, D, N = 2, 7, 8, 5
    x = rng.standard_normal((T, B, D))
    a = -np.abs(rng.standard_normal(D))
    bp = rng.standard_normal((T, B, N))
    cp = rng.standard_normal((T, B, N))
    dt = np.abs(rng.standard_normal((T, B, D))) * 0.2
    reference = SSMStateHandle(B, D, N, a).decode_block(dt, x, bp, cp)
    handle = rt.apple_gpu_fused_ssm_state_handle(
        B, D, N, a, capacity=16, compute_dtype=compute_dtype)
    try:
        assert set_dispatch_telemetry_enabled(True)
        clear_dispatch_telemetry()
        output = handle.decode_block(dt, x, bp, cp)
        record = read_dispatch_telemetry()
        assert handle.last_block_execution == "native_gpu"
        assert record["device_time_ns"] > 0
        assert record["resources"]["threadgroup"] == [B * D, 1, 1]
        assert record["resources"]["thread_execution_width"] > 0
        assert np.max(np.abs(output - reference)) < tolerance
    finally:
        set_dispatch_telemetry_enabled(False)


def test_block_decode_out_of_native_envelope_is_explicit_reference():
    """N>256 must not retain a native label through the legacy fallback ABI."""
    B, T, D, N = 1, 2, 2, 257
    a = -np.ones(D)
    dt = np.full((T, B, D), 0.1)
    x = np.ones((T, B, D))
    bp = np.ones((T, B, N))
    cp = np.ones((T, B, N))
    reference = SSMStateHandle(B, D, N, a).decode_block(dt, x, bp, cp)
    handle = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, compute_dtype="fp32")
    output = handle.decode_block(dt, x, bp, cp)
    assert handle.last_block_execution == "reference_cpu"
    np.testing.assert_allclose(output, reference, rtol=0, atol=0)
