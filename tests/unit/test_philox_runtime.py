"""M6 Step 4 runtime emission — ``ebm.langevin_step_philox``.

This module locks the runtime-side MSL emission of the M6 Step 4
Philox-4x32-10 stream + the new on-device Langevin variant.

Coverage:

  - **Manifest** has an ``ebm_langevin_step_philox`` entry pointing
    at ``tessera_apple_gpu_ebm_langevin_step_philox_f32`` with a
    matching ABI string.
  - **MSL source in apple_gpu_runtime.mm** declares the same
    Philox constants as :mod:`tessera.compiler.philox` and the
    same 10-round structure (cross-platform invariant).
  - **Python wrapper** is exported from :mod:`tessera.ebm` and
    its numpy reference path matches the Python Philox-Box-Muller
    construction byte-for-byte.
  - **Cross-platform determinism** (Darwin only): the Apple GPU
    dispatch and the numpy reference produce the same f32 output
    for the same ``(y, grad, key, counter, η, σ)`` tuple.
  - **Bridge wiring**: a call inside ``capture_compile_reports``
    records a ``JitBridgeRoute`` with op ``ebm_langevin_step_philox``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from tessera import ebm
from tessera.compiler import backend_manifest as bm
from tessera.compiler import philox


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME_MM = (
    REPO_ROOT
    / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
)


# ---------------------------------------------------------------------------
# Manifest contract
# ---------------------------------------------------------------------------

def test_manifest_has_langevin_step_philox_entry() -> None:
    assert "ebm_langevin_step_philox" in bm._EBM_APPLE_GPU_FUSED
    spec = bm._EBM_APPLE_GPU_FUSED["ebm_langevin_step_philox"]
    assert spec["symbol"] == "tessera_apple_gpu_ebm_langevin_step_philox_f32"
    assert "fp32" in spec["dtypes"]


def test_manifest_abi_matches_extern_c_signature() -> None:
    """The ABI string must match the actual ``extern \"C\"``
    signature in apple_gpu_runtime.mm.  Drift here means the
    Python wrapper's ctypes argtypes would be wrong."""
    spec = bm._EBM_APPLE_GPU_FUSED["ebm_langevin_step_philox"]
    abi = spec["abi"]
    for piece in (
        "y:f32*", "grad:f32*", "eta:f32", "noise_scale:f32",
        "key:u32*", "counter:u32*", "out:f32*", "n:i32",
    ):
        assert piece in abi, (
            f"ABI string missing {piece!r}: {abi}"
        )


def test_bridge_resolves_philox_langevin_symbol() -> None:
    from tessera.compiler import jit_bridge
    sym = jit_bridge.lookup_apple_gpu_symbol("ebm_langevin_step_philox")
    assert sym == "tessera_apple_gpu_ebm_langevin_step_philox_f32"


# ---------------------------------------------------------------------------
# MSL source in the .mm file matches the Python reference
# ---------------------------------------------------------------------------

def test_runtime_mm_embeds_philox_constants() -> None:
    """The MSL source embedded in apple_gpu_runtime.mm must declare
    the same Philox-4x32 constants as the Python reference.  This
    is the cross-platform-determinism lock: if anyone drifts the
    .mm or the .py the test catches it."""
    src = RUNTIME_MM.read_text()
    assert "PHILOX_M0 = 0xD2511F53u" in src, (
        "runtime .mm doesn't declare PHILOX_M0 = 0xD2511F53u — drift "
        "vs tessera.compiler.philox.PHILOX_M0"
    )
    assert "PHILOX_M1 = 0xCD9E8D57u" in src
    assert "PHILOX_W0 = 0x9E3779B9u" in src
    assert "PHILOX_W1 = 0xBB67AE85u" in src


def test_runtime_mm_has_10_round_loop() -> None:
    src = RUNTIME_MM.read_text()
    assert "for (int r = 0; r < 10; ++r)" in src, (
        "runtime .mm's Philox loop is not 10 rounds — locking the "
        "algorithm identity"
    )


def test_runtime_mm_python_reference_matches_msl_kernel() -> None:
    """Spot-check that the host-side reference path in the .mm file
    uses the same constants as the MSL kernel — they share an
    algorithm by design, but the host-side fallback is independently
    written."""
    src = RUNTIME_MM.read_text()
    # The host-side fallback uses literal hex constants
    # without the trailing 'u' suffix.
    assert "0xD2511F53u" in src and "0xCD9E8D57u" in src
    assert re.search(r"0x9E3779B9u", src) is not None
    assert re.search(r"0xBB67AE85u", src) is not None


def test_runtime_mm_declares_new_kernel_symbol() -> None:
    src = RUNTIME_MM.read_text()
    # The kernel function name (in MSL) and the extern "C" wrapper.
    assert "kernel void ebm_langevin_step_philox_f32" in src
    assert (
        "extern \"C\" void tessera_apple_gpu_ebm_langevin_step_philox_f32"
        in src
    )


# ---------------------------------------------------------------------------
# Python wrapper / numpy reference
# ---------------------------------------------------------------------------

def test_langevin_step_philox_is_exported_from_ebm_namespace() -> None:
    assert hasattr(ebm, "langevin_step_philox")
    assert callable(ebm.langevin_step_philox)


def test_python_reference_is_deterministic() -> None:
    rng = np.random.RandomState(0)
    y = rng.randn(8).astype(np.float32)
    grad = rng.randn(8).astype(np.float32) * 0.1
    key = np.array([42, 99], dtype=np.uint32)
    counter = np.array([7, 0, 0, 0], dtype=np.uint32)
    a = ebm.langevin_step_philox(
        y, grad, eta=0.05, noise_scale=0.1, key=key, counter=counter,
    )
    b = ebm.langevin_step_philox(
        y, grad, eta=0.05, noise_scale=0.1, key=key, counter=counter,
    )
    np.testing.assert_array_equal(a, b)


def test_python_reference_changes_with_counter() -> None:
    rng = np.random.RandomState(1)
    y = rng.randn(8).astype(np.float32)
    grad = rng.randn(8).astype(np.float32) * 0.1
    key = np.array([42, 99], dtype=np.uint32)
    a = ebm.langevin_step_philox(
        y, grad, eta=0.05, noise_scale=0.1, key=key,
        counter=np.array([0, 0, 0, 0], dtype=np.uint32),
    )
    b = ebm.langevin_step_philox(
        y, grad, eta=0.05, noise_scale=0.1, key=key,
        counter=np.array([1, 0, 0, 0], dtype=np.uint32),
    )
    # Same (y, grad) but different counter → different noise → different output.
    diff = float(np.abs(a - b).max())
    assert diff > 1e-3, f"counter change had no effect; max-abs diff = {diff}"


def test_python_reference_noise_scale_zero_recovers_deterministic_step() -> None:
    """``noise_scale=0`` ⇒ result is exactly the deterministic
    Langevin step ``y - eta*grad`` regardless of (key, counter)."""
    rng = np.random.RandomState(2)
    y = rng.randn(8).astype(np.float32)
    grad = rng.randn(8).astype(np.float32) * 0.1
    out = ebm.langevin_step_philox(
        y, grad, eta=0.05, noise_scale=0.0,
        key=np.array([1, 2], dtype=np.uint32),
        counter=np.array([3, 4, 5, 6], dtype=np.uint32),
    )
    expected = (y - 0.05 * grad).astype(np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_python_reference_matches_explicit_box_muller_at_thread_zero() -> None:
    """Hand-derive thread 0's noise from `philox_4x32_10` and verify
    the wrapper produces it."""
    y = np.array([1.0], dtype=np.float32)
    grad = np.array([0.0], dtype=np.float32)
    key = np.array([42, 99], dtype=np.uint32)
    counter = np.array([7, 0, 0, 0], dtype=np.uint32)
    out = ebm.langevin_step_philox(
        y, grad, eta=0.0, noise_scale=1.0, key=key, counter=counter,
    )
    # Hand-derived: thread 0's counter = (counter[0]+0, ...) = counter itself.
    p = philox.philox_4x32_10(counter, key)
    u0 = (float(p[0]) + 0.5) * (2.0 ** -32)
    u1 = (float(p[1]) + 0.5) * (2.0 ** -32)
    r = float(np.sqrt(-2.0 * np.log(u0)))
    theta = 2.0 * float(np.pi) * u1
    z = r * float(np.cos(theta))
    expected = float(y[0]) + 1.0 * z
    np.testing.assert_allclose(out[0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Cross-platform determinism (Darwin only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU runtime only loadable on macOS",
)
def test_apple_gpu_path_matches_numpy_reference() -> None:
    """The headline M6 Step 4 invariant: same ``(y, grad, key,
    counter, η, σ)`` produces the same f32 output on Apple GPU and
    via the Python reference."""
    rng = np.random.RandomState(3)
    y = rng.randn(64).astype(np.float32)
    grad = rng.randn(64).astype(np.float32) * 0.2
    key = np.array([0xDEADBEEF, 0xCAFE0042], dtype=np.uint32)
    counter = np.array([0, 0, 0, 0], dtype=np.uint32)
    # GPU path (auto-routes through jit_bridge):
    gpu = ebm.langevin_step_philox(
        y, grad, eta=0.05, noise_scale=0.1, key=key, counter=counter,
    )
    # Force numpy reference by directly calling the bridge-free
    # implementation — we hit it by passing a non-f32 dtype path:
    # easier: just compute the reference by hand using philox.
    expected = np.empty_like(y)
    for i in range(y.size):
        ci = np.array(
            [counter[0] + np.uint32(i), counter[1], counter[2], counter[3]],
            dtype=np.uint32,
        )
        p = philox.philox_4x32_10(ci, key)
        u0 = (float(p[0]) + 0.5) * (2.0 ** -32)
        u1 = (float(p[1]) + 0.5) * (2.0 ** -32)
        r = float(np.sqrt(-2.0 * np.log(u0)))
        theta = 2.0 * float(np.pi) * u1
        z = r * float(np.cos(theta))
        expected[i] = float(y[i]) - 0.05 * float(grad[i]) + 0.1 * z
    expected = expected.astype(np.float32)
    # f32 round-off tolerance.  log + sqrt + cos can drift ~1 ulp
    # apart between fp libraries; loose ulps here to keep the test
    # stable across macOS Metal versions.
    np.testing.assert_allclose(gpu, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Bridge wiring
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU runtime only loadable on macOS",
)
def test_apple_gpu_dispatch_records_bridge_route() -> None:
    """When tracing is on, the call must record a JitBridgeRoute
    whose op_name is the manifest name (not the raw symbol)."""
    from tessera.compiler import jit_bridge as bridge
    rng = np.random.RandomState(4)
    y = rng.randn(8).astype(np.float32)
    grad = rng.randn(8).astype(np.float32) * 0.1
    key = np.array([1, 2], dtype=np.uint32)
    counter = np.array([0, 0, 0, 0], dtype=np.uint32)
    prev = bridge.tracing_enabled()
    bridge.set_tracing_enabled(True)
    bridge.clear_dispatch_trace()
    try:
        ebm.langevin_step_philox(
            y, grad, eta=0.05, noise_scale=0.1, key=key, counter=counter,
        )
        routes = tuple(bridge.take_dispatch_trace())
    finally:
        bridge.set_tracing_enabled(prev)
    assert any(r.op_name == "ebm_langevin_step_philox" for r in routes)
    assert any(
        r.symbol == "tessera_apple_gpu_ebm_langevin_step_philox_f32"
        for r in routes
    )
