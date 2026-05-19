"""M7 follow-up — MSL kernels for the conformal-primitive surface.

Coverage:

  - **Manifest contract**: every shipped complex op has an
    ``apple_gpu`` entry with the expected symbol + ABI.
  - **Bridge routes**: ``jit_bridge.lookup_apple_gpu_symbol``
    resolves each op to its C ABI symbol.
  - **MSL source in apple_gpu_runtime.mm**: each kernel function
    is declared with the expected signature.
  - **Cross-platform determinism** (Darwin): Apple GPU output
    matches the numpy reference for each shipped op.
  - **Conformal-jacobian / laplacian-2d are intentionally CPU-only**
    — the manifest has no apple_gpu entry; lookup returns None.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tessera import complex as tc
from tessera.compiler import backend_manifest as bm
from tessera.compiler import jit_bridge


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME_MM = (
    REPO_ROOT
    / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
)


# ---------------------------------------------------------------------------
# Manifest contract — the 4 shipped MSL kernels are exactly what M7's
# follow-up promised; the 2 host-only ops are explicitly NOT here.
# ---------------------------------------------------------------------------

_SHIPPED_COMPLEX_OPS = (
    "complex_mul",
    "complex_exp",
    "complex_stereographic",
    "complex_mobius",
)


_HOST_ONLY_COMPLEX_OPS = (
    "complex_conjugate",
    "complex_abs",
    "conformal_jacobian",
    "laplacian_2d",
)


@pytest.mark.parametrize("op", _SHIPPED_COMPLEX_OPS)
def test_manifest_has_complex_op_with_apple_gpu_entry(op) -> None:
    """Each shipped complex op resolves through the bridge."""
    sym = jit_bridge.lookup_apple_gpu_symbol(op)
    assert sym is not None, f"{op} has no apple_gpu fast path"
    assert sym.startswith("tessera_apple_gpu_")
    assert sym.endswith("_f32")


@pytest.mark.parametrize("op", _HOST_ONLY_COMPLEX_OPS)
def test_host_only_complex_ops_have_no_apple_gpu_entry(op) -> None:
    """``complex_conjugate`` is one float negation, ``complex_abs``
    is sqrt(re²+im²), ``conformal_jacobian`` is 4 host calls, and
    ``laplacian_2d`` is a tiny stencil — none benefit from GPU at
    our typical sizes.  The manifest documents this honestly."""
    sym = jit_bridge.lookup_apple_gpu_symbol(op)
    assert sym is None, (
        f"{op} resolved to {sym!r} but should be CPU-only by design"
    )


def test_complex_manifest_routes_through_dedicated_table() -> None:
    """The `manifest_for("complex_*")` lookup goes through
    `complex_manifest_for`, not the generic OP_SPECS path."""
    entries = bm.manifest_for("complex_mul")
    targets = {e.target for e in entries}
    assert "apple_gpu" in targets
    assert "x86" in targets
    assert "apple_cpu" in targets


# ---------------------------------------------------------------------------
# MSL source in the .mm file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel_name", [
    "complex_mul_f32",
    "complex_exp_f32",
    "complex_stereographic_f32",
    "complex_mobius_f32",
])
def test_runtime_mm_declares_kernel(kernel_name) -> None:
    src = RUNTIME_MM.read_text()
    assert f"kernel void {kernel_name}" in src, (
        f"{kernel_name} kernel not found in apple_gpu_runtime.mm"
    )


@pytest.mark.parametrize("symbol", [
    "tessera_apple_gpu_complex_mul_f32",
    "tessera_apple_gpu_complex_exp_f32",
    "tessera_apple_gpu_complex_stereographic_f32",
    "tessera_apple_gpu_complex_mobius_f32",
])
def test_runtime_mm_declares_extern_wrapper(symbol) -> None:
    src = RUNTIME_MM.read_text()
    assert f"extern \"C\" void {symbol}" in src


# ---------------------------------------------------------------------------
# Cross-platform determinism (Darwin)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU runtime only loadable on macOS",
)
def test_complex_mul_apple_gpu_matches_reference() -> None:
    rng = np.random.RandomState(0)
    a = (rng.randn(64) + 1j * rng.randn(64)).astype(np.complex64)
    b = (rng.randn(64) + 1j * rng.randn(64)).astype(np.complex64)
    out = tc.complex_mul(tc.from_numpy(a), tc.from_numpy(b)).to_numpy()
    expected = (a * b).astype(np.complex128)
    np.testing.assert_allclose(out, expected, atol=1e-5)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU runtime only loadable on macOS",
)
def test_complex_exp_apple_gpu_matches_reference() -> None:
    rng = np.random.RandomState(1)
    z = (rng.randn(64) + 1j * rng.randn(64)).astype(np.complex64)
    out = tc.complex_exp(tc.from_numpy(z)).to_numpy()
    expected = np.exp(z).astype(np.complex128)
    np.testing.assert_allclose(out, expected, atol=1e-4)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU runtime only loadable on macOS",
)
def test_complex_stereographic_apple_gpu_matches_reference() -> None:
    """Sample points on the unit sphere, project via the GPU
    kernel, compare to the numpy reference."""
    rng = np.random.RandomState(2)
    pts = rng.randn(32, 3).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=-1, keepdims=True)
    # Avoid the north pole for stability.
    pts = pts[pts[:, 2] < 0.9]
    out = tc.stereographic(pts).to_numpy()
    # Numpy reference.
    denom = 1.0 - pts[:, 2]
    expected_re = pts[:, 0] / denom
    expected_im = pts[:, 1] / denom
    expected = expected_re + 1j * expected_im
    np.testing.assert_allclose(out, expected, atol=1e-4)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU runtime only loadable on macOS",
)
def test_complex_mobius_apple_gpu_matches_reference() -> None:
    """f(z; 1, 2, 0, 3) = (z + 2) / 3 — applied to a batch of
    f32 inputs."""
    rng = np.random.RandomState(3)
    z = (rng.randn(64) + 1j * rng.randn(64)).astype(np.complex64)
    out = tc.mobius(
        tc.from_numpy(z), a=1.0, b=2.0, c=0.0, d=3.0,
    ).to_numpy()
    expected = ((z.astype(np.complex128) + 2.0) / 3.0)
    np.testing.assert_allclose(out, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Bridge wiring (Darwin) — auto-emit captures the route
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU runtime only loadable on macOS",
)
def test_complex_mul_records_bridge_route() -> None:
    from tessera.compiler import jit_bridge as bridge
    rng = np.random.RandomState(4)
    a = (rng.randn(8) + 1j * rng.randn(8)).astype(np.complex64)
    b = (rng.randn(8) + 1j * rng.randn(8)).astype(np.complex64)
    prev = bridge.tracing_enabled()
    bridge.set_tracing_enabled(True)
    bridge.clear_dispatch_trace()
    try:
        tc.complex_mul(tc.from_numpy(a), tc.from_numpy(b))
        routes = tuple(bridge.take_dispatch_trace())
    finally:
        bridge.set_tracing_enabled(prev)
    assert any(r.op_name == "complex_mul" for r in routes)
    assert any(
        r.symbol == "tessera_apple_gpu_complex_mul_f32" for r in routes
    )


# ---------------------------------------------------------------------------
# Numpy fallback on non-Darwin / non-f32 inputs
# ---------------------------------------------------------------------------

def test_complex_mul_falls_back_to_numpy_for_non_f32() -> None:
    """fp64 inputs route to the numpy path (no f32 GPU kernel for
    them); the result should still be correct."""
    a = np.array([1.0 + 2j], dtype=np.complex128)
    b = np.array([3.0 + 4j], dtype=np.complex128)
    out = tc.complex_mul(tc.from_numpy(a), tc.from_numpy(b)).to_numpy()
    np.testing.assert_allclose(out, a * b, atol=1e-12)
