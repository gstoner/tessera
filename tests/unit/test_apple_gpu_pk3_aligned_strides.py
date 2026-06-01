"""Phase 2 stride-alignment wire-up — PK3 prepare_tensors opt-in.

The ``tessera_apple_gpu_row_major_strides_aligned`` helper has been
available since Project-2 but only used inside structural tests.
Phase 2 wires it through the PK3 ``prepare_tensors`` path: callers
can opt in via ``Pipeline.set_aligned_strides(True)`` and the
runtime sets ``MTLTensorDescriptor.strides`` explicitly to honor
Apple's 64-byte / 128-byte alignment rules.

Default (flag off) is fully backward-compatible: Metal uses its
implicit strides (the existing PK3 behavior). Tests:

* **Symbol availability** — new C ABI symbol binds.
* **Default off** — pipelines built without the explicit setter
  call behave identically to the existing PK3 path (no perf or
  numerical change).
* **Opt-in path** — ``set_aligned_strides(True)`` then
  ``prepare_tensors`` then ``dispatch`` produces numerically
  identical output to the dense path on Apple's bundled sample
  matmul (the 4×4 input is below the alignment threshold so the
  strides for fp32 round 4→16; storage is larger but the math
  result is the same).
* **Lifecycle** — calling the setter on a destroyed pipeline
  raises.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
from tessera.apple_mlpkg import (
    Pipeline,
    compile_mlpackage,
    last_error_kind,
    packaged_ml_available,
    packaged_ml_skip_reason,
)


_FIXTURES_DIR = (Path(__file__).resolve().parent.parent
                 / "fixtures" / "apple_gpu")


def _find_mtlpackage() -> Path | None:
    if not _FIXTURES_DIR.is_dir():
        return None
    for entry in _FIXTURES_DIR.iterdir():
        if entry.suffix == ".mtlpackage" and entry.is_dir():
            return entry
    return None


# ---- Symbol availability -----------------------------------------------

def test_set_aligned_strides_symbol_resolves():
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    fn = bind_symbol(
        "tessera_apple_gpu_mlpkg_set_aligned_strides",
        (ctypes.c_void_p, ctypes.c_int32), ctypes.c_int32)
    assert fn is not None


# ---- Lifecycle ---------------------------------------------------------

def test_set_aligned_strides_on_destroyed_raises():
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    with pytest.raises(RuntimeError, match="already destroyed"):
        pipe.set_aligned_strides(True)


# ---- Default off — backward-compat ------------------------------------

def test_default_strides_path_unchanged():
    """When set_aligned_strides is NOT called, the pipeline uses
    Metal's implicit strides (the pre-Phase-2 behavior). Verify by
    running a normal dispatch — must succeed numerically as before."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(f"no .mtlpackage fixture in {_FIXTURES_DIR}")
    pipe = compile_mlpackage(pkg, function_name="main",
                             input_dimensions={0: (4, 4), 1: (4, 4)})
    if pipe is None:
        pytest.fail(f"compile failed; last_error_kind={last_error_kind()}")
    try:
        # Default (no set_aligned_strides call) — implicit strides.
        assert pipe.prepare_tensors() is True

        rng = np.random.default_rng(0xC0FFEE)
        A_np = rng.standard_normal((4, 4), dtype=np.float32)
        B_np = rng.standard_normal((4, 4), dtype=np.float32)
        assert pipe.fill_input("inputA", A_np.tobytes())
        assert pipe.fill_input("inputB", B_np.tobytes())
        assert pipe.dispatch(timeout_ms=30_000)
        raw = pipe.read_output("output", 4 * 4 * 4)
        out = np.frombuffer(raw, dtype=np.float32).reshape(4, 4)
        np.testing.assert_allclose(out, A_np @ B_np,
                                    rtol=1e-3, atol=1e-3)
    finally:
        pipe.destroy()


# ---- Opt-in: aligned strides path -------------------------------------

def test_aligned_strides_via_buffer_backed_tensor_succeeds():
    """Project 1 (2026-06-01) — PK3 now uses the buffer-backed tensor
    creation path (``[MTLBuffer newTensorWithDescriptor:offset:error:]``)
    when ``useAlignedStrides`` is enabled. That API accepts explicit
    strides; the buffer is sized via ``aligned_buffer_nbytes`` to
    cover the padded layout.

    The 4×4 fp32 inputs: natural stride[1]=4, aligned stride[1]=16
    (64-byte alignment). Allocated buffer is 16 * 4 * 4 = 256 bytes
    (vs dense 64 bytes), but the math result is identical because
    the kernel reads only the valid 4×4 slice via the tensor strides.

    Pins the wire-up end-to-end: setter → buffer-backed alloc →
    dispatch → correct output."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(f"no .mtlpackage fixture in {_FIXTURES_DIR}")
    pipe = compile_mlpackage(pkg, function_name="main",
                             input_dimensions={0: (4, 4), 1: (4, 4)})
    if pipe is None:
        pytest.fail(f"compile failed; last_error_kind={last_error_kind()}")
    try:
        assert pipe.set_aligned_strides(True) is True
        assert pipe.prepare_tensors() is True

        rng = np.random.default_rng(0xA1167)
        A_np = rng.standard_normal((4, 4), dtype=np.float32)
        B_np = rng.standard_normal((4, 4), dtype=np.float32)
        assert pipe.fill_input("inputA", A_np.tobytes())
        assert pipe.fill_input("inputB", B_np.tobytes())
        assert pipe.dispatch(timeout_ms=30_000)
        raw = pipe.read_output("output", 4 * 4 * 4)
        out = np.frombuffer(raw, dtype=np.float32).reshape(4, 4)
        np.testing.assert_allclose(out, A_np @ B_np,
                                    rtol=1e-3, atol=1e-3)
    finally:
        pipe.destroy()


def test_set_aligned_strides_returns_true_when_runtime_available():
    """The setter itself succeeds when called on a real pipeline
    handle, regardless of whether prepare_tensors then succeeds.
    Verifies the C ABI is bound + reachable."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(f"no .mtlpackage fixture in {_FIXTURES_DIR}")
    pipe = compile_mlpackage(pkg, function_name="main",
                             input_dimensions={0: (4, 4), 1: (4, 4)})
    if pipe is None:
        pytest.fail(f"compile failed; last_error_kind={last_error_kind()}")
    try:
        assert pipe.set_aligned_strides(True) is True
        assert pipe.set_aligned_strides(False) is True
    finally:
        pipe.destroy()
