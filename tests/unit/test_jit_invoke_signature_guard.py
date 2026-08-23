"""Boundary signature validation for ``jb.invoke`` (RUNTIME_ABI_SPEC §12.6).

The compiled code bakes static extents into its indexing math (identity
layout, §12.4), so a buffer whose shape disagrees with the compiled signature
is guaranteed out-of-bounds access — historically silent heap corruption
(a 2-element array against a ``tensor<7xf32>`` module aborted the process in
``malloc_consolidate``). These tests pin the fail-closed contract at both
layers: the Python ``invoke`` validation (rich diagnostics) and the C-side
``tessera_jit_invoke`` backstop (memory safety for callers that bypass
Python).
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tessera import _jit_boundary as jb


pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


def _binary_module(op: str = "arith.addf", tensor: str = "tensor<7xf32>") -> str:
    return f"""
module {{
  func.func @pointwise(%a: {tensor}, %b: {tensor}) -> {tensor} {{
    %r = {op} %a, %b : {tensor}
    return %r : {tensor}
  }}
}}
"""


def _f32(*values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


A7 = _f32(1, 2, 3, 4, 5, 6, 7)
B7 = _f32(7, 6, 5, 4, 3, 2, 1)


def test_correct_shapes_still_execute():
    handle = jb.compile_module(_binary_module())
    out = np.empty_like(A7)
    jb.invoke(handle, "pointwise", [A7, B7], out)
    np.testing.assert_array_equal(out, A7 + B7)


def test_short_input_is_rejected_not_heap_corruption():
    # The original defect: 2 elements against tensor<7xf32> wrote out of
    # bounds and aborted the process later in malloc. Must now raise.
    handle = jb.compile_module(_binary_module())
    short = _f32(1, 2)
    out = np.empty_like(A7)
    with pytest.raises(jb.TesseraJitError, match="shape mismatch"):
        jb.invoke(handle, "pointwise", [short, B7], out)
    # The process must remain healthy: a correct invoke still works.
    jb.invoke(handle, "pointwise", [A7, B7], out)
    np.testing.assert_array_equal(out, A7 + B7)


def test_oversized_input_is_rejected():
    handle = jb.compile_module(_binary_module())
    with pytest.raises(jb.TesseraJitError, match="shape mismatch"):
        jb.invoke(handle, "pointwise", [np.zeros(9, np.float32), B7], np.empty_like(A7))


def test_short_output_buffer_is_rejected():
    handle = jb.compile_module(_binary_module())
    with pytest.raises(jb.TesseraJitError, match="output 0 shape mismatch"):
        jb.invoke(handle, "pointwise", [A7, B7], np.empty(2, np.float32))


def test_wrong_rank_is_rejected():
    handle = jb.compile_module(_binary_module())
    square = np.zeros((7, 7), np.float32)
    with pytest.raises(jb.TesseraJitError, match="rank mismatch"):
        jb.invoke(handle, "pointwise", [square, square], np.empty_like(square))


def test_wrong_arity_is_rejected():
    handle = jb.compile_module(_binary_module())
    with pytest.raises(jb.TesseraJitError, match="expects 2 input"):
        jb.invoke(handle, "pointwise", [A7], np.empty_like(A7))
    with pytest.raises(jb.TesseraJitError, match="expects 1 output"):
        jb.invoke(handle, "pointwise", [A7, B7], [np.empty_like(A7), np.empty_like(A7)])


def test_wrong_dtype_is_rejected():
    handle = jb.compile_module(_binary_module())
    a64 = A7.astype(np.float64)
    b64 = B7.astype(np.float64)
    with pytest.raises(jb.TesseraJitError, match="dtype mismatch"):
        jb.invoke(handle, "pointwise", [a64, b64], np.empty_like(a64))


def test_unknown_symbol_is_rejected_before_dispatch():
    handle = jb.compile_module(_binary_module())
    with pytest.raises(jb.TesseraJitError, match="unknown function"):
        jb.invoke(handle, "no_such_function", [A7, B7], np.empty_like(A7))


def test_dynamic_extents_accept_any_length_but_still_check_dtype():
    handle = jb.compile_module(_binary_module(tensor="tensor<?xf32>"))
    for n in (2, 7, 11):
        a = np.arange(n, dtype=np.float32)
        out = np.empty_like(a)
        jb.invoke(handle, "pointwise", [a, a], out)
        np.testing.assert_array_equal(out, a + a)
    bad = np.arange(3, dtype=np.float64)
    with pytest.raises(jb.TesseraJitError, match="dtype mismatch"):
        jb.invoke(handle, "pointwise", [bad, bad], np.empty_like(bad))


def test_c_backstop_rejects_bad_extent_without_python_validation():
    """Callers that bypass invoke() (foreign languages, raw ctypes) hit the
    C-side backstop: rc != 0 with a descriptive last_error, not corruption."""
    handle = jb.compile_module(_binary_module())
    lib = jb._load()

    short = _f32(1, 2)
    out = np.empty_like(A7)
    descs = [jb._make_descriptor(a) for a in (short, B7, out)]
    packed, _keep = jb._build_packed_args(descs)
    rc = lib.tessera_jit_invoke(handle, b"pointwise", packed, len(descs))
    assert rc == 1
    err = lib.tessera_jit_last_error().decode("utf-8", "replace")
    assert "expects extent 7, got 2" in err


def test_c_backstop_rejects_bad_arity():
    handle = jb.compile_module(_binary_module())
    lib = jb._load()
    descs = [jb._make_descriptor(a) for a in (A7, B7)]  # missing the out-param
    packed, _keep = jb._build_packed_args(descs)
    rc = lib.tessera_jit_invoke(handle, b"pointwise", packed, len(descs))
    assert rc == 1
    assert "expects 3 arguments" in lib.tessera_jit_last_error().decode()


def test_signature_abi_reports_compiled_types():
    handle = jb.compile_module(_binary_module(tensor="tensor<2x?xf32>"))
    lib = jb._load()
    raw = lib.tessera_jit_signature(handle, b"pointwise")
    assert raw is not None
    assert raw.decode() == "tensor<2x?xf32>;tensor<2x?xf32>|tensor<2x?xf32>"
    assert lib.tessera_jit_signature(handle, b"missing") is None
