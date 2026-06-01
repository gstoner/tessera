"""PK3 — Tensor creation + MTL4ArgumentTable binding from reflection.

Third step of the packaged-kernel sprint. Mirrors Apple's sample at
``MLMatrixMultiplier.m:configureWithMatrix1:`` (lines 161-207):

* For each reflected tensor binding, create a device-backed
  ``MTLTensor`` matching the binding's shape + dtype.
* Build a ``MTL4ArgumentTable`` sized to the highest binding index.
* Bind each tensor's ``gpuResourceID`` at the binding's
  kernel-side index.

PK3 also adds ``fill_input`` and ``read_output`` so a fill→read
roundtrip can validate the tensor lifecycle without GPU dispatch
(execution lands in PK4).

Tests pin:

* **Symbol resolution** — the 4 new C ABI symbols bind cleanly.
* **Idempotency** — calling ``prepare_tensors()`` twice is a no-op
  the second time.
* **Pre-/post-condition contract** — ``argument_table_ready()``
  flips ``False → True`` only after a successful prepare.
* **Byte-count validation** — ``fill_input()`` rejects payloads that
  don't match the tensor's reflected size.
* **Fill → read roundtrip** — given a real ``.mtlpackage``, filling
  an input tensor and reading it back returns identical bytes. This
  proves both directions of the host↔tensor data path work
  without executing the GPU kernel (which is PK4's job).

Like PK1 / PK2, real-artifact tests skip when no fixture is present.
"""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path

import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
from tessera.apple_mlpkg import (
    Pipeline,
    compile_mlpackage,
    last_error_kind,
)


_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "apple_gpu"


def _find_mtlpackage() -> Path | None:
    if not _FIXTURES_DIR.is_dir():
        return None
    for entry in _FIXTURES_DIR.iterdir():
        if entry.suffix == ".mtlpackage" and entry.is_dir():
            return entry
    return None


# ---- Symbol resolution -------------------------------------------------

def test_pk3_symbols_resolve():
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    for sym in (
        "tessera_apple_gpu_mlpkg_prepare_tensors",
        "tessera_apple_gpu_mlpkg_argument_table_ready",
        "tessera_apple_gpu_mlpkg_fill_input",
        "tessera_apple_gpu_mlpkg_read_output",
    ):
        fn = bind_symbol(sym, (ctypes.c_void_p,), ctypes.c_int32)
        assert fn is not None, f"PK3 symbol {sym!r} missing"


# ---- Destroyed-handle contract -----------------------------------------

def test_prepare_tensors_on_destroyed_raises():
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    with pytest.raises(RuntimeError, match="already destroyed"):
        pipe.prepare_tensors()


def test_fill_input_on_destroyed_raises():
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    with pytest.raises(RuntimeError, match="already destroyed"):
        pipe.fill_input("inputA", b"")


def test_read_output_on_destroyed_raises():
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    with pytest.raises(RuntimeError, match="already destroyed"):
        pipe.read_output("output", 4)


def test_argument_table_ready_on_destroyed_returns_false():
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    assert pipe.argument_table_ready() is False


# ---- Real-artifact path (skipped when no fixture) ----------------------

def _open_pipe_or_skip() -> Pipeline:
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(
            f"No .mtlpackage fixture in {_FIXTURES_DIR} — drop one in to "
            f"exercise PK3 tensor + argument-table path")
    pipe = compile_mlpackage(pkg, function_name="main",
                              input_dimensions={0: (4, 4), 1: (4, 4)})
    if pipe is None:
        pytest.fail(f"compile_mlpackage failed; last_error_kind="
                    f"{last_error_kind()}")
    return pipe


def test_prepare_tensors_builds_argument_table():
    pipe = _open_pipe_or_skip()
    try:
        # Pre-condition: not yet ready.
        assert pipe.argument_table_ready() is False
        # Prepare succeeds.
        assert pipe.prepare_tensors() is True
        # Post-condition: argument table built.
        assert pipe.argument_table_ready() is True
    finally:
        pipe.destroy()


def test_prepare_tensors_is_idempotent():
    pipe = _open_pipe_or_skip()
    try:
        assert pipe.prepare_tensors() is True
        # Second call: still True, no side effects.
        assert pipe.prepare_tensors() is True
        assert pipe.argument_table_ready() is True
    finally:
        pipe.destroy()


def test_fill_input_rejects_wrong_byte_count():
    """The runtime validates byte_count vs tensor's expected size.
    Passing the wrong number of bytes returns False — no crash, no
    silent truncation."""
    pipe = _open_pipe_or_skip()
    try:
        assert pipe.prepare_tensors() is True
        bindings = pipe.bindings()
        # Pick any input binding (anything not literally named "output").
        target = None
        for name, b in bindings.items():
            if "output" not in name.lower():
                target = (name, b)
                break
        if target is None:
            pytest.skip("no non-output binding in this package")
        name, _b = target
        # Way too small.
        assert pipe.fill_input(name, b"\x00\x00\x00\x00") is False
        # Way too large.
        assert pipe.fill_input(name, b"\x00" * 1_000_000) is False
    finally:
        pipe.destroy()


def test_fill_then_read_roundtrip_is_bit_exact():
    """Fill an input tensor with a known byte pattern, read it back —
    the bytes must match exactly. Proves the host↔tensor data path
    works in both directions without GPU dispatch."""
    pipe = _open_pipe_or_skip()
    try:
        assert pipe.prepare_tensors() is True
        bindings = pipe.bindings()
        # Find an input binding with a known size.
        target_name = None
        target_binding = None
        for name, b in bindings.items():
            if "output" not in name.lower():
                target_name = name
                target_binding = b
                break
        if target_name is None:
            pytest.skip("no non-output binding in this package")
        # Compute expected byte count from the reflection.
        dtype_byte_size = {
            "fp32": 4, "fp16": 2, "bf16": 2,
            "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
            "int32": 4, "uint32": 4,
        }.get(target_binding.dtype)
        if dtype_byte_size is None:
            pytest.skip(f"unhandled dtype: {target_binding.dtype}")
        elem_count = 1
        for d in target_binding.dims:
            elem_count *= d
        byte_count = elem_count * dtype_byte_size
        # Generate a deterministic byte pattern (not all zero — that
        # would mask a bug where the tensor was never written).
        pattern = bytes((i & 0xFF) for i in range(byte_count))
        assert pipe.fill_input(target_name, pattern) is True
        # Read back.
        roundtrip = pipe.read_output(target_name, byte_count)
        assert roundtrip is not None
        assert len(roundtrip) == byte_count
        assert roundtrip == pattern, (
            f"fill→read roundtrip mismatch: first 16 bytes "
            f"in={pattern[:16].hex()} out={roundtrip[:16].hex()}")
    finally:
        pipe.destroy()


def test_unknown_tensor_name_fails_cleanly():
    """``fill_input`` / ``read_output`` with a name that doesn't
    appear in the bindings must return False / None — no crash."""
    pipe = _open_pipe_or_skip()
    try:
        assert pipe.prepare_tensors() is True
        assert pipe.fill_input("__not_a_binding__", b"\x00") is False
        assert pipe.read_output("__not_a_binding__", 4) is None
    finally:
        pipe.destroy()


# ---- prepare_tensors fails cleanly when reflection has no bindings ----

def test_prepare_tensors_without_compile_returns_false(monkeypatch):
    """Calling ``prepare_tensors`` via the C ABI on a NULL handle (which
    is what happens when the Python wrapper holds handle=0) must
    return False without crashing."""
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    # We can't call .prepare_tensors() on handle=0 (raises) — drive
    # the C ABI directly to verify the NULL-handle path.
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    fn = bind_symbol(
        "tessera_apple_gpu_mlpkg_prepare_tensors",
        (ctypes.c_void_p,), restype=ctypes.c_int32)
    assert int(fn(ctypes.c_void_p(0))) == 0
