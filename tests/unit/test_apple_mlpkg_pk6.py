"""PK6 — Reflection validation gate (audit Action 4).

The audit's "Action 4 — Reflection as ABI verification" framing:

  > When Tessera loads or generates an Apple ML package, it should
  > verify compiler-expected bindings against reflected bindings
  > before marking the artifact executable.

PK6 ships ``validate_bindings(pipeline, expected, *, strict_extra)``:

* ``expected`` is a list of ``ExpectedBinding(name, rank, dtype,
  buffer_index, dims)`` declarations — each field optional, ``None``
  means "don't check this axis".
* Returns ``BindingValidation`` with ``missing`` / ``extra`` /
  ``mismatched`` lists + a ``first_failure_reason`` named-gate
  property.

These tests use the bundled Apple matrix-multiplication package as
the proving ground:

* **Correct expectations pass** — every field matches reflection.
* **Wrong dtype → mismatch caught** with precise field name.
* **Wrong rank / buffer_index / dims → mismatch caught**.
* **Missing binding caught** (expected says ``foo``, package has none).
* **Extra binding is soft by default, hard with ``strict_extra=True``**.
* **Partial expectations work** — caller can skip axes with ``None``.
* **First-failure reason names the diff class** (the audit gate).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime
from tessera.apple_mlpkg import (
    BindingMismatch,
    BindingValidation,
    ExpectedBinding,
    Pipeline,
    compile_mlpackage,
    last_error_kind,
    validate_bindings,
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


def _open_matmul_pipe_or_skip() -> Pipeline:
    """Open the bundled Apple matmul package at the canonical 4x4
    shape (matches PK4 helpers). Skips if the package or runtime
    isn't available."""
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(f"no .mtlpackage fixture in {_FIXTURES_DIR}")
    pipe = compile_mlpackage(
        pkg, function_name="main",
        input_dimensions={0: (4, 4), 1: (4, 4)})
    if pipe is None:
        pytest.fail(f"compile_mlpackage failed; last_error_kind="
                    f"{last_error_kind()}")
    return pipe


# ---- Happy path: correct expectations validate clean -------------------

def test_correct_expectations_validate_ok():
    """Apple's bundled matmul package has three bindings at well-known
    layout: inputA @ buffer 0 = (4, 4) fp32, inputB @ buffer 1 =
    (4, 4) fp32, output @ buffer 2 = (4, 4) fp32. A complete expected
    schema must validate clean."""
    pipe = _open_matmul_pipe_or_skip()
    try:
        result = validate_bindings(pipe, [
            ExpectedBinding(name="inputA", rank=2, dtype="fp32",
                            buffer_index=0, dims=(4, 4)),
            ExpectedBinding(name="inputB", rank=2, dtype="fp32",
                            buffer_index=1, dims=(4, 4)),
            ExpectedBinding(name="output", rank=2, dtype="fp32",
                            buffer_index=2, dims=(4, 4)),
        ])
        assert result.ok is True, result
        assert result.missing == ()
        assert result.extra == ()
        assert result.mismatched == ()
        assert result.first_failure_reason is None
    finally:
        pipe.destroy()


# ---- Partial expectations (None on a field = "don't check it") ---------

def test_partial_expectation_only_checks_named_fields():
    """A caller that only cares about name + dtype passes other
    fields as None — those axes are skipped during validation."""
    pipe = _open_matmul_pipe_or_skip()
    try:
        result = validate_bindings(pipe, [
            ExpectedBinding(name="inputA", dtype="fp32"),
            ExpectedBinding(name="output", dtype="fp32"),
        ])
        assert result.ok is True
        # inputB is "extra" (we didn't expect it), but soft mode → ok.
        assert "inputB" in result.extra
    finally:
        pipe.destroy()


# ---- Missing-binding detection -----------------------------------------

def test_missing_binding_is_named_in_first_failure_reason():
    pipe = _open_matmul_pipe_or_skip()
    try:
        result = validate_bindings(pipe, [
            ExpectedBinding(name="inputA", dtype="fp32"),
            ExpectedBinding(name="this_binding_does_not_exist"),
        ])
        assert result.ok is False
        assert "this_binding_does_not_exist" in result.missing
        assert (result.first_failure_reason and
                "this_binding_does_not_exist"
                in result.first_failure_reason)
        assert "missing bindings" in result.first_failure_reason
    finally:
        pipe.destroy()


# ---- Mismatch detection (per-field) -----------------------------------

def test_dtype_mismatch_is_caught_with_precise_field_name():
    pipe = _open_matmul_pipe_or_skip()
    try:
        result = validate_bindings(pipe, [
            ExpectedBinding(name="inputA", dtype="fp16"),  # actual is fp32
        ])
        assert result.ok is False
        assert len(result.mismatched) == 1
        m = result.mismatched[0]
        assert m.name == "inputA"
        assert m.field == "dtype"
        assert m.expected == "fp16"
        assert m.actual == "fp32"
        assert "dtype mismatch" in (result.first_failure_reason or "")
    finally:
        pipe.destroy()


def test_rank_mismatch_is_caught():
    pipe = _open_matmul_pipe_or_skip()
    try:
        result = validate_bindings(pipe, [
            ExpectedBinding(name="inputA", rank=3),  # actual rank is 2
        ])
        assert result.ok is False
        assert any(m.field == "rank" for m in result.mismatched)
    finally:
        pipe.destroy()


def test_buffer_index_mismatch_is_caught():
    """Buffer-index drift is the audit's headline concern: a package
    that re-orders bindings between builds would silently misroute
    data. Pin this with a precise diff."""
    pipe = _open_matmul_pipe_or_skip()
    try:
        result = validate_bindings(pipe, [
            ExpectedBinding(name="output", buffer_index=99),  # actual is 2
        ])
        assert result.ok is False
        m = [m for m in result.mismatched if m.field == "buffer_index"]
        assert len(m) == 1
        assert m[0].expected == 99
        assert m[0].actual == 2
    finally:
        pipe.destroy()


def test_dims_mismatch_is_caught():
    pipe = _open_matmul_pipe_or_skip()
    try:
        result = validate_bindings(pipe, [
            ExpectedBinding(name="inputA", dims=(8, 8)),  # actual is (4, 4)
        ])
        assert result.ok is False
        m = [m for m in result.mismatched if m.field == "dims"]
        assert len(m) == 1
        assert m[0].expected == (8, 8)
        assert m[0].actual == (4, 4)
    finally:
        pipe.destroy()


# ---- Extra-binding handling (soft default, hard with strict_extra) -----

def test_extra_binding_is_warning_by_default():
    pipe = _open_matmul_pipe_or_skip()
    try:
        # Expect only inputA — inputB + output become "extra".
        result = validate_bindings(pipe, [
            ExpectedBinding(name="inputA"),
        ])
        assert result.ok is True  # extras don't fail soft mode
        assert "inputB" in result.extra
        assert "output" in result.extra
        # ``first_failure_reason`` is None because ok=True.
        assert result.first_failure_reason is None
    finally:
        pipe.destroy()


def test_extra_binding_is_hard_failure_with_strict_extra():
    pipe = _open_matmul_pipe_or_skip()
    try:
        result = validate_bindings(pipe, [
            ExpectedBinding(name="inputA"),
        ], strict_extra=True)
        assert result.ok is False
        assert "inputB" in result.extra
        assert "unexpected bindings" in (result.first_failure_reason or "")
    finally:
        pipe.destroy()


# ---- Failure ordering (missing > mismatched > extra) -------------------

def test_first_failure_reason_prioritizes_missing_over_mismatched():
    """When both missing AND mismatched are present, missing wins —
    the package literally doesn't have a binding we expected, which
    is a more fundamental contract break."""
    pipe = _open_matmul_pipe_or_skip()
    try:
        result = validate_bindings(pipe, [
            ExpectedBinding(name="this_does_not_exist"),
            ExpectedBinding(name="inputA", dtype="fp16"),  # mismatched
        ])
        assert result.ok is False
        assert "missing" in (result.first_failure_reason or "")
        # Both diff buckets populated.
        assert result.missing == ("this_does_not_exist",)
        assert len(result.mismatched) >= 1
    finally:
        pipe.destroy()


# ---- Multiple mismatches: first one wins in the reason ----------------

def test_first_failure_reason_is_first_mismatch_when_no_missing():
    pipe = _open_matmul_pipe_or_skip()
    try:
        result = validate_bindings(pipe, [
            ExpectedBinding(name="inputA", dtype="fp16", rank=3),
        ])
        assert result.ok is False
        # Both rank and dtype mismatches recorded.
        assert len(result.mismatched) >= 2
        # First reason names the FIRST mismatch (dtype is checked
        # before rank in our impl, so dtype comes first).
        assert (result.first_failure_reason or "").startswith(
            "binding 'inputA' ")
    finally:
        pipe.destroy()
