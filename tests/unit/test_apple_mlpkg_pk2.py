"""PK2 — reflection extraction from a loaded MTL4 ML pipeline.

The packaged-kernel sprint's second step: walk the pipeline's
``reflection.bindings``, filter tensor bindings, and surface each as
a structured Python ``TensorBinding`` (name + buffer_index + rank +
dims + dtype). Mirrors Apple's sample at
``MLMatrixMultiplier+TensorSetup.m::extractTensorBindingsFromPipelineState``.

Tests pin:

* **Symbol resolution** — the two new C ABI symbols
  (``binding_count`` and ``binding_info``) bind cleanly + the dtype
  probe returns sensible raw enum values for known dtypes.
* **No-pipeline → no bindings** — calling ``Pipeline.bindings()`` on
  a destroyed handle raises with a clear message.
* **Dtype decoding** — known dtypes (Float32 / Float16 / BFloat16)
  decode to Tessera's canonical names; unknown raws round-trip as
  ``"raw=<N>"``.
* **Real-artifact reflection** — when a ``.mtlpackage`` fixture is
  present, ``bindings()`` returns a non-empty dict whose entries
  have plausible metadata (rank > 0, dtype not "raw=0", buffer_index
  >= 0).

Like PK1, the real-artifact tests skip cleanly when no fixture is
present (drop in
``tests/fixtures/apple_gpu/<something>.mtlpackage`` to exercise).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
from tessera.apple_mlpkg import (
    packaged_ml_available,
    packaged_ml_skip_reason,
    Pipeline,
    TensorBinding,
    compile_mlpackage,
    _DTYPE_TAG_BY_NAME,
    _dtype_name_for_raw,
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


# ---- C ABI symbol resolution + dtype probe -----------------------------

def test_pk2_symbols_resolve():
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    for sym, args, ret in (
        ("tessera_apple_gpu_mlpkg_binding_count",
         (ctypes.c_void_p,), ctypes.c_int32),
        ("tessera_apple_gpu_mlpkg_dtype_raw_for_tag",
         (ctypes.c_int32,), ctypes.c_int32),
    ):
        fn = bind_symbol(sym, args, ret)
        assert fn is not None, f"PK2 symbol {sym!r} missing"


def test_dtype_probe_returns_distinct_raws_for_known_dtypes():
    """The runtime's ``mlpkg_dtype_raw_for_tag`` probe returns
    Apple's ``MTLTensorDataType`` raw enum value for each named tag.
    On this host they should all be non-negative AND distinct from
    each other — a collision would mean the SDK collapsed two
    canonical dtypes onto one enum value (which would be a bug).
    Skip when MTL4 packaged ML isn't available — the probe is
    ``@available(macOS 26.0)`` and returns -1 on older OSes."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    probe = bind_symbol(
        "tessera_apple_gpu_mlpkg_dtype_raw_for_tag",
        (ctypes.c_int32,), restype=ctypes.c_int32)
    raws_by_name = {}
    for name, tag in _DTYPE_TAG_BY_NAME.items():
        r = int(probe(ctypes.c_int32(tag)))
        if r != -1:
            raws_by_name[name] = r
    # Float32 / Float16 / BFloat16 are the dtypes the audit deep-review
    # called out as table-stakes — at minimum these three must be
    # available + distinct.
    for n in ("fp32", "fp16", "bf16"):
        assert n in raws_by_name, f"dtype {n!r} missing from host SDK"
    distinct = set(raws_by_name.values())
    assert len(distinct) == len(raws_by_name), (
        f"dtype raw-value collision: {raws_by_name}")


# ---- Dtype name decoding -----------------------------------------------

def test_dtype_name_for_known_raws_returns_canonical_names():
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    probe = bind_symbol(
        "tessera_apple_gpu_mlpkg_dtype_raw_for_tag",
        (ctypes.c_int32,), restype=ctypes.c_int32)
    for name, tag in _DTYPE_TAG_BY_NAME.items():
        raw = int(probe(ctypes.c_int32(tag)))
        if raw == -1:
            continue
        decoded = _dtype_name_for_raw(raw)
        assert decoded == name, f"raw={raw} decoded {decoded!r}, expected {name!r}"


def test_dtype_name_for_unknown_raw_returns_sentinel():
    """An enum value the host SDK uses but we haven't named must round-trip
    as ``"raw=<N>"`` so the caller sees something deterministic (not a
    crash, not an empty string)."""
    decoded = _dtype_name_for_raw(99999)
    assert decoded.startswith("raw=")
    assert "99999" in decoded


# ---- Pipeline.bindings() on a destroyed handle -------------------------

def test_bindings_on_destroyed_pipeline_raises():
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    with pytest.raises(RuntimeError, match="already destroyed"):
        pipe.bindings()


# ---- Real-artifact reflection (skipped when no fixture) ----------------

def test_reflection_extraction_from_real_mlpackage():
    """When a ``.mtlpackage`` fixture is present, ``bindings()`` must
    return a non-empty dict. Apple's sample matrix-multiplication
    package has three bindings: ``inputA`` / ``inputB`` / ``output``,
    each 2D float32. Any device_verified_jit Metal package will have at least
    one binding (otherwise the kernel can't take or produce data)."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(
            f"No .mtlpackage fixture in {_FIXTURES_DIR} — drop one in to "
            f"exercise PK2's reflection path. Apple's sample "
            f"`matrix-multiplication.mtlpackage` is known-good.")
    pipe = compile_mlpackage(pkg, function_name="main",
                              input_dimensions={0: (4, 4), 1: (4, 4)})
    if pipe is None:
        pytest.fail(f"compile_mlpackage failed; last_error_kind="
                    f"{last_error_kind()}")
    try:
        bindings = pipe.bindings()
        assert isinstance(bindings, dict)
        assert len(bindings) >= 1, "expected at least one tensor binding"
        for name, b in bindings.items():
            assert isinstance(b, TensorBinding)
            assert b.name == name
            assert b.buffer_index >= 0, b
            assert b.rank > 0, b
            assert b.rank == len(b.dims), b
            # The dtype must decode to a known name OR a raw=<N>
            # sentinel — never empty.
            assert b.dtype, b
            assert b.dtype_raw > 0, (
                f"dtype_raw=0 means MTLTensorDataTypeNone — invalid for a "
                f"real binding (binding {name})")
    finally:
        pipe.destroy()


def test_reflection_bindings_are_stable_across_calls():
    """Calling ``bindings()`` twice on the same pipeline returns the
    same shape. Catches a regression where the underlying reflection
    array gets re-allocated and the indices shift."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip("no .mtlpackage fixture available")
    pipe = compile_mlpackage(pkg, function_name="main",
                              input_dimensions={0: (4, 4), 1: (4, 4)})
    if pipe is None:
        pytest.skip(f"compile failed; last_error_kind={last_error_kind()}")
    try:
        b1 = pipe.bindings()
        b2 = pipe.bindings()
        assert b1 == b2
    finally:
        pipe.destroy()
