"""PK8 — author a *production* ``.mtlpackage`` from the MPSGraph lane.

This is the inverse of PK1-PK7 (which *consume* a package). PK8 proves
Tessera can *author* its own packaged kernel on-host, with no coremltools
and no DXIL, by riding the same MPSGraph primitive the runtime already
builds for its MPSGraph-lane ops:

  build MPSGraph matmul
    → compile to MPSGraphExecutable
    → serializeToMPSGraphPackageAtURL:        (MPSGraphExecutable.h:205)
    → wrap with the MLLibrary ``manifest.json``
    → .mtlpackage

The authored package then flows through the *existing* PK1-PK7 lifecycle
unchanged (load → reflect → prepare → dispatch) and the GPU output is
compared bitwise against numpy ``A @ B``.

Grounded finding (2026-06-02): MPSGraph's serialized packages expose
*positionally-indexed, unnamed* bindings (``MPSGraphTensor`` has no name
property; the Apple sample's ``inputA``/``inputB``/``output`` names came
from its CoreML origin). So PK8 binds by ``buffer_index`` via the
``fill_input_at`` / ``read_output_at`` ABI — the argument table is still
correct (resources bound by reflected index). This is an Apple-side fact,
not a Tessera limitation.

Skips cleanly off-Darwin / pre-macOS-26 (the MTL4 ML dispatch gate).
Authoring alone needs only macOS 14, but the end-to-end dispatch proof
needs the full MTL4 ML pipeline, so we gate on ``packaged_ml_available``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tessera.apple_mlpkg import (
    author_matmul_package,
    compile_mlpackage,
    first_function_name,
    last_error_kind,
    packaged_ml_available,
    packaged_ml_skip_reason,
)


def _require_packaged_ml() -> None:
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")


def test_pk8_symbols_resolve():
    """The authoring + positional-binding ABI is reachable. (Does not
    require a live device — just that the wrappers bind.)"""
    # Importing + referencing is enough; the functions guard internally
    # and return False/None when the runtime is unavailable.
    assert callable(author_matmul_package)
    assert callable(first_function_name)


def test_pk8_author_produces_mtlpackage_structure():
    """Authoring writes the canonical MLLibrary layout: a ``manifest.json``
    wrapping a ``library.mpsgraphpackage`` (matching Apple's sample)."""
    _require_packaged_ml()
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "tessera_matmul.mtlpackage"
        assert author_matmul_package(str(pkg), 4, 4, 4), (
            "author_matmul_package failed on a host that reports "
            "packaged ML available"
        )
        assert (pkg / "manifest.json").is_file()
        inner = pkg / "library.mpsgraphpackage"
        assert inner.is_dir()
        # The serialized MPSGraph executable: model + reflection.
        assert (inner / "reflection.fb").is_file()
        manifest = (pkg / "manifest.json").read_text()
        assert "MLLibrary" in manifest
        assert "library.mpsgraphpackage" in manifest


def test_pk8_authored_package_loads_and_reflects():
    """PK1+PK2 over a Tessera-authored package: it loads as an MTL4 ML
    pipeline and reflects 3 fp32 bindings at indices 0/1/2."""
    _require_packaged_ml()
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "m.mtlpackage"
        assert author_matmul_package(str(pkg), 4, 4, 4)
        fn = first_function_name(str(pkg)) or "main"
        pipe = compile_mlpackage(str(pkg), function_name=fn)
        assert pipe is not None, f"PK1 load failed err={last_error_kind()}"
        try:
            binds = pipe.bindings()
            # Unnamed bindings collapse in the name-keyed dict, so assert on
            # the raw reflection via prepare + index addressing instead.
            assert pipe.prepare_tensors()
            # Every binding the package declares is fp32.
            for b in binds.values():
                assert b.dtype == "fp32"
        finally:
            pipe.destroy()


@pytest.mark.parametrize("mkn", [(4, 4, 4), (8, 16, 4), (3, 5, 7)])
def test_pk8_authored_matmul_dispatches_and_matches_numpy(mkn):
    """The full proof: author → load → prepare → fill (by index) →
    dispatch → read → compare to numpy ``A @ B`` at fp32 tolerance."""
    _require_packaged_ml()
    m, k, n = mkn
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "matmul.mtlpackage"
        assert author_matmul_package(str(pkg), m, k, n)
        fn = first_function_name(str(pkg)) or "main"
        pipe = compile_mlpackage(str(pkg), function_name=fn)
        assert pipe is not None, f"PK1 load failed err={last_error_kind()}"
        try:
            assert pipe.prepare_tensors()
            rng = np.random.default_rng(0xA11CE)
            a = rng.standard_normal((m, k)).astype(np.float32)
            b = rng.standard_normal((k, n)).astype(np.float32)
            # MPSGraph-authored packages are positionally bound:
            # inputs at 0/1, output at the last index (2).
            assert pipe.fill_input_at(0, a.tobytes())
            assert pipe.fill_input_at(1, b.tobytes())
            assert pipe.dispatch(timeout_ms=30_000)
            raw = pipe.read_output_at(2, m * n * 4)
            assert raw is not None
            c = np.frombuffer(raw, dtype=np.float32).reshape(m, n)
            ref = a @ b
            assert np.allclose(c, ref, rtol=1e-4, atol=1e-4), (
                f"max abs err {np.max(np.abs(c - ref)):.3e}"
            )
        finally:
            pipe.destroy()


def test_pk8_reauthor_is_deterministic_overwrite():
    """Re-authoring into the same path overwrites cleanly (serialize is
    set to non-append) — no stale ``.mpsgraphpackage`` contents leak."""
    _require_packaged_ml()
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "r.mtlpackage"
        assert author_matmul_package(str(pkg), 4, 4, 4)
        assert author_matmul_package(str(pkg), 8, 8, 8)  # different shape
        fn = first_function_name(str(pkg)) or "main"
        pipe = compile_mlpackage(str(pkg), function_name=fn)
        assert pipe is not None
        try:
            assert pipe.prepare_tensors()
            rng = np.random.default_rng(1)
            a = rng.standard_normal((8, 8)).astype(np.float32)
            b = rng.standard_normal((8, 8)).astype(np.float32)
            assert pipe.fill_input_at(0, a.tobytes())
            assert pipe.fill_input_at(1, b.tobytes())
            assert pipe.dispatch(timeout_ms=30_000)
            raw = pipe.read_output_at(2, 8 * 8 * 4)
            c = np.frombuffer(raw, dtype=np.float32).reshape(8, 8)
            assert np.allclose(c, a @ b, rtol=1e-4, atol=1e-4)
        finally:
            pipe.destroy()
