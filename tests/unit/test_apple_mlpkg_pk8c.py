"""PK8c — committed Tessera-authored production ``.mtlpackage`` fixtures.

Unlike PK8/PK8a/PK8b (which author into a temp dir at test time), this
loads the *committed* production packages — proving the on-disk artifacts
declared in ``PACKAGED_PRODUCTION_KERNELS`` are real, loadable, and
numerically correct. Regenerate the fixtures with
``python3 scripts/author_apple_packages.py``.

Portability: a serialized MPSGraph package targets the current platform's
latest deployment version and may not load on an *older* macOS. So a load
failure here is treated as a **skip** (host/SDK drift), not a hard failure;
a successful load asserts numerics.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.apple_mlpkg import (
    compile_mlpackage,
    first_function_name,
    packaged_ml_available,
    packaged_ml_skip_reason,
)
from tessera.compiler.apple_packaged_manifest import (
    PACKAGED_PRODUCTION_KERNELS,
    repo_root,
    resolve_packaged_path,
    validate_packaged_entry,
)


def _entry(substr):
    for e in PACKAGED_PRODUCTION_KERNELS:
        if substr in (e.packaged_pipeline_path or ""):
            return e
    raise AssertionError(f"no production entry matching {substr!r}")


# ── committed-artifact existence (no device needed) ──────────────────────


def test_production_fixtures_exist_on_disk():
    assert len(PACKAGED_PRODUCTION_KERNELS) >= 2
    for e in PACKAGED_PRODUCTION_KERNELS:
        path = resolve_packaged_path(e)
        assert path.is_dir(), f"missing committed package: {path}"
        assert (path / "manifest.json").is_file()
        assert (path / "library.mpsgraphpackage").is_dir()


def test_production_entries_pass_drift_gate():
    for e in PACKAGED_PRODUCTION_KERNELS:
        ok, reason = validate_packaged_entry(e)
        assert ok, f"{e.packaged_pipeline_path}: {reason}"


# ── load + dispatch the committed packages (gated) ───────────────────────


def _require_packaged_ml():
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")


def _load_or_skip(entry):
    path = resolve_packaged_path(entry)
    fn = first_function_name(path) or "main"
    pipe = compile_mlpackage(path, function_name=fn)
    if pipe is None:
        pytest.skip(
            f"committed package failed to load on this host (SDK/version "
            f"drift) — regenerate via scripts/author_apple_packages.py: "
            f"{path}")
    return pipe


def test_committed_matmul_dispatches_and_matches_numpy():
    _require_packaged_ml()
    entry = _entry("tessera_authored_matmul_8x8x8")
    pipe = _load_or_skip(entry)
    try:
        assert pipe.prepare_tensors()
        rng = np.random.default_rng(40)
        a = rng.standard_normal((8, 8)).astype(np.float32)
        b = rng.standard_normal((8, 8)).astype(np.float32)
        assert pipe.fill_input_at(0, a.tobytes())
        assert pipe.fill_input_at(1, b.tobytes())
        assert pipe.dispatch(timeout_ms=30_000)
        raw = pipe.read_output_at(2, 8 * 8 * 4)
        c = np.frombuffer(raw, dtype=np.float32).reshape(8, 8)
        assert np.allclose(c, a @ b, rtol=1e-4, atol=2e-4)
    finally:
        pipe.destroy()


def test_committed_matmul_softmax_chain_dispatches_and_matches_numpy():
    _require_packaged_ml()
    entry = _entry("tessera_authored_matmul_softmax_4x6x5")
    pipe = _load_or_skip(entry)
    try:
        assert pipe.prepare_tensors()
        rng = np.random.default_rng(41)
        M, K, N = 4, 6, 5
        a = rng.standard_normal((M, K)).astype(np.float32)
        b = rng.standard_normal((K, N)).astype(np.float32)
        assert pipe.fill_input_at(0, a.tobytes())
        assert pipe.fill_input_at(1, b.tobytes())
        assert pipe.dispatch(timeout_ms=30_000)
        raw = pipe.read_output_at(2, M * N * 4)
        out = np.frombuffer(raw, dtype=np.float32).reshape(M, N)
        ab = a @ b
        e = np.exp(ab - ab.max(axis=1, keepdims=True))
        ref = e / e.sum(axis=1, keepdims=True)
        assert np.allclose(out, ref, rtol=1e-4, atol=2e-4)
    finally:
        pipe.destroy()


# ── expanded coverage: one representative per authorable family ──────────
#
# Each spec: (fixture substr, build_inputs(rng) -> list[np.ndarray],
#             ref(inputs) -> np.ndarray, output_shape). Inputs are filled
# positionally (fill_input_at), output read from the last binding index.


def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


_EXPANDED_SPECS = [
    (
        "tessera_authored_silu_8x16",
        lambda r: [r.standard_normal((8, 16)).astype(np.float32)],
        lambda a: _silu(a[0]),
        (8, 16),
    ),
    (
        "tessera_authored_softmax_8x16",
        lambda r: [r.standard_normal((8, 16)).astype(np.float32)],
        lambda a: _softmax(a[0]),
        (8, 16),
    ),
    (
        "tessera_authored_rmsnorm_8x16",
        lambda r: [r.standard_normal((8, 16)).astype(np.float32),
                   r.standard_normal((16,)).astype(np.float32)],
        lambda a: a[0] / np.sqrt((a[0] ** 2).mean(axis=1, keepdims=True)
                                 + 1e-5) * a[1],
        (8, 16),
    ),
    (
        "tessera_authored_matmul_softmax_matmul_4x6x5x3",
        lambda r: [r.standard_normal((4, 6)).astype(np.float32),
                   r.standard_normal((6, 5)).astype(np.float32),
                   r.standard_normal((5, 3)).astype(np.float32)],
        lambda a: _softmax(a[0] @ a[1]) @ a[2],
        (4, 3),
    ),
    (
        "tessera_authored_rmsnorm_matmul_4x6x5",
        lambda r: [r.standard_normal((4, 6)).astype(np.float32),
                   r.standard_normal((6,)).astype(np.float32),
                   r.standard_normal((6, 5)).astype(np.float32)],
        lambda a: (a[0] / np.sqrt((a[0] ** 2).mean(axis=1, keepdims=True)
                                  + 1e-5) * a[1]) @ a[2],
        (4, 5),
    ),
]


@pytest.mark.parametrize(
    "substr,build_inputs,ref_fn,out_shape", _EXPANDED_SPECS,
    ids=[s[0].replace("tessera_authored_", "") for s in _EXPANDED_SPECS])
def test_committed_expanded_fixture_dispatches_and_matches_numpy(
        substr, build_inputs, ref_fn, out_shape):
    """Each Tessera-authored production fixture (silu / softmax / rmsnorm /
    attention block / rmsnorm→matmul) loads + dispatches + matches numpy."""
    _require_packaged_ml()
    entry = _entry(substr)
    pipe = _load_or_skip(entry)
    try:
        assert pipe.prepare_tensors()
        rng = np.random.default_rng(hash(substr) & 0xFFFF)
        inputs = build_inputs(rng)
        for i, arr in enumerate(inputs):
            assert pipe.fill_input_at(i, np.ascontiguousarray(arr).tobytes())
        assert pipe.dispatch(timeout_ms=30_000)
        nbytes = int(np.prod(out_shape)) * 4
        raw = pipe.read_output_at(len(inputs), nbytes)
        out = np.frombuffer(raw, dtype=np.float32).reshape(out_shape)
        assert np.allclose(out, ref_fn(inputs), rtol=1e-4, atol=2e-4)
    finally:
        pipe.destroy()
