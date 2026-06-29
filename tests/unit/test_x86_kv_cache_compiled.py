"""kv_cache state_update lane on x86 AVX-512 (P13 of S_SERIES_GAP_CLOSURE_PLAN)
— the paged scatter-copy into a KV buffer: append (the AVX-512 row-scatter, set
mode), read (the gather), prune (host page compaction). Reachable via
`compiler_path="x86_kv_cache_compiled"`. Validated vs a numpy paged-buffer
reference. Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, n_operands, kwargs):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_kv_cache_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _run(rt, op, *arrs, **kwargs):
    res = rt.launch(_art(rt, op, len(arrs), kwargs), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_kv_cache_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(47)


def test_append_writes_rows():
    rt = _rt_or_skip()
    buf = _RNG.standard_normal((8, 4, 16)).astype(np.float32)   # [S, H, D]
    pos = np.array([5, 0, 3], np.int64)
    upd = _RNG.standard_normal((3, 4, 16)).astype(np.float32)
    got = _run(rt, "tessera.kv_cache.append", buf, pos, upd)
    ref = buf.copy()
    ref[pos] = upd
    np.testing.assert_array_equal(got, ref)


def test_read_gathers_rows():
    rt = _rt_or_skip()
    buf = _RNG.standard_normal((10, 2, 8)).astype(np.float32)
    pos = np.array([9, 1, 1, 4], np.int64)
    got = _run(rt, "tessera.kv_cache.read", buf, pos)
    np.testing.assert_array_equal(got, buf[pos])


def test_prune_compacts():
    rt = _rt_or_skip()
    buf = _RNG.standard_normal((12, 3, 4)).astype(np.float32)
    got = _run(rt, "tessera.kv_cache.prune", buf, keep=5)
    np.testing.assert_array_equal(got, buf[:5])


def test_append_then_read_roundtrip():
    rt = _rt_or_skip()
    buf = np.zeros((6, 2, 4), np.float32)
    pos = np.array([2, 4], np.int64)
    upd = _RNG.standard_normal((2, 2, 4)).astype(np.float32)
    appended = _run(rt, "tessera.kv_cache.append", buf, pos, upd)
    back = _run(rt, "tessera.kv_cache.read", appended, pos)
    np.testing.assert_array_equal(back, upd)
