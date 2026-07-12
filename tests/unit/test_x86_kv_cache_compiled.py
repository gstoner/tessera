"""Native x86 KV-cache ABI execution and numerical provenance."""

from __future__ import annotations

import numpy as np
import pytest

import tessera


def _runtime_or_skip():
    from tessera import runtime as rt

    lib = rt._load_x86_elementwise()
    if lib is None or not hasattr(lib, "tessera_x86_kv_cache_read_f32"):
        pytest.skip("native x86 KV-cache ABI is not built")
    return rt


def _artifact(rt, op: str, operand_names: list[str], kwargs: dict):
    return rt.RuntimeArtifact(metadata={
        "target": "x86",
        "compiler_path": "x86_kv_cache_compiled",
        "executable": True,
        "execution_kind": "native_cpu",
        "execution_mode": "cpu_avx512",
        "arg_names": operand_names,
        "output_name": "out",
        "ops": [{
            "op_name": op,
            "result": "out",
            "operands": operand_names,
            "kwargs": kwargs,
        }],
    })


def _launch(rt, op: str, operands: tuple[np.ndarray, ...], **kwargs):
    names = ["cache", "rows"][:len(operands)]
    result = rt.launch(_artifact(rt, op, names, kwargs), operands)
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "x86_kv_cache_compiled"
    assert result["execution_kind"] == "native_cpu"
    return np.asarray(result["output"], np.float32)


def test_x86_kv_cache_read_matches_handle_reference() -> None:
    rt = _runtime_or_skip()
    rng = np.random.default_rng(73)
    handle = tessera.cache.KVCacheHandle(num_heads=3, head_dim=8, max_seq=32)
    keys = rng.standard_normal((12, 3, 8)).astype(np.float32)
    handle.append(keys, keys + 100.0)
    expected, _ = handle.read(2, 9)

    output = _launch(rt, "tessera.kv_cache.read",
                     (handle.keys.astype(np.float32),), start=2, end=9)
    np.testing.assert_allclose(output, np.asarray(expected, np.float32), rtol=0, atol=0)


def test_x86_kv_cache_append_and_prune_match_handle_reference() -> None:
    rt = _runtime_or_skip()
    rng = np.random.default_rng(79)
    handle = tessera.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
    initial = rng.standard_normal((6, 2, 4)).astype(np.float32)
    added = rng.standard_normal((3, 2, 4)).astype(np.float32)
    handle.append(initial, initial)
    cache = handle.keys.astype(np.float32).copy()

    output = _launch(rt, "tessera.kv_cache.append", (cache, added), start=6)
    handle.append(added, added)
    np.testing.assert_allclose(output, handle.keys.astype(np.float32), rtol=0, atol=0)

    output = _launch(rt, "tessera.kv_cache.prune", (output,),
                     current_seq=handle.current_seq, limit=4)
    handle.prune(4)
    np.testing.assert_allclose(output, handle.keys.astype(np.float32), rtol=0, atol=0)


def test_x86_kv_cache_read_rejects_invalid_bounds() -> None:
    rt = _runtime_or_skip()
    cache = np.zeros((8, 2, 4), np.float32)
    result = rt.launch(
        _artifact(rt, "tessera.kv_cache.read", ["cache"],
                  {"start": 7, "end": 10}),
        (cache,),
    )
    assert result["ok"] is False
