"""ODS-WIRE-3: `tessera.cache.commit` / `tessera.cache.rollback` on x86.

`TileToX86Pass` lowers both ops to the KV-cache handle ABI in
`kv_cache_f32.cpp` (`tessera_x86_kv_cache_{commit,rollback}_f32`: handle +
count in, updated handle out; lit: `phase2/x86_kv_cache_cursor_abi.mlir`).
This file checks the runtime half:

* host-free: the ctypes mirror of the handle struct and the ABI version agree
  with the C++ source, and every handle the ABI cannot represent is refused
  with `X86_KV_CACHE_HANDLE_REFUSED` before any native call;
* Zen 5 only: the native ABI and the bufferized `x86_kv_cache_compiled` lane
  match the Python references (`tessera.ops.cache_commit` /
  `cache_rollback`) bit for bit. The Mac is arm64 and never runs them.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pytest

import tessera
from tessera import runtime as rt

_ROOT = Path(__file__).resolve().parents[2]
_KV_SOURCE = (
    _ROOT / "src/compiler/codegen/tessera_x86_backend/src/kernels/kv_cache_f32.cpp"
)


def _native_or_skip():
    lib = rt._load_x86_elementwise()
    if lib is None or not hasattr(lib, "tessera_x86_kv_cache_commit_f32"):
        pytest.skip("native x86 KV-cache handle ABI is not built on this host")
    return lib


def _filled(seed: int, current: int, *, max_seq: int = 16):
    rng = np.random.default_rng(seed)
    handle = tessera.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=max_seq)
    keys = rng.standard_normal((current, 2, 4)).astype(np.float32)
    handle.append(keys, keys + 10.0)
    return handle


def _assert_same(native, reference):
    assert native.current_seq == reference.current_seq
    np.testing.assert_array_equal(native.keys, reference.keys)
    np.testing.assert_array_equal(native.values, reference.values)


# ── host-free ───────────────────────────────────────────────────────────────

def test_ctypes_mirror_matches_the_cpp_handle_struct():
    text = _KV_SOURCE.read_text()
    body = re.search(
        r"struct tessera_x86_kv_cache_f32_handle \{(.*?)\};", text, re.S
    ).group(1)
    cpp_fields = re.findall(r"^\s*(?:int64_t|float \*)\s*(\w+);", body, re.M)
    assert cpp_fields == [name for name, _ in rt._X86KVCacheF32Handle._fields_]
    version = re.search(r"kKvCacheF32HandleAbi = (\d+);", text).group(1)
    assert int(version) == rt._X86_KV_CACHE_F32_HANDLE_ABI


@pytest.mark.parametrize("make,why", [
    (lambda: tessera.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=8,
                                         quantize_bits=8), "quantized"),
    (lambda: tessera.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=8,
                                         dtype="fp16"), "contiguous f32"),
    (lambda: object(), "KV"),
])
def test_unrepresentable_handles_are_refused(make, why):
    with pytest.raises(ValueError, match=f"X86_KV_CACHE_HANDLE_REFUSED.*{why}"):
        rt.x86_kv_cache_cursor(make(), "tessera.cache.commit", 0)


def test_a_non_contiguous_buffer_is_refused():
    handle = _filled(3, 4)
    handle.keys = np.asfortranarray(handle.keys)
    with pytest.raises(ValueError, match="X86_KV_CACHE_HANDLE_REFUSED"):
        rt.x86_kv_cache_cursor(handle, "tessera.cache.rollback", 1)


@pytest.mark.parametrize("op,kwargs", [
    ("tessera.cache.commit", {"accepted_length": 2}),
    ("tessera.cache.rollback", {"current_seq": 4}),
])
def test_the_lane_never_defaults_a_semantic_key(op, kwargs):
    buffers = [np.zeros((8, 2, 4), np.float32)] * 2
    with pytest.raises(ValueError, match="they never default"):
        rt._execute_x86_compiled_kv_cache_cursor(op, buffers, kwargs)


def test_an_unknown_cursor_op_is_refused():
    with pytest.raises(ValueError, match="x86 KV-cache cursor handles"):
        rt.x86_kv_cache_cursor(_filled(5, 3), "tessera.cache.page_lookup", 1)


# ── Zen 5: native numerics against the Python references ────────────────────

@pytest.mark.compiler_avx512
@pytest.mark.parametrize("accepted", [0, 3, 7])
def test_native_commit_matches_the_reference(accepted):
    _native_or_skip()
    handle = _filled(11, 7)
    expected = tessera.ops.cache_commit(copy.deepcopy(handle), accepted)
    returned = rt.x86_kv_cache_cursor(handle, "tessera.cache.commit", accepted)
    assert returned is handle
    _assert_same(handle, expected)


@pytest.mark.compiler_avx512
@pytest.mark.parametrize("rejected", [0, 2, 7, 30])
def test_native_rollback_matches_the_reference(rejected):
    _native_or_skip()
    handle = _filled(13, 7)
    expected = tessera.ops.cache_rollback(copy.deepcopy(handle), rejected)
    rt.x86_kv_cache_cursor(handle, "tessera.cache.rollback", rejected)
    _assert_same(handle, expected)


@pytest.mark.compiler_avx512
def test_native_commit_then_rollback_chain_matches_the_reference():
    _native_or_skip()
    handle = _filled(17, 9)
    expected = tessera.ops.cache_rollback(
        tessera.ops.cache_commit(copy.deepcopy(handle), 6), 2
    )
    rt.x86_kv_cache_cursor(
        rt.x86_kv_cache_cursor(handle, "tessera.cache.commit", 6),
        "tessera.cache.rollback", 2,
    )
    _assert_same(handle, expected)


@pytest.mark.compiler_avx512
@pytest.mark.parametrize("op,count", [
    ("tessera.cache.commit", 8),    # beyond current_seq (the reference raises)
    ("tessera.cache.commit", -1),
    ("tessera.cache.rollback", -1),
])
def test_native_abi_rejects_what_the_reference_rejects(op, count):
    _native_or_skip()
    handle = _filled(19, 7)
    before = copy.deepcopy(handle)
    reference = (tessera.ops.cache_commit if op == "tessera.cache.commit"
                 else tessera.ops.cache_rollback)
    with pytest.raises(ValueError):
        reference(copy.deepcopy(handle), count)
    with pytest.raises(ValueError, match="handle ABI rejected"):
        rt.x86_kv_cache_cursor(handle, op, count)
    _assert_same(handle, before)


@pytest.mark.compiler_avx512
@pytest.mark.parametrize("op,count_key,count", [
    ("tessera.cache.commit", "accepted_length", 4),
    ("tessera.cache.rollback", "num_rejected", 3),
])
def test_bufferized_lane_matches_the_reference(op, count_key, count):
    _native_or_skip()
    handle = _filled(23, 8)
    reference = (tessera.ops.cache_commit if op == "tessera.cache.commit"
                 else tessera.ops.cache_rollback)
    expected = reference(copy.deepcopy(handle), count)
    artifact = rt.RuntimeArtifact(metadata={
        "target": "x86",
        "compiler_path": "x86_kv_cache_compiled",
        "executable": True,
        "execution_kind": "native_cpu",
        "execution_mode": "cpu_avx512",
        "arg_names": ["keys", "values"],
        "output_name": "out",
        "ops": [{
            "op_name": op, "result": "out", "operands": ["keys", "values"],
            "kwargs": {"current_seq": handle.current_seq, count_key: count},
        }],
    })
    result = rt.launch(artifact, (handle.keys, handle.values))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "x86_kv_cache_compiled"
    assert result["execution_kind"] == "native_cpu"
    keys, values, current_seq = result["output"]
    assert int(current_seq) == expected.current_seq
    np.testing.assert_array_equal(keys, expected.keys)
    np.testing.assert_array_equal(values, expected.values)
    # The bufferized lane is value-semantic: the bound buffers are untouched.
    assert handle.current_seq == 8
