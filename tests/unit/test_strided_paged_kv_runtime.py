"""Strided page metadata admission and exact-device native execution."""
import copy
import ctypes as ct
import os
from unittest.mock import patch

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.native_artifact import BufferBinding, LaunchDescriptor, ScalarArgument
from tessera.compiler.rocm_native import GFX_PAGED_KV_STRIDED_F32_ABI
from tessera.compiler.scheduled_paged_kv import project_paged_host_storage
from benchmarks.rocm.benchmark_rocm_e2e_movement import _paged_module
from tests.unit.test_public_movement_frontend import paged


def pages(kind):
    rng = np.random.default_rng(81520)
    if kind == "compact":
        return rng.normal(size=(4, 4, 3, 8)).astype(np.float32)
    if kind == "padded":
        return rng.normal(size=(5, 5, 3, 18)).astype(np.float32)[1:, 1:, :, 1:17:2]
    if kind == "permuted":
        return rng.normal(size=(8, 3, 4, 4)).astype(np.float32).transpose(3, 2, 1, 0)
    if kind == "fortran":
        return np.array(pages("compact"), order="F")
    raise ValueError(kind)


def descriptor():
    return LaunchDescriptor(
        image_digest="0"*64, entry_symbol="strided", abi_id=GFX_PAGED_KV_STRIDED_F32_ABI,
        buffers=(BufferBinding(0, "pages", "input", "fp32", 4, "strided", 4),),
        scalars=tuple(ScalarArgument(i+1, name, "int64")
                      for i, name in enumerate(("StrideP", "StridePage", "StrideH", "StrideD"))),
    )


@pytest.mark.parametrize("kind", ["compact", "padded", "permuted", "fortran"])
def test_descriptor_storage_facts_match_the_actual_strides(kind):
    x = pages(kind)
    scalars = dict(zip(("StrideP", "StridePage", "StrideH", "StrideD"),
                       (s//4 for s in x.strides), strict=True))
    values, facts, actual = rt._split_native_arguments(descriptor(), {"pages": x, **scalars})
    assert values["pages"] is x
    assert facts["pages"].layout == "strided"
    assert actual == scalars


@pytest.mark.parametrize("stride", [0, 1, -1, True])
def test_serialized_stride_scalars_cannot_override_storage_facts(stride):
    x = pages("padded")
    scalars = dict(zip(("StrideP", "StridePage", "StrideH", "StrideD"),
                       (s//4 for s in x.strides), strict=True))
    scalars["StrideP"] = stride
    with pytest.raises(ValueError, match="stride scalars"):
        rt._split_native_arguments(descriptor(), {"pages": x, **scalars})


def test_storage_projection_is_pitch_independent_and_does_not_mutate_the_trace():
    module = _paged_module(4, 4, 3, 8, 1, 5)
    original = copy.deepcopy(module)
    table = np.array([2, 0, 3, 1], np.int32)
    padded = project_paged_host_storage(module, (pages("padded"), table))
    permuted = project_paged_host_storage(module, (pages("permuted"), table))
    assert module == original
    assert padded == permuted
    assert padded.functions[0].args[0].layout == "strided"


@pytest.mark.skipif(os.environ.get("TESSERA_ROCM_MOVEMENT_DEVICE_PROOF") != "1",
                    reason="requires owning ROCm native movement proof")
@pytest.mark.parametrize("kind", ["padded", "permuted", "fortran"])
def test_public_jit_serialized_and_resident_strided_pages_execute(kind):
    arch = rt._rocm_live_arch()
    assert arch == os.environ["TESSERA_ROCM_CHIP"] and arch in {"gfx1151", "gfx1201"}
    x = pages(kind)
    table = np.array([2, 0, 3, 1], np.int32)
    expected = x[table].reshape(16, 3, 8)[1:6]
    fn = ts.jit(target="rocm_"+arch, native_required=True)(paged)
    first = fn(x, table)
    assert fn.compile_result.executable
    descriptor = fn.compile_result.launch_descriptor
    assert descriptor.abi_id == GFX_PAGED_KV_STRIDED_F32_ABI
    np.testing.assert_array_equal(first, expected)
    original = first.copy()
    artifact = fn.compile_result.to_runtime_artifact()
    with patch("subprocess.run", side_effect=AssertionError("warm path invoked compiler")):
        x[...] += np.float32(0.5)
        expected = x[table].reshape(16, 3, 8)[1:6]
        np.testing.assert_array_equal(fn(x, table), expected)
        np.testing.assert_array_equal(first, original)
        scalars = dict(zip(("P", "LP", "PageSize", "H", "D", "Start", "Tokens",
                           "StrideP", "StridePage", "StrideH", "StrideD"),
                          (4, 4, 4, 3, 8, 1, 5, *(s//4 for s in x.strides)), strict=True))
        out = np.empty_like(expected)
        inputs = {descriptor.buffers[0].name: x, descriptor.buffers[1].name: table,
                  descriptor.buffers[2].name: out}
        receipt = rt.launch(artifact, {"buffers": inputs, "scalars": scalars})
        assert receipt["ok"], receipt
        np.testing.assert_array_equal(receipt["output"], expected)
        readonly = np.empty_like(expected)
        readonly.setflags(write=False)
        invalid = dict(inputs)
        invalid[descriptor.buffers[2].name] = readonly
        rejected = rt.launch(artifact, {"buffers": invalid, "scalars": scalars})
        assert not rejected["ok"] and "writable" in rejected["reason"]
        # The raw native service must refuse output aliases of the borrowed
        # physical span even though it stages inputs before executing.
        aliased = np.ctypeslib.as_array((ct.c_float*expected.size).from_address(x.ctypes.data)).reshape(expected.shape)
        invalid[descriptor.buffers[2].name] = aliased
        rejected = rt.launch(artifact, {"buffers": invalid, "scalars": scalars})
        assert not rejected["ok"] and "rc=1" in rejected["reason"]
        owner = fn.prepare_native_movement(x, table)
        try:
            assert owner.capture() == 1
            result, _ = owner.execute(captured=True)
            np.testing.assert_array_equal(result, expected)
            x[...] *= np.float32(0.5)
            table[:] = [1, 3, 0, 2]
            owner.upload((x, table))
            result, _ = owner.execute(captured=True)
            np.testing.assert_array_equal(result, x[table].reshape(16, 3, 8)[1:6])
            with pytest.raises(RuntimeError, match="upload failed"):
                owner.upload((np.array(x, order="C"), table))
        finally:
            owner.close()


@pytest.mark.skipif(os.environ.get("TESSERA_ROCM_MOVEMENT_DEVICE_PROOF") != "1",
                    reason="requires owning ROCm native movement proof")
@pytest.mark.parametrize("kind", ["padded", "permuted", "fortran"])
def test_strided_readonly_pages_default_end_and_reordered_arguments(kind):
    from tests.unit.test_public_movement_frontend import paged_default
    arch = rt._rocm_live_arch()
    assert arch == os.environ["TESSERA_ROCM_CHIP"]
    x = pages(kind)
    x.setflags(write=False)
    table = np.array([2, 0, 3, 1], np.int32)
    expected = x[table].reshape(16, 3, 8)[3:4]
    fn = ts.jit(target="rocm_"+arch, native_required=True)(paged_default)
    np.testing.assert_array_equal(fn(table=table, pages=x), expected)
    with patch("subprocess.run", side_effect=AssertionError("warm path invoked compiler")):
        owner = fn.prepare_native_movement(table, x)
        try:
            assert owner.capture() == 1
            np.testing.assert_array_equal(owner.execute(captured=True)[0], expected)
        finally:
            owner.close()


@pytest.mark.skipif(os.environ.get("TESSERA_ROCM_MOVEMENT_DEVICE_PROOF") != "1",
                    reason="requires owning ROCm native movement proof")
@pytest.mark.parametrize("kind", ["padded", "permuted", "fortran"])
def test_strided_native_paged_softmax_consumes_the_owned_compact_tensor(kind):
    from benchmarks.rocm.benchmark_paged_softmax_edge import normalized
    arch = rt._rocm_live_arch()
    assert arch == os.environ["TESSERA_ROCM_CHIP"]
    x = pages(kind)
    table = np.array([2, 0, 3, 1], np.int32)
    values = x[table].reshape(16, 3, 8)[1:6].astype(np.float64)
    exponent = np.exp(values-values.max(-1, keepdims=True))
    expected = exponent/exponent.sum(-1, keepdims=True)
    fn = ts.jit(target="rocm_"+arch, native_required=True)(paged)
    consumer = ts.jit(target="rocm_"+arch, native_required=True)(normalized)
    owner = fn.prepare_native_paged_softmax(consumer, x, table)
    try:
        assert owner.capture() == 2
        with patch("subprocess.run", side_effect=AssertionError("warm edge invoked compiler")):
            result, receipt = owner.execute(captured=True)
        np.testing.assert_allclose(result, expected, rtol=3e-5, atol=2e-6)
        assert receipt["device_event_scope"] == "whole_sequence"
    finally:
        owner.close()


@pytest.mark.skipif(os.environ.get("TESSERA_ROCM_MOVEMENT_DEVICE_PROOF") != "1",
                    reason="requires owning ROCm native movement proof")
def test_readonly_self_aliasing_positive_stride_pages_have_a_bounded_native_span():
    arch = rt._rocm_live_arch()
    assert arch == os.environ["TESSERA_ROCM_CHIP"]
    backing = np.arange(16, dtype=np.float32)
    x = np.lib.stride_tricks.as_strided(
        backing, shape=(4, 4, 3, 8), strides=(4, 4, 4, 4), writeable=False)
    table = np.array([2, 0, 3, 1], np.int32)
    expected = x[table].reshape(16, 3, 8)[1:6]
    fn = ts.jit(target="rocm_"+arch, native_required=True)(paged)
    np.testing.assert_array_equal(fn(x, table), expected)
    with patch("subprocess.run", side_effect=AssertionError("warm alias invoked compiler")):
        owner = fn.prepare_native_movement(x, table)
        try:
            assert owner.capture() == 1
            np.testing.assert_array_equal(owner.execute(captured=True)[0], expected)
        finally:
            owner.close()
