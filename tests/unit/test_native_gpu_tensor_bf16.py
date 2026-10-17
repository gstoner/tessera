"""Semantic BF16 must retain its native two-byte tensor ABI without reinterpretation."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tessera.compiler.native_gpu_tensor import IndexSpec, NativeTensorCall, TensorSpec
from tests.unit.test_native_gpu_tensor import binding


def typed_binding(dtype="bf16"):
    original = binding()
    return NativeTensorCall(original.package, original.signature,
        (TensorSpec("output", dtype, ("n",), True), IndexSpec("n")),
        grid=original.grid, block=original.block)


def tensor(*, semantic=True, typestr=None, **changes):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    dtype = np.dtype(ml_dtypes.bfloat16)
    interface = dict(version=3, shape=(32,), typestr=typestr or dtype.str,
                     data=(4096, False), strides=None)
    interface.update(changes)
    values = dict(__cuda_array_interface__=interface)
    if semantic:
        values["dtype"] = dtype
    return SimpleNamespace(**values)


@pytest.mark.parametrize("typestr", [None, "|V2", "=V2"])
def test_bf16_semantic_storage_retains_two_byte_span(typestr):
    value = tensor(typestr=typestr)
    raw, spans, grid, block, outputs = typed_binding().prepare(value, 32)
    assert raw == (4096, 32)
    assert spans == [(4096, 64)]
    assert grid == (1, 1, 1) and block == (32, 1, 1)
    assert outputs == (value,)


@pytest.mark.parametrize("kind", ["missing", "uint16", "fp16", "physical_uint16", "foreign_endian"])
def test_bf16_requires_consistent_semantic_and_physical_storage(kind):
    value = tensor(semantic=kind != "missing")
    if kind == "uint16":
        value.dtype = np.dtype("uint16")
    elif kind == "fp16":
        value.dtype = np.dtype("float16")
    elif kind == "physical_uint16":
        value.__cuda_array_interface__["typestr"] = np.dtype("uint16").str
    elif kind == "foreign_endian":
        value.__cuda_array_interface__["typestr"] = (">" if sys.byteorder == "little" else "<") + "V2"
    with pytest.raises(ValueError):
        typed_binding().prepare(value, 32)


@pytest.mark.parametrize("changes,message", [
    (dict(data=(4097, False)), "misaligned"),
    (dict(data=(4096, True)), "read-only"),
    (dict(strides=(4,)), "contiguous"),
])
def test_bf16_preserves_alignment_readonly_and_stride_guards(changes, message):
    with pytest.raises(ValueError, match=message):
        typed_binding().prepare(tensor(**changes), 32)


def test_fp16_storage_still_uses_its_physical_dtype_without_semantic_override():
    value = SimpleNamespace(__cuda_array_interface__=dict(
        version=3, shape=(32,), typestr=np.dtype("float16").str,
        data=(4096, False), strides=(2,)))
    assert typed_binding("fp16").prepare(value, 32)[1] == [(4096, 64)]


@pytest.mark.parametrize("semantic", ["bf16", "bfloat16"])
def test_bf16_semantic_aliases_preserve_resident_tape_storage(semantic):
    value = tensor()
    value.dtype = semantic
    assert typed_binding().prepare(value, 32)[1] == [(4096, 64)]
