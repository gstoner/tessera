"""Compile the production byte packer; prove spans and refusal before writes."""
import ctypes as c
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from tessera.compiler.native_scaled_program import pack_host_view
from tessera.compiler.paged_host_span import checked_host_span


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("native host C++ compiler required")
    root = Path(__file__).resolve().parents[2]
    source = root / "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.cpp"
    text = source.read_text()
    begin = text.index('extern "C" int tessera_rocm_program_pack_host_view(')
    end = text.index('extern "C" int tessera_rocm_program_prepare(', begin)
    folder = tmp_path_factory.mktemp("native-host-view")
    cpp = folder / "pack.cpp"
    cpp.write_text("#include <cstdint>\n#include <cstring>\n#include <type_traits>\n" + text[begin:end])
    output = folder / "pack.so"
    subprocess.run([compiler, "-O2", "-std=c++17", "-shared", "-fPIC",
                    str(cpp), "-o", str(output)], check=True, capture_output=True)
    return c.CDLL(str(output))


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("layout", ["permuted", "padded", "owned_permutation", "self_alias"])
def test_native_pack_moves_exact_bytes_and_recopies_changed_source(library, dtype, layout):
    source = np.arange(2*3*8, dtype=dtype).reshape(2, 3, 8)
    if layout == "permuted":
        view = source.transpose(2, 0, 1)
    elif layout == "padded":
        view = source[..., 1::2].transpose(2, 0, 1)
    elif layout == "owned_permutation":
        view = np.stack([source[0].T, source[1].T], axis=1)
        assert view.flags.owndata and not view.flags.c_contiguous and not view.flags.f_contiguous
    else:
        view = np.lib.stride_tricks.as_strided(
            source, shape=(3, 3, 8), strides=(source.itemsize*8, source.itemsize*8, source.itemsize))
    packed = pack_host_view(library, view)
    saved = packed.copy()
    assert packed.flags.c_contiguous
    assert not np.shares_memory(view, packed)
    np.testing.assert_array_equal(packed, view)
    view[...] = 7
    np.testing.assert_array_equal(pack_host_view(library, view), view)
    np.testing.assert_array_equal(packed, saved)


@pytest.mark.parametrize("stride", [-4, 0, 3, 2**63-4])
def test_unproved_tensor_stride_is_refused_without_native_read(library, stride):
    source = np.empty(4, np.float32)
    view = np.lib.stride_tricks.as_strided(source, shape=(2, 2), strides=(stride, 4))
    with pytest.raises(ValueError, match="strides|extent"):
        pack_host_view(library, view)


@pytest.mark.parametrize("pitch", [16, 64])
def test_forged_capacity_is_refused_before_pack(library, pitch):
    source = np.empty(4, np.float32)
    view = np.lib.stride_tricks.as_strided(source, shape=(4, 4), strides=(pitch, 4))
    with pytest.raises(ValueError, match="backing allocation"):
        pack_host_view(library, view)


def test_numpy_owned_positive_self_alias_uses_its_physical_span():
    source = np.ndarray((2, 3), dtype=np.float32, strides=(4, 4))
    assert source.flags.owndata
    assert checked_host_span(source)[0] == 16


@pytest.mark.parametrize("mutation", ["span", "capacity", "zero_extent", "overflow",
                                     "fractional", "overlap", "rank", "itemsize"])
def test_native_admission_refuses_before_destination_write(library, mutation):
    source = np.arange(24, dtype=np.float32).reshape(4, 6).T
    destination = np.full(source.shape, -99, np.float32)
    span, strides = checked_host_span(source)
    shape = list(source.shape)
    rank, itemsize, capacity = 2, 4, destination.nbytes
    if mutation == "span":
        span -= 4
    elif mutation == "capacity":
        capacity -= 4
    elif mutation == "zero_extent":
        shape[0] = 0
    elif mutation == "overflow":
        shape[0] = 2**63-1
    elif mutation == "fractional":
        strides = (3, strides[1])
    elif mutation == "overlap":
        destination = source
    elif mutation == "rank":
        rank = 33
    else:
        itemsize = 0
    before = destination.copy()
    pack = library.tessera_rocm_program_pack_host_view
    pack.argtypes = [c.c_void_p, c.c_uint64, c.c_uint32, c.POINTER(c.c_uint64),
                    c.POINTER(c.c_uint64), c.c_uint32, c.c_void_p, c.c_uint64]
    result = pack(c.c_void_p(source.ctypes.data), span, rank,
                  (c.c_uint64*2)(*shape), (c.c_uint64*2)(*strides), itemsize,
                  c.c_void_p(destination.ctypes.data), capacity)
    assert result == 1
    np.testing.assert_array_equal(destination, before)
