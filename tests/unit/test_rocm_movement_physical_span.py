"""Native byte-footprint proof before any HIP or pointer access."""
import ctypes as ct
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def span(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host C++ compiler")
    root = Path(__file__).resolve().parents[2]
    include = root / "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip"
    temporary = tmp_path_factory.mktemp("native-physical-span")
    source = temporary / "span.cpp"
    library = temporary / "span.so"
    source.write_text("""
#include "MovementPhysicalSpan.h"
extern "C" bool span(const int64_t *dimensions, size_t *bytes) {
  return tessera::rocm::pagedPhysicalSpan(dimensions, *bytes);
}
""")
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-I", str(include),
                    str(source), "-o", str(library)], check=True, capture_output=True, text=True)
    lib = ct.CDLL(str(library))
    lib.span.argtypes = [ct.POINTER(ct.c_int64), ct.POINTER(ct.c_size_t)]
    lib.span.restype = ct.c_bool

    def check(shape, strides):
        p, page, h, d = shape
        dimensions = (ct.c_int64 * 11)(p, p, page, h, d, 0, 1, *strides)
        size = ct.c_size_t()
        return lib.span(dimensions, ct.byref(size)), size.value

    return check


@pytest.mark.parametrize("shape,strides,bytes_", [
    ((4, 3, 2, 5), (30, 10, 5, 1), 480),
    ((4, 3, 2, 5), (48, 16, 8, 1), 756),
    ((4, 3, 2, 5), (1, 4, 12, 24), 480),
    ((4, 3, 2, 5), (64, 1, 16, 3), 892),
    # Read-only self-aliasing is legal; output aliasing is checked elsewhere.
    ((4, 3, 2, 5), (1, 1, 1, 1), 44),
])
def test_addressed_span_covers_padded_permuted_and_aliased_pages(span, shape, strides, bytes_):
    assert span(shape, strides) == (True, bytes_)


@pytest.mark.parametrize("shape,strides", [
    ((0, 3, 2, 5), (30, 10, 5, 1)),
    ((4, -3, 2, 5), (30, 10, 5, 1)),
    ((4, 3, 2, 5), (0, 10, 5, 1)),
    ((4, 3, 2, 5), (30, -10, 5, 1)),
    ((2, 1, 1, 1), (2**63-1, 1, 1, 1)),
    ((2**63-1, 2, 1, 1), (1, 1, 1, 1)),
])
def test_invalid_or_overflowing_span_is_rejected_before_pointer_access(span, shape, strides):
    assert not span(shape, strides)[0]


def test_exact_signed_byte_extent_boundary(span):
    max_elements = (2**63-1)//4
    ok, bytes_ = span((2, 1, 1, 1), (max_elements-1, 1, 1, 1))
    assert ok and bytes_ == max_elements*4
    assert not span((2, 1, 1, 1), (max_elements, 1, 1, 1))[0]
