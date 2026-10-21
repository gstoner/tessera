"""Execute compiler-generated shape MLIR through LLVM, independently of CUDA."""
import base64
import ctypes as ct
from dataclasses import replace
import json
from pathlib import Path
import re
import tempfile

import pytest

from tessera.compiler.native_tensor_shape import build_shape_companion
from tests.unit.test_native_tensor_program_pipeline import _run
from tests.unit.test_native_tensor_shape_ownership import shape_source

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


@pytest.fixture(scope="module", params=["norm_lhs", "rhs_add", "chain", "two_matmuls"])
def native_shape(request):
    result = _run(shape_source(request.param), "--verify-each", "--tessera-graph-to-schedule",
                  "--tessera-schedule-to-tile", "--tessera-lower-to-nvidia-sm120")
    assert result.returncode == 0, result.stderr
    record = re.search(r'tessera.native.tensor_program_frame_json = "([^"]+)"', result.stdout)
    assert record
    frame = json.loads(base64.b64decode(record[1], validate=True))
    companion = build_shape_companion(frame["shape_program"])
    with tempfile.TemporaryDirectory(prefix="tessera-shape-test-") as tmp:
        library_path = Path(tmp) / "shape.so"
        library_path.write_bytes(companion.image)
        library = ct.CDLL(str(library_path))
        function = getattr(library, companion.entry)
        function.argtypes = [ct.POINTER(ct.c_int64), ct.POINTER(ct.c_int64)]
        function.restype = ct.c_int32
        yield request.param, frame, companion, function


def oracle(case, m, n, k):
    a, b = [m, k], [k, n]
    if case == "two_matmuls":
        roots = [[m, 32], [32, k], b]
        private = [[m, k], [m, k], [m, n]]
        scalar_values = [[m, k, 32, 32, k, k], [m, k], [m, n, k]]
        grids = [[(k + 7) // 8, (m + 15) // 16, 1], [(m + 127) // 128, 1, 1],
                 [(n + 7) // 8, (m + 15) // 16, 1]]
    else:
        roots = [a, b] + ([b] if case == "rhs_add" else [a] if case == "chain" else [])
        private = ([b, a] if case == "rhs_add" else [a, a] if case == "chain" else [a]) + [[m, n]]
        scalar_values = ([[k * n], [m, k]] if case == "rhs_add" else
                         [[m * k], [m, k]] if case == "chain" else [[m, k]])
        grids = ([[(k * n + 127) // 128, 1, 1], [(m + 127) // 128, 1, 1]] if case == "rhs_add" else
                 [[(m * k + 127) // 128, 1, 1], [(m + 127) // 128, 1, 1]] if case == "chain" else
                 [[(m + 127) // 128, 1, 1]])
        scalar_values.append([m, n, k, k, n, n])
        grids.append([(n + 7) // 8, (m + 15) // 16, 1])
    words = [value for shape in roots + private for value in shape]
    for scalars, grid in zip(scalar_values, grids, strict=True):
        words.extend(scalars + [0] * (8 - len(scalars)) + grid)
    return roots, words


def invoke(native_shape, roots):
    _, frame, companion, function = native_shape
    source = (ct.c_int64 * companion.root_words)(*(value for shape in roots for value in shape))
    sentinel = -0x12345678
    storage = (ct.c_int64 * (companion.plan_words + 2))(*([sentinel] * (companion.plan_words + 2)))
    destination = ct.cast(ct.byref(storage, ct.sizeof(ct.c_int64)), ct.POINTER(ct.c_int64))
    before = tuple(source)
    status = function(source, destination)
    assert tuple(source) == before
    assert storage[0] == sentinel and storage[-1] == sentinel
    assert companion.plan_words == 2 * len(frame["buffers"]) + 11 * len(frame["stages"])
    return status, list(storage)[1:-1], sentinel


@pytest.mark.parametrize("mnk", [(1, 1, 1), (17, 11, 23), (65, 47, 64), (5, 31, 7), (17, 17, 17)])
def test_compiled_shape_plan_matches_independent_oracle(native_shape, mnk):
    case, _, _, _ = native_shape
    roots, expected = oracle(case, *mnk)
    status, actual, _ = invoke(native_shape, roots)
    assert status == 0
    assert actual == expected


@pytest.mark.parametrize("fault", ["zero", "negative", "m_capacity", "n_capacity", "k_capacity",
                                   "contraction_mismatch", "overflow", "producer_mismatch"])
def test_invalid_root_shape_leaves_all_plan_words_untouched(native_shape, fault):
    case, _, _, _ = native_shape
    roots, _ = oracle(case, 17, 11, 23)
    roots = [list(value) for value in roots]
    if fault == "zero":
        roots[0][0] = 0
    elif fault == "negative":
        roots[0][0] = -1
    elif fault == "m_capacity":
        roots[0][0] = 66
    elif fault == "n_capacity":
        roots[-1 if case == "two_matmuls" else 1][1] = 48
    elif fault == "k_capacity":
        roots[1 if case == "two_matmuls" else 0][1] = 65
    elif fault == "overflow":
        roots[0][0] = 2**63 - 1
    elif fault == "contraction_mismatch":
        roots[-1 if case == "two_matmuls" else 1][0] += 1
    elif case in {"chain", "rhs_add"}:
        roots[2][0] += 1
    elif case == "two_matmuls":
        roots[0][1] = 31
    else:
        roots[1][0] = 32
    status, actual, sentinel = invoke(native_shape, roots)
    assert status != 0
    assert all(value == sentinel for value in actual)


def test_companion_binds_typed_ir_and_native_library(native_shape):
    _, frame, companion, _ = native_shape
    companion.validate(frame["shape_program"])
    with pytest.raises(ValueError, match="differs"):
        replace(companion, image=companion.image + b"changed").validate(frame["shape_program"])
    changed = {**frame["shape_program"], "ir": frame["shape_program"]["ir"] + "\n"}
    with pytest.raises(ValueError, match="transport differs"):
        build_shape_companion(changed)


@pytest.mark.parametrize("fault", ["division_by_zero", "memory_read", "extra_function", "wrong_arity"])
def test_native_export_rejects_unsafe_member_shape_ir(fault):
    result = _run(shape_source("norm_lhs"), "--verify-each", "--tessera-graph-to-schedule",
                  "--tessera-schedule-to-tile")
    assert result.returncode == 0, result.stderr
    count = 4 if fault == "wrong_arity" else 5
    body = ""
    returned = "%a"
    if fault == "division_by_zero":
        body = "%zero = arith.constant 0 : i64\n%v = arith.divui %a, %zero : i64\n"
        returned = "%v"
    elif fault == "memory_read":
        body = "%ptr = llvm.mlir.undef : !llvm.ptr\n%v = llvm.load %ptr : !llvm.ptr -> i64\n"
        returned = "%v"
    extra = "func.func private @external()\n" if fault == "extra_function" else ""
    signature = ", ".join(["i64"] * count)
    values = ", ".join([returned] * count)
    source = (f"module {{ {extra}func.func @member(%a:i64,%b:i64,%c:i64,%d:i64) -> ({signature}) {{\n"
              f"{body}return {values} : {signature}\n}} }}")
    literal = '"' + source.replace("\\", "\\5C").replace('"', "\\22").replace("\n", "\\0A") + '"'
    changed, replacements = re.subn(r'((?:tessera\.native\.program_)?shape_ir = )"(?:\\.|[^"\\])*"',
                                    lambda match: match[1] + literal, result.stdout)
    assert replacements == 4
    refused = _run(changed, "--verify-each", "--tessera-schedule-to-tile")
    assert refused.returncode > 0, refused.stderr
    assert "native shape export requires verified pure member shape functions" in refused.stderr
