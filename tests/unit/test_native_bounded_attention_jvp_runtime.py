"""Compiler-route proof of bounded host preflight; no GPU execution claim."""
from pathlib import Path
import os

import ml_dtypes
import numpy as np
import pytest
import tessera as ts

from tessera.compiler import native_attention_jvp_runtime as runtime
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.device.nvidia.test_public_typed_attention_vjp import function

pytestmark = pytest.mark.compiler_route


@pytest.fixture(scope="module", params=["float16", "bfloat16"])
def bounded_program(request):
    compiler = find_tessera_opt()
    if compiler is None:
        pytest.skip("matching native compiler required")
    dtype = np.dtype(ml_dtypes.bfloat16 if request.param == "bfloat16" else request.param)
    roots = {name: np.zeros(shape, dtype) for name, shape in {
        "q": (1, 2, 1, 16), "k": (1, 1, 1, 16), "v": (1, 1, 1, 16),
    }.items()}
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("q", "k", "v"))(
        function(True, False, True))
    program = fn.compile_native_attention_jvp(
        **roots, compiler=Path(compiler), llvm_bin=Path(os.environ["TESSERA_LLVM_BIN"]),
        sequence_bounds=(7, 9))
    metadata = dict(program_json=program.to_json(), program_digest=program.program_digest,
                    arg_names=["primal_0", "primal_1", "primal_2",
                               "tangent_0", "tangent_1", "tangent_2"])
    return metadata, dtype


@pytest.fixture(autouse=True)
def native_preparation_boundary(monkeypatch):
    runtime.clear_prepared()
    def checked_native_preparation(owner):
        raise RuntimeError("checked native preparation reached")
    monkeypatch.setattr(runtime.PreparedAttentionJVP, "_prepare", checked_native_preparation)
    yield
    runtime.clear_prepared()


def inputs(dtype, sq, sk):
    primals = tuple(np.zeros(shape, dtype) for shape in
                    ((1, 2, sq, 16), (1, 1, sk, 16), (1, 1, sk, 16)))
    return primals + tuple(np.zeros_like(value) for value in primals)


@pytest.mark.parametrize("sq,sk", [(1, 1), (3, 4), (7, 9)])
def test_bounded_host_shape_reaches_native_preparation(bounded_program, sq, sk):
    metadata, dtype = bounded_program
    with pytest.raises(RuntimeError, match="checked native preparation reached"):
        runtime.execute(metadata, inputs(dtype, sq, sk))


@pytest.mark.parametrize("sq,sk", [(0, 4), (8, 4), (3, 0), (3, 10)])
def test_bounded_host_capacity_rejects_before_native_setup(bounded_program, sq, sk):
    metadata, dtype = bounded_program
    with pytest.raises(ValueError, match="sequence|capacity|envelope"):
        runtime.execute(metadata, inputs(dtype, sq, sk))
