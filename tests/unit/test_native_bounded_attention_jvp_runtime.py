"""Host-free: bounded preflight and policy guards; marked cases need a compiler."""
from pathlib import Path
import os

import ml_dtypes
import numpy as np
import pytest
import tessera as ts

from tessera.compiler import native_attention_jvp_runtime as runtime
from tessera.compiler.scheduled_matmul import find_tessera_opt

def attention(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=True, lse_checkpoint="saved")


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
        attention)
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


@pytest.mark.compiler_route
@pytest.mark.parametrize("sq,sk", [(1, 1), (3, 4), (7, 9)])
def test_bounded_host_shape_reaches_native_preparation(bounded_program, sq, sk):
    metadata, dtype = bounded_program
    with pytest.raises(RuntimeError, match="checked native preparation reached"):
        runtime.execute(metadata, inputs(dtype, sq, sk))


@pytest.mark.compiler_route
@pytest.mark.parametrize("sq,sk", [(0, 4), (8, 4), (3, 0), (3, 10)])
def test_bounded_host_capacity_rejects_before_native_setup(bounded_program, sq, sk):
    metadata, dtype = bounded_program
    with pytest.raises(ValueError, match="sequence|capacity|envelope"):
        runtime.execute(metadata, inputs(dtype, sq, sk))


@pytest.mark.parametrize("bounds", [(0, 9), (7,), (True, 9), (7, 9.0), "7,9"])
def test_public_invalid_bounds_reject_before_frontend(bounds, monkeypatch):
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("q", "k", "v"))(
        attention)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid sequence capacity reached tracing")
    monkeypatch.setattr(fn, "_specialized_autodiff_module", forbidden)
    with pytest.raises(ts.TesseraJitError, match="positive Sq/Sk"):
        fn.native_jvp(tangents=(), sequence_bounds=bounds)


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151", "rocm_gfx1201", "apple_gpu"])
def test_public_bounded_jvp_keeps_owning_target(target):
    fn = ts.jit(target=target, autodiff="forward", wrt=("q", "k", "v"))(
        attention)
    with pytest.raises(ts.TesseraJitError, match="exact nvidia_sm120"):
        fn.native_jvp(tangents=(), sequence_bounds=(7, 9))


def shape_sensitive_attention(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=q.shape[2] == 1, lse_checkpoint="saved")


@pytest.mark.parametrize("sq", [1, 3])
def test_shape_dependent_python_policy_needs_a_bounded_graph_contract(sq, monkeypatch):
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("q", "k", "v"))(
        shape_sensitive_attention)
    def forbidden(*args, **kwargs):
        pytest.fail("trace-dependent numerical policy reached native compilation")
    monkeypatch.setattr(fn, "compile_native_attention_jvp", forbidden)
    roots = tuple(np.zeros(shape, np.float16) for shape in
                  ((1, 2, sq, 16), (1, 1, 4, 16), (1, 1, 4, 16)))
    with pytest.raises(ts.TesseraJitError, match="frontend|certificate|structural"):
        fn.native_jvp(*roots, tangents=tuple(np.zeros_like(x) for x in roots),
                      sequence_bounds=(7, 9))
