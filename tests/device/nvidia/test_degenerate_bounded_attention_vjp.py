"""Trace-at-one bias storage survives native bounded reverse replay."""
import os
import subprocess

import numpy as np
import pytest
import tessera as ts

from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lse_cotangent_native import oracle

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("trace,broadcast", [((1, 4), "query"), ((3, 1), "key"), ((1, 1), "both")])
@pytest.mark.parametrize("compact", [False, True])
def test_degenerate_bias_axes_keep_native_reverse_storage(trace, broadcast, compact, monkeypatch):
    if not nvidia_cuda_host_ready():
        pytest.skip("exact SM120 CUDA host required")

    @ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("bias", "v", "q", "k"))
    def attention(q, k, v, bias):
        return ts.ops.flash_attn(q, k, v, attn_bias=bias, causal=True, lse_checkpoint="saved")

    def inputs(sq, sk):
        rng = np.random.default_rng(202610142 + sq * 31 + sk)
        physical = (1, 2, 1 if broadcast in {"query", "both"} else sq,
                    1 if broadcast in {"key", "both"} else sk)
        values = [rng.normal(0, .2, shape).astype(np.float32) for shape in
                  ((1, 2, sq, 8), (1, 1, sk, 8), (1, 1, sk, 6), physical)]
        seeds = [rng.normal(0, .1, shape).astype(np.float32) for shape in
                 ((1, 2, sq, 6), (1, 2, sq))]
        return values, seeds

    traced, _ = inputs(*trace)
    program = attention.compile_native_attention_vjp(
        *traced, compiler=os.environ["TESSERA_OPT"], sequence_bounds=(9, 11),
        compact_gradients=compact)
    program = NativeAttentionVJPProgram.from_json(program.to_json(),
                                                  expected_digest=program.program_digest)
    identity = program.program_digest
    monkeypatch.setattr(subprocess, "run",
        lambda *a, **k: pytest.fail("compiler recovery during trace-at-one reverse replay"))
    with NvidiaDeviceSession() as session:
        for sq, sk in (trace, (3, 5), (9, 11), (1, 11), (9, 1)):
            values, seeds = inputs(sq, sk)
            output, lse, gradients = oracle(*values[:3], *seeds, True, values[3])
            axes = tuple(i for i, size in enumerate(values[3].shape) if size == 1)
            gradients = (*gradients[:3], gradients[3].sum(axis=axes, keepdims=True))
            with program.capture(*(session.upload(value) for value in values)) as frame:
                for actual, wanted in zip(frame.primal, (output, lse), strict=True):
                    np.testing.assert_allclose(download(session, actual), wanted, rtol=4e-5, atol=4e-5)
                got = frame.backward(tuple(session.upload(value) for value in seeds))
                for actual, role in zip(got, program.active, strict=True):
                    np.testing.assert_allclose(download(session, actual), gradients[role],
                                               rtol=4e-5, atol=4e-5)
            assert program.program_digest == identity
