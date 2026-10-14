"""Exact RTX5070 bounded saved-LSE forward/JVP generations."""
import copy
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import tessera as ts

from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_program import NativeAttentionJVPProgram
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_resident_attention_forward import oracle

pytestmark = pytest.mark.hardware_nvidia


def function(bias, activity):
    if bias:
        @ts.jit(target="nvidia_sm120", autodiff="forward", wrt=activity)
        def attention(v, q, k, bias):
            return ts.ops.flash_attn(q, k, v, attn_bias=bias, causal=True,
                                     lse_checkpoint="saved")
    else:
        @ts.jit(target="nvidia_sm120", autodiff="forward", wrt=activity)
        def attention(v, q, k):
            return ts.ops.flash_attn(q, k, v, causal=True, lse_checkpoint="saved")
    return attention


def inputs(sq, sk, biased, activity):
    rng = np.random.default_rng(20261011 + 31 * sq + sk)
    shapes = {"v": (1, 1, sk, 3), "q": (1, 2, sq, 4), "k": (1, 1, sk, 4)}
    if biased:
        shapes["bias"] = (1, 2, 1, sk)
    values = {name: rng.normal(0, .2, shape).astype(np.float32)
              for name, shape in shapes.items()}
    seeds = {name: rng.normal(0, .1, shapes[name]).astype(np.float32)
             for name in activity}
    return values, seeds


def reference(values, seeds):
    h = 1e-4
    upper = {name: value.astype(np.float64) + h * seeds.get(name, 0)
             for name, value in values.items()}
    lower = {name: value.astype(np.float64) - h * seeds.get(name, 0)
             for name, value in values.items()}
    primal = oracle(tuple(values[name] for name in ("q", "k", "v")), values.get("bias"))
    positive = oracle(tuple(upper[name] for name in ("q", "k", "v")), upper.get("bias"))
    negative = oracle(tuple(lower[name] for name in ("q", "k", "v")), lower.get("bias"))
    tangent = tuple((a - b) / (2 * h) for a, b in zip(positive, negative, strict=True))
    return primal, tangent


@pytest.mark.parametrize("bounds", [(9, 11), (17, 129)])
@pytest.mark.parametrize("biased,activity", [
    (False, ("k", "v", "q")), (False, ("v",)),
    (True, ("k", "v", "q", "bias")), (True, ("v",)), (True, ("bias",)),
])
def test_bounded_saved_lse_replays_one_package_and_keeps_generations(
        biased, activity, bounds, monkeypatch):
    if not nvidia_cuda_host_ready():
        pytest.skip("exact SM120 CUDA host required")
    fn = function(biased, activity)
    original = copy.deepcopy(fn.graph_ir)
    traced, _ = inputs(3, 4, biased, activity)
    program = fn.compile_native_attention_jvp(
        *traced.values(), compiler=Path(os.environ["TESSERA_OPT"]),
        llvm_bin=Path(os.environ["LLVM_BIN"]), sequence_bounds=bounds)
    assert fn.graph_ir == original
    assert program.saved_lse is True
    assert program.pair.forward.descriptor.provenance["shape_bounds"] == [1, 2, 1, *bounds, 4, 3]
    program = NativeAttentionJVPProgram.from_json(
        program.to_json(), expected_digest=program.program_digest)
    identities = (program.program_digest, program.pair.forward.image.image_digest,
                  program.tangent.binding_digest)
    with NvidiaDeviceSession() as session:
        held = []
        try:
            monkeypatch.setattr(subprocess, "run",
                lambda *a, **k: pytest.fail("compiler recovery during bounded saved-LSE replay"))
            for sq, sk in ((1, 1), (3, 4), bounds, (bounds[0], 1), (1, bounds[1]), (3, 4)):
                values, seeds = inputs(sq, sk, biased, activity)
                expected = reference(values, seeds)
                resident = [session.upload(value) for value in values.values()]
                directions = [session.upload(seeds[name]) for name in activity]
                frame = program.capture(*resident)
                held.append(frame)
                tangent = frame.jvp(*directions)
                for actual, wanted in zip((*frame.primal, *tangent),
                                          (*expected[0], *expected[1]), strict=True):
                    np.testing.assert_allclose(download(session, actual), wanted,
                                               rtol=4e-5, atol=3e-6)
                if activity == ("v",):
                    np.testing.assert_array_equal(download(session, tangent[1]),
                                                  np.zeros((1, 2, sq), np.float32))
                old = tuple(download(session, value) for value in tangent)
                changed = [session.upload(seeds[name] * -.7) for name in activity]
                repeated = frame.jvp(*changed)
                for actual, wanted in zip(repeated, expected[1], strict=True):
                    np.testing.assert_allclose(download(session, actual), -.7 * wanted,
                                               rtol=4e-5, atol=3e-6)
                for actual, wanted in zip(tangent, old, strict=True):
                    np.testing.assert_array_equal(download(session, actual), wanted)
                if len(held) == 1:
                    first = (frame, tangent, old)
                else:
                    for actual, wanted in zip(first[1], first[2], strict=True):
                        np.testing.assert_array_equal(download(session, actual), wanted)
                assert identities == (program.program_digest,
                    program.pair.forward.image.image_digest, program.tangent.binding_digest)
            oversized, _ = inputs(bounds[0] + 1, bounds[1], biased, activity)
            with pytest.raises(ValueError, match="envelope"):
                program.capture(*(session.upload(value) for value in oversized.values()))
        finally:
            for frame in reversed(held):
                frame.close()
            fn.close_native_storage()


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("bias_axes", ["key", "query", "both"])
def test_bounded_broadcast_bias_direct_forward_backward_launch_abi(resident, bias_axes):
    from tessera import runtime as rt
    from tessera.compiler import nvidia_native, scheduled_checkpoint
    from tests.device.nvidia.test_bounded_attention_packages import artifact
    from tests.device.nvidia.test_lse_checkpoint_native import _reference
    from tests.unit.test_bounded_attention_checkpoint import graph
    from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
    if not nvidia_cuda_host_ready():
        pytest.skip("exact SM120 CUDA host required")
    physical = {"key": "1x2x1x?", "query": "1x2x?x1", "both": "1x2x?x?"}[bias_axes]

    def scheduled(backward):
        text = graph(backward, bias=True).replace("1x2x?x?xf32", physical + "xf32")
        schedule = run_tessera_opt(find_tessera_opt(), text, "--tessera-graph-to-schedule")
        tile = run_tessera_opt(find_tessera_opt(), schedule, "--tessera-schedule-to-tile")
        return scheduled_checkpoint._decode_checkpoint(text, schedule, tile, backward=backward)

    pair = nvidia_native.package_scheduled_checkpoint_pair(
        scheduled(False), scheduled(True), pipeline_name="tessera-nvidia-pipeline-sm120")
    assert len(pair.forward.descriptor.scalars) == len(pair.backward.descriptor.scalars) == (7 if bias_axes == 'both' else 11)
    assert "_compact_m" not in pair.backward.descriptor.entry_symbol
    with NvidiaDeviceSession() as session:
        for sq, sk in ((3, 4), (9, 11), (9, 1), (1, 11)):
            rng = np.random.default_rng(202610112 + sq + sk)
            q, k, v, seed = [rng.normal(0, .2, shape).astype(np.float32) for shape in
                ((1, 2, sq, 8), (1, 1, sk, 8), (1, 1, sk, 6), (1, 2, sq, 6))]
            bias_shape = {"key": (1, 2, 1, sk), "query": (1, 2, sq, 1),
                          "both": (1, 2, sq, sk)}[bias_axes]
            bias = rng.normal(0, .1, bias_shape).astype(np.float32)
            output, lse, gradients = _reference(q, k, v, seed,
                np.broadcast_to(bias, (1, 2, sq, sk)), scale=.5)
            host = dict(q=q, k=k, v=v, bias=bias, do=seed,
                        o=np.empty_like(output, dtype=np.float32),
                        lse=np.empty_like(lse, dtype=np.float32),
                        dq=np.empty_like(q), dk=np.empty_like(k), dv=np.empty_like(v))
            buffers = {name: session.upload(value) if resident else value
                       for name, value in host.items()}
            scalars = dict(zip(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"),
                              (1, 2, 1, sq, sk, 8, 6), strict=True))
            scalars.update(zip(("BiasB", "BiasH", "BiasQ", "BiasK"), bias_shape, strict=True))
            for package, wants in ((pair.forward, {"o": output, "lse": lse}),
                    (pair.backward, dict(zip(("dq", "dk", "dv"), gradients, strict=True)))):
                args = {binding.name: buffers[binding.name] for binding in package.descriptor.buffers}
                args.update({item.name: scalars[item.name] for item in package.descriptor.scalars})
                result = rt.launch(artifact(package), args, stream=session.stream if resident else None)
                assert result["ok"], result
                assert result["execution_kind"] == "native_gpu"
                if resident:
                    assert session.synchronize() == 0
                for name, wanted in wants.items():
                    actual = session.download(buffers[name]) if resident else buffers[name]
                    np.testing.assert_allclose(actual, wanted, rtol=4e-5, atol=3e-6)
