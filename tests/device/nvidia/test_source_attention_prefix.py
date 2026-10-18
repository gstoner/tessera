"""Exact SM120 native-private RMSNorm to saved-LSE attention composition."""
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts

from tests._support.nvidia import nvidia_cuda_host_ready
from tessera.compiler.native_source_program import outline_source_program
from tests.unit.test_native_attention_prefix_program import (
    prefix_attention, prefix_attention_permuted, prefix_attention_swapped_outputs)

pytestmark = pytest.mark.hardware_nvidia


@pytest.fixture(autouse=True)
def owning_sm120():
    if not nvidia_cuda_host_ready():
        pytest.skip("requires live CUDA SM120 host")
    capabilities = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True)
    if capabilities.strip().splitlines()[0] != "12.0":
        pytest.skip("requires owning SM120 device")


def cooperative_prefix(q, k, v):
    qn = ts.ops.rmsnorm(q, eps=1e-5, numeric_policy={"accum": "fp32"})
    return ts.ops.flash_attn(qn, k, v, causal=False, lse_checkpoint="saved")


def trace(function, roots):
    fn = ts.jit(target="nvidia_sm120")(function)
    graph = fn._traced_autodiff_module(roots, {})
    graph.module_attrs.update({"tessera.target": '"nvidia_sm120"',
                               "tessera.arch": '"sm_120"'})
    return graph.to_mlir(target="nvidia_sm120", canonical=True)


def oracle(q, k, v):
    qf = q.astype(np.float32)
    normalized = (qf / np.sqrt(np.mean(qf * qf, axis=-1, keepdims=True)
                              + np.float32(1e-5))).astype(q.dtype)
    scores = normalized.astype(np.float64) @ k.astype(np.float64).swapaxes(-1, -2)
    scores *= q.shape[-1] ** -0.5
    maximum = scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores - maximum)
    total = weights.sum(axis=-1, keepdims=True)
    return weights / total @ v.astype(np.float64), maximum[..., 0] + np.log(total[..., 0])


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("variant", ["normal", "permuted", "swapped", "cooperative"])
def test_composed_native_frame_keeps_private_edge_and_independent_outputs(storage, variant):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    wide = variant == "cooperative"
    sq, sk, d, dv = (17, 19, 257, 13) if wide else (3, 5, 4, 3)
    rng = np.random.default_rng(20261018)
    q = rng.normal(size=(1, 2, sq, d)).astype(dtype)
    k = rng.normal(size=(1, 1, sk, d)).astype(dtype)
    v = rng.normal(size=(1, 1, sk, dv)).astype(dtype)
    function = {"normal": prefix_attention, "permuted": prefix_attention_permuted,
                "swapped": prefix_attention_swapped_outputs,
                "cooperative": cooperative_prefix}[variant]
    roots = (v, q, k) if variant == "permuted" else (q, k, v)
    frozen = tuple(value.copy() for value in roots)
    program = outline_source_program(trace(function, roots))
    with program.prepare() as frame:
        assert frame.cooperative == wide
        first, receipt = frame(*roots)
        o, lse = first[::-1] if variant == "swapped" else first
        expected_o, expected_lse = oracle(q, k, v)
        np.testing.assert_allclose(o.astype(np.float32), expected_o,
                                   rtol=0.02 if storage == "bf16" else 0.003,
                                   atol=0.01 if storage == "bf16" else 0.001)
        np.testing.assert_allclose(lse, expected_lse, rtol=2e-5, atol=2e-5)
        retained = tuple(value.copy() for value in first)
        second, repeat = frame(*roots)
        assert receipt["execution_kind"] == "native_gpu"
        assert receipt["private_intermediate"] == "native_owned_normalized_q"
        assert receipt["member_images"] == repeat["member_images"]
        assert all(receipt[key] >= 0 for key in
                   ("norm_device_ms", "attention_device_ms", "program_device_ms"))
        for original, saved in zip(roots, frozen, strict=True):
            np.testing.assert_array_equal(original, saved)
        for prior, saved, current in zip(first, retained, second, strict=True):
            np.testing.assert_array_equal(prior, saved)
            assert not np.shares_memory(prior, current)
        with pytest.raises(ValueError, match="shape or storage"):
            wrong = list(roots)
            wrong[0] = wrong[0].astype(np.float32)
            frame(*wrong)
    frame.close()
    for prior, saved in zip(first, retained, strict=True):
        np.testing.assert_array_equal(prior, saved)
    with pytest.raises(ValueError, match="closed"):
        frame(*roots)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("mode", ["different", "legacy", "per_thread"])
def test_composed_native_frame_orders_pending_resident_roots(storage, mode, monkeypatch):
    from contextlib import ExitStack
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload

    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(3301)
    roots = tuple(rng.normal(size=shape).astype(dtype) for shape in
                  ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3)))
    program = outline_source_program(trace(prefix_attention, roots))
    with ExitStack() as stack:
        sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in roots]
        buffers = [session.upload(np.zeros_like(root))
                   for session, root in zip(sessions, roots, strict=True)]
        for session in sessions:
            assert session.synchronize() == 0
        streams = [1 if mode == "legacy" else 2 if mode == "per_thread" else session.stream
                   for session in sessions]
        borrowed = tuple(Borrowed(buffer, stream)
                         for buffer, stream in zip(buffers, streams, strict=True))
        frame = stack.enter_context(program.prepare())
        frame(*borrowed)  # Prime every private allocation before delayed writes.
        completed = [[], [], []]
        callbacks = [queue_delayed_upload(session, buffer, root, stream, done)
                     for session, buffer, root, stream, done in
                     zip(sessions, buffers, roots, streams, completed, strict=True)]
        assert not any(completed)

        def forbidden(*args, **kwargs):
            pytest.fail("resident composed replay invoked a compiler")

        with monkeypatch.context() as guarded:
            guarded.setattr(subprocess, "run", forbidden)
            (o, lse), receipt = frame(*borrowed)
        assert all(completed) and len(callbacks) == 3
        expected_o, expected_lse = oracle(*roots)
        np.testing.assert_allclose(o.astype(np.float32), expected_o,
                                   rtol=0.02 if storage == "bf16" else 0.003,
                                   atol=0.01 if storage == "bf16" else 0.001)
        np.testing.assert_allclose(lse, expected_lse, rtol=2e-5, atol=2e-5)
        assert receipt["host_preparation"] == "native_ordered_resident_snapshot"


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("variant", ["normal", "permuted", "swapped"])
def test_public_jit_source_attention_uses_native_members(storage, variant, monkeypatch):
    from tessera import runtime as rt

    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(9042)
    q, k, v = tuple(rng.normal(size=shape).astype(dtype) for shape in
                    ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3)))
    function = {"normal": prefix_attention, "permuted": prefix_attention_permuted,
                "swapped": prefix_attention_swapped_outputs}[variant]
    roots = (v, q, k) if variant == "permuted" else (q, k, v)
    fn = ts.jit(target="nvidia_sm120")(function)
    fn._trace_frontend_capture(roots, {})

    def forbidden(*args, **kwargs):
        pytest.fail("public compiled source attention used eager math or a warm compiler")

    with monkeypatch.context() as guarded:
        guarded.setattr(ts.ops, "rmsnorm", forbidden)
        guarded.setattr(ts.ops, "flash_attn", forbidden)
        first = fn(*roots)
        guarded.setattr(subprocess, "run", forbidden)
        second = fn(*roots)
    assert fn.execution_kind == "native_gpu"
    artifact = rt.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
    assert artifact.metadata["compiler_path"] == "nvidia_sm120_source_attention_compiled"
    o, lse = second[::-1] if variant == "swapped" else second
    expected_o, expected_lse = oracle(q, k, v)
    np.testing.assert_allclose(o.astype(np.float32), expected_o,
                               rtol=0.02 if storage == "bf16" else 0.003,
                               atol=0.01 if storage == "bf16" else 0.001)
    np.testing.assert_allclose(lse, expected_lse, rtol=2e-5, atol=2e-5)
    fn.close_native_storage()
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(a, b)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_portable_source_attention_replays_without_compiler(storage, monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.prepared_source_attention import PreparedSourceAttention

    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(9051)
    roots = tuple(rng.normal(size=shape).astype(dtype) for shape in
                  ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3)))
    program = outline_source_program(trace(prefix_attention, roots))
    with program.prepare(argument_names=("q", "k", "v")) as initial:
        expected, _ = initial(*roots)
        manifest = initial.manifest()
        artifact = rt.RuntimeArtifact.from_json(initial.runtime_artifact().to_json())

    def forbidden(*args, **kwargs):
        pytest.fail("portable source attention invoked a compiler")

    with monkeypatch.context() as guarded:
        guarded.setattr(subprocess, "run", forbidden)
        with PreparedSourceAttention.from_manifest(manifest) as restored:
            result, receipt = restored(*roots)
        launched = rt.launch(artifact, dict(zip(("q", "k", "v"), roots, strict=True)))
    assert receipt["execution_kind"] == "native_gpu"
    assert launched["ok"] and launched["execution_kind"] == "native_gpu", launched
    assert len(launched["component_receipts"]) == 2
    for a, b, c in zip(expected, result, launched["output"], strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_public_source_attention_borrows_resident_roots(storage):
    from contextlib import ExitStack
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed

    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(9084)
    roots = tuple(rng.normal(size=shape).astype(dtype) for shape in
                  ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3)))
    with ExitStack() as stack:
        sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in roots]
        borrowed = tuple(Borrowed(session.upload(root))
                         for session, root in zip(sessions, roots, strict=True))
        fn = ts.jit(target="nvidia_sm120")(prefix_attention)
        o, lse = fn(*borrowed)
        expected_o, expected_lse = oracle(*roots)
        np.testing.assert_allclose(o.astype(np.float32), expected_o,
                                   rtol=0.02 if storage == "bf16" else 0.003,
                                   atol=0.01 if storage == "bf16" else 0.001)
        np.testing.assert_allclose(lse, expected_lse, rtol=2e-5, atol=2e-5)
        assert fn.execution_kind == "native_gpu"
        assert fn._native_source_attention_last_receipt["host_preparation"] == "native_ordered_resident_snapshot"
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_portable_source_attention_fresh_process_has_no_compiler(storage, tmp_path):
    import json
    import os
    import sys
    from tessera import runtime as rt

    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(9085)
    roots = tuple(rng.normal(size=shape).astype(dtype) for shape in
                  ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3)))
    program = outline_source_program(trace(prefix_attention, roots))
    owner = program.prepare(argument_names=("q", "k", "v"))
    expected, _ = owner(*roots)
    owner.close()
    artifact = owner.runtime_artifact()  # Native handles are not artifact data.
    (tmp_path / "artifact.json").write_text(artifact.to_json())
    np.savez(tmp_path / "roots.npz", **{name: value.view(np.uint16) for name, value in
                                      zip(("q", "k", "v"), roots, strict=True)})
    child = """
import sys,subprocess,numpy as np,ml_dtypes
from pathlib import Path
from tessera import runtime as rt
def forbidden(*a,**k): raise AssertionError("portable replay invoked external tool")
subprocess.run=forbidden
root=Path(sys.argv[1])
artifact=rt.RuntimeArtifact.from_json((root/"artifact.json").read_text())
with np.load(root/"roots.npz") as arrays:
    dtype=np.float16 if sys.argv[2]=="f16" else ml_dtypes.bfloat16
    result=rt.launch(artifact,{name:value.view(dtype) for name,value in arrays.items()})
assert result["ok"] and result["execution_kind"]=="native_gpu",result
np.savez(root/"result.npz",o=result["output"][0].view(np.uint16),lse=result["output"][1])
"""
    env = dict(os.environ, TESSERA_OPT="/unavailable/compiler",
               TESSERA_NVIDIA_OPT="/unavailable/compiler")
    process = subprocess.run([sys.executable, "-c", child, str(tmp_path), storage],
                             env=env, text=True, capture_output=True, timeout=120)
    assert process.returncode == 0, process.stderr + process.stdout
    with np.load(tmp_path / "result.npz") as replay:
        np.testing.assert_array_equal(replay["o"].view(dtype), expected[0])
        np.testing.assert_array_equal(replay["lse"], expected[1])
    # A mismatching parent Graph cannot borrow a valid native payload.
    data = json.loads(artifact.to_json())
    data.pop("artifact_hash")
    data["graph_ir"] += "\n"
    changed = rt.RuntimeArtifact.from_dict(data)
    with pytest.raises(ValueError, match="parent Graph"):
        from tessera.compiler.source_attention_runtime import execute
        execute(changed, roots)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_public_source_attention_revisits_cached_static_shape(storage, monkeypatch):
    from tessera import runtime as rt

    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(9087)
    frames = []
    for sq, sk in ((3, 5), (7, 9)):
        frames.append(tuple(rng.normal(size=shape).astype(dtype) for shape in
                            ((1, 2, sq, 4), (1, 1, sk, 4), (1, 1, sk, 3))))
    fn = ts.jit(target="nvidia_sm120")(prefix_attention)
    held = fn(*frames[0])
    original_graph = fn.runtime_artifact().graph_ir
    fn(*frames[1])

    def forbidden(*args, **kwargs):
        pytest.fail("cached source shape invoked a compiler or retracer")

    with monkeypatch.context() as guarded:
        guarded.setattr(subprocess, "run", forbidden)
        guarded.setattr(fn, "_trace_frontend_capture", forbidden)
        result = fn(*frames[0])
    assert fn.graph_ir.functions[0].args[0].ir_type.shape == ("1", "2", "3", "4")
    assert fn.runtime_artifact().graph_ir == original_graph
    artifact = rt.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
    replay = rt.launch(artifact, frames[0])
    assert replay["ok"], replay
    for a, b, c in zip(held, result, replay["output"], strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
    fn.close_native_storage()


def test_public_source_attention_does_not_reuse_native_witness_for_other_storage(monkeypatch):
    rng = np.random.default_rng(9123)
    half = tuple(rng.normal(size=shape).astype(np.float16) for shape in
                 ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3)))
    fn = ts.jit(target="nvidia_sm120")(prefix_attention)
    fn(*half)
    source_graph = fn.runtime_artifact().graph_ir
    full = tuple(value.astype(np.float32) for value in half)
    output = fn(*full)
    expected = oracle(*full)
    for actual, reference in zip(output, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
    assert fn.execution_kind != "native_gpu"
    assert fn.runtime_artifact().metadata.get("compiler_path") != "nvidia_sm120_source_attention_compiled"

    def forbidden(*args, **kwargs):
        pytest.fail("returning to a cached half source invoked a compiler")

    with monkeypatch.context() as guarded:
        guarded.setattr(subprocess, "run", forbidden)
        fn(*half)
    assert fn.execution_kind == "native_gpu"
    assert fn.runtime_artifact().graph_ir == source_graph
    fn.close_native_storage()
