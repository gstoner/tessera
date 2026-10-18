"""Public certified source VJP and serialized compiler-free exact-device replay."""
import copy
import json
import subprocess

import pytest
import tessera as ts

from tessera import runtime as rt
from tessera.compiler.prepared_nvidia_source_reverse import PreparedSourceReverse
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_vjp_plugins import validate_native_vjp_execution_certificate
from benchmarks.nvidia.active_source_jvp_reference import product, product_reversed
from tests.device.nvidia.test_source_reverse_program import inputs, check

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("resident", [False, True])
def test_public_source_reverse_certifies_and_reuses_native_owner(storage, permuted, resident, monkeypatch):
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("a",))(
        product_reversed if permuted else product)
    try:
        with NvidiaDeviceSession() as session:
            for seed in (87, 193):
                roots = inputs(storage, permuted, seed)
                values = tuple(session.upload(v) for v in roots) if resident else roots
                if seed == 87:
                    actual = fn.native_backward(*values[:2], out_cotangents=values[2])
                else:
                    with monkeypatch.context() as warm:
                        warm.setattr(subprocess, "run", lambda *a, **k: pytest.fail("warm VJP compiled"))
                        warm.setattr(fn, "_trace_frontend_capture", lambda *a, **k: pytest.fail("warm VJP retraced"))
                        actual = fn.native_backward(*values[:2], out_cotangents=values[2])
                check(actual, roots, permuted)
                certificate = fn.last_backward_execution["execution_certificate"]
                validate_native_vjp_execution_certificate(certificate)
                assert certificate["evidence_scope"] == "exact_device"
                if resident:
                    assert certificate["proof_mode"] == "abstract_metadata_structural"
                    assert certificate["source_reexecution"] == "prohibited"
                assert len(fn._native_prepared_vjp_calls) == 1
            owner = next(iter(fn._native_prepared_vjp_calls.values()))
            artifact = rt.RuntimeArtifact.from_json(fn.native_backward_runtime_artifact().to_json())
            with monkeypatch.context() as borrowed:
                borrowed.setattr(PreparedSourceReverse, "from_manifest",
                                 classmethod(lambda *a, **k: pytest.fail("prepared a duplicate live owner")))
                result = rt.launch(artifact, dict(zip(
                    ("primal_0", "primal_1", "cotangent"), values, strict=True)))
            assert result["ok"], result
            assert len(result["component_receipts"]) == 2
            check(result["output"], roots, permuted)
            handle = owner.handle
            fn.close_native_storage()
            assert not owner.finalizer.alive
            with monkeypatch.context() as portable:
                portable.setattr(subprocess, "run", lambda *a, **k: pytest.fail("portable replay compiled"))
                from tessera.compiler import nvidia_native
                portable.setattr(nvidia_native, "_compile_tile_ir",
                                 lambda *a, **k: pytest.fail("portable replay packaged"))
                result = rt.launch(artifact, values)
            assert result["ok"], result
            check(result["output"], roots, permuted)
            assert owner.handle == handle and not owner.finalizer.alive
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("fault", ["target", "root_names", "graph", "placement"])
def test_portable_parent_mutation_refuses_before_provider(fault, monkeypatch):
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("a",))(product)
    roots = inputs("f16", False)
    try:
        fn.native_backward(*roots[:2], out_cotangents=roots[2])
        artifact = fn.native_backward_runtime_artifact()
        metadata = copy.deepcopy(artifact.metadata)
        if fault == "target":
            metadata["target"] = "rocm_gfx1201"
        elif fault == "root_names":
            metadata["arg_names"] = ["primal_1", "primal_0", "cotangent"]
        elif fault == "placement":
            metadata["result_placement"] = "cuda"
        malformed = rt.RuntimeArtifact(
            graph_ir="module {}" if fault == "graph" else artifact.graph_ir, metadata=metadata)
        from tessera.compiler.nvidia_source_reverse_runtime import execute
        with monkeypatch.context() as refused:
            refused.setattr(rt, "_load_nvidia_ptx_launch", lambda: pytest.fail("invalid parent reached provider"))
            with pytest.raises(ValueError):
                execute(malformed, roots)
    finally:
        fn.close_native_storage()


def test_public_artifact_can_be_validated_without_compiler_or_device(monkeypatch):
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("a",))(product)
    roots = inputs("f16", False)
    try:
        fn.native_backward(*roots[:2], out_cotangents=roots[2])
        artifact = fn.native_backward_runtime_artifact()
        from tessera.compiler.portable_nvidia_source_reverse import restore
        with monkeypatch.context() as offline:
            offline.setattr(rt, "_load_nvidia_ptx_launch", lambda: pytest.fail("metadata validation loaded device"))
            offline.setattr(subprocess, "run", lambda *a, **k: pytest.fail("metadata validation ran compiler"))
            native, members = restore(json.loads(json.dumps(artifact.metadata["native_program"])))
        assert native.manifest["outputs"] == [4] and len(members) == 2
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("slot", [0, 1])
def test_public_reverse_borrows_owned_cotangent_after_producer_close(storage, slot, monkeypatch):
    from benchmarks.nvidia.active_source_jvp_reference import inputs as upstream_inputs
    from tessera.compiler.native_tensor_result import NativeTensorResult
    a, b, da = upstream_inputs({"shape_mnk": (17, 19, 35), "storage": storage})
    producer = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("a",))(product)
    pair = producer.native_jvp(a, b, tangents=(da,), result_placement="cuda")
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("a",))(product)
    try:
        # Read once to define the independent downstream oracle, then prohibit
        # source downloads during the compiled consumer and portable replay.
        seed = pair[slot].to_host()
        host = (*inputs(storage, False)[:2], seed)
        producer.close_native_storage()
        with NvidiaDeviceSession() as session:
            values = (session.upload(host[0]), session.upload(host[1]), pair[slot])
            with monkeypatch.context() as resident:
                resident.setattr(NativeTensorResult, "to_host",
                                 lambda *a, **k: pytest.fail("reverse downloaded its cotangent"))
                actual = fn.native_backward(*values[:2], out_cotangents=values[2])
                check(actual, host, False)
                artifact = rt.RuntimeArtifact.from_json(fn.native_backward_runtime_artifact().to_json())
                fn.close_native_storage()
                resident.setattr(subprocess, "run", lambda *a, **k: pytest.fail("resident replay compiled"))
                replay = rt.launch(artifact, values)
                assert replay["ok"], replay
                check(replay["output"], host, False)
    finally:
        fn.close_native_storage()
        for value in pair:
            value.close()
        producer.close_native_storage()
