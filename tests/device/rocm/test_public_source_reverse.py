"""Ordinary public native source VJP through the checked HIP program owner."""
import os
import subprocess

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from benchmarks.nvidia.active_source_jvp_reference import product, product_reversed
from benchmarks.math.source_reverse_reference import inputs, check
from tessera.compiler.native_vjp_plugins import validate_native_vjp_execution_certificate

pytestmark = [
    pytest.mark.hardware_rocm,
    pytest.mark.skipif(os.environ.get("TESSERA_ROCM_SOURCE_REVERSE_DEVICE_PROOF") != "1",
                       reason="explicit owning ROCm source-pullback proof required"),
]


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("permuted", [False, True])
def test_public_source_reverse_retains_owner_and_replays_portable(storage, permuted, monkeypatch):
    arch = os.environ["TESSERA_ROCM_CHIP"]
    assert rt._rocm_live_arch() == arch
    fn = ts.jit(target="rocm_" + arch, autodiff="reverse", wrt=("a",))(
        product_reversed if permuted else product)
    try:
        roots = inputs(storage, permuted)
        actual = fn.native_backward(*roots[:2], out_cotangents=roots[2])
        check(actual, roots, permuted)
        snapshot = actual[0].copy()
        roots = inputs(storage, permuted, 193)
        with monkeypatch.context() as warm:
            warm.setattr(subprocess, "run", lambda *a, **k: pytest.fail("warm public VJP compiled"))
            warm.setattr(fn, "_trace_frontend_capture", lambda *a, **k: pytest.fail("warm VJP retraced"))
            from tessera.compiler import rocm_native
            warm.setattr(rocm_native, "_run_opt", lambda *a, **k: pytest.fail("warm VJP lowered"))
            warm.setattr(rt, "_launch_native_descriptor", lambda *a, **k: pytest.fail("Python member launch"))
            result = fn.native_backward(*roots[:2], out_cotangents=roots[2])
        check(result, roots, permuted)
        np.testing.assert_array_equal(actual[0], snapshot)
        assert not np.shares_memory(actual[0], result[0])
        certificate = fn.last_backward_execution["execution_certificate"]
        validate_native_vjp_execution_certificate(certificate)
        assert certificate["evidence_scope"] == "exact_device"
        assert certificate["evidence_target"] == "rocm_" + arch
        assert len(fn._native_prepared_vjp_calls) == 1
        artifact = rt.RuntimeArtifact.from_json(fn.native_backward_runtime_artifact().to_json())
        fn.close_native_storage()
        with monkeypatch.context() as offline:
            offline.setattr(subprocess, "run", lambda *a, **k: pytest.fail("portable public VJP compiled"))
            launched = rt.launch(artifact, roots)
        assert launched["ok"], launched
        assert len(launched["component_receipts"]) == 2
        check(launched["output"], roots, permuted)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("fault", ["target", "root_names", "graph", "placement", "image"])
def test_public_source_reverse_refuses_mutated_parent_before_provider(fault, monkeypatch):
    import copy
    from tessera.compiler.rocm_source_reverse_runtime import execute
    arch = os.environ["TESSERA_ROCM_CHIP"]
    fn = ts.jit(target="rocm_" + arch, autodiff="reverse", wrt=("a",))(product)
    roots = inputs("f16", False)
    try:
        fn.native_backward(*roots[:2], out_cotangents=roots[2])
        artifact = fn.native_backward_runtime_artifact()
        metadata = copy.deepcopy(artifact.metadata)
        graph = artifact.graph_ir
        if fault == "target":
            metadata["target"] = "rocm_gfx1151" if arch == "gfx1201" else "rocm_gfx1201"
        elif fault == "root_names":
            metadata["arg_names"] = ["primal_1", "primal_0", "cotangent"]
        elif fault == "graph":
            graph = "module {}"
        elif fault == "placement":
            metadata["result_placement"] = "cuda"
        else:
            metadata["native_program"]["row_image"] = "AA=="
        malformed = rt.RuntimeArtifact(graph_ir=graph, metadata=metadata)
        monkeypatch.setattr(rt, "_load_rocm_native_movement_runtime",
                            lambda: pytest.fail("malformed public artifact reached provider"))
        with pytest.raises(ValueError):
            execute(malformed, roots)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_public_source_reverse_fresh_process_replay(storage, tmp_path):
    import sys
    arch = os.environ["TESSERA_ROCM_CHIP"]
    fn = ts.jit(target="rocm_" + arch, autodiff="reverse", wrt=("a",))(product)
    roots = inputs(storage, False)
    try:
        fn.native_backward(*roots[:2], out_cotangents=roots[2])
        path = tmp_path / "public-vjp.json"
        path.write_text(fn.native_backward_runtime_artifact().to_json())
    finally:
        fn.close_native_storage()
    code = """
import sys
from pathlib import Path
from tessera import runtime as rt
from benchmarks.math.source_reverse_reference import inputs, check
artifact = rt.RuntimeArtifact.from_json(Path(sys.argv[1]).read_text())
roots = inputs(sys.argv[2], False, 911)
launched = rt.launch(artifact, roots)
assert launched["ok"], launched
check(launched["output"], roots, False)
print("public compiler-free source VJP passed")
"""
    env = dict(os.environ, TESSERA_OPT="/missing/compiler", TESSERA_ROCM_OPT="/missing/compiler",
               TESSERA_LLVM_BIN="/missing/llvm", TESSERA_NVIDIA_OPT="/missing/compiler")
    result = subprocess.run([sys.executable, "-c", code, str(path), storage],
                            env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "public compiler-free source VJP passed" in result.stdout


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 7, 1), (129, 33, 65), (5, 67, 257)])
def test_public_source_reverse_static_boundaries(storage, permuted, shape):
    arch = os.environ["TESSERA_ROCM_CHIP"]
    assert rt._rocm_live_arch() == arch
    fn = ts.jit(target="rocm_" + arch, autodiff="reverse", wrt=("a",))(
        product_reversed if permuted else product)
    try:
        for seed in (87, 193):
            roots = inputs(storage, permuted, seed, shape=shape)
            actual = fn.native_backward(*roots[:2], out_cotangents=roots[2])
            check(actual, roots, permuted)
    finally:
        fn.close_native_storage()
