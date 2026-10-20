"""Ordinary frontend calls must consume the compiler-owned native program."""
import subprocess

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests.device.nvidia.test_native_tensor_program_package_execution import (
    CASES, FUNCTIONS, oracle, roots, values,
)

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("kind,storage", CASES)
def test_ordinary_jit_uses_native_program_and_replays_without_tools(kind, storage, monkeypatch):
    assert rt._nvidia_device_name() == "sm_120"
    fn = ts.jit(target="nvidia_sm120")(FUNCTIONS[kind])
    try:
        a, b = values(storage)
        first = fn(*roots(kind, a, b))
        np.testing.assert_allclose(first, oracle(kind, a, b), rtol=.004, atol=2e-5)
        assert fn.execution_kind == "native_gpu"
        artifact = fn.runtime_artifact()
        assert "tessera_nvidia.program_invoke " in artifact.target_ir
        assert "tessera.matmul" in artifact.graph_ir
        assert "schedule." in artifact.schedule_ir
        assert "tile.program_invoke " in artifact.tile_ir
        assert fn.schedule_ir == artifact.schedule_ir
        assert fn.tile_ir == artifact.tile_ir
        assert fn.target_ir == artifact.target_ir
        assert fn.ir_text() == artifact.graph_ir
        stages = fn.lowering_artifacts()
        assert all(b.input_digest == a.output_digest for a, b in zip(stages, stages[1:]))
        assert fn.compile_report().plan_hash == artifact.artifact_hash
        if kind == "norm":
            assert 'accum = "fp32"' in artifact.graph_ir
        saved = first.copy()

        def forbidden(*args, **kwargs):
            pytest.fail("ordinary native JIT replay invoked tracing or an external compiler")

        with monkeypatch.context() as guarded:
            guarded.setattr(subprocess, "run", forbidden)
            guarded.setattr(subprocess, "Popen", forbidden)
            guarded.setattr(fn, "_trace_frontend_capture", forbidden)
            for seed in (71, 72):
                a, b = values(storage, seed)
                before = a.copy(), b.copy()
                result = fn(*roots(kind, a, b))
                np.testing.assert_allclose(result, oracle(kind, a, b), rtol=.004, atol=2e-5)
                np.testing.assert_array_equal(a, before[0])
                np.testing.assert_array_equal(b, before[1])
                np.testing.assert_array_equal(first, saved)
                assert not np.shares_memory(first, result)
            fn.close_native_storage()
            # Closing resources preserves the verified compiler product, so
            # rebinding the same envelope does not retrace or recompile.
            result = fn(*roots(kind, a, b))
            np.testing.assert_allclose(result, oracle(kind, a, b), rtol=.004, atol=2e-5)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("kind", ["relu", "norm"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_ordinary_jit_runtime_artifact_executes_verified_source(kind, storage):
    assert rt._nvidia_device_name() == "sm_120"
    fn = ts.jit(target="nvidia_sm120")(FUNCTIONS[kind])
    a, b = values(storage, 73)
    try:
        fn(*roots(kind, a, b))
        artifact = rt.RuntimeArtifact.from_dict(fn.runtime_artifact().to_dict())
        receipt = rt.launch(artifact, roots(kind, a, b))
        assert receipt.get("ok") is True, receipt
        assert receipt["execution_kind"] == "native_gpu"
        assert receipt["compiler_path"] == "canonical_native_tensor_program"
        np.testing.assert_allclose(receipt["output"], oracle(kind, a, b), rtol=.004, atol=2e-5)
        refused = rt.launch(artifact, roots(kind, a, b), stream=1)
        assert refused.get("ok") is False
    finally:
        fn.close_native_storage()
