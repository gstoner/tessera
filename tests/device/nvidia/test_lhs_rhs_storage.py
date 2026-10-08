"""Physical storage and caller-owned edge proof for the named LHS program."""
import numpy as np
import pytest
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import rms_lhs, _oracle, _storage

pytestmark = pytest.mark.skipif(not nvidia_cuda_host_ready(), reason="owning NVIDIA host required")


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_direct_and_padded_resident_storage(order, dtype):
    rng = np.random.default_rng(5070)
    storage = _storage(dtype)
    source = (rng.normal(size=(17,35)) * .2).astype(storage)
    rhs = np.array(rng.normal(size=(35,19)) * .2, dtype=storage, order=order)
    program = rms_lhs.compile_native_lhs_matmul(source, rhs)
    expected = _oracle(source, rhs, "rmsnorm")
    with program.edge.execute(source, rhs) as direct:
        np.testing.assert_allclose(direct.output, expected, atol=.015, rtol=.015)
        assert direct.consumer_receipt["execution_kind"] == "native_gpu"
    # The uploaded allocation must use the compiler layout, never caller pitch.
    backing = np.zeros((35,24) if order == "C" else (40,19), dtype=storage, order=order)
    padded = backing[:35,:19]
    padded[:] = rhs
    assert not padded.flags.c_contiguous and not padded.flags.f_contiguous
    with program.edge.execute_resident(source, padded) as resident:
        actual = resident.device_session.download(resident.output)
        np.testing.assert_allclose(actual, expected, atol=.015, rtol=.015)
        assert resident.producer_receipt["execution_kind"] == "native_gpu"
        assert resident.consumer_receipt["execution_kind"] == "native_gpu"


@pytest.mark.parametrize("order", ["C", "F"])
def test_direct_edge_alias_refused_before_launch(order, monkeypatch):
    from tessera import runtime as rt
    source = np.ones((17,35), np.float16)
    rhs = np.ones((35,19), np.float16, order=order)
    program = rms_lhs.compile_native_lhs_matmul(source, rhs)
    def forbidden(*args, **kwargs):
        raise AssertionError("aliased frame reached launch")
    monkeypatch.setattr(rt, "launch", forbidden)
    with pytest.raises(ValueError, match="must not alias"):
        program.edge.execute(source, rhs, intermediate=source)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_fresh_process_portable_layout_without_compiler(order, dtype, tmp_path):
    import os
    import subprocess
    import sys
    from tessera.compiler.nvidia_tensor_lhs import runtime_artifact
    storage = _storage(dtype)
    source = np.ones((17,35), storage)
    rhs = np.ones((35,19), storage, order=order)
    program = rms_lhs.compile_native_lhs_matmul(source, rhs)
    path = tmp_path / "program.json"
    path.write_text(runtime_artifact(program).to_json())
    script = """
import sys
from pathlib import Path
import numpy as np
from tessera import runtime as rt
from tessera.compiler import nvidia_native, nvidia_tensor_lhs, scheduled_matmul
def forbidden(*args, **kwargs):
    raise AssertionError("fresh portable replay invoked compiler")
for module in (scheduled_matmul, nvidia_tensor_lhs):
    module.find_tessera_opt = forbidden
    module.run_tessera_opt = forbidden
nvidia_native.package_scheduled_tensor_matmul = forbidden
dtype, order = sys.argv[2:]
if dtype == "bf16":
    import ml_dtypes
    storage = ml_dtypes.bfloat16
else:
    storage = np.float16
source = np.ones((17,35), storage)
rhs = np.ones((35,19), storage, order=order)
artifact = rt.RuntimeArtifact.from_json(Path(sys.argv[1]).read_text())
receipt = rt.launch(artifact, (source,rhs))
assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
assert len(receipt["component_receipts"]) == 2
expected = float(np.array(1 / np.sqrt(1 + 1e-5), dtype=storage)) * 35
np.testing.assert_allclose(receipt["output"], expected, atol=.015, rtol=.015)
layout = "row_major" if order == "C" else "col_major"
assert artifact.metadata["native_program"]["semantics"]["consumer_attrs"]["rhs_storage_order"] == layout
"""
    env = dict(os.environ, TESSERA_OPT="/nonexistent/forbidden-compiler")
    result = subprocess.run([sys.executable, "-c", script, str(path), dtype, order],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
