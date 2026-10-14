"""Canonical SM120 package execution without a live compiler PATH probe."""
import json
import subprocess

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import pipeline_gates as pg
from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.driver import compile_graph_module
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.unit.test_scheduled_matmul_consumers import _module

pytestmark = pytest.mark.hardware_nvidia


def test_actual_native_package_executes_after_compiler_driver_is_unavailable(monkeypatch):
    if not nvidia_cuda_host_ready():
        pytest.skip("requires the owning RTX5070 runtime and compiler")
    gpu = subprocess.check_output([
        "/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,compute_cap",
        "--format=csv,noheader"], text=True).strip()
    assert "RTX 5070" in gpu and gpu.endswith("12.0"), gpu
    module = _module(target="nvidia_sm120", shape=(17, 23, 19),
                     dtype="fp16", output_dtype="fp32")
    bundle = compile_graph_module(module, source_origin="native-package-gate-proof",
        target="nvidia_sm120", options={"package_native": True},
        enable_tool_validation=False)
    real_which = pg.shutil.which
    monkeypatch.setattr(pg.shutil, "which",
        lambda name, *args, **kwargs: None if name == "nvcc" else
        real_which(name, *args, **kwargs))

    def forbidden(*args, **kwargs):
        pytest.fail("materialized package admission/launch invoked a compiler process")
    monkeypatch.setattr(subprocess, "run", forbidden)
    result = compile_result_from_bundle(bundle, module=module)
    assert result.executable and result.program_executable, result.reason
    assert result.native_image.image_digest == bundle.native_image.image_digest
    rng = np.random.default_rng(20261011)
    a = rng.normal(0, .2, (17, 23)).astype(np.float16)
    b = np.asfortranarray(rng.normal(0, .2, (23, 19)).astype(np.float16))
    out = np.full((17, 19), np.nan, np.float32)
    bindings = dict(zip((v.name for v in result.launch_descriptor.buffers),
                        (a, b, out), strict=True))
    bindings.update(M=17, N=19, K=23)
    receipt = rt.launch(result.to_runtime_artifact(), bindings)
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", json.dumps(receipt, default=str)
    np.testing.assert_allclose(out, a.astype(np.float64) @ b.astype(np.float64),
                               rtol=4e-5, atol=4e-5)
