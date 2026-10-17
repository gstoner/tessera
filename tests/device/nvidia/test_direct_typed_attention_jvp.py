"""Source-backed tape preparation preserves captured native storage roles."""
from pathlib import Path
import os

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_native import package_generated_attention_forward_checkpoint
from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from tests.device.nvidia.test_public_typed_attention_jvp import reference
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare
from tests.unit.test_saved_lse_half_backward import source

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.compiler_route]


@pytest.mark.parametrize("storage,wide_output", [("f16", False), ("f16", True), ("bf16", False), ("bf16", True), ("f32", False)])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_direct_tape_prepares_jvp_from_captured_storage(
    storage, wide_output, saved, bias, bounded, monkeypatch,
):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype({"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}[storage])
    dims = (1, 2, 1, "?", "?", 4, 3) if bounded else (1, 2, 1, 3, 5, 4, 3)
    original = source(storage, saved=saved, bias=bias,
                      output="f32" if wide_output else storage, dims=dims)
    if bounded:
        original = original.replace('tessera.arch = "sm_120"',
            'tessera.arch = "sm_120", tessera.attention_shape_bounds = array<i64: 1, 2, 1, 7, 9, 4, 3>')
    pair = package_generated_attention_forward_checkpoint(
        original, pipeline_name="tessera-nvidia-pipeline-sm120")
    forward_request = original.replace('"reverse"', '"forward"')
    rng = np.random.default_rng(20261017)
    with NvidiaDeviceSession() as session:
        for sq, sk in (((3, 5), (7, 9)) if bounded else ((3, 5),)):
            values = {n: rng.normal(0, .4, shape).astype(dtype) for n, shape in {
                "q": (1, 2, sq, 4), "k": (1, 1, sk, 4), "v": (1, 1, sk, 3)}.items()}
            if bias:
                values["bias"] = rng.normal(0, .1, (1, 2, 1, sk)).astype(np.float32)
            seeds = {n: rng.normal(0, .1, x.shape).astype(x.dtype) for n, x in values.items()}
            expected = reference(*(values[n] for n in ("q", "k", "v")),
                *(seeds[n] for n in ("q", "k", "v")),
                bias=values.get("bias"), dbias=seeds.get("bias"))[1]
            roots = {n: session.upload(x) for n, x in values.items()}
            directions = {n: session.upload(x) for n, x in seeds.items()}
            with pair.capture(*(roots[n] for n in ("q", "k", "v")),
                              **({"bias": roots["bias"]} if bias else {})) as frame:
                digest = frame.prepare_jvp(source=forward_request,
                    compiler=Path(os.environ["TESSERA_OPT"]),
                    llvm_bin=Path(os.environ["TESSERA_LLVM_BIN"]))
                assert len(digest) == 64
                def forbidden(*args, **kwargs):
                    pytest.fail("prepared direct tape JVP attempted compilation")
                with monkeypatch.context() as guard:
                    guard.setattr("subprocess.run", forbidden)
                    result = frame.jvp(*(directions[n] for n in ("q", "k", "v")),
                                       **({"dbias": directions["bias"]} if bias else {}))
                for i, (actual, target) in enumerate(zip(
                    result if saved else (result,), expected if saved else expected[:1], strict=True,
                )):
                    got = download(session, actual)
                    assert got.dtype == (np.dtype("float32") if i or wide_output else dtype)
                    compare(got, target)
