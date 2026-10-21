"""Missing shape-companion tools leave automatic admission unclaimed."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tessera as ts
from tessera import runtime
from tessera.compiler import llvm_tools, native_tensor_program, scheduled_matmul
from tessera.compiler.jit import _JIT_FALLBACK


def norm_lhs(a, b):
    return ts.ops.matmul(ts.ops.rmsnorm(a), b, output_dtype="fp32")


@pytest.fixture
def shape_preflight(monkeypatch, tmp_path):
    tools = {}
    for name in ("mlir-opt", "mlir-translate", "clang"):
        path = tmp_path / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        tools[name] = path
    monkeypatch.setattr(llvm_tools, "find_llvm_tool", lambda name: tools[name])
    monkeypatch.setattr(scheduled_matmul, "find_tessera_opt", lambda: Path(__file__))
    provider = SimpleNamespace(tessera_nvidia_native_frame_prepare_v3=object(),
                               tessera_nvidia_native_frame_shapes_v3=object())
    monkeypatch.setattr(runtime, "_load_nvidia_ptx_launch", lambda: provider)
    def forbidden(*args, **kwargs):
        pytest.fail("shape preflight compiled a program")
    monkeypatch.setattr(native_tensor_program, "compile_native_tensor_program", forbidden)
    fn = ts.jit(target="nvidia_sm120")(norm_lhs)
    return tools, fn, (np.ones((17, 32), np.float16), np.ones((32, 11), np.float16))


@pytest.mark.parametrize("name", ["mlir-opt", "mlir-translate", "clang"])
@pytest.mark.parametrize("fault", ["not_found", "missing", "not_executable"])
def test_missing_shape_tool_leaves_route_unclaimed(shape_preflight, monkeypatch, name, fault):
    tools, fn, args = shape_preflight
    if fault == "not_found":
        tools[name] = None
    elif fault == "missing":
        tools[name].unlink()
    else:
        tools[name].chmod(0o644)
    def forbidden(*args, **kwargs):
        pytest.fail("missing shape tool traced frontend source")
    monkeypatch.setattr(fn, "_traced_autodiff_module", forbidden)
    assert fn._try_native_tensor_program_call(args, {}, shape_bounds={"M": 65, "N": 47, "K": 64}) is _JIT_FALLBACK
    assert not fn._native_tensor_program_cache


def test_complete_shape_toolchain_reaches_frontend_admission(shape_preflight, monkeypatch):
    _, fn, args = shape_preflight
    class AdmissionReached(Exception):
        pass
    def reached(*args, **kwargs):
        raise AdmissionReached
    monkeypatch.setattr(fn, "_traced_autodiff_module", reached)
    with pytest.raises(AdmissionReached):
        fn._try_native_tensor_program_call(args, {}, shape_bounds={"M": 65, "N": 47, "K": 64})
    assert not fn._native_tensor_program_cache
