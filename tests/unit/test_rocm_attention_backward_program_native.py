from __future__ import annotations

import numpy as np
import pytest

from benchmarks.rocm.benchmark_rocm_attention_backward_program import (
    _module,
    _reference,
)
from tessera.compiler.rocm_native import (
    package_attention_backward,
    tools_available,
)
from tessera.runtime import (
    _rocm_wmma_runtime_available,
    _submit_rocm_gfx1151_attention_backward_program,
)


@pytest.mark.compiler_rocm
def test_gfx1151_lse_checkpoint_auto_selector_tracks_measured_threshold() -> None:
    if not tools_available():
        pytest.skip("tessera-opt is unavailable")
    short = package_attention_backward(
        _module(1, 4, 2, 17, 19, 64, lse_checkpoint="auto"),
        pipeline_name="tessera-lower-to-rocm",
    )
    long = package_attention_backward(
        _module(1, 4, 2, 256, 256, 64, lse_checkpoint="auto"),
        pipeline_name="tessera-lower-to-rocm",
    )
    assert short.descriptors[0].provenance["lse_checkpoint"] == "recompute"
    assert long.descriptors[0].provenance["lse_checkpoint"] == "saved"
    assert short.descriptors[0].provenance["lse_checkpoint_policy"] == "auto"
    assert long.descriptors[0].provenance["lse_checkpoint_policy"] == "auto"
    assert all(binding.name != "row_lse" for binding in short.descriptors[0].buffers)
    assert any(binding.name == "row_lse" for binding in long.descriptors[0].buffers)


@pytest.mark.compiler_rocm
@pytest.mark.hardware_rocm
@pytest.mark.performance
def test_gfx1151_compiler_owned_backward_program_executes_and_times() -> None:
    if not tools_available():
        pytest.skip("tessera-opt is unavailable")
    if not _rocm_wmma_runtime_available():
        pytest.skip("a usable gfx1151 ROCm device is unavailable")
    shape = (1, 4, 2, 17, 19, 64)
    rng = np.random.default_rng(20260726)
    q = rng.normal(0.0, 0.25, (shape[0], shape[1], shape[3], shape[5])).astype(np.float16)
    key = rng.normal(0.0, 0.25, (shape[0], shape[2], shape[4], shape[5])).astype(np.float16)
    value = rng.normal(0.0, 0.25, key.shape).astype(np.float16)
    do = rng.normal(0.0, 0.25, q.shape).astype(np.float16)
    bias = rng.normal(0.0, 0.1, (shape[0], shape[1], shape[3], shape[4])).astype(np.float32)
    outputs = (
        np.empty(q.shape, dtype=np.float32),
        np.empty(key.shape, dtype=np.float32),
        np.empty(value.shape, dtype=np.float32),
    )
    program = package_attention_backward(_module(*shape), pipeline_name="tessera-lower-to-rocm")
    result = _submit_rocm_gfx1151_attention_backward_program(
        program,
        {
            "do": do,
            "q": q,
            "key": key,
            "v": value,
            "bias": bias,
            "dq": outputs[0],
            "dk": outputs[1],
            "dv": outputs[2],
        },
        warmup=2,
        iterations=3,
    )
    reference = _reference(do, q, key, value, bias)
    for actual, expected in zip(result["outputs"], reference, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-2)
    assert len(result["entry_symbols"]) == 5
    assert result["workspace_bytes"] == program.workspace.bytes
    assert all(sample > 0.0 for sample in result["kernel_wall_samples_ms"])
