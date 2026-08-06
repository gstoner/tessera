from __future__ import annotations

import dataclasses
import math
import platform

import numpy as np
import pytest

from benchmarks.rocm.benchmark_rocm_attention_backward_program import _module
from tessera.compiler.attention_contract import (
    reference_attention_backward_split_reduced,
)
from tessera.compiler.rocm_native import package_scheduled_attention_backward as package_rocm
from tessera.compiler.scheduled_attention_backward import (
    lower_scheduled_attention_backward,
    supports_scheduled_attention_backward,
)
from tessera.compiler.x86_native import package_scheduled_attention_backward as package_x86
from tessera.runtime import (
    _rocm_wmma_runtime_available,
    _submit_rocm_gfx1151_attention_backward_program,
    _submit_x86_native,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt

# These lower through the production `tessera-opt`, which the CI unit lane does
# not build. The library correctly RAISES rather than silently degrading, so
# the tests must skip there rather than fail.
_needs_opt = pytest.mark.skipif(
    find_tessera_opt() is None, reason="tessera-opt not built"
)


def _x86_module(hq: int, hkv: int, sq: int, sk: int, *, d: int = 16):
    module = _module(1, hq, hkv, sq, sk, d, dtype="fp32", lse_checkpoint="auto")
    module.functions[0].body[0].kwargs["window"] = 3
    module.functions[0].body[0].kwargs["softcap"] = 4.0
    return module


def _x86_reference(do, q, key, value, bias, *, scale, window, softcap):
    b, hq, sq, d = q.shape
    _, hkv, sk, _ = key.shape
    dq, dk, dv = np.zeros_like(q), np.zeros_like(key), np.zeros_like(value)
    row_lse = np.empty((b, hq, sq), np.float32)
    offset = max(sk - sq, 0)
    for batch in range(b):
        for head in range(hq):
            kv_head = head * hkv // hq
            for row in range(sq):
                begin = max(0, row + offset - window + 1)
                end = min(sk - 1, row + offset)
                keys = key[batch, kv_head, begin : end + 1]
                scores = scale * (keys @ q[batch, head, row])
                derivative = np.ones_like(scores)
                if softcap > 0:
                    tanh = np.tanh(scores / softcap)
                    scores = softcap * tanh
                    derivative = 1 - tanh * tanh
                scores += bias[batch, head, row, begin : end + 1]
                row_lse[batch, head, row] = np.logaddexp.reduce(scores)
                probabilities = np.exp(scores - row_lse[batch, head, row])
                values = value[batch, kv_head, begin : end + 1]
                dp = values @ do[batch, head, row]
                ds = probabilities * (dp - np.sum(probabilities * dp)) * derivative
                dq[batch, head, row] = scale * (ds @ keys)
                dk[batch, kv_head, begin : end + 1] += (
                    scale * ds[:, None] * q[batch, head, row]
                )
                dv[batch, kv_head, begin : end + 1] += (
                    probabilities[:, None] * do[batch, head, row]
                )
    return (dq, dk, dv), row_lse


@_needs_opt
@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_attention_backward_schedule_is_one_content_addressed_three_result_program(target: str) -> None:
    module = _x86_module(4, 2, 17, 19) if target == "x86" else _module(1, 4, 2, 17, 19, 16)
    artifact = lower_scheduled_attention_backward(module, target=target)
    assert artifact.schedule_ir.count("schedule.attention_backward") == 1
    assert artifact.tile_ir.count("tile.attention_backward_kernel") == 1
    assert "tessera_attn.backward" not in artifact.tile_ir
    assert artifact.output_names == ("dq", "dk", "dv")
    assert artifact.reduction_order == (0, 1)
    assert artifact.workspace_bytes > 0
    artifact.validate()


def test_attention_backward_fails_closed_for_unmeasured_rdna4_profiles() -> None:
    module = _module(1, 4, 2, 17, 19, 16)
    assert not supports_scheduled_attention_backward(module, target="rocm_gfx1200")
    assert not supports_scheduled_attention_backward(module, target="rocm_gfx1250")
    with pytest.raises(ValueError, match="architecture-owned profiles"):
        lower_scheduled_attention_backward(module, target="rocm_gfx1200")


@_needs_opt
def test_attention_backward_rejects_stale_multi_output_identity() -> None:
    artifact = lower_scheduled_attention_backward(_x86_module(2, 2, 16, 16), target="x86")
    with pytest.raises(ValueError, match="content identity"):
        dataclasses.replace(artifact, schedule_digest="0" * 64).validate()


@pytest.mark.parametrize(
    ("hq", "hkv", "sq", "sk"),
    [(2, 2, 16, 16), (4, 2, 17, 19), (4, 1, 15, 21)],
)
@_needs_opt
def test_zen5_scheduled_attention_backward_exact_mha_gqa_mqa(
    hq: int, hkv: int, sq: int, sk: int
) -> None:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        pytest.skip("Zen 5/AVX-512 validation requires an x86_64 host")
    artifact = lower_scheduled_attention_backward(_x86_module(hq, hkv, sq, sk), target="x86")
    package = package_x86(artifact, pipeline_name="tessera-lower-to-x86")
    rng = np.random.default_rng(20260805 + hkv + sq)
    d = artifact.dims[-1]
    q = rng.normal(0, 0.2, (1, hq, sq, d)).astype(np.float32)
    key = rng.normal(0, 0.2, (1, hkv, sk, d)).astype(np.float32)
    value = rng.normal(0, 0.2, key.shape).astype(np.float32)
    do = rng.normal(0, 0.2, q.shape).astype(np.float32)
    bias = rng.normal(0, 0.05, (1, hq, sq, sk)).astype(np.float32)
    expected, row_lse = _x86_reference(
        do, q, key, value, bias, scale=artifact.scale, window=3, softcap=4.0
    )
    outputs = (np.empty_like(q), np.empty_like(key), np.empty_like(value))
    actual = _submit_x86_native(
        package.image, package.descriptor,
        {"do": do, "q": q, "key": key, "v": value, "bias": bias,
         "row_lse": row_lse, "dq": outputs[0], "dk": outputs[1], "dv": outputs[2]},
        {"B": 1, "Hq": hq, "Hkv": hkv, "Sq": sq, "Sk": sk, "D": d, "Dv": d},
        None,
    )
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=2e-4, atol=3e-5)


@_needs_opt
@pytest.mark.compiler_rocm
def test_gfx1151_scheduled_attention_backward_packages_exact_tile_program() -> None:
    artifact = lower_scheduled_attention_backward(
        _module(1, 4, 2, 17, 19, 16), target="rocm_gfx1151"
    )
    program = package_rocm(artifact, pipeline_name="tessera-lower-to-rocm")
    assert len(program.descriptors) == 5
    assert program.descriptors[0].provenance["tile_ir_digest"] == artifact.tile_digest
    assert program.descriptors[0].provenance["lse_checkpoint"] == "recompute"


@pytest.mark.compiler_rocm
@pytest.mark.hardware_rocm
@pytest.mark.parametrize(
    ("hq", "hkv", "sq", "sk"),
    [(2, 2, 16, 16), (4, 2, 17, 19), (4, 1, 15, 21)],
)
@_needs_opt
def test_gfx1151_scheduled_attention_backward_exact_mha_gqa_mqa(
    hq: int, hkv: int, sq: int, sk: int
) -> None:
    if not _rocm_wmma_runtime_available():
        pytest.skip("a WSL-visible gfx1151 device is unavailable")
    artifact = lower_scheduled_attention_backward(
        _module(1, hq, hkv, sq, sk, 16), target="rocm_gfx1151"
    )
    program = package_rocm(artifact, pipeline_name="tessera-lower-to-rocm")
    rng = np.random.default_rng(20260805 + hkv + sk)
    q = rng.normal(0, 0.2, (1, hq, sq, 16)).astype(np.float16)
    key = rng.normal(0, 0.2, (1, hkv, sk, 16)).astype(np.float16)
    value = rng.normal(0, 0.2, key.shape).astype(np.float16)
    do = rng.normal(0, 0.2, q.shape).astype(np.float16)
    bias = rng.normal(0, 0.05, (1, hq, sq, sk)).astype(np.float32)
    outputs = (np.empty(q.shape, np.float32), np.empty(key.shape, np.float32), np.empty(value.shape, np.float32))
    result = _submit_rocm_gfx1151_attention_backward_program(
        program,
        {"do": do, "q": q, "key": key, "v": value, "bias": bias,
         "dq": outputs[0], "dk": outputs[1], "dv": outputs[2]},
        warmup=1, iterations=1,
    )
    expected = reference_attention_backward_split_reduced(
        do, q, key, value, split_count=2, scale=math.sqrt(1 / 16), bias=bias,
        causal=True, window_left=8, window_right=0, softcap=8.0,
        dropout_seed=37,
    )
    for observed, reference in zip(result["outputs"], expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=0, atol=2e-2)
