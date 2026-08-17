"""Canonical tensor-valued attention backward on the x86 AVX-512 image."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _runtime():
    from tessera import runtime

    if not runtime._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return runtime


def _artifact(runtime, *, bias: bool = False, **kwargs):
    names = ["do", "q", "k", "v"] + (["bias"] if bias else [])
    operands = ["q", "k", "v"] + (["bias"] if bias else [])
    return runtime.RuntimeArtifact(
        metadata={
            "target": "x86",
            "compiler_path": "x86_flash_attn_bwd_compiled",
            "executable": True,
            "execution_kind": "native_cpu",
            "arg_names": names,
            "output_name": "grads",
            "ops": [
                {
                    "op_name": "tessera.flash_attn",
                    "result": "grads",
                    "operands": operands,
                    "kwargs": kwargs,
                }
            ],
        }
    )


def _reference(dout, q, k, v, *, scale, causal, window=0, softcap=0.0, bias=None):
    b, hq, sq, d = q.shape
    _, hkv, sk, _ = k.shape
    dv = v.shape[-1]
    dq = np.zeros_like(q)
    dk = np.zeros_like(k)
    dv_out = np.zeros_like(v)
    offset = max(sk - sq, 0)
    for batch in range(b):
        for head in range(hq):
            kv_head = head * hkv // hq
            for row in range(sq):
                begin, end = 0, sk - 1
                if causal:
                    end = row + offset
                if window > 0:
                    if causal:
                        begin, end = row + offset - window + 1, row + offset
                    else:
                        begin = row + offset - window // 2
                        end = row + offset + window // 2
                begin, end = max(begin, 0), min(end, sk - 1)
                keys = k[batch, kv_head, begin : end + 1]
                scores = scale * (keys @ q[batch, head, row])
                derivative = np.ones_like(scores)
                if softcap > 0.0:
                    tanh = np.tanh(scores / softcap)
                    scores = softcap * tanh
                    derivative = 1.0 - tanh * tanh
                if bias is not None:
                    scores = scores + bias[batch, head, row, begin : end + 1]
                probabilities = np.exp(scores - np.max(scores))
                probabilities /= probabilities.sum()
                values = v[batch, kv_head, begin : end + 1]
                dp = values @ dout[batch, head, row]
                ds = probabilities * (dp - np.sum(probabilities * dp)) * derivative
                dq[batch, head, row] = scale * (ds @ keys)
                dk[batch, kv_head, begin : end + 1] += (
                    scale * ds[:, None] * q[batch, head, row]
                )
                dv_out[batch, kv_head, begin : end + 1] += (
                    probabilities[:, None] * dout[batch, head, row]
                )
    return dq, dk, dv_out


@pytest.mark.parametrize("checkpoint", ["recompute", "saved"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 2, 7, 9, 8, 6),
        (1, 4, 2, 8, 8, 16, 16),
        (2, 4, 1, 6, 10, 8, 5),
    ],
)
def test_attention_backward_matches_reference(shape, checkpoint):
    runtime = _runtime()
    b, hq, hkv, sq, sk, d, dv = shape
    rng = np.random.default_rng(sum(shape))
    q = (rng.standard_normal((b, hq, sq, d)) * 0.3).astype(np.float32)
    k = (rng.standard_normal((b, hkv, sk, d)) * 0.3).astype(np.float32)
    v = (rng.standard_normal((b, hkv, sk, dv)) * 0.3).astype(np.float32)
    dout = (rng.standard_normal((b, hq, sq, dv)) * 0.3).astype(np.float32)
    scale = 1.0 / np.sqrt(d)
    result = runtime.launch(
        _artifact(
            runtime,
            scale=scale,
            causal=True,
            lse_checkpoint=checkpoint,
        ),
        (dout, q, k, v),
    )
    assert result["ok"] is True, result.get("reason")
    expected = _reference(dout, q, k, v, scale=scale, causal=True)
    for actual, reference in zip(result["output"], expected):
        np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=3e-5)


def test_attention_backward_preserves_bias_window_softcap_semantics():
    runtime = _runtime()
    rng = np.random.default_rng(91)
    b, hq, hkv, sq, sk, d = 1, 4, 2, 7, 9, 8
    q = rng.standard_normal((b, hq, sq, d)).astype(np.float32)
    k = rng.standard_normal((b, hkv, sk, d)).astype(np.float32)
    v = rng.standard_normal((b, hkv, sk, d)).astype(np.float32)
    dout = rng.standard_normal((b, hq, sq, d)).astype(np.float32)
    bias = (rng.standard_normal((b, hq, sq, sk)) * 0.2).astype(np.float32)
    kwargs = {
        "scale": 1.0 / np.sqrt(d),
        "causal": False,
        "window": 5,
        "softcap": 3.0,
        "lse_checkpoint": "recompute",
    }
    result = runtime.launch(
        _artifact(runtime, bias=True, **kwargs), (dout, q, k, v, bias)
    )
    assert result["ok"] is True, result.get("reason")
    expected = _reference(dout, q, k, v, bias=bias, **{
        key: kwargs[key] for key in ("scale", "causal", "window", "softcap")
    })
    for actual, reference in zip(result["output"], expected):
        np.testing.assert_allclose(actual, reference, rtol=4e-5, atol=4e-5)


def test_parallel_attention_backward_reduction_is_bit_deterministic():
    runtime = _runtime()
    rng = np.random.default_rng(701)
    b, hq, hkv, sq, sk, d = 1, 8, 2, 32, 35, 16
    q = rng.standard_normal((b, hq, sq, d)).astype(np.float32)
    k = rng.standard_normal((b, hkv, sk, d)).astype(np.float32)
    v = rng.standard_normal((b, hkv, sk, d)).astype(np.float32)
    dout = rng.standard_normal((b, hq, sq, d)).astype(np.float32)
    artifact = _artifact(
        runtime,
        scale=1.0 / np.sqrt(d),
        causal=True,
        lse_checkpoint="recompute",
    )
    first = runtime.launch(artifact, (dout, q, k, v))["output"]
    second = runtime.launch(artifact, (dout, q, k, v))["output"]
    for lhs, rhs in zip(first, second):
        np.testing.assert_array_equal(lhs, rhs)


def test_exact_zen5_lse_packet_selects_saved_checkpoint():
    packet = json.loads(
        (
            Path(__file__).parents[2]
            / "benchmarks/baselines/x86_avx512_attention_lse_2026_07_30.json"
        ).read_text()
    )
    assert packet["work_item"] == "X86-LSE-1"
    assert packet["architecture"] == "zen5_avx512"
    assert packet["verdict"] == "saved"
    assert all(row["winner"] == "saved" for row in packet["rows"])
