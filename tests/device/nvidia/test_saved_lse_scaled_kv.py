"""Exact-SM120 scaled FP8 K/V saved-LSE forward (SM120-SAVED-LSE-SCALED-FP8-KV-20261025)."""
import json

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests.device.nvidia.test_saved_lse_storage import compare, reference

pytestmark = pytest.mark.hardware_nvidia


def scaled(q, k8, v8, ks, vs):
    k = ts.ops.dequantize_scaled(k8, ks)
    v = ts.ops.dequantize_scaled(v8, vs)
    return ts.ops.flash_attn(q, k, v, causal=True, lse_checkpoint="saved")


def scaled_open(q, k8, v8, ks, vs):
    k = ts.ops.dequantize_scaled(k8, ks)
    v = ts.ops.dequantize_scaled(v8, vs)
    return ts.ops.flash_attn(q, k, v, causal=False, lse_checkpoint="saved")


def scaled_bias(q, k8, v8, ks, vs, bias):
    k = ts.ops.dequantize_scaled(k8, ks)
    v = ts.ops.dequantize_scaled(v8, vs)
    return ts.ops.flash_attn(q, k, v, bias=bias, causal=True, lse_checkpoint="saved")


def _dtype(name):
    return np.dtype(getattr(ml_dtypes, name) if name in (
        "bfloat16", "float8_e4m3fn", "float8_e5m2") else name)


#: (K scale shape, V scale shape) templates over [B, Hkv, Sk, D|Dv].
GRANULARITY = {
    "per_tensor": lambda sk, d, dv: ((1, 1, 1, 1), (1, 1, 1, 1)),
    "per_head": lambda sk, d, dv: ((1, 2, 1, 1), (1, 2, 1, 1)),
    "per_token": lambda sk, d, dv: ((1, 2, sk, 1), (1, 2, sk, 1)),
    "per_channel": lambda sk, d, dv: ((1, 1, 1, d), (1, 1, 1, dv)),
    "full": lambda sk, d, dv: ((1, 2, sk, d), (1, 2, sk, dv)),
}


def _inputs(q_name, kv_name, granularity, sk, bias):
    rng = np.random.default_rng(20261025)
    hq, hkv, sq, d, dv = 4, 2, 3, 8, 6
    kv = _dtype(kv_name)
    ks_shape, vs_shape = GRANULARITY[granularity](sk, d, dv)
    values = {
        "q": rng.normal(0, .3, (1, hq, sq, d)).astype(_dtype(q_name)),
        "k8": rng.normal(0, 3, (1, hkv, sk, d)).astype(kv),
        "v8": rng.normal(0, 3, (1, hkv, sk, dv)).astype(kv),
        "ks": rng.uniform(.02, .3, ks_shape).astype(np.float32),
        "vs": rng.uniform(.02, .3, vs_shape).astype(np.float32),
    }
    if bias:
        values["bias"] = rng.normal(0, .1, (1, hq, 1, 1)).astype(np.float32)
    return values


def _expected(values, causal):
    # The declared semantics: each K/V element is f32(code) * scale rounded
    # once to f32 (the dequantize_scaled reference), then fp64 attention.
    dense = {"q": values["q"], "k": ts.ops.dequantize_scaled(values["k8"], values["ks"]),
             "v": ts.ops.dequantize_scaled(values["v8"], values["vs"])}
    if "bias" in values:
        dense["bias"] = values["bias"]
    return reference(dense, causal)


@pytest.mark.parametrize("q_name", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("kv_name", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("granularity", sorted(GRANULARITY))
@pytest.mark.parametrize("sk", [5, 129])
@pytest.mark.parametrize("variant", ["causal", "open", "bias"])
def test_scaled_kv_public_and_serialized(q_name, kv_name, granularity, sk, variant):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    values = _inputs(q_name, kv_name, granularity, sk, variant == "bias")
    function = {"causal": scaled, "open": scaled_open, "bias": scaled_bias}[variant]
    causal = variant != "open"
    expected = _expected(values, causal)
    fn = ts.jit(target="nvidia_sm120")(function)
    host = fn(**values)
    receipt = fn._native_descriptor_last_receipt
    assert receipt and receipt["execution_kind"] == "native_gpu", receipt
    artifact = fn._cached_artifact
    assert "_scaled_" in artifact.launch_descriptor.entry_symbol
    compare(host, expected, _dtype(q_name))
    # Serialized replay through the checked launch, with guard bytes.
    artifact = rt.RuntimeArtifact.from_dict(json.loads(json.dumps(artifact.to_dict())))
    descriptor = artifact.launch_descriptor
    bindings = sorted(descriptor.buffers, key=lambda binding: binding.ordinal)
    # Input bindings are the tracer's positional names (`aN` = frontend
    # argument N) in entry order (Q, K, V, [bias], K scale, V scale), which
    # differs from the signature order.
    inputs = [binding for binding in bindings if binding.direction == "input"]
    assert sorted(binding.name for binding in inputs) == [f"a{i}" for i in range(len(values))]
    arguments = {binding.name: values[fn.arg_names[int(binding.name[1:])]] for binding in inputs}
    arenas = [np.full(value.size + 16, 37, value.dtype) for value in host]
    returned = tuple(arena[8:-8].reshape(value.shape) for arena, value in zip(arenas, host, strict=True))
    outputs = [binding for binding in bindings if binding.direction == "output"]
    arguments.update((binding.name, value) for binding, value in zip(outputs, returned, strict=True))
    scalar_names = [scalar.name for scalar in sorted(descriptor.scalars, key=lambda value: value.ordinal)]
    provenance = descriptor.provenance
    bias_values = tuple(provenance["bias_shape"]) if "BiasB" in scalar_names else ()
    scalar_values = (*provenance["shape"], *bias_values, *provenance["kv_scale_shapes"])
    arguments.update(zip(scalar_names, scalar_values, strict=True))
    result = rt.launch(artifact, arguments)
    assert result["ok"] and result["execution_kind"] == "native_gpu", result
    for actual, original in zip(returned, host, strict=True):
        np.testing.assert_array_equal(actual, original)
    for arena in arenas:
        np.testing.assert_array_equal(arena[:8], np.full(8, 37, arena.dtype))
        np.testing.assert_array_equal(arena[-8:], np.full(8, 37, arena.dtype))


@pytest.mark.parametrize("kv_name", ["float8_e4m3fn", "float8_e5m2"])
def test_scaled_kv_applies_each_scale_exactly(kv_name):
    """One key per batch element (softmax weight exactly 1): O must equal the
    f32-rounded product f32(v_code) * v_scale and LSE must equal
    0.5 * f32(k_code) * k_scale, bit for bit, for every finite code."""
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    kv = _dtype(kv_name)
    codes = np.arange(256, dtype=np.uint8).view(kv)
    finite = codes[np.isfinite(codes.astype(np.float64))]
    batch = finite.size
    rng = np.random.default_rng(7)
    k8 = np.zeros((batch, 1, 1, 4), kv)
    k8[:, 0, 0, 0] = finite
    v8 = np.repeat(finite[:, None], 3, axis=1).reshape(batch, 1, 1, 3).astype(kv)
    ks = rng.uniform(.01, 2, (batch, 1, 1, 1)).astype(np.float32)
    vs = rng.uniform(.01, 2, (batch, 1, 1, 3)).astype(np.float32)
    q = np.zeros((batch, 1, 1, 4), np.float32)
    q[..., 0] = 1.0
    fn = ts.jit(target="nvidia_sm120")(scaled_open)
    output, lse = fn(q=q, k8=k8, v8=v8, ks=ks, vs=vs)
    assert fn._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
    key = ts.ops.dequantize_scaled(k8, ks)[:, 0, 0, 0]
    value = ts.ops.dequantize_scaled(v8, vs)[:, 0, 0, :]
    np.testing.assert_array_equal(output[:, 0, 0, :], value)
    np.testing.assert_array_equal(lse[:, 0, 0], key * np.float32(0.5))
