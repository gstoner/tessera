"""Exact RTX5070 proof of native recompute Graph ancestry and policy."""
import numpy as np
import pytest

from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.driver import compile_graph_module
from tessera.runtime import launch
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.unit.test_nvidia_native_recompute_schedule import recompute_module


def reference(q, k, v, do, bias, *, scale, causal, window, softcap, dropout, seed):
    q, k, v, do = (np.asarray(x, dtype=np.float64) for x in (q, k, v, do))
    b, hq, sq, _ = q.shape
    _, hkv, sk, _ = k.shape
    result = [np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)]
    keys = np.arange(sk)
    for batch in range(b):
        for head in range(hq):
            kv = head * hkv // hq
            for row in range(sq):
                aligned = row + max(sk - sq, 0)
                legal = np.ones(sk, dtype=bool)
                if causal:
                    legal &= keys <= aligned
                if window[0] >= 0:
                    legal &= keys >= aligned - window[0]
                if window[1] >= 0:
                    legal &= keys <= aligned + window[1]
                raw = scale * (k[batch, kv] @ q[batch, head, row])
                if bias is not None:
                    raw += np.asarray(bias[batch, head, row], dtype=np.float64)
                t = np.tanh(raw / softcap) if softcap else None
                scores = softcap * t if softcap else raw
                scores = np.where(legal, scores, -np.inf)
                if not legal.any():
                    continue
                p = np.exp(scores - scores.max())
                p /= p.sum()
                counter = (((batch * hq + head) * sq + row) * sk + keys)
                hashes = (counter * 1664525 + (seed & 0xffffffff) + 1013904223) & 0xffffffff
                multiplier = (hashes >= int(dropout * 4294967296)).astype(np.float64) / (1 - dropout)
                dp = multiplier * (v[batch, kv] @ do[batch, head, row])
                ds = p * (dp - np.sum(p * dp))
                if softcap:
                    ds *= 1 - t * t
                result[0][batch, head, row] += scale * (ds @ k[batch, kv])
                result[1][batch, kv] += scale * ds[:, None] * q[batch, head, row]
                result[2][batch, kv] += (p * multiplier)[:, None] * do[batch, head, row]
    return result


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("advanced", [False, True])
def test_native_recompute_policy_and_ssa_roles(dtype, bias, permuted, advanced, monkeypatch):
    _exercise(dtype, bias, permuted, advanced, monkeypatch)


def _exercise(dtype, bias, permuted, advanced, monkeypatch, seed_override=None):
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    from tessera.compiler import nvidia_native as native
    from tessera import runtime as rt
    assert rt._nvidia_device_name() == "sm_120"
    monkeypatch.setattr(native, "emit_attention_backward_tile_ir",
                        lambda **kwargs: pytest.fail("legacy Python Tile constructor"))
    module = recompute_module(dtype, bias, permuted)
    policy = dict(scale=.5, causal=True, window=(-1, -1), softcap=0.,
                  dropout=0., seed=0)
    if advanced:
        policy.update(scale=.3, window=(2, 0), softcap=.7, dropout=.2, seed=-182400000)
    if seed_override is not None:
        policy["seed"] = seed_override
    module.functions[0].body[0].kwargs.update(policy)
    bundle = compile_graph_module(module, source_origin="NVIDIA-LSE-1", target="nvidia_sm120",
                                  options={"package_native": True}, enable_tool_validation=False)
    provenance = bundle.launch_descriptor.provenance
    assert provenance["compiler_route"] == "canonical_scheduled_tile_consumer"
    for key in ("graph_ir_digest", "schedule_digest", "schedule_ir_digest", "tile_ir_digest"):
        assert len(provenance[key]) == 64
    storage = {"fp16": np.float16, "fp32": np.float32}.get(dtype)
    if dtype == "bf16":
        from ml_dtypes import bfloat16
        storage = bfloat16
    rng = np.random.default_rng(120106)
    q, k, v, do = ((rng.normal(size=shape) * .2).astype(storage)
                  for shape in ((1, 2, 3, 4), (1, 1, 4, 4), (1, 1, 4, 3), (1, 2, 3, 3)))
    b = (rng.normal(size=(1, 2, 3, 4)) * .3).astype(np.float32) if bias else None
    outputs = [np.empty_like(x) for x in (q, k, v)]
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    dims = dict(zip(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"), (1, 2, 1, 3, 4, 4, 3)))
    bindings = dict(do=do, q=q, k=k, v=v, dq=outputs[0], dk=outputs[1], dv=outputs[2], **dims)
    if bias:
        bindings["bias"] = b
    for iteration in range(2):
        if iteration:
            q *= storage(.75)
            v *= storage(-.5)
        result = launch(artifact, bindings)
        assert result.get("ok"), result
        assert result.get("execution_kind") == "native_gpu"
        expected = reference(q, k, v, do, b, **policy)
        for actual, ref in zip(outputs, expected, strict=True):
            # Compare against independently rounded output-storage values.
            rounded = ref.astype(storage).astype(np.float64)
            atol, rtol = ((2e-6, 2e-4) if dtype == "fp32" else
                          (3e-5, 4e-3) if dtype == "fp16" else (2e-4, 2e-2))
            np.testing.assert_allclose(actual.astype(np.float64), rounded, atol=atol, rtol=rtol)

@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("seed", [-(1 << 63), (1 << 63) - 1, -182400000])
def test_native_recompute_seed_modulo32_extremes(seed, monkeypatch):
    _exercise("fp32", True, True, True, monkeypatch, seed_override=seed)
