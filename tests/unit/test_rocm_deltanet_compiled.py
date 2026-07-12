"""Compiler-generated gated/delta linear-attention scan on gfx1151 — the first
RECURRENT device_verified_jit ROCm kernel (gated_deltanet / kimi_delta_attention /
modified_delta_attention).

The `tessera_rocm.deltanet` directive expands (via
`generate-rocm-deltanet-kernel`) into a causal sequential-scan kernel — one
workgroup per (b,h), one thread per value-column, LDS state. Reachable through
`runtime.launch()` via `compiler_path="rocm_deltanet_compiled"`; operands
[Q,K,V,gate?,beta?,decay?] with has_gate/has_beta/has_decay + erase kwargs;
`modified` from the op name. f16/bf16/f32 storage, f32 compute.

Validated vs the numpy `_delta_attention_impl` reference (ported here). Keys are
L2-normalized — real DeltaNet needs that for f32/half conditioning (the L1.1
finding).

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _dn_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _ref(Q, K, V, *, gate=None, beta=None, decay=None, erase=False,
         modified=False):
    """Causal path of _delta_attention_impl, in f64."""
    Q = Q.astype(np.float64); K = K.astype(np.float64); V = V.astype(np.float64)
    B, H, S, Dqk = Q.shape
    Dv = V.shape[-1]
    St = np.zeros((B, H, Dqk, Dv), np.float64)
    O = np.zeros((B, H, S, Dv), np.float64)
    dec = None if decay is None else decay.astype(np.float64)
    bet = None if beta is None else beta.astype(np.float64)
    for t in range(S):
        k_t = K[:, :, t, :]
        target = V[:, :, t, :]
        if erase:
            vhat = np.einsum("bhd,bhde->bhe", k_t, St)
            a_s = (dec[:, :, t][:, :, None] if dec is not None else 1.0)
            target = target - a_s * vhat
        if dec is not None:
            St = dec[:, :, t][:, :, None, None] * St
        weight = 1.0 if bet is None else bet[:, :, t][:, :, None, None]
        delta = np.einsum("bhd,bhe->bhde", k_t, target)
        if modified:
            delta = delta / (1.0 + np.linalg.norm(delta, axis=(-2, -1),
                                                  keepdims=True))
        St = St + weight * delta
        O[:, :, t, :] = np.einsum("bhd,bhde->bhe", Q[:, :, t, :], St)
    if gate is not None:
        g = 1.0 / (1.0 + np.exp(-gate.astype(np.float64)))
        O = O * g
    return O


def _l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-6)


def _artifact(rt, op_name, operands, kwargs):
    names = [f"x{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_deltanet_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": names, "kwargs": kwargs}],
    }), tuple(operands)


_OPS = ["tessera.gated_deltanet", "tessera.kimi_delta_attention",
        "tessera.modified_delta_attention"]


def _dtypes():
    out = [(np.float32, 2e-3)]
    out.append((np.float16, 4e-2))
    out.append((pytest.importorskip("ml_dtypes").bfloat16, 1.5e-1))
    return out


@pytest.mark.parametrize("op_name", _OPS)
@pytest.mark.parametrize("dtype,tol", _dtypes())
@pytest.mark.parametrize("shape", [(1, 2, 8, 16), (2, 1, 12, 16)])
def test_deltanet_plain(op_name, dtype, tol, shape):
    """Bare recurrence (no gate/beta/decay/erase)."""
    rt = _dn_or_skip()
    rng = np.random.default_rng(3 + shape[2])
    B, H, S, D = shape
    q = (rng.standard_normal(shape) * 0.5).astype(dtype)
    k = _l2(rng.standard_normal(shape)).astype(dtype)
    v = (rng.standard_normal(shape) * 0.5).astype(dtype)
    art, ops = _artifact(rt, op_name, [q, k, v], {"causal": True})
    res = rt.launch(art, ops)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_deltanet_compiled"
    out = res["output"].astype(np.float32)
    modified = op_name.endswith("modified_delta_attention")
    ref = _ref(q, k, v, modified=modified).astype(np.float32)
    np.testing.assert_allclose(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype,tol", _dtypes())
def test_deltanet_gated_beta_decay(dtype, tol):
    """gated_deltanet with gate + beta + decay all present (erase off)."""
    rt = _dn_or_skip()
    rng = np.random.default_rng(17)
    B, H, S, D = 1, 2, 10, 16
    q = (rng.standard_normal((B, H, S, D)) * 0.5).astype(dtype)
    k = _l2(rng.standard_normal((B, H, S, D))).astype(dtype)
    v = (rng.standard_normal((B, H, S, D)) * 0.5).astype(dtype)
    gate = (rng.standard_normal((B, H, S, D)) * 0.5).astype(np.float32)
    beta = (rng.uniform(0.2, 0.9, (B, H, S))).astype(np.float32)
    decay = (rng.uniform(0.85, 0.99, (B, H, S))).astype(np.float32)
    art, ops = _artifact(
        rt, "tessera.gated_deltanet", [q, k, v, gate, beta, decay],
        {"causal": True, "has_gate": True, "has_beta": True, "has_decay": True})
    res = rt.launch(art, ops)
    assert res["ok"] is True, res.get("reason")
    out = res["output"].astype(np.float32)
    ref = _ref(q, k, v, gate=gate, beta=beta, decay=decay).astype(np.float32)
    np.testing.assert_allclose(out, ref, atol=max(tol, 5e-2), rtol=max(tol, 5e-2))


@pytest.mark.parametrize("dtype,tol", _dtypes())
def test_deltanet_erase_true(dtype, tol):
    """The genuine delta rule (erase=True) with beta + decay."""
    rt = _dn_or_skip()
    rng = np.random.default_rng(29)
    B, H, S, D = 1, 2, 10, 16
    q = (rng.standard_normal((B, H, S, D)) * 0.5).astype(dtype)
    k = _l2(rng.standard_normal((B, H, S, D))).astype(dtype)
    v = (rng.standard_normal((B, H, S, D)) * 0.5).astype(dtype)
    beta = (rng.uniform(0.2, 0.9, (B, H, S))).astype(np.float32)
    decay = (rng.uniform(0.85, 0.99, (B, H, S))).astype(np.float32)
    art, ops = _artifact(
        rt, "tessera.gated_deltanet", [q, k, v, beta, decay],
        {"causal": True, "erase": True, "has_beta": True, "has_decay": True})
    res = rt.launch(art, ops)
    assert res["ok"] is True, res.get("reason")
    out = res["output"].astype(np.float32)
    ref = _ref(q, k, v, beta=beta, decay=decay, erase=True).astype(np.float32)
    np.testing.assert_allclose(out, ref, atol=max(tol, 5e-2), rtol=max(tol, 5e-2))


def test_deltanet_non_causal_rejected():
    from tessera import runtime as rt
    z = np.zeros((1, 1, 4, 16), np.float32)
    art, ops = _artifact(rt, "tessera.gated_deltanet", [z, z, z],
                         {"causal": False})
    with pytest.raises(ValueError, match="causal-only"):
        rt._execute_rocm_compiled_deltanet(art, ops)


# ── GPU-free codegen gate (needs only tessera-opt, not a GPU) ────────────────
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_OPT = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"


def _gen(directive, *passes):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(_OPT), "-", *passes],
                          input=directive, capture_output=True, text=True)


def _directive(**attrs):
    body = ", ".join(
        f'{k} = {v}' for k, v in
        [("name", '"dn"'), ("d_qk", "16 : i64"), ("d_v", "16 : i64")]
        + [(k, ("true" if v else "false")) for k, v in attrs.items()]
        + [("dtype", '"f32"')])
    return f'module {{\n  "tessera_rocm.deltanet"() {{{body}}} : () -> ()\n}}\n'


@pytest.mark.parametrize("attrs", [
    {}, {"erase": True}, {"modified": True},
    {"has_gate": True, "has_beta": True, "has_decay": True},
])
def test_deltanet_codegen_signature_and_lowers(attrs):
    """The directive expands to an 8-arg scan kernel and lowers to ROCDL."""
    import re
    ir = _gen(_directive(**attrs), "--generate-rocm-deltanet-kernel")
    assert ir.returncode == 0, ir.stderr
    m = re.search(r"gpu\.func @dn\(([^)]*)\)", ir.stdout)
    # (Q,K,V,O,gate : memref, beta,decay : memref, S : index) -> 8 args
    assert m and len([a for a in m.group(1).split(",") if a.strip()]) == 8
    assert "scf.for" in ir.stdout and "gpu.barrier" in ir.stdout
    low = _gen(_directive(**attrs),
               "--pass-pipeline=builtin.module(generate-rocm-deltanet-kernel,"
               "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
               "reconcile-unrealized-casts))")
    assert low.returncode == 0, low.stderr
    assert "llvm." in low.stdout


def test_deltanet_codegen_bad_dims_rejected():
    bad = ('module {\n  "tessera_rocm.deltanet"() {name = "dn", d_qk = 0 : i64, '
           'd_v = 16 : i64} : () -> ()\n}\n')
    r = _gen(bad, "--generate-rocm-deltanet-kernel")
    assert r.returncode != 0 and "d_qk and d_v must be positive" in r.stderr
