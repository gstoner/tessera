"""Native gfx1151 recurrent-cell device kernels (generate-rocm-recurrent-cell-
kernel): simple_rnn_cell + gru_cell.

These promote the two cells off the host-reference structured-compute lane onto a
real fused device kernel — the two gate GEMMs and the elementwise gate math run in
one hsaco, so the `native_gpu` provenance is genuine (previously the lane reported
native_gpu while running host numpy). f16/bf16/f32 storage, f32 accumulate.

Locks: (1) the native path matches nn.functional across dtypes, activations, and
bias variants, and reports `native_gpu`; (2) a shape/dtype outside the native
contract cleanly demotes to the host reference (`reference_cpu`) — GPU-free, so it
runs everywhere. Skip-clean without tessera-opt / a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.nn import functional as F


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op_name, names, kwargs=None):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_structured_compute_compiled",
        "executable": True, "arg_names": list(names), "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": list(names),
                 "kwargs": dict(kwargs or {})}]})


def _run(rt, op_name, names, args, kwargs=None):
    res = rt.launch(_art(rt, op_name, names, kwargs), args)
    assert res["ok"] is True, res.get("reason")
    return np.asarray(res["output"], np.float32), res.get("execution_kind")


_RNG = np.random.default_rng(31)
B, In, H = 4, 6, 5


def _dtypes():
    from tessera import runtime as rt
    out = [("f32", np.float32, 3e-4), ("f16", np.float16, 3e-2)]
    bf16 = rt._bfloat16_dtype()
    if bf16 is not None:
        out.append(("bf16", bf16, 3e-2))
    return out


@pytest.mark.parametrize("tag,dt,tol", _dtypes())
def test_simple_rnn_native_matches_reference(tag, dt, tol):
    rt = _rocm_or_skip()
    x = _RNG.standard_normal((B, In)).astype(dt)
    h = _RNG.standard_normal((B, H)).astype(dt)
    Wih = _RNG.standard_normal((In, H)).astype(dt)
    Whh = _RNG.standard_normal((H, H)).astype(dt)
    got, kind = _run(rt, "tessera.simple_rnn_cell", ("x", "h", "Wih", "Whh"),
                     (x, h, Wih, Whh))
    ref = F.simple_rnn_cell(x.astype(np.float32), h.astype(np.float32),
                            Wih.astype(np.float32), Whh.astype(np.float32))
    assert kind == "native_gpu"        # genuine device execution, not host numpy
    np.testing.assert_allclose(got, ref, atol=tol, rtol=tol)


def test_simple_rnn_relu_with_bias_native(rt=None):
    rt = _rocm_or_skip()
    x = _RNG.standard_normal((B, In)).astype(np.float32)
    h = _RNG.standard_normal((B, H)).astype(np.float32)
    Wih = _RNG.standard_normal((In, H)).astype(np.float32)
    Whh = _RNG.standard_normal((H, H)).astype(np.float32)
    bias = _RNG.standard_normal((H,)).astype(np.float32)
    got, kind = _run(rt, "tessera.simple_rnn_cell", ("x", "h", "Wih", "Whh", "b"),
                     (x, h, Wih, Whh, bias), {"activation": "relu"})
    assert kind == "native_gpu"
    np.testing.assert_allclose(
        got, F.simple_rnn_cell(x, h, Wih, Whh, bias=bias, activation="relu"),
        atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("tag,dt,tol", _dtypes())
def test_gru_native_matches_reference(tag, dt, tol):
    rt = _rocm_or_skip()
    x = _RNG.standard_normal((B, In)).astype(dt)
    h = _RNG.standard_normal((B, H)).astype(dt)
    Wih = _RNG.standard_normal((In, 3 * H)).astype(dt)
    Whh = _RNG.standard_normal((H, 3 * H)).astype(dt)
    got, kind = _run(rt, "tessera.gru_cell", ("x", "h", "Wih", "Whh"),
                     (x, h, Wih, Whh))
    ref = F.gru_cell(x.astype(np.float32), h.astype(np.float32),
                     Wih.astype(np.float32), Whh.astype(np.float32))
    assert kind == "native_gpu"
    np.testing.assert_allclose(got, ref, atol=tol, rtol=tol)


def test_gru_with_biases_native():
    rt = _rocm_or_skip()
    x = _RNG.standard_normal((B, In)).astype(np.float32)
    h = _RNG.standard_normal((B, H)).astype(np.float32)
    Wih = _RNG.standard_normal((In, 3 * H)).astype(np.float32)
    Whh = _RNG.standard_normal((H, 3 * H)).astype(np.float32)
    bih = _RNG.standard_normal((3 * H,)).astype(np.float32)
    bhh = _RNG.standard_normal((3 * H,)).astype(np.float32)
    got, kind = _run(rt, "tessera.gru_cell", ("x", "h", "Wih", "Whh", "bih", "bhh"),
                     (x, h, Wih, Whh, bih, bhh))
    assert kind == "native_gpu"
    np.testing.assert_allclose(got, F.gru_cell(x, h, Wih, Whh, bih, bhh),
                               atol=3e-4, rtol=3e-4)


# ── fallback to the host reference (GPU-free — runs everywhere) ────────────────

def test_native_kernel_rejects_bad_shape_for_host_fallback():
    # A weight shape outside the native contract must raise _RocmCompiledUnavailable
    # (BEFORE any hsaco/GPU work) so the dispatch cleanly demotes to the host
    # reference — never a silent wrong answer. This is the provenance gate that
    # keeps `native_gpu` honest.
    from tessera import runtime as rt

    x = np.zeros((2, 6), np.float32)
    h = np.zeros((2, 5), np.float32)
    Wih_bad = np.zeros((6, 4), np.float32)      # H mismatch (4 != 5)
    Whh = np.zeros((5, 5), np.float32)
    with pytest.raises(rt._RocmCompiledUnavailable):
        rt._rocm_recurrent_cell("simple_rnn", [x, h, Wih_bad, Whh], {}, np)
    # gru weights must be (In,3H)/(H,3H).
    with pytest.raises(rt._RocmCompiledUnavailable):
        rt._rocm_recurrent_cell("gru", [x, h, np.zeros((6, 5), np.float32),
                                        np.zeros((5, 15), np.float32)], {}, np)


def test_simple_rnn_unsupported_activation_falls_back():
    # sigmoid isn't in the native kernel's {tanh, relu}; must demote (host handles
    # it) rather than silently produce tanh output.
    from tessera import runtime as rt

    x = np.zeros((2, 6), np.float32)
    h = np.zeros((2, 5), np.float32)
    Wih = np.zeros((6, 5), np.float32)
    Whh = np.zeros((5, 5), np.float32)
    with pytest.raises(rt._RocmCompiledUnavailable):
        rt._rocm_recurrent_cell("simple_rnn", [x, h, Wih, Whh],
                                {"activation": "sigmoid"}, np)
