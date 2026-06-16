"""M4 — whole-graph pointwise MSL emitter (the GPU `tessera_jit` foundation).

An arbitrary connected DAG of same-shape elementwise ops compiles to ONE Metal
kernel run in a single dispatch, instead of N separate MPSGraph dispatches with
host round-trips between them — the GPU analogue of the CPU `run_graph_ops` lane.
See docs/audit/backend/apple/APPLE_GPU_CODEGEN_PLAN.md (M4).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import fusion as F

_RNG = np.random.default_rng(20260615)


def _gelu(v):
    t = np.clip(0.7978845608028654 * (v + 0.044715 * v**3), -30.0, 30.0)
    return 0.5 * v * (1.0 + np.tanh(t))


# ── discoverer ──────────────────────────────────────────────────────────────
def test_discovers_connected_pointwise_dag():
    ops = [F._Op("tessera.mul", ("x", "a"), "m"),
           F._Op("tessera.add", ("m", "b"), "s"),
           F._Op("tessera.gelu", ("s",), "o")]
    (regs,) = F.discover_pointwise_graph(ops)
    idxs, region = regs
    assert idxs == [0, 1, 2]
    assert region.inputs == ("x", "a", "b") and region.output == "o"


def test_diamond_reuses_inputs():
    # mul(add(x,y), sub(x,y)) — x,y feed two ops; one region, inputs (x,y).
    ops = [F._Op("tessera.add", ("x", "y"), "s1"),
           F._Op("tessera.sub", ("x", "y"), "s2"),
           F._Op("tessera.mul", ("s1", "s2"), "o")]
    (regs,) = F.discover_pointwise_graph(ops)
    _idxs, region = regs
    assert set(region.inputs) == {"x", "y"} and region.output == "o"


def test_single_op_not_fused():
    # A lone pointwise op stays on the MPSGraph elementwise lane.
    assert F.discover_pointwise_graph([F._Op("tessera.relu", ("x",), "o")]) == []


def test_non_pointwise_op_bounds_the_region():
    # The matmul isn't pointwise → only the gelu+add after it could fuse, but
    # that's a 2-op pointwise tail consuming the matmul result (external input).
    ops = [F._Op("tessera.matmul", ("x", "w"), "m"),
           F._Op("tessera.add", ("m", "b"), "s"),
           F._Op("tessera.gelu", ("s",), "o")]
    regs = F.discover_pointwise_graph(ops)
    assert len(regs) == 1
    idxs, region = regs[0]
    assert idxs == [1, 2]                       # matmul excluded
    assert "m" in region.inputs                 # matmul result is an external input


def test_skip_excludes_claimed_ops():
    ops = [F._Op("tessera.mul", ("x", "a"), "m"),
           F._Op("tessera.add", ("m", "b"), "o")]
    assert F.discover_pointwise_graph(ops, skip={0}) == []  # mul claimed → no 2-op region


# ── synthesizer ─────────────────────────────────────────────────────────────
def test_synthesizer_emits_one_kernel_with_n_input_buffers():
    region = F.PointwiseGraphRegion(
        ops=(("mul", ("x", "a"), "m"), ("add", ("m", "b"), "s"),
             ("gelu", ("s",), "o")),
        inputs=("x", "a", "b"), output="o")
    src = F.synthesize_pointwise_graph_msl(region)
    assert "in0 [[buffer(0)]]" in src and "in2 [[buffer(2)]]" in src
    assert "O   [[buffer(3)]]" in src           # output after the 3 inputs
    assert src.count("float t") == 3            # one temp per op


# ── runner: executes on Metal, matches numpy ────────────────────────────────
def test_run_pointwise_dag_matches_numpy():
    region = F.PointwiseGraphRegion(
        ops=(("mul", ("x", "a"), "m"), ("add", ("m", "b"), "s"),
             ("gelu", ("s",), "o")),
        inputs=("x", "a", "b"), output="o")
    x, a, b = (_RNG.standard_normal((4, 16)).astype(np.float32) for _ in range(3))
    out, ex = F.run_pointwise_graph(region, [x, a, b])
    assert ex in ("metal_runtime", "reference")
    np.testing.assert_allclose(np.asarray(out), _gelu(x * a + b),
                               rtol=1e-5, atol=1e-5)


# ── M4 broadcast operands (per-feature bias/scale) ──────────────────────────
def test_synthesizer_broadcast_input_indexes_modulo_cols():
    # A broadcast input must read in{i}[gid % C], a full input in{i}[gid]; the
    # cols-modulus buffer is declared iff some input is broadcast.
    region = F.PointwiseGraphRegion(
        ops=(("mul", ("x", "s"), "m"), ("add", ("m", "bias"), "o")),
        inputs=("x", "s", "bias"), output="o")
    src = F.synthesize_pointwise_graph_msl(region, "f32", (False, True, True))
    assert "in0[gid]" in src                  # x is full
    assert "in1[gid % (uint)C]" in src        # scale broadcasts
    assert "in2[gid % (uint)C]" in src        # bias broadcasts
    assert "constant int&       C" in src     # cols buffer declared
    # No-broadcast emission has no C buffer (regression on the original path).
    full = F.synthesize_pointwise_graph_msl(region, "f32")
    assert "% (uint)C" not in full and "constant int&       C" not in full


@pytest.mark.parametrize("bias_shape", [(64,), (1, 64)])
def test_run_pointwise_per_feature_bias_scale_on_metal(bias_shape):
    # relu(x * scale + bias) with per-feature scale/bias — the ubiquitous case.
    region = F.PointwiseGraphRegion(
        ops=(("mul", ("x", "s"), "m"), ("add", ("m", "b"), "t"),
             ("relu", ("t",), "o")),
        inputs=("x", "s", "b"), output="o")
    x = _RNG.standard_normal((32, 64)).astype(np.float32)
    s = _RNG.standard_normal(bias_shape).astype(np.float32)
    b = _RNG.standard_normal(bias_shape).astype(np.float32)
    out, ex = F.run_pointwise_graph(region, [x, s, b])
    ref = np.maximum(x * s + b, 0.0)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)
    if sys.platform == "darwin":
        assert ex == "metal_runtime"           # broadcast fused on Metal


def test_run_pointwise_per_row_broadcast_declines_to_reference():
    # (rows,1) broadcast is NOT last-dim-aligned → must decline (never mis-index).
    region = F.PointwiseGraphRegion(
        ops=(("mul", ("x", "s"), "m"), ("gelu", ("m",), "o")),
        inputs=("x", "s"), output="o")
    x = _RNG.standard_normal((32, 64)).astype(np.float32)
    s = _RNG.standard_normal((32, 1)).astype(np.float32)
    out, ex = F.run_pointwise_graph(region, [x, s])
    assert ex == "reference"
    np.testing.assert_allclose(np.asarray(out), _gelu(x * s), rtol=1e-5, atol=1e-5)


# ── M5 pointwise -> plain row-reduction fusion ──────────────────────────────
def test_pointwise_reduce_msl_has_accumulator_no_score_array():
    region = F.PointwiseReduceRegion(
        ops=(("mul", ("x", "x"), "sq"),), inputs=("x",), output="sq", reduce="sum")
    src = F.synthesize_pointwise_reduce_msl(region, "f32")
    assert "float acc = 0.0f;" in src              # sum init
    assert "acc = acc + v;" in src                 # accumulate
    assert "for (int c = 0; c < cols" in src       # reduce over the row
    # mean appends a /cols finalize; amax/amin use INFINITY inits.
    assert "/= float(cols)" in F.synthesize_pointwise_reduce_msl(
        F.PointwiseReduceRegion((("abs", ("x",), "a"),), ("x",), "a", "mean"), "f32")
    assert "-INFINITY" in F.synthesize_pointwise_reduce_msl(
        F.PointwiseReduceRegion((("exp", ("x",), "e"),), ("x",), "e", "amax"), "f32")


def test_pointwise_reduce_region_validation():
    with pytest.raises(ValueError):
        F.PointwiseReduceRegion((("abs", ("x",), "a"),), ("x",), "a", "nope")


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
@pytest.mark.parametrize("reduce,fn", [
    ("sum", lambda a: a.sum(-1)),
    ("mean", lambda a: a.mean(-1)),
    ("amax", lambda a: a.max(-1)),
    ("amin", lambda a: a.min(-1)),
])
def test_pointwise_reduce_on_metal(reduce, fn):
    # reduce(x*x) over the last axis, fused into one kernel. Output drops the
    # last axis; matches numpy.
    region = F.PointwiseReduceRegion(
        ops=(("mul", ("x", "x"), "sq"),), inputs=("x",), output="sq", reduce=reduce)
    x = (_RNG.standard_normal((32, 64))).astype(np.float32)
    out, ex = F.run_pointwise_reduce(region, [x])
    assert ex == "metal_runtime"
    assert np.asarray(out).shape == (32,)          # last axis dropped
    np.testing.assert_allclose(np.asarray(out), fn(x * x), rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
def test_pointwise_reduce_3d_reduces_last_axis():
    # 3D input → output keeps leading dims, drops the last.
    region = F.PointwiseReduceRegion(
        ops=(("abs", ("x",), "a"),), inputs=("x",), output="a", reduce="mean")
    x = (_RNG.standard_normal((8, 4, 32))).astype(np.float32)
    out, ex = F.run_pointwise_reduce(region, [x])
    assert ex == "metal_runtime"
    assert np.asarray(out).shape == (8, 4)
    np.testing.assert_allclose(np.asarray(out), np.abs(x).mean(-1), rtol=1e-4, atol=1e-3)


# ── end-to-end @jit(apple_gpu) ──────────────────────────────────────────────
def _dag(x, a, b):
    return ts.ops.gelu(ts.ops.add(ts.ops.mul(x, a), b))


def _chain(x):
    return ts.ops.tanh(ts.ops.silu(ts.ops.relu(x)))


def test_jit_pointwise_dag_matches_numpy():
    fn = ts.jit(target="apple_gpu")(_dag)
    x, a, b = (_RNG.standard_normal((8, 32)).astype(np.float32) for _ in range(3))
    np.testing.assert_allclose(np.asarray(fn(x, a, b)), _gelu(x * a + b),
                               rtol=1e-4, atol=1e-4)


def test_jit_pointwise_chain_matches_numpy():
    fn = ts.jit(target="apple_gpu")(_chain)
    x = _RNG.standard_normal((8, 32)).astype(np.float32)
    r = np.maximum(x, 0.0)
    s = r / (1.0 + np.exp(-r))
    np.testing.assert_allclose(np.asarray(fn(x)), np.tanh(s), rtol=1e-4, atol=1e-4)


def test_jit_pointwise_f16():
    fn = ts.jit(target="apple_gpu")(_dag)
    x, a, b = (_RNG.standard_normal((8, 32)).astype(np.float16) for _ in range(3))
    got = np.asarray(fn(x, a, b)).astype(np.float32)
    ref = _gelu(x.astype(np.float32) * a.astype(np.float32) + b.astype(np.float32))
    np.testing.assert_allclose(got, ref, rtol=3e-2, atol=3e-2)
