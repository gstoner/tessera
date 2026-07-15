import time

import numpy as np
import tessera as ts

from tessera import runtime as rt
from tessera.stdlib import moe


def _artifact(op_name, arg_names):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_moe_transport_compiled",
        "executable": True,
        "arg_names": list(arg_names),
        "ops": [{"op_name": op_name}],
    })


def _median_ms(fn, reps=7):
    vals = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(vals))


def _plan(seed=41, tokens=12, experts=4, top_k=2, capacity=5):
    rng = np.random.default_rng(seed)
    eids = rng.integers(0, experts, size=(tokens, top_k), dtype=np.int64)
    weights = rng.random((tokens, top_k), dtype=np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    return moe.plan_dispatch(eids, weights, experts, capacity=capacity)


def _expect_native():
    """The transport ops run natively iff tessera-opt is built AND a usable AMD
    GPU is present (else the executor falls back to the stdlib oracle)."""
    return (rt._tessera_opt_path() is not None
            and rt._rocm_wmma_runtime_available())


def test_rocm_moe_dispatch_runtime_matches_dispatch_plan_oracle():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((12, 8)).astype(np.float32)
    plan = _plan()

    res = rt.launch(
        _artifact("tessera.moe_dispatch", ["x", "plan"]),
        {"x": x, "plan": plan},
    )

    assert res["ok"]
    assert res["compiler_path"] == "rocm_moe_transport_compiled"
    # dispatch is a pure row gather — native on-box, oracle off-box; exact either
    # way (device gather copies rows bit-for-bit).
    assert res["execution_kind"] == ("native_gpu" if _expect_native()
                                     else "reference_cpu")
    np.testing.assert_allclose(res["output"], moe.dispatch(x, plan), rtol=0, atol=0)


def test_rocm_moe_combine_runtime_matches_weighted_combine_oracle():
    rng = np.random.default_rng(43)
    x = rng.standard_normal((12, 8)).astype(np.float32)
    plan = _plan(seed=43)
    packed = moe.dispatch(x, plan)
    partials = packed * rng.normal(1.0, 0.05, size=packed.shape).astype(np.float32)

    res = rt.launch(
        _artifact("tessera.moe_combine", ["partials", "plan"]),
        (partials, plan),
    )

    assert res["ok"]
    assert res["execution_kind"] == ("native_gpu" if _expect_native()
                                     else "reference_cpu")
    # native combine accumulates in f32 (device scatter-add) vs the f64 oracle —
    # matches within f32 tolerance; the tiny tolerance still pins the plan
    # permutation / route weights / capacity drops.
    np.testing.assert_allclose(res["output"], moe.combine(partials, plan),
                               rtol=1e-5, atol=1e-6)


def test_rocm_grouped_swiglu_runtime_matches_grouped_gemm_oracle():
    rng = np.random.default_rng(44)
    hidden, ffn, experts = 10, 14, 3
    group_sizes = np.array([2, 0, 4], dtype=np.int64)
    x_packed = rng.standard_normal((int(group_sizes.sum()), hidden)).astype(np.float32)
    scale_h = 1.0 / np.sqrt(hidden)
    scale_f = 1.0 / np.sqrt(ffn)
    w_gate = (rng.standard_normal((experts, hidden, ffn)) * scale_h).astype(np.float32)
    w_up = (rng.standard_normal((experts, hidden, ffn)) * scale_h).astype(np.float32)
    w_down = (rng.standard_normal((experts, ffn, hidden)) * scale_f).astype(np.float32)

    res = rt.launch(
        _artifact(
            "tessera.grouped_swiglu",
            ["x_packed", "w_gate", "w_up", "w_down", "group_sizes"],
        ),
        {
            "x_packed": x_packed,
            "w_gate": w_gate,
            "w_up": w_up,
            "w_down": w_down,
            "group_sizes": group_sizes,
        },
    )

    assert res["ok"]
    # grouped_swiglu now runs the three expert GEMMs on the f32 device GEMM
    # kernel (native on-box), silu*mul host-side; f32 vs the f64 oracle.
    assert res["execution_kind"] == ("native_gpu" if _expect_native()
                                     else "reference_cpu")
    np.testing.assert_allclose(
        res["output"],
        moe.grouped_swiglu(x_packed, w_gate, w_up, w_down, group_sizes),
        rtol=1e-4,
        atol=1e-5,
    )


def test_dk3_rocm_moe_transport_perf_baseline_is_bounded():
    rng = np.random.default_rng(45)
    x = rng.standard_normal((24, 32)).astype(np.float32)
    plan = _plan(seed=45, tokens=24, experts=6, top_k=2, capacity=8)
    art = _artifact("tessera.moe_dispatch", ["x", "plan"])

    direct_ms = _median_ms(lambda: moe.dispatch(x, plan), reps=9)
    launch_ms = _median_ms(lambda: rt.launch(art, (x, plan)), reps=9)

    assert launch_ms < max(75.0, direct_ms * 4.0)


def test_rocm_grouped_gemm_uses_one_native_offsets_argument():
    from tessera.compiler.grouped_layout import reference_grouped_gemm

    rng = np.random.default_rng(47)
    sizes = np.array([2, 0, 3], dtype=np.int64)
    x = rng.standard_normal((5, 7)).astype(np.float32)
    weights = rng.standard_normal((3, 7, 6)).astype(np.float32)
    res = rt.launch(
        _artifact("tessera.grouped_gemm", ["x", "weights", "group_sizes"]),
        {"x": x, "weights": weights, "group_sizes": sizes},
    )
    assert res["ok"]
    assert res["execution_kind"] == ("native_gpu" if _expect_native()
                                     else "reference_cpu")
    np.testing.assert_allclose(
        res["output"], reference_grouped_gemm(x, weights, sizes),
        rtol=2e-5, atol=2e-5)


def test_jit_grouped_gemm_stamps_native_rocm_lane(monkeypatch):
    import importlib

    jit_module = importlib.import_module("tessera.compiler.jit")

    monkeypatch.setattr(jit_module, "_rocm_compiled_lane_available", lambda: True)

    @ts.jit(target="rocm")
    def grouped(x, weights, group_sizes):
        return ts.ops.grouped_gemm(x, weights, group_sizes)

    artifact = grouped.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "rocm_moe_transport_compiled"
    assert artifact.metadata["grouped_argument_layout"] == "device_offsets[E+1]"
    assert artifact.metadata["ops"][0]["op_name"] == "tessera.grouped_gemm"


def test_rocm_grouped_gemm_tile_ratchet_boundaries():
    from tessera import runtime as rt

    assert rt._rocm_grouped_gemm_tile_n(128, 256) == 1
    assert rt._rocm_grouped_gemm_tile_n(128, 512) == 2
    assert rt._rocm_grouped_gemm_tile_n(256, 512) == 4
    # Tiled kernels have an exact output-divisibility contract.
    assert rt._rocm_grouped_gemm_tile_n(1024, 513) == 1


def test_rocm_f32_gemm_tile_ratchet_boundaries():
    from tessera import runtime as rt

    assert rt._rocm_f32_gemm_tile(128, 256, 512) == (4, 4)
    assert rt._rocm_f32_gemm_tile(256, 256, 256) == (2, 2)
    assert rt._rocm_f32_gemm_tile(512, 512, 512) == (4, 4)
