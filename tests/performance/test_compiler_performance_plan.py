from __future__ import annotations

import statistics
import time

import numpy as np

import tessera as ts
from benchmarks.benchmark_attention import FlashAttnBenchmark
from benchmarks.benchmark_collective import CollectiveBenchmark, CollectiveOp
from benchmarks.benchmark_gemm import GEMMBenchmark
from benchmarks.run_all import run_all_benchmarks


def _compile_transformer_proxy():
    @ts.jit(cpu_tile=(128, 128, 64))
    def transformer_proxy(x, wq, wk, wv, wo):
        q = ts.ops.matmul(x, wq)
        k = ts.ops.matmul(x, wk)
        v = ts.ops.matmul(x, wv)
        scores = ts.ops.matmul(q, ts.ops.transpose(k))
        probs = ts.ops.softmax(scores)
        ctx = ts.ops.matmul(probs, v)
        return ts.ops.matmul(ctx, wo)

    return transformer_proxy


def test_jit_transformer_proxy_compile_latency_and_artifact_size_budget():
    timings = []
    compiled = None
    for _ in range(5):
        start = time.perf_counter()
        compiled = _compile_transformer_proxy()
        timings.append(time.perf_counter() - start)

    median_s = statistics.median(timings)
    assert median_s < 0.25
    assert compiled is not None
    assert compiled.uses_compiled_path

    artifact_bytes = sum(len(artifact.text.encode("utf-8")) for artifact in compiled.lowering_artifacts())
    assert artifact_bytes < 20 * 1024


def test_transformer_proxy_execution_latency_is_small_for_tiny_cpu_case():
    compiled = _compile_transformer_proxy()
    x = np.ones((8, 16), dtype=np.float32)
    w = np.eye(16, dtype=np.float32)

    timings = []
    for _ in range(10):
        start = time.perf_counter()
        out = compiled(x, w, w, w, w)
        timings.append(time.perf_counter() - start)

    np.testing.assert_allclose(out, compiled(x, w, w, w, w))
    assert statistics.median(timings) < 0.02


def test_gemm_roofline_proxy_covers_memory_and_compute_bound_regions():
    bench = GEMMBenchmark(dtype="bf16", peak_tflops=989.0, peak_membw_gbps=3350.0)

    tiny = bench.run_single(128, 128, 128)
    large = bench.run_single(4096, 4096, 4096)

    assert tiny.roofline_bound == "memory"
    assert large.roofline_bound == "compute"
    assert 0 < tiny.tflops <= bench.peak_tflops
    assert 0 < large.tflops <= bench.peak_tflops


def test_attention_roofline_proxy_reports_bounded_mfu_and_positive_tokens():
    bench = FlashAttnBenchmark(causal=True, peak_tflops=989.0, peak_membw_gbps=3350.0)
    result = bench.run_single(batch=2, heads=16, seq_len=1024, head_dim=128)

    assert result.tokens_per_sec > 0
    assert result.tflops > 0
    assert 0 < result.mfu <= 1.0


def test_collective_proxy_reports_bounded_utilization_across_ops():
    bench = CollectiveBenchmark(peak_bw_gbps=600.0, latency_us=5.0)
    results = [
        bench.run_single(op, num_ranks=8, message_bytes=64 * 1024 * 1024)
        for op in (
            CollectiveOp.ALL_REDUCE,
            CollectiveOp.REDUCE_SCATTER,
            CollectiveOp.ALL_GATHER,
        )
    ]

    for result in results:
        assert result.latency_ms > 0
        assert result.bus_bw_gbps > 0
        assert 0 < bench.bus_utilization(result) <= 1.0

    assert results[0].config.bus_bytes() == 2 * results[1].config.bus_bytes()


def test_combined_benchmark_suite_schema_has_all_compiler_perf_families():
    suite = run_all_benchmarks(
        peak_tflops=989.0,
        peak_membw_gbps=3350.0,
        peak_bw_gbps=600.0,
        gemm_sizes=[(1024, 1024, 1024)],
        attn_configs=[(1, 8, 512, 64)],
        collective_ops=[CollectiveOp.ALL_REDUCE],
        collective_ranks=[8],
        collective_sizes=[16 * 1024 * 1024],
        verbose=False,
    )
    data = suite.to_dict()

    assert data["summary"]["gemm_count"] == 1
    assert data["summary"]["attn_count"] == 1
    assert data["summary"]["collective_count"] == 1
    assert data["gemm"][0]["roofline_bound"] in {"compute", "memory"}
    assert data["attention"][0]["mfu"] > 0
    assert data["collective"][0]["bus_bw_gbps"] > 0
