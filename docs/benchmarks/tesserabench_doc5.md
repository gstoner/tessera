# TesseraBench - Document 5: Distributed Benchmarking

This document covers TesseraBench's distributed benchmarking suite: multi-rank
correctness and throughput measurement for data-parallel (DP), tensor-parallel
(TP), and pipeline-parallel (PP) workloads, collective communication profiling,
and topology-aware sweep configuration for NVLink and PCIe clusters.

This document uses the current Tessera API from `docs/CANONICAL_API.md`:
`import tessera`, `@tessera.jit`, `@tessera.kernel`, `tessera.ops`,
`tessera.domain.Rect`, `tessera.dist.Block` / `Cyclic` / `Replicated`,
`tessera.array.from_domain`, and `tessera.index_launch`. Multi-rank execution
uses `MockRankGroup` (Phase 1 — CPU threads, no NCCL required) or the
NCCL/RCCL adapters (Phase 4 planned).

> **Phase status:** `MockRankGroup`-based benchmarks run today (Phase 1).
> NCCL/RCCL adapters, `DistributedPlan`, pipeline-parallel stages, and Cyclic
> MoE are Phase 4 planned.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│              TesseraBench Distributed Benchmarking                  │
├──────────────────────┬──────────────────────┬───────────────────────┤
│   Rank Management    │  Collective Profiler  │  Topology Planner     │
│   MockRankGroup      │  AllReduce / RS / AG  │  NVLink / PCIe / RDMA │
│   NCCLAdapter (P4)   │  ChunkPlanner (P4)    │  BusBandwidth metrics │
├──────────────────────┴──────────────────────┴───────────────────────┤
│           Tessera Distributed Programming Model                     │
├───────────────┬──────────────────────┬──────────────────────────────┤
│  Data Parallel│  Tensor Parallel      │  Pipeline Parallel           │
│  dist.Block   │  index_launch(tp)     │  1F1B schedule (P4)          │
│  all_reduce   │  reduce_scatter/AG    │  PipelineStageInsert (P4)    │
├───────────────┴──────────────────────┴──────────────────────────────┤
│  MockRankGroup (Phase 1+)  │  NCCL/RCCL Adapters (Phase 4 planned)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Rank Management and Test Infrastructure

### MockRankGroup (Phase 1 — Available Today)

`MockRankGroup` simulates multi-rank execution using Python threads sharing
in-process memory. No NCCL, MPI, or GPU is required. All distributed
benchmarks in this document can run with `MockRankGroup`.

```python
import tessera
from tessera.testing import MockRankGroup
from tessera.distributed import DistributedArray
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DistributedBenchmarkResult:
    """Result from one distributed benchmark run."""
    benchmark_name: str
    n_ranks: int
    mesh_axes: Dict[str, int]
    latency_ms: float
    throughput_gbs: float
    efficiency_pct: float   # actual / ideal linear scaling
    collective_overhead_ms: float
    topology: str           # "mock", "nvlink", "pcie", "rdma"
    notes: str = ""


class DistributedBenchmarkHarness:
    """
    Runs distributed benchmarks across a configurable rank group.

    Phase 1: uses MockRankGroup (in-process threads, no NCCL).
    Phase 4+: accepts NCCLAdapter or RCCLAdapter for real multi-GPU runs.
    """

    def __init__(self, n_ranks: int, mesh_axes: Dict[str, int],
                 topology: str = "mock"):
        """
        Args:
            n_ranks:    total number of ranks (product of mesh_axes values).
            mesh_axes:  e.g. {"dp": 4, "tp": 2} → 8 ranks total.
            topology:   "mock" | "nvlink" | "pcie" | "rdma"
        """
        assert n_ranks == 1 or n_ranks == _product(mesh_axes.values()), (
            "n_ranks must equal the product of mesh_axes values"
        )
        self.n_ranks = n_ranks
        self.mesh_axes = mesh_axes
        self.topology = topology

        if topology == "mock":
            self.rank_group = MockRankGroup(n=n_ranks, mesh_axes=mesh_axes)
        else:
            # Phase 4: real NCCL/RCCL adapter
            raise NotImplementedError(
                f"Topology '{topology}' requires Phase 4 NCCL/RCCL adapters."
            )

    def run(self, fn, *args, warmup: int = 3, repeat: int = 10, **kwargs
            ) -> List[float]:
        """Run fn(rank_group, *args) and return per-run latencies in ms."""
        latencies = []
        for i in range(warmup + repeat):
            t0 = time.perf_counter()
            self.rank_group.run(fn, *args, **kwargs)
            t1 = time.perf_counter()
            if i >= warmup:
                latencies.append((t1 - t0) * 1000)
        return latencies


def _product(values):
    result = 1
    for v in values:
        result *= v
    return result
```

---

## 2. Data-Parallel GEMM Benchmark

Measures how GEMM throughput scales with data-parallel rank count. Shards the
batch dimension using `tessera.dist.Block` and dispatches with `index_launch`.

```python
import tessera
from tessera.testing import MockRankGroup
from tessera.compiler.gpu_target import GPUTargetProfile, ISA
import numpy as np


@tessera.jit
def dp_gemm_step(
    A: tessera.Region["read"],
    B: tessera.Region["read"],
    C: tessera.Region["write"],
):
    """Single-step GEMM for data-parallel sweep."""
    C[:] = tessera.ops.gemm(A, B)


@tessera.kernel
def dp_gemm_shard(
    A_shard: tessera.f16[..., ...],
    B:       tessera.f16[..., ...],
    C_shard: tessera.mut_f32[..., ...],
):
    """Per-rank tile kernel dispatched by index_launch."""
    C_shard[:] = tessera.ops.gemm(A_shard, B)


class DataParallelGEMMBenchmark:
    """
    Benchmarks GEMM scaling under data parallelism.

    Sweeps:
      - rank counts:  [1, 2, 4, 8]
      - matrix sizes: [(1024,1024,1024), (4096,4096,4096)]
      - dtype:        bf16 (storage), fp32 (accumulator)
    """

    RANK_COUNTS   = [1, 2, 4, 8]
    MATRIX_SHAPES = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    def run_sweep(self) -> List[DistributedBenchmarkResult]:
        results = []
        # Baseline: single-rank GEMM for efficiency calculation
        baseline_ms = self._run_single_rank(self.MATRIX_SHAPES[0])

        for n_ranks in self.RANK_COUNTS:
            for M, N, K in self.MATRIX_SHAPES:
                result = self._run_dp_gemm(n_ranks, M, N, K, baseline_ms)
                results.append(result)
                print(
                    f"  dp={n_ranks:2d}  M={M} N={N} K={K}"
                    f"  {result.latency_ms:.2f} ms"
                    f"  eff={result.efficiency_pct:.1f}%"
                )
        return results

    def _run_single_rank(self, shape) -> float:
        M, N, K = shape
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(K, N).astype(np.float16)
        C = np.zeros((M, N), dtype=np.float32)

        # Warm up
        for _ in range(3):
            dp_gemm_step(A, B, C)

        t0 = time.perf_counter()
        for _ in range(10):
            dp_gemm_step(A, B, C)
        return (time.perf_counter() - t0) * 100   # ms per call

    def _run_dp_gemm(self, n_ranks: int, M: int, N: int, K: int,
                     baseline_ms: float) -> DistributedBenchmarkResult:

        harness = DistributedBenchmarkHarness(
            n_ranks=n_ranks,
            mesh_axes={"dp": n_ranks},
            topology="mock",
        )

        # Shard batch dimension (M) across dp ranks
        D    = tessera.domain.Rect((M, K))
        dist = tessera.dist.Block(mesh_axes=("dp",))
        A    = tessera.array.from_domain(D, dtype="f16", distribution=dist)
        B_full = np.random.randn(K, N).astype(np.float16)
        C_out  = tessera.array.from_domain(
            tessera.domain.Rect((M, N)), dtype="f32", distribution=dist
        )

        def _dispatch(rank_group):
            tessera.index_launch(axis="dp")(dp_gemm_shard)(
                A.parts("dp"),
                [B_full] * n_ranks,
                C_out.parts("dp"),
            )

        latencies = harness.run(_dispatch, warmup=3, repeat=10)
        avg_ms    = sum(latencies) / len(latencies)

        # Efficiency: ideal is baseline / n_ranks
        ideal_ms     = baseline_ms / n_ranks
        efficiency   = (ideal_ms / avg_ms) * 100.0 if avg_ms > 0 else 0.0

        flops        = 2 * M * N * K
        tflops       = flops / (avg_ms * 1e9)   # TFLOPS

        return DistributedBenchmarkResult(
            benchmark_name=f"dp_gemm_{M}x{N}x{K}",
            n_ranks=n_ranks,
            mesh_axes={"dp": n_ranks},
            latency_ms=avg_ms,
            throughput_gbs=tflops,
            efficiency_pct=efficiency,
            collective_overhead_ms=0.0,   # pure DP: no collective in forward pass
            topology="mock",
        )
```

---

## 3. Tensor-Parallel GEMM Benchmark

Splits the weight matrix column-wise across TP ranks. Each rank computes a
partial output shard; an all-reduce combines the partial sums.

```python
@tessera.kernel
def tp_gemm_shard(
    X:       tessera.f16[..., ...],
    W_shard: tessera.f16[..., ...],
    Y_shard: tessera.mut_f32[..., ...],
):
    """Per-rank column-parallel GEMM shard."""
    Y_shard[:] = tessera.ops.gemm(X, W_shard)


class TensorParallelGEMMBenchmark:
    """
    Benchmarks column-parallel GEMM under tensor parallelism.

    Weight matrix W is sharded along its column dimension across tp ranks.
    After the local GEMM each rank holds a partial Y_shard; an all_reduce
    across the tp axis completes the computation.

    Phase 1 (today):   MockRankGroup all_reduce via shared memory barrier + sum.
    Phase 4 (planned): NCCLAdapter all_reduce over real NVLink/PCIe.
    """

    TP_COUNTS     = [1, 2, 4, 8]
    GEMM_SHAPES   = [(4096, 4096, 4096), (8192, 8192, 8192)]

    def run_sweep(self) -> List[DistributedBenchmarkResult]:
        results = []
        for n_tp in self.TP_COUNTS:
            for M, N, K in self.GEMM_SHAPES:
                result = self._run_tp_gemm(n_tp, M, N, K)
                results.append(result)
                print(
                    f"  tp={n_tp:2d}  M={M} N={N} K={K}"
                    f"  {result.latency_ms:.2f} ms  "
                    f"  collective={result.collective_overhead_ms:.2f} ms"
                )
        return results

    def _run_tp_gemm(self, n_tp: int, M: int, N: int, K: int
                     ) -> DistributedBenchmarkResult:

        harness = DistributedBenchmarkHarness(
            n_ranks=n_tp,
            mesh_axes={"tp": n_tp},
            topology="mock",
        )

        X = np.random.randn(M, K).astype(np.float16)

        # Shard W along N (column-parallel)
        W_domain = tessera.domain.Rect((K, N))
        W_dist   = tessera.dist.Block(mesh_axes=("tp",))
        W        = tessera.array.from_domain(W_domain, dtype="f16", distribution=W_dist)

        # Y shards: each rank produces M × (N/tp)
        Y_domain = tessera.domain.Rect((M, N))
        Y        = tessera.array.from_domain(Y_domain, dtype="f32", distribution=W_dist)

        compute_latencies   = []
        collective_latencies = []

        def _dispatch(rank_group):
            # Local GEMM
            t_compute = time.perf_counter()
            tessera.index_launch(axis="tp")(tp_gemm_shard)(
                [X] * n_tp,
                W.parts("tp"),
                Y.parts("tp"),
            )
            compute_latencies.append((time.perf_counter() - t_compute) * 1000)

            # Collective: all_reduce across tp axis (Phase 1: mock barrier+sum)
            t_coll = time.perf_counter()
            rank_group.all_reduce(Y, op="sum", axis="tp")
            collective_latencies.append((time.perf_counter() - t_coll) * 1000)

        harness.run(_dispatch, warmup=3, repeat=10)

        avg_compute    = sum(compute_latencies[-10:]) / 10
        avg_collective = sum(collective_latencies[-10:]) / 10
        avg_total      = avg_compute + avg_collective

        return DistributedBenchmarkResult(
            benchmark_name=f"tp_gemm_{M}x{N}x{K}",
            n_ranks=n_tp,
            mesh_axes={"tp": n_tp},
            latency_ms=avg_total,
            throughput_gbs=(2 * M * N * K) / (avg_total * 1e9),
            efficiency_pct=(avg_compute / avg_total) * 100.0,
            collective_overhead_ms=avg_collective,
            topology="mock",
        )
```

---

## 4. Collective Communication Benchmark

Measures raw collective throughput across rank counts and message sizes,
matching the `benchmarks/benchmark_collective.py` runner.

```python
import tessera
from tessera.testing import MockRankGroup
import numpy as np
import time
from dataclasses import dataclass
from typing import List


@dataclass
class CollectiveResult:
    op: str                 # "all_reduce", "reduce_scatter", "all_gather"
    n_ranks: int
    message_bytes: int
    topology: str           # "mock", "nvlink", "pcie"
    latency_ms: float
    bus_bw_gbs: float       # (n-1)/n × 2 × bytes / time for AR; n-1/n × bytes/time for RS/AG


class CollectiveBenchmark:
    """
    Benchmarks all_reduce, reduce_scatter, and all_gather across rank counts
    and message sizes.

    Bus bandwidth formula (consistent with NCCL convention):
      AllReduce:      bw = 2 × (n-1)/n × bytes / latency
      ReduceScatter:    bw =   (n-1)/n × bytes / latency
      AllGather:        bw =   (n-1)/n × bytes / latency

    Topology-specific expected peaks (Phase 4, real hardware):
      NVLink (NVL72):  ~900 GB/s bus bandwidth per all_reduce
      NVLink (8-GPU):  ~300 GB/s
      PCIe Gen5:       ~60 GB/s

    Phase 1 (today): MockRankGroup uses shared memory; bandwidth reflects
    CPU memory, not GPU interconnect. Use for correctness verification only.
    """

    RANK_COUNTS    = [2, 4, 8, 16, 32, 64, 128]
    MESSAGE_SIZES  = [
        1   * 1024 * 1024,    #   1 MB
        16  * 1024 * 1024,    #  16 MB
        64  * 1024 * 1024,    #  64 MB
        256 * 1024 * 1024,    # 256 MB
        512 * 1024 * 1024,    # 512 MB
    ]
    OPS            = ["all_reduce", "reduce_scatter", "all_gather"]

    def run_sweep(self, rank_counts=None, message_sizes=None, ops=None,
                  topology: str = "mock", warmup: int = 5, repeat: int = 20,
                  ) -> List[CollectiveResult]:

        rank_counts   = rank_counts   or self.RANK_COUNTS
        message_sizes = message_sizes or self.MESSAGE_SIZES
        ops           = ops           or self.OPS
        results       = []

        for n_ranks in rank_counts:
            if n_ranks > 8 and topology == "mock":
                # MockRankGroup is thread-based; high rank counts may be slow
                print(f"  Warning: {n_ranks} ranks with mock topology is CPU-only.")

            harness = DistributedBenchmarkHarness(
                n_ranks=n_ranks,
                mesh_axes={"dp": n_ranks},
                topology=topology,
            )

            for nbytes in message_sizes:
                for op in ops:
                    result = self._bench_collective(
                        harness, op, n_ranks, nbytes, topology, warmup, repeat
                    )
                    results.append(result)
                    print(
                        f"  {op:20s}  ranks={n_ranks:4d}"
                        f"  size={nbytes//1024//1024:4d} MB"
                        f"  {result.latency_ms:8.3f} ms"
                        f"  {result.bus_bw_gbs:6.1f} GB/s"
                    )

        return results

    def _bench_collective(
        self, harness: DistributedBenchmarkHarness,
        op: str, n_ranks: int, nbytes: int,
        topology: str, warmup: int, repeat: int,
    ) -> CollectiveResult:

        n_elems = nbytes // 4  # fp32 elements
        buf     = np.ones(n_elems, dtype=np.float32)

        # Allocate Tessera array partitioned across dp axis
        domain   = tessera.domain.Rect((n_elems,))
        dist     = tessera.dist.Block(mesh_axes=("dp",))
        arr      = tessera.array.from_domain(domain, dtype="f32", distribution=dist,
                                              fill="ones")

        def _run(rank_group):
            if op == "all_reduce":
                rank_group.all_reduce(arr, op="sum", axis="dp")
            elif op == "reduce_scatter":
                rank_group.reduce_scatter(arr, op="sum", axis="dp")
            elif op == "all_gather":
                rank_group.all_gather(arr, axis="dp")

        latencies = harness.run(_run, warmup=warmup, repeat=repeat)
        avg_ms    = sum(latencies) / len(latencies)

        # Bus bandwidth (NCCL convention)
        t_s = avg_ms / 1000.0
        if op == "all_reduce":
            bus_bw = (2 * (n_ranks - 1) / n_ranks * nbytes) / t_s / 1e9
        else:
            bus_bw = ((n_ranks - 1) / n_ranks * nbytes) / t_s / 1e9

        return CollectiveResult(
            op=op,
            n_ranks=n_ranks,
            message_bytes=nbytes,
            topology=topology,
            latency_ms=avg_ms,
            bus_bw_gbs=bus_bw,
        )
```

---

## 5. Topology-Aware Chunk Planning

The `ChunkPlanner` (Phase 4) selects the optimal chunk size and in-flight count
based on link topology. The benchmark below validates and measures the effect
of chunk size on effective bandwidth.

```python
# Phase 4 planned — ChunkPlanner from src/collectives/lib/ChunkPlanner.cpp
# The constants below reflect the spec from docs/CLAUDE.md:
#   NVLink:  512 KiB per chunk
#   PCIe:    128 KiB per chunk
#   RDMA:    256 KiB per chunk

CHUNK_BYTES = {
    "nvlink": 512 * 1024,
    "pcie":   128 * 1024,
    "rdma":   256 * 1024,
    "mock":     64 * 1024,   # conservative for thread-based mock
}

MAX_INFLIGHT = {
    "nvlink": 8,
    "pcie":   4,
    "rdma":   6,
    "mock":   2,
}


class TopologyAwareBenchmark:
    """
    Validates ChunkPlanner chunk selection and measures the bandwidth
    improvement from topology-aware chunking vs a flat transfer.

    Phase 1 (today): runs with mock topology to verify the interface.
    Phase 4 (planned): runs against real NVLink / PCIe topologies via
                       NCCLAdapter + ChunkPlanner C++ implementation.
    """

    def run_chunk_sensitivity(self, topology: str = "mock",
                               n_ranks: int = 4,
                               message_bytes: int = 256 * 1024 * 1024,
                               ) -> Dict[str, Any]:

        chunk_sizes = [
            32  * 1024,
            64  * 1024,
            128 * 1024,
            256 * 1024,
            512 * 1024,
            1   * 1024 * 1024,
        ]
        expected_optimal = CHUNK_BYTES[topology]

        harness = DistributedBenchmarkHarness(
            n_ranks=n_ranks,
            mesh_axes={"dp": n_ranks},
            topology=topology,
        )

        domain = tessera.domain.Rect((message_bytes // 4,))
        dist   = tessera.dist.Block(mesh_axes=("dp",))
        arr    = tessera.array.from_domain(domain, dtype="f32", distribution=dist,
                                            fill="ones")

        bandwidth_by_chunk = {}
        for chunk_bytes in chunk_sizes:
            def _run(rank_group, chunk=chunk_bytes):
                # Phase 4: rank_group.all_reduce(arr, op="sum", axis="dp",
                #                                chunk_bytes=chunk)
                # Phase 1 mock ignores chunk size:
                rank_group.all_reduce(arr, op="sum", axis="dp")

            latencies  = harness.run(_run, warmup=3, repeat=10)
            avg_ms     = sum(latencies) / len(latencies)
            t_s        = avg_ms / 1000.0
            bus_bw     = (2 * (n_ranks - 1) / n_ranks * message_bytes) / t_s / 1e9
            bandwidth_by_chunk[chunk_bytes] = bus_bw

        best_chunk = max(bandwidth_by_chunk, key=bandwidth_by_chunk.get)
        return {
            "topology": topology,
            "expected_optimal_chunk_bytes": expected_optimal,
            "measured_best_chunk_bytes": best_chunk,
            "bandwidth_by_chunk_gbs": bandwidth_by_chunk,
            "match": best_chunk == expected_optimal,
        }
```

---

## 6. Cyclic Distribution (MoE) Benchmark

Measures token routing throughput for Mixture-of-Experts workloads using
`tessera.dist.Cyclic`. Cyclic distributes tokens in round-robin fashion so
each expert receives a uniform load. An `all_to_all` collective rebalances
tokens after routing.

```python
import tessera
from tessera.testing import MockRankGroup
import numpy as np


@tessera.kernel
def expert_ffn_shard(
    tokens:   tessera.bf16[..., ...],
    W_gate:   tessera.bf16[..., ...],
    W_up:     tessera.bf16[..., ...],
    W_down:   tessera.bf16[..., ...],
    out:      tessera.mut_f32[..., ...],
):
    """Single expert FFN: gate × SiLU(up) → down projection."""
    gate = tessera.ops.gemm(tokens, W_gate)
    up   = tessera.ops.gemm(tokens, W_up)
    activated = tessera.ops.gelu(up) * gate          # SwiGLU-style
    out[:] = tessera.ops.gemm(activated, W_down)


class MoEBenchmark:
    """
    Benchmarks MoE token routing and expert computation with Cyclic distribution.

    Phase 1 (today):  MockRankGroup + Cyclic sharding, sequential dispatch.
    Phase 4 (planned): real all_to_all collective + NCCLAdapter.

    Sweep parameters:
      - n_experts:   [8, 16, 32, 64]
      - seq_len:     [1024, 4096, 8192]
      - hidden_dim:  [2048, 4096]
      - top_k:       2 (each token routed to 2 experts)
    """

    N_EXPERTS_LIST = [8, 16, 32, 64]
    SEQ_LENGTHS    = [1024, 4096, 8192]
    HIDDEN_DIMS    = [2048, 4096]
    TOP_K          = 2

    def run_sweep(self) -> List[DistributedBenchmarkResult]:
        results = []
        for n_experts in self.N_EXPERTS_LIST:
            for seq_len in self.SEQ_LENGTHS:
                for hidden_dim in self.HIDDEN_DIMS:
                    result = self._run_moe(n_experts, seq_len, hidden_dim)
                    results.append(result)
                    print(
                        f"  experts={n_experts:2d}  seq={seq_len:5d}"
                        f"  hidden={hidden_dim:5d}"
                        f"  {result.latency_ms:.2f} ms"
                    )
        return results

    def _run_moe(self, n_experts: int, seq_len: int, hidden_dim: int,
                 ) -> DistributedBenchmarkResult:

        harness = DistributedBenchmarkHarness(
            n_ranks=n_experts,
            mesh_axes={"ep": n_experts},   # expert-parallel axis
            topology="mock",
        )

        ffn_dim = hidden_dim * 4

        # Token tensor — Cyclic distributes in round-robin across experts
        token_domain = tessera.domain.Rect((seq_len * self.TOP_K, hidden_dim))
        token_dist   = tessera.dist.Cyclic(mesh_axes=("ep",))
        tokens       = tessera.array.from_domain(
            token_domain, dtype="bf16", distribution=token_dist
        )

        # Expert weights — replicated per rank
        W_gate_list = [
            np.random.randn(hidden_dim, ffn_dim).astype(np.float16)
            for _ in range(n_experts)
        ]
        W_up_list   = [
            np.random.randn(hidden_dim, ffn_dim).astype(np.float16)
            for _ in range(n_experts)
        ]
        W_down_list = [
            np.random.randn(ffn_dim, hidden_dim).astype(np.float16)
            for _ in range(n_experts)
        ]
        out_domain  = tessera.domain.Rect((seq_len * self.TOP_K, hidden_dim))
        out_dist    = tessera.dist.Cyclic(mesh_axes=("ep",))
        out         = tessera.array.from_domain(out_domain, dtype="f32",
                                                 distribution=out_dist)

        def _dispatch(rank_group):
            # Phase 4: all_to_all rebalances Cyclic tokens before dispatch.
            # Phase 1 mock: parts() returns the round-robin slices directly.
            tessera.index_launch(axis="ep")(expert_ffn_shard)(
                tokens.parts("ep"),
                W_gate_list,
                W_up_list,
                W_down_list,
                out.parts("ep"),
            )

        latencies = harness.run(_dispatch, warmup=3, repeat=10)
        avg_ms    = sum(latencies) / len(latencies)

        # Compute: each token processed by top_k experts
        token_flops = 2 * hidden_dim * ffn_dim * 3   # gate + up + down
        total_flops = token_flops * seq_len * self.TOP_K
        tflops      = total_flops / (avg_ms * 1e9)

        return DistributedBenchmarkResult(
            benchmark_name=f"moe_e{n_experts}_s{seq_len}_h{hidden_dim}",
            n_ranks=n_experts,
            mesh_axes={"ep": n_experts},
            latency_ms=avg_ms,
            throughput_gbs=tflops,
            efficiency_pct=0.0,   # set by caller if baseline known
            collective_overhead_ms=0.0,
            topology="mock",
            notes=f"top_k={self.TOP_K}, ffn_dim={ffn_dim}",
        )
```

---

## 7. Pipeline-Parallel Benchmark (Phase 4 Planned)

Pipeline parallelism uses the 1F1B (one-forward-one-backward) schedule. Each
rank hosts one pipeline stage; micro-batches flow through stages with
`send`/`recv` activation handoffs.

```python
# Phase 4 planned — requires PipelineStageInsertionPass and NCCLAdapter.
# This class documents the intended interface; run() raises NotImplementedError
# until Phase 4 is complete.

@tessera.jit
def transformer_layer(
    x:    tessera.Region["read"],
    W_q:  tessera.Region["read"],
    W_k:  tessera.Region["read"],
    W_v:  tessera.Region["read"],
    W_o:  tessera.Region["read"],
    W_ff: tessera.Region["read"],
    out:  tessera.Region["write"],
):
    """Single transformer layer: self-attention + FFN."""
    q   = tessera.ops.gemm(x, W_q)
    k   = tessera.ops.gemm(x, W_k)
    v   = tessera.ops.gemm(x, W_v)
    attn = tessera.ops.flash_attn(q, k, v, causal=True)
    proj = tessera.ops.gemm(attn, W_o)
    out[:] = tessera.ops.gelu(tessera.ops.gemm(proj, W_ff))


class PipelineParallelBenchmark:
    """
    Benchmarks pipeline-parallel execution using the 1F1B schedule.

    Phase 4 requirements:
      - PipelineStageInsertionPass: partitions schedule.pipeline.region
        across ranks, inserts send/recv micro-batch ops.
      - NCCLAdapter: activation P2P transfers between adjacent pipeline stages.

    Until Phase 4 is available, this benchmark runs a mock sequential
    simulation to document expected latency formulas.

    1F1B latency formula (ideal):
        T_1F1B = (n_stages - 1) × T_bubble + n_microbatches × T_stage
        T_bubble = (n_stages - 1) × T_stage
    """

    STAGE_COUNTS    = [2, 4, 8, 16]
    MICROBATCH_COUNTS = [4, 8, 16, 32]

    def estimate_1f1b_latency(
        self,
        n_stages: int,
        n_microbatches: int,
        t_stage_ms: float,
    ) -> Dict[str, float]:
        """
        Analytical 1F1B schedule estimate.
        Returns dict with breakdown of compute vs bubble time.
        """
        t_bubble_total = (n_stages - 1) * t_stage_ms
        t_compute      = n_microbatches * t_stage_ms
        t_total        = t_bubble_total + t_compute
        efficiency     = t_compute / t_total * 100.0

        return {
            "n_stages":          n_stages,
            "n_microbatches":    n_microbatches,
            "t_stage_ms":        t_stage_ms,
            "t_bubble_total_ms": t_bubble_total,
            "t_compute_ms":      t_compute,
            "t_total_ms":        t_total,
            "pipeline_efficiency_pct": efficiency,
        }

    def run(self, *args, **kwargs):
        raise NotImplementedError(
            "PipelineParallelBenchmark requires Phase 4 "
            "PipelineStageInsertionPass and NCCLAdapter. "
            "Use estimate_1f1b_latency() for analytical estimates."
        )

    def print_efficiency_table(self):
        """Print 1F1B efficiency for a range of stages and microbatches."""
        t_stage_ms = 5.0   # representative 5ms per stage per microbatch
        print(f"{'Stages':>8} {'μBatches':>10} {'Efficiency':>12} {'Bubble':>12}")
        print("-" * 46)
        for n_stages in self.STAGE_COUNTS:
            for n_mb in self.MICROBATCH_COUNTS:
                est = self.estimate_1f1b_latency(n_stages, n_mb, t_stage_ms)
                print(
                    f"{n_stages:>8}  {n_mb:>10}"
                    f"  {est['pipeline_efficiency_pct']:>10.1f}%"
                    f"  {est['t_bubble_total_ms']:>10.1f} ms"
                )
```

---

## 8. Distributed Benchmark Runner

Orchestrates all distributed benchmarks and produces a unified JSON report.

```python
import json
import datetime
import tessera


class DistributedBenchmarkRunner:
    """
    Top-level runner for the TesseraBench distributed benchmark suite.

    Usage:
        runner = DistributedBenchmarkRunner(topology="mock")
        report = runner.run_all()
        runner.save(report, "distributed_benchmarks.json")
    """

    def __init__(self, topology: str = "mock",
                 rank_counts: List[int] = None,
                 skip: List[str] = None):
        self.topology    = topology
        self.rank_counts = rank_counts or [1, 2, 4, 8]
        self.skip        = set(skip or [])

    def run_all(self) -> Dict[str, Any]:
        report = {
            "tessera_version": getattr(tessera, "__version__", "dev"),
            "topology":        self.topology,
            "timestamp":       datetime.datetime.utcnow().isoformat(),
            "benchmarks":      {},
        }

        if "dp_gemm" not in self.skip:
            print("\n=== Data-Parallel GEMM ===")
            bench   = DataParallelGEMMBenchmark()
            results = bench.run_sweep()
            report["benchmarks"]["dp_gemm"] = [self._to_dict(r) for r in results]

        if "tp_gemm" not in self.skip:
            print("\n=== Tensor-Parallel GEMM ===")
            bench   = TensorParallelGEMMBenchmark()
            results = bench.run_sweep()
            report["benchmarks"]["tp_gemm"] = [self._to_dict(r) for r in results]

        if "collectives" not in self.skip:
            print("\n=== Collective Communication ===")
            bench   = CollectiveBenchmark()
            results = bench.run_sweep(
                rank_counts=self.rank_counts,
                topology=self.topology,
            )
            report["benchmarks"]["collectives"] = [self._to_dict(r) for r in results]

        if "moe" not in self.skip:
            print("\n=== MoE / Cyclic Distribution ===")
            bench   = MoEBenchmark()
            results = bench.run_sweep()
            report["benchmarks"]["moe"] = [self._to_dict(r) for r in results]

        if "pipeline" not in self.skip:
            print("\n=== Pipeline Parallel (analytical) ===")
            bench = PipelineParallelBenchmark()
            bench.print_efficiency_table()
            report["benchmarks"]["pipeline_efficiency"] = [
                bench.estimate_1f1b_latency(s, m, 5.0)
                for s in PipelineParallelBenchmark.STAGE_COUNTS
                for m in PipelineParallelBenchmark.MICROBATCH_COUNTS
            ]

        return report

    def save(self, report: Dict[str, Any], path: str):
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved → {path}")

    @staticmethod
    def _to_dict(result) -> Dict[str, Any]:
        return {k: v for k, v in result.__dict__.items()}


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TesseraBench Distributed Benchmarks")
    parser.add_argument("--topology", default="mock",
                        choices=["mock", "nvlink", "pcie", "rdma"])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Benchmarks to skip: dp_gemm tp_gemm collectives moe pipeline")
    parser.add_argument("--output", default="distributed_benchmarks.json")
    args = parser.parse_args()

    runner = DistributedBenchmarkRunner(
        topology=args.topology,
        rank_counts=args.ranks,
        skip=args.skip,
    )
    report = runner.run_all()
    runner.save(report, args.output)
```

---

## 9. Expected Results Reference

### Collective Bandwidth (real hardware, Phase 4)

| Topology | Collective | Message | Expected Bus BW |
|----------|-----------|---------|-----------------|
| NVLink (8-GPU, NVSwitch) | AllReduce | 256 MB | ~280 GB/s |
| NVLink (NVL72 rack) | AllReduce | 256 MB | ~850 GB/s |
| PCIe Gen5 (8-GPU) | AllReduce | 256 MB | ~50 GB/s |
| PCIe Gen5 (8-GPU) | ReduceScatter | 256 MB | ~48 GB/s |
| NVLink (8-GPU) | AllGather | 256 MB | ~270 GB/s |

### Data-Parallel Scaling Efficiency

Ideal linear scaling assumes zero collective overhead. Real DP efficiency
drops due to all_reduce of gradients after the backward pass:

| Ranks | Expected DP Efficiency (NVLink) | Expected DP Efficiency (PCIe) |
|-------|-------------------------------|-------------------------------|
| 2 | ~98% | ~92% |
| 4 | ~96% | ~85% |
| 8 | ~93% | ~74% |
| 16 | ~89% | ~61% |

### 1F1B Pipeline Efficiency

| Stages | Micro-batches | Efficiency |
|--------|--------------|------------|
| 4 | 4 | 57% |
| 4 | 8 | 73% |
| 4 | 16 | 84% |
| 8 | 8 | 53% |
| 8 | 16 | 69% |
| 8 | 32 | 82% |

---

## 10. Cross-References

- [`docs/CANONICAL_API.md`](../CANONICAL_API.md) — authoritative API names
- [`docs/spec/CONFORMANCE.md`](../spec/CONFORMANCE.md) — T2 profile: collective, Cyclic, pipeline requirements
- [`docs/spec/LANGUAGE_AND_IR_SPEC.md §11`](../spec/LANGUAGE_AND_IR_SPEC.md) — distributed semantics and mesh rules
- [`docs/guides/Tessera_QA_Reliability_Guide.md`](../guides/Tessera_QA_Reliability_Guide.md) — collective correctness and regression testing
- [`benchmarks/benchmark_collective.py`](../../benchmarks/benchmark_collective.py) — standalone collective benchmark runner (Phase 6)
- [`docs/benchmarks/tesserabench_doc1.md`](tesserabench_doc1.md) — TesseraBench core architecture
- [`docs/benchmarks/tesserabench_doc6.md`](tesserabench_doc6.md) — advanced Tessera integration features
