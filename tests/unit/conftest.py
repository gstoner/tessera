"""Shared fixtures for flattened Python unit tests."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import tessera
from benchmarks.benchmark_attention import FlashAttnBenchmark
from benchmarks.benchmark_collective import CollectiveBenchmark
from benchmarks.benchmark_gemm import GEMMBenchmark
from tessera.compiler.attn_lower import FlashAttnLoweringConfig
from tessera.compiler.autotune_v2 import BayesianAutotuner, GEMMWorkload
from tessera.compiler.checkpoint import (
    CheckpointIRAnnotator,
    CheckpointPolicy,
    CollectiveCheckpointConfig,
)
from tessera.compiler.graph_ir import GraphIRBuilder
from tessera.compiler.gpu_target import GPUTargetProfile, ISA
from tessera.compiler.solver_config import (
    DeploymentManifest,
    PrecondType,
    ResilienceConfig,
    RNGBackend,
    RNGStreamPlan,
    SolverConfig,
    SolverVariant,
    ZeROConfig,
)
from tessera.diagnostics import DiagnosticLevel, ErrorReporter, ShapeInferenceEngine
from tessera.distributed.array import DistributedArray
from tessera.distributed.domain import Block, Rect, Replicated
from tessera.distributed.shard import MeshSpec
from tessera.runtime import DeviceKind, TesseraRuntime, TsrStatus
from tessera.testing import MockRankGroup
from tessera.testing.mock_collective import MockRankGroup as CollectiveMockRankGroup


@pytest.fixture
def mesh4() -> MeshSpec:
    return MeshSpec({"dp": 4})


@pytest.fixture
def mesh2x2() -> MeshSpec:
    return MeshSpec({"dp": 2, "tp": 2})


@pytest.fixture
def mesh_tp8() -> MeshSpec:
    return MeshSpec({"tp": 8})


@pytest.fixture
def group4() -> MockRankGroup:
    return MockRankGroup(n=4, mesh_axes={"dp": 4})


@pytest.fixture
def group2x2() -> MockRankGroup:
    return MockRankGroup(n=4, mesh_axes={"dp": 2, "tp": 2})


@pytest.fixture
def group1() -> MockRankGroup:
    return MockRankGroup(n=1, mesh_axes={"dp": 1})


@pytest.fixture
def rect_3d() -> Rect:
    return Rect((4, 128, 256))


@pytest.fixture
def rect_2d() -> Rect:
    return Rect((8, 256))


@pytest.fixture
def block_dp_tp() -> Block:
    return Block(mesh_axes=("dp", "tp"))


@pytest.fixture
def block_dp() -> Block:
    return Block(mesh_axes=("dp",))


@pytest.fixture
def replicated() -> Replicated:
    return Replicated()


@pytest.fixture
def bf16_array_3d(rect_3d, block_dp) -> DistributedArray:
    return DistributedArray.from_domain(rect_3d, dtype="bf16", distribution=block_dp)


@pytest.fixture
def fp32_array_2d(rect_2d, block_dp) -> DistributedArray:
    return DistributedArray.from_domain(rect_2d, dtype="fp32", distribution=block_dp)


@pytest.fixture
def builder():
    return GraphIRBuilder()


@pytest.fixture
def simple_gemm_ir(builder):
    from tessera.distributed.region import Region

    def step(W: Region["read"], X: Region["read"], Y: Region["write"]):
        Y[:] = tessera.ops.gemm(X, W)

    builder.lower(step)
    return builder.module()


@pytest.fixture
def sm90_profile():
    return GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4)


@pytest.fixture
def sm80_profile():
    return GPUTargetProfile(isa=ISA.SM_80, warps_per_cta=4)


@pytest.fixture
def causal_attn_config():
    return FlashAttnLoweringConfig(tile_q=64, tile_kv=64, causal=True)


@pytest.fixture
def flash_attn_ir(sm90_profile):
    @tessera.jit(target=sm90_profile)
    def flash_attn_fwd(
        Q: tessera.Tensor["B", "H", "S", "D"],
        K: tessera.Tensor["B", "H", "S", "D"],
        V: tessera.Tensor["B", "H", "S", "D"],
    ):
        tessera.require(tessera.constraint.Divisible("D", 64))
        return tessera.ops.flash_attn(Q, K, V, causal=True)

    return flash_attn_fwd.graph_ir


@pytest.fixture
def ranks_4():
    return CollectiveMockRankGroup(n=4, mesh_axes={"dp": 4})


@pytest.fixture
def ranks_8():
    return CollectiveMockRankGroup(n=8, mesh_axes={"dp": 4, "tp": 2})


@pytest.fixture
def ranks_8_pp():
    return CollectiveMockRankGroup(n=8, mesh_axes={"dp": 2, "tp": 2, "pp": 2})


@pytest.fixture
def small_grad():
    return np.ones((64, 32), dtype=np.float32)


@pytest.fixture
def default_solver_config():
    return SolverConfig(sparse_threshold=0.05, max_iter=500, num_ranks=4)


@pytest.fixture
def zero2_config():
    return ZeROConfig(stage=2, dp_axis="dp", num_dp_ranks=4)


@pytest.fixture
def resilience_cfg():
    return ResilienceConfig(checkpoint_interval=100, restart_policy="last", max_restarts=3)


@pytest.fixture
def small_manifest():
    manifest = DeploymentManifest()
    manifest.add_mesh_axis("dp", 4).add_mesh_axis("tp", 2)
    manifest.add_collective("reduce_scatter").add_collective("all_gather")
    manifest.add_optimizer_shard("fc1.weight", "dp")
    manifest.add_checkpoint("layer_0.attn")
    return manifest


@pytest.fixture
def gemm_workload_4k():
    return GEMMWorkload(M=4096, N=4096, K=4096)


@pytest.fixture
def simple_layer_names():
    return [
        "embed",
        "layer_0.attn",
        "layer_0.fc1",
        "layer_0.fc2",
        "layer_0.norm",
        "layer_1.attn",
        "layer_1.fc1",
        "layer_1.fc2",
        "layer_1.norm",
        "lm_head",
    ]


@pytest.fixture
def mock_runtime():
    runtime = TesseraRuntime(mock=True)
    runtime.init()
    yield runtime
    runtime.shutdown()


@pytest.fixture
def fresh_runtime():
    return TesseraRuntime(mock=True)


@pytest.fixture
def reporter():
    return ErrorReporter()


@pytest.fixture
def engine(reporter):
    return ShapeInferenceEngine(reporter)


@pytest.fixture
def gemm_bench():
    return GEMMBenchmark(dtype="bf16", peak_tflops=312.0, peak_membw_gbps=2000.0)


@pytest.fixture
def attn_bench():
    return FlashAttnBenchmark(causal=True, peak_tflops=312.0, peak_membw_gbps=2000.0)


@pytest.fixture
def coll_bench():
    return CollectiveBenchmark(peak_bw_gbps=600.0, latency_us=5.0)
