"""
Phase 5 conftest — shared fixtures.
"""
import pytest
import numpy as np

from tessera.compiler.solver_config import (
    SolverConfig, ZeROConfig, ResilienceConfig, DeploymentManifest,
    RNGStreamPlan, RNGBackend, PrecondType, SolverVariant,
)
from tessera.compiler.checkpoint import (
    CollectiveCheckpointConfig, CheckpointPolicy, CheckpointIRAnnotator,
)
from tessera.compiler.autotune_v2 import BayesianAutotuner, GEMMWorkload


@pytest.fixture
def default_solver_config():
    return SolverConfig(sparse_threshold=0.05, max_iter=500, num_ranks=4)


@pytest.fixture
def zero2_config():
    return ZeROConfig(stage=2, dp_axis="dp", num_dp_ranks=4)


@pytest.fixture
def resilience_cfg():
    return ResilienceConfig(checkpoint_interval=100, restart_policy="last",
                            max_restarts=3)


@pytest.fixture
def small_manifest():
    m = DeploymentManifest()
    m.add_mesh_axis("dp", 4).add_mesh_axis("tp", 2)
    m.add_collective("reduce_scatter").add_collective("all_gather")
    m.add_optimizer_shard("fc1.weight", "dp")
    m.add_checkpoint("layer_0.attn")
    return m


@pytest.fixture
def gemm_workload_4k():
    return GEMMWorkload(M=4096, N=4096, K=4096)


@pytest.fixture
def simple_layer_names():
    return ["embed", "layer_0.attn", "layer_0.fc1", "layer_0.fc2",
            "layer_0.norm", "layer_1.attn", "layer_1.fc1", "layer_1.fc2",
            "layer_1.norm", "lm_head"]
