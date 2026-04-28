"""
solver_config.py — Python-layer solver configuration objects (Phase 5)

Mirrors the configuration that the C++ solver passes consume via IR attributes.
Provides:

- ``SolverConfig``          — sparse tolerance, RNG policy, Newton convergence
- ``SparseAnalysisResult``  — per-op sparsity annotation
- ``ZeROConfig``            — ZeRO-2/3 optimizer state partitioning
- ``ResilienceConfig``      — checkpoint/restart policy
- ``DeploymentManifest``    — full deployment descriptor (mesh + collectives + ckpt)
- ``RNGStreamPlan``         — per-rank stream ID assignment
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PrecondType(Enum):
    """Preconditioner selection for sparse linear solvers."""
    NONE    = "none"
    JACOBI  = "jacobi"    # diagonal scaling — fast, works for diag-dominant
    ILU     = "ilu"       # incomplete LU — general sparse matrices
    AMG     = "amg"       # algebraic multigrid — very sparse / SPD matrices


class SolverVariant(Enum):
    """Iterative solver algorithm."""
    CG       = "cg"        # Conjugate Gradient (requires SPD)
    GMRES    = "gmres"     # GMRES — general non-symmetric
    BICGSTAB = "bicgstab"  # BiCGSTAB — non-symmetric, lower memory than GMRES


class RNGBackend(Enum):
    """Parallel random number generator backend."""
    PHILOX   = "philox"     # counter-based, GPU-friendly
    THREEFRY = "threefry"   # counter-based, alternative
    XOSHIRO  = "xoshiro"    # fast, good statistical properties (CPU)


# ---------------------------------------------------------------------------
# Sparse analysis result
# ---------------------------------------------------------------------------

@dataclass
class SparseAnalysisResult:
    """Analysis result for a single tensor op."""

    op_name: str
    fill_fraction: float       # fraction of non-zero elements (0–1)
    is_sparse: bool            # True when fill_fraction < sparse_threshold
    precond: PrecondType
    solver_variant: SolverVariant

    def to_ir_attr(self) -> str:
        """Emit a tessera_solver attribute string."""
        parts = [
            f'op = "{self.op_name}"',
            f'fill_fraction = {self.fill_fraction:.4f}',
            f'precond = "{self.precond.value}"',
            f'solver = "{self.solver_variant.value}"',
        ]
        if self.is_sparse:
            parts.append("tessera_solver.sparse_hint")
        return "{" + ", ".join(parts) + "}"

    def __repr__(self) -> str:
        return (
            f"SparseAnalysisResult(op={self.op_name!r}, "
            f"fill={self.fill_fraction:.3f}, "
            f"sparse={self.is_sparse}, precond={self.precond.value}, "
            f"solver={self.solver_variant.value})"
        )


# ---------------------------------------------------------------------------
# Main solver configuration
# ---------------------------------------------------------------------------

@dataclass
class SolverConfig:
    """
    Configuration for the Tessera scientific solver suite.

    Parameters
    ----------
    sparse_threshold : float
        Ops with fill_fraction < this value are tagged sparse (default 5%).
    max_iter : int
        Maximum Newton / iterative refinement iterations.
    tolerance : float
        Convergence tolerance (residual norm).
    default_precond : PrecondType
        Fallback preconditioner when auto-selection does not apply.
    default_solver : SolverVariant
        Fallback iterative solver.
    rng_backend : RNGBackend
        RNG algorithm for ``tessera_rng.*`` ops.
    global_seed : int
        Base seed for all RNG streams.
    num_ranks : int
        Total number of parallel ranks (used to compute stream IDs).
    """

    sparse_threshold: float = 0.05
    max_iter: int = 500
    tolerance: float = 1e-8
    default_precond: PrecondType = PrecondType.ILU
    default_solver: SolverVariant = SolverVariant.GMRES
    rng_backend: RNGBackend = RNGBackend.PHILOX
    global_seed: int = 0
    num_ranks: int = 1

    def __post_init__(self) -> None:
        if not (0.0 < self.sparse_threshold < 1.0):
            raise ValueError(
                f"sparse_threshold={self.sparse_threshold} must be in (0, 1)"
            )
        if self.max_iter < 1:
            raise ValueError(f"max_iter={self.max_iter} must be >= 1")
        if self.tolerance <= 0:
            raise ValueError(f"tolerance={self.tolerance} must be > 0")
        if self.num_ranks < 1:
            raise ValueError(f"num_ranks={self.num_ranks} must be >= 1")

    # ------------------------------------------------------------------
    # Sparsity analysis
    # ------------------------------------------------------------------

    def analyze_op(self, op_name: str, fill_fraction: float) -> SparseAnalysisResult:
        """
        Apply sparsity heuristics to a named op.

        Preconditioner selection rules:

        - fill < 1%   → AMG (very sparse, likely SPD)
        - fill < 3%   → ILU (moderately sparse, general)
        - fill < threshold → Jacobi (lightly sparse, diagonal-dominant)
        - fill ≥ threshold → NONE (dense)

        Solver selection rules:

        - AMG precond  → CG  (AMG is typically for SPD systems)
        - ILU / Jacobi → GMRES
        - NONE         → default_solver
        """
        if not (0.0 <= fill_fraction <= 1.0):
            raise ValueError(
                f"fill_fraction={fill_fraction} must be in [0, 1]"
            )

        is_sparse = fill_fraction < self.sparse_threshold

        if is_sparse:
            if fill_fraction < 0.01:
                precond = PrecondType.AMG
            elif fill_fraction < 0.03:
                precond = PrecondType.ILU
            else:
                precond = PrecondType.JACOBI
        else:
            precond = PrecondType.NONE

        if precond == PrecondType.AMG:
            variant = SolverVariant.CG
        elif precond in (PrecondType.ILU, PrecondType.JACOBI):
            variant = SolverVariant.GMRES
        else:
            variant = self.default_solver

        return SparseAnalysisResult(
            op_name=op_name,
            fill_fraction=fill_fraction,
            is_sparse=is_sparse,
            precond=precond,
            solver_variant=variant,
        )

    # ------------------------------------------------------------------
    # RNG stream assignment
    # ------------------------------------------------------------------

    def rng_stream_id(self, rank: int) -> int:
        """
        Compute a deterministic per-rank stream ID.

        Formula: ``global_seed * num_ranks + rank``

        This guarantees that streams are independent across ranks and
        reproducible for a given seed.
        """
        if not (0 <= rank < self.num_ranks):
            raise ValueError(
                f"rank={rank} must be in [0, {self.num_ranks})"
            )
        return self.global_seed * self.num_ranks + rank

    def rng_stream_plan(self) -> "RNGStreamPlan":
        """Build a full RNGStreamPlan for all ranks."""
        return RNGStreamPlan(
            backend=self.rng_backend,
            global_seed=self.global_seed,
            num_ranks=self.num_ranks,
        )

    # ------------------------------------------------------------------
    # IR serialisation
    # ------------------------------------------------------------------

    def to_mlir_attrs(self) -> str:
        return (
            f"{{tessera.solver_config = {{"
            f"sparse_threshold = {self.sparse_threshold}, "
            f"max_iter = {self.max_iter}, "
            f"tolerance = {self.tolerance:.2e}, "
            f"precond = \"{self.default_precond.value}\", "
            f"solver = \"{self.default_solver.value}\", "
            f"rng_backend = \"{self.rng_backend.value}\"}}}}"
        )

    def __repr__(self) -> str:
        return (
            f"SolverConfig(sparse_threshold={self.sparse_threshold}, "
            f"max_iter={self.max_iter}, tol={self.tolerance:.2e}, "
            f"precond={self.default_precond.value}, "
            f"solver={self.default_solver.value})"
        )


# ---------------------------------------------------------------------------
# RNG stream plan
# ---------------------------------------------------------------------------

@dataclass
class RNGStreamPlan:
    """
    Maps every rank to a unique Philox / Threefry stream ID.

    Stream IDs are computed as: ``global_seed * num_ranks + rank``
    """

    backend: RNGBackend = RNGBackend.PHILOX
    global_seed: int = 0
    num_ranks: int = 1

    def __post_init__(self) -> None:
        if self.num_ranks < 1:
            raise ValueError(f"num_ranks={self.num_ranks} must be >= 1")

    def stream_id(self, rank: int) -> int:
        if not (0 <= rank < self.num_ranks):
            raise ValueError(f"rank={rank} out of range [0, {self.num_ranks})")
        return self.global_seed * self.num_ranks + rank

    def all_stream_ids(self) -> List[int]:
        return [self.stream_id(r) for r in range(self.num_ranks)]

    def streams_are_unique(self) -> bool:
        ids = self.all_stream_ids()
        return len(set(ids)) == len(ids)

    def to_ir_attr(self) -> str:
        return (
            f'{{tessera_rng.stream_plan = {{'
            f'backend = "{self.backend.value}", '
            f'global_seed = {self.global_seed}, '
            f'num_ranks = {self.num_ranks}}}}}'
        )

    def __repr__(self) -> str:
        return (
            f"RNGStreamPlan(backend={self.backend.value!r}, "
            f"global_seed={self.global_seed}, num_ranks={self.num_ranks})"
        )


# ---------------------------------------------------------------------------
# ZeRO optimizer shard configuration
# ---------------------------------------------------------------------------

@dataclass
class ZeROConfig:
    """
    ZeRO optimizer state partitioning configuration.

    - Stage 1: partition optimizer states across DP ranks
    - Stage 2: also partition gradients (default)
    - Stage 3: also partition parameters
    """

    stage: int = 2
    dp_axis: str = "dp"
    num_dp_ranks: int = 1
    partition_optimizer_states: bool = True
    partition_gradients: bool = True    # ZeRO-2+
    partition_parameters: bool = False  # ZeRO-3 only

    def __post_init__(self) -> None:
        if self.stage not in (1, 2, 3):
            raise ValueError(f"stage={self.stage} must be 1, 2, or 3")
        if self.num_dp_ranks < 1:
            raise ValueError(f"num_dp_ranks={self.num_dp_ranks} must be >= 1")
        if self.stage < 3 and self.partition_parameters:
            raise ValueError("partition_parameters requires stage=3")

    def partitioned_param_count(self, total_params: int) -> int:
        """Parameters held by each DP rank (ceiling division)."""
        return math.ceil(total_params / self.num_dp_ranks)

    def memory_reduction_factor(self) -> float:
        """
        Approximate optimizer-state memory reduction vs. full replication.
        Stage 1 → 1/num_dp_ranks of optimizer states.
        Stage 2 → also 1/num_dp_ranks of gradients.
        Stage 3 → also 1/num_dp_ranks of parameters.
        """
        return 1.0 / self.num_dp_ranks

    def to_ir_attr(self) -> str:
        return (
            f'{{tessera_sr.zero_config = {{'
            f'stage = {self.stage}, '
            f'dp_axis = "{self.dp_axis}", '
            f'num_ranks = {self.num_dp_ranks}}}}}'
        )

    def __repr__(self) -> str:
        return (
            f"ZeROConfig(stage={self.stage}, "
            f"dp_axis={self.dp_axis!r}, num_dp_ranks={self.num_dp_ranks})"
        )


# ---------------------------------------------------------------------------
# Resilience / checkpoint-restart configuration
# ---------------------------------------------------------------------------

@dataclass
class ResilienceConfig:
    """Configuration for the ResilienceRestartPass."""

    checkpoint_interval: int = 100   # training steps between checkpoints
    restart_policy: str = "last"     # "last" | "best" | "epoch"
    max_restarts: int = 3
    fault_barrier_enabled: bool = True
    save_dir: str = "/tmp/tessera_ckpt"

    _VALID_POLICIES = ("last", "best", "epoch")

    def __post_init__(self) -> None:
        if self.checkpoint_interval < 1:
            raise ValueError(
                f"checkpoint_interval={self.checkpoint_interval} must be >= 1"
            )
        if self.max_restarts < 0:
            raise ValueError(
                f"max_restarts={self.max_restarts} must be >= 0"
            )
        if self.restart_policy not in self._VALID_POLICIES:
            raise ValueError(
                f"restart_policy must be one of {self._VALID_POLICIES}, "
                f"got {self.restart_policy!r}"
            )

    def to_ir_attr(self) -> str:
        return (
            f'{{tessera_sr.resilience = {{'
            f'interval = {self.checkpoint_interval}, '
            f'policy = "{self.restart_policy}", '
            f'max_restarts = {self.max_restarts}}}}}'
        )

    def __repr__(self) -> str:
        return (
            f"ResilienceConfig(interval={self.checkpoint_interval}, "
            f"policy={self.restart_policy!r}, max_restarts={self.max_restarts})"
        )


# ---------------------------------------------------------------------------
# Deployment manifest
# ---------------------------------------------------------------------------

@dataclass
class DeploymentManifest:
    """
    Full deployment descriptor mirroring ExportDeploymentManifestPass output.

    Captures mesh topology, collective routes, optimizer shards, and
    checkpoint configuration for deployment tooling.
    """

    version: str = "v1.1"
    mesh: Dict[str, int] = field(default_factory=dict)
    collectives: List[str] = field(default_factory=list)
    optimizer_shards: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)

    def add_mesh_axis(self, name: str, size: int) -> "DeploymentManifest":
        if size < 1:
            raise ValueError(f"mesh axis size must be >= 1, got {size}")
        self.mesh[name] = size
        return self

    def add_collective(self, name: str) -> "DeploymentManifest":
        self.collectives.append(name)
        return self

    def add_optimizer_shard(
        self, layer: str, axis: str, policy: str = "zero2"
    ) -> "DeploymentManifest":
        self.optimizer_shards.append(f"{layer}@{axis}:{policy}")
        return self

    def add_checkpoint(
        self, layer: str, policy: str = "selective"
    ) -> "DeploymentManifest":
        self.checkpoints.append(f"{layer}:{policy}")
        return self

    def total_ranks(self) -> int:
        result = 1
        for v in self.mesh.values():
            result *= v
        return result

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "mesh": dict(self.mesh),
            "total_ranks": self.total_ranks(),
            "collectives": list(self.collectives),
            "optimizer_shards": list(self.optimizer_shards),
            "checkpoints": list(self.checkpoints),
        }

    def to_ir_attr(self) -> str:
        return (
            f'{{tessera.deployment_manifest = {{'
            f'version = "{self.version}", '
            f'total_ranks = {self.total_ranks()}}}}}'
        )

    def __repr__(self) -> str:
        return (
            f"DeploymentManifest(version={self.version!r}, "
            f"mesh={self.mesh}, "
            f"collectives={len(self.collectives)}, "
            f"checkpoints={len(self.checkpoints)})"
        )
