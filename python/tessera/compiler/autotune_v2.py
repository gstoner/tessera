"""
autotune_v2.py — Bayesian Autotuner (Phase 5)

BayesianAutotuner wraps an Optuna TPE sampler (Hyperband pruning) to search
tile/warp/stage configurations for a GEMM workload. Falls back to a grid
search when optuna is not installed. Warm-starts from the existing SQLite
tuning cache.

Usage::

    from tessera.compiler.autotune_v2 import BayesianAutotuner, GEMMWorkload

    tuner = BayesianAutotuner(GEMMWorkload(M=4096, N=4096, K=4096))
    result = tuner.tune(max_trials=50)
    print(result.tflops)          # best throughput found
    tuner.save_to_cache("tuning_cache.db")
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Workload description
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GEMMWorkload:
    """Describes a GEMM problem (M × K) × (K × N) = (M × N)."""

    M: int
    N: int
    K: int
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        for name, val in [("M", self.M), ("N", self.N), ("K", self.K)]:
            if val <= 0:
                raise ValueError(f"{name}={val} must be > 0")
        if self.dtype not in (
            "bf16",
            "fp16",
            "fp32",
            "fp8",
            "fp8_e4m3",
            "fp8_e5m2",
            "fp6_e2m3",
            "fp6_e3m2",
            "fp4_e2m1",
            "nvfp4",
            "int8",
        ):
            raise ValueError(f"dtype={self.dtype!r} not supported")

    def flops(self) -> int:
        """Multiply-accumulate count (2 × M × N × K)."""
        return 2 * self.M * self.N * self.K

    def tflops_at(self, latency_ms: float) -> float:
        """Compute TFLOPs/s given a latency in milliseconds."""
        if latency_ms <= 0:
            raise ValueError("latency_ms must be > 0")
        return self.flops() / (latency_ms * 1e-3) / 1e12

    def __repr__(self) -> str:
        return f"GEMMWorkload(M={self.M}, N={self.N}, K={self.K}, dtype={self.dtype!r})"


# ---------------------------------------------------------------------------
# Tile configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TuningConfig:
    """A concrete tile/warp/stage configuration for a GEMM kernel."""

    tile_m: int
    tile_n: int
    tile_k: int
    num_warps: int = 4
    num_stages: int = 2

    def __post_init__(self) -> None:
        for name, val in [("tile_m", self.tile_m), ("tile_n", self.tile_n),
                          ("tile_k", self.tile_k)]:
            if val <= 0 or (val & (val - 1)) != 0:
                raise ValueError(f"{name}={val} must be a positive power of 2")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError(f"num_warps={self.num_warps} must be in {{1,2,4,8}}")
        if self.num_stages < 1:
            raise ValueError(f"num_stages={self.num_stages} must be >= 1")

    def smem_bytes(self) -> int:
        """Approximate shared-memory footprint (bytes, BF16)."""
        return 2 * (self.tile_m * self.tile_k + self.tile_k * self.tile_n) * self.num_stages

    def to_ir_attr(self) -> str:
        return (
            f"{{tessera.tile_config = {{tile_m = {self.tile_m}, "
            f"tile_n = {self.tile_n}, tile_k = {self.tile_k}, "
            f"num_warps = {self.num_warps}, num_stages = {self.num_stages}}}}}"
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    def __repr__(self) -> str:
        return (f"TuningConfig(tile_m={self.tile_m}, tile_n={self.tile_n}, "
                f"tile_k={self.tile_k}, num_warps={self.num_warps}, "
                f"num_stages={self.num_stages})")


# ---------------------------------------------------------------------------
# Tuning result
# ---------------------------------------------------------------------------

@dataclass
class TuningResult:
    """One evaluated configuration and its measured (or estimated) throughput."""

    config: TuningConfig
    latency_ms: float
    tflops: float
    sampled_at: float = field(default_factory=time.time)
    trial_id: int = 0

    def __repr__(self) -> str:
        return (f"TuningResult(tflops={self.tflops:.2f}, "
                f"latency_ms={self.latency_ms:.4f}, config={self.config!r})")


# ---------------------------------------------------------------------------
# Autotuner
# ---------------------------------------------------------------------------

class BayesianAutotuner:
    """
    Bayesian autotuner using Optuna TPE + Hyperband pruning.

    Falls back to a deterministic grid search when ``optuna`` is not installed.
    Warm-starts from an SQLite cache that shares the schema with the v1 autotuner.

    Parameters
    ----------
    workload:
        The GEMM workload to optimise.
    peak_tflops:
        Hardware peak (TFLOPs/s); used by the synthetic latency model.
        Default 312 = NVIDIA A100 BF16.
    seed:
        Random seed for reproducibility.
    smem_budget_bytes:
        Shared-memory budget (bytes). Configurations that exceed this are
        pruned immediately.  Default 98304 (96 KiB, A100 limit per SM).
    """

    _TILE_CHOICES: Sequence[int] = (32, 64, 128, 256)
    _WARP_CHOICES: Sequence[int] = (1, 2, 4, 8)
    _STAGE_CHOICES: Sequence[int] = (1, 2, 3, 4)

    def __init__(
        self,
        workload: GEMMWorkload,
        *,
        peak_tflops: float = 312.0,
        seed: int = 42,
        smem_budget_bytes: int = 98_304,
    ) -> None:
        self.workload = workload
        self.peak_tflops = peak_tflops
        self.seed = seed
        self.smem_budget_bytes = smem_budget_bytes
        self._results: List[TuningResult] = []
        self._best: Optional[TuningResult] = None

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def warm_start_from_cache(self, db_path: str) -> int:
        """
        Load prior results from SQLite cache.

        Compatible with the v1 schema; silently ignores missing tables/columns.

        Returns the number of results loaded.
        """
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.execute(
                "SELECT tile_m, tile_n, tile_k, num_warps, num_stages, "
                "latency_ms, tflops, sampled_at, trial_id "
                "FROM tuning_results "
                "WHERE M=? AND N=? AND K=? AND dtype=?",
                (self.workload.M, self.workload.N, self.workload.K,
                 self.workload.dtype),
            )
            count = 0
            for row in cur.fetchall():
                try:
                    cfg = TuningConfig(
                        tile_m=int(row[0]), tile_n=int(row[1]),
                        tile_k=int(row[2]), num_warps=int(row[3]),
                        num_stages=int(row[4]),
                    )
                    res = TuningResult(
                        config=cfg,
                        latency_ms=float(row[5]),
                        tflops=float(row[6]),
                        sampled_at=float(row[7]),
                        trial_id=int(row[8]),
                    )
                    self._results.append(res)
                    if self._best is None or res.tflops > self._best.tflops:
                        self._best = res
                    count += 1
                except (ValueError, TypeError):
                    continue  # skip corrupted rows
            conn.close()
            return count
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return 0

    def save_to_cache(self, db_path: str) -> None:
        """Persist all evaluated results to SQLite cache (upsert-style)."""
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tuning_results (
                M INT, N INT, K INT, dtype TEXT,
                tile_m INT, tile_n INT, tile_k INT,
                num_warps INT, num_stages INT,
                latency_ms REAL, tflops REAL,
                sampled_at REAL, trial_id INT
            )
            """
        )
        for r in self._results:
            conn.execute(
                "INSERT INTO tuning_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self.workload.M, self.workload.N,
                    self.workload.K, self.workload.dtype,
                    r.config.tile_m, r.config.tile_n, r.config.tile_k,
                    r.config.num_warps, r.config.num_stages,
                    r.latency_ms, r.tflops,
                    r.sampled_at, r.trial_id,
                ),
            )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Synthetic latency model (used in place of real kernel measurement)
    # ------------------------------------------------------------------

    def _mock_latency(self, cfg: TuningConfig) -> float:
        """
        Analytical roofline model with tile- and stage-efficiency factors.

        Optimal point: tile_m=128, tile_n=128, tile_k=32, num_warps=4, num_stages=2.
        Deviations incur multiplicative penalties to simulate realistic behaviour.
        """
        roofline_ms = self.workload.flops() / (self.peak_tflops * 1e12) * 1e3

        # Tile efficiency: normalise to 128×128
        tile_area = cfg.tile_m * cfg.tile_n
        tile_eff = min(math.sqrt(tile_area) / 128.0, 1.0)

        # Stage penalty: 2 stages is optimal for most A100 configs
        stage_pen = 1.0 + abs(cfg.num_stages - 2) * 0.08

        # Warp penalty: 4 warps is optimal
        warp_pen = 1.0 + abs(math.log2(cfg.num_warps) - 2) * 0.05

        # Small-K penalty: tile_k < 32 is wasteful
        k_pen = 1.0 + max(0, math.log2(32) - math.log2(cfg.tile_k)) * 0.04

        return roofline_ms / max(tile_eff, 1e-6) * stage_pen * warp_pen * k_pen

    def _evaluate(self, cfg: TuningConfig, trial_id: int) -> TuningResult:
        latency_ms = self._mock_latency(cfg)
        tflops = self.workload.tflops_at(latency_ms)
        return TuningResult(cfg, latency_ms, tflops, time.time(), trial_id)

    def _is_feasible(self, cfg: TuningConfig) -> bool:
        return cfg.smem_bytes() <= self.smem_budget_bytes

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def tune(self, max_trials: int = 50) -> TuningResult:
        """
        Search for the best TuningConfig.

        Uses Optuna TPE + Hyperband pruning when available; degrades
        gracefully to grid search otherwise.

        Parameters
        ----------
        max_trials : int
            Maximum number of kernel evaluations.

        Returns
        -------
        TuningResult
            The best configuration found.
        """
        try:
            import optuna  # noqa: F401
            return self._tune_optuna(max_trials)
        except ImportError:
            return self._tune_grid(max_trials)

    def _tune_optuna(self, max_trials: int) -> TuningResult:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            cfg = TuningConfig(
                tile_m=trial.suggest_categorical("tile_m",
                                                 list(self._TILE_CHOICES)),
                tile_n=trial.suggest_categorical("tile_n",
                                                 list(self._TILE_CHOICES)),
                tile_k=trial.suggest_categorical("tile_k",
                                                 list(self._TILE_CHOICES)),
                num_warps=trial.suggest_categorical("num_warps",
                                                    list(self._WARP_CHOICES)),
                num_stages=trial.suggest_categorical("num_stages",
                                                     list(self._STAGE_CHOICES)),
            )
            if not self._is_feasible(cfg):
                raise optuna.TrialPruned()
            res = self._evaluate(cfg, trial.number)
            self._results.append(res)
            if self._best is None or res.tflops > self._best.tflops:
                self._best = res
            return res.tflops  # maximise

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max_trials)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )

        # Enqueue warm-start configurations (best 10 from cache)
        warm = sorted(self._results, key=lambda r: -r.tflops)[:10]
        for r in warm:
            study.enqueue_trial({
                "tile_m": r.config.tile_m,
                "tile_n": r.config.tile_n,
                "tile_k": r.config.tile_k,
                "num_warps": r.config.num_warps,
                "num_stages": r.config.num_stages,
            })

        study.optimize(objective, n_trials=max_trials, show_progress_bar=False)
        return self._best

    def _tune_grid(self, max_trials: int) -> TuningResult:
        """Deterministic grid search fallback."""
        trial_id = len(self._results)
        candidates = [
            TuningConfig(tm, tn, tk, nw, ns)
            for tm in self._TILE_CHOICES
            for tn in self._TILE_CHOICES
            for tk in [32, 64, 128]
            for nw in self._WARP_CHOICES
            for ns in self._STAGE_CHOICES
            if TuningConfig(tm, tn, tk, nw, ns).smem_bytes() <= self.smem_budget_bytes
        ]
        # Prefer larger tiles (better for large GEMMs)
        candidates.sort(key=lambda c: -(c.tile_m * c.tile_n))

        for cfg in candidates[:max_trials]:
            res = self._evaluate(cfg, trial_id)
            self._results.append(res)
            if self._best is None or res.tflops > self._best.tflops:
                self._best = res
            trial_id += 1

        return self._best

    # ------------------------------------------------------------------
    # Properties / serialisation
    # ------------------------------------------------------------------

    @property
    def best(self) -> Optional[TuningResult]:
        """Best TuningResult seen so far (None if tune() not yet called)."""
        return self._best

    @property
    def num_trials(self) -> int:
        """Total number of configurations evaluated."""
        return len(self._results)

    @property
    def results(self) -> List[TuningResult]:
        """All evaluated TuningResults in order."""
        return list(self._results)

    def to_mlir_attrs(self) -> str:
        """Serialise the best config to a tessera MLIR attribute string."""
        if self._best is None:
            return "{tessera.autotune = {}}"
        c = self._best.config
        schedule_hash = self.schedule_hash()
        return (
            f"{{tessera.autotune = {{tile_m = {c.tile_m}, tile_n = {c.tile_n}, "
            f"tile_k = {c.tile_k}, num_warps = {c.num_warps}, "
            f"num_stages = {c.num_stages}, "
            f"schedule_hash = \"{schedule_hash}\", "
            f"tflops = {self._best.tflops:.2f}}}}}"
        )

    def schedule_artifact(self, arch: str = "generic") -> Dict[str, object]:
        """Return a reproducible schedule artifact for deployment bundles."""
        tile = self._best.config.to_dict() if self._best is not None else {}
        artifact = {
            "version": 1,
            "kind": "tessera.schedule_artifact",
            "arch": arch,
            "shape": {
                "M": self.workload.M,
                "N": self.workload.N,
                "K": self.workload.K,
                "dtype": self.workload.dtype,
            },
            "numeric_policy": {
                "storage": self.workload.dtype,
                "accum": "f32" if self.workload.dtype != "int8" else "s32",
                "rounding": "nearest_even",
                "deterministic": True,
            },
            "movement": {
                "prefetch": "auto",
                "overlap": "compute",
                "stages": tile.get("num_stages", 0),
            },
            "tile": tile,
        }
        artifact["hash"] = self._hash_payload(artifact)
        return artifact

    def schedule_hash(self, arch: str = "generic") -> str:
        """Stable hash for the current best schedule artifact."""
        return str(self.schedule_artifact(arch=arch)["hash"])

    def to_schedule_artifact_mlir(self, arch: str = "generic") -> str:
        """Serialise the schedule artifact as a Schedule IR op."""
        artifact = self.schedule_artifact(arch=arch)
        shape = artifact["shape"]
        tile = artifact["tile"]
        shape_key = (
            f"M={shape['M']};N={shape['N']};K={shape['K']};dtype={shape['dtype']}"
        )
        tile_attr = (
            "{"
            + ", ".join(f"{k} = {v}" for k, v in tile.items())
            + "}"
            if tile
            else "{}"
        )
        return (
            "schedule.artifact "
            f"{{hash = \"{artifact['hash']}\", arch = \"{arch}\", "
            f"shape_key = \"{shape_key}\", tile = {tile_attr}, "
            "movement = {prefetch = \"auto\", overlap = \"compute\"}, "
            f"numeric_policy = \"{shape['dtype']}@accum(f32)\"}}"
        )

    def cost_measurements(self) -> List[Dict[str, float]]:
        """Return autotuner measurements for learned surrogate training."""
        rows: List[Dict[str, float]] = []
        bytes_moved = 2.0 * (
            self.workload.M * self.workload.K
            + self.workload.K * self.workload.N
            + self.workload.M * self.workload.N
        )
        flops = float(self.workload.flops())
        for r in self._results:
            rows.append({
                "M": float(self.workload.M),
                "N": float(self.workload.N),
                "K": float(self.workload.K),
                "flops": flops,
                "bytes_moved": bytes_moved,
                "tile_m": float(r.config.tile_m),
                "tile_n": float(r.config.tile_n),
                "tile_k": float(r.config.tile_k),
                "num_warps": float(r.config.num_warps),
                "num_stages": float(r.config.num_stages),
                "latency_ms": float(r.latency_ms),
                "tflops": float(r.tflops),
            })
        return rows

    @staticmethod
    def _hash_payload(payload: Dict[str, object]) -> str:
        stable = dict(payload)
        stable.pop("hash", None)
        raw = json.dumps(stable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def __repr__(self) -> str:
        return (f"BayesianAutotuner(workload={self.workload!r}, "
                f"trials={self.num_trials}, best={self._best!r})")
