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

import contextlib
import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..telemetry import make_event
from .rounding import RTNE


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
    layout: str = "row_major"
    arch: str = "generic"
    movement: Mapping[str, object] = field(default_factory=lambda: {"prefetch": "auto", "overlap": "compute"})

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
    status: str = "ok"
    reason: str = ""
    method: str = "roofline"

    def __repr__(self) -> str:
        suffix = f", status={self.status!r}" if self.status != "ok" else ""
        return (f"TuningResult(tflops={self.tflops:.2f}, "
                f"latency_ms={self.latency_ms:.4f}, config={self.config!r}{suffix})")


@dataclass(frozen=True)
class CandidateRejection:
    """Why a tuning candidate was excluded before evaluation."""

    config: TuningConfig
    reason: str


class LegalGEMMCandidateGenerator:
    """Generate target-legal GEMM tuning candidates."""

    def __init__(
        self,
        workload: GEMMWorkload,
        *,
        tile_choices: Sequence[int],
        warp_choices: Sequence[int],
        stage_choices: Sequence[int],
        smem_budget_bytes: int,
    ) -> None:
        self.workload = workload
        self.tile_choices = tuple(tile_choices)
        self.warp_choices = tuple(warp_choices)
        self.stage_choices = tuple(stage_choices)
        self.smem_budget_bytes = smem_budget_bytes
        self.rejections: list[CandidateRejection] = []

    def candidates(self) -> list[TuningConfig]:
        self.rejections.clear()
        legal: list[TuningConfig] = []
        for tm in self.tile_choices:
            for tn in self.tile_choices:
                for tk in self.tile_choices:
                    for nw in self.warp_choices:
                        for ns in self.stage_choices:
                            cfg = TuningConfig(tm, tn, tk, nw, ns)
                            reason = self.rejection_reason(cfg)
                            if reason:
                                self.rejections.append(CandidateRejection(cfg, reason))
                            else:
                                legal.append(cfg)
        legal.sort(key=lambda c: (-(c.tile_m * c.tile_n), c.tile_k, c.num_warps, c.num_stages))
        return legal

    def rejection_reason(self, cfg: TuningConfig) -> str:
        if cfg.tile_m > self.workload.M or cfg.tile_n > self.workload.N or cfg.tile_k > self.workload.K:
            return "tile_exceeds_problem_shape"
        if self.workload.M % cfg.tile_m or self.workload.N % cfg.tile_n or self.workload.K % cfg.tile_k:
            return "shape_not_divisible_by_tile"
        if cfg.smem_bytes() > self.smem_budget_bytes:
            return "shared_memory_budget_exceeded"
        return ""


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
        self._rejections: List[CandidateRejection] = []

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def warm_start_from_cache(self, db_path: str) -> int:
        """
        Load prior results from SQLite cache.

        Compatible with the v1 schema; silently ignores missing tables/columns.

        Returns the number of results loaded.
        """
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            columns = _table_columns(conn, "tuning_results")
            if not columns:
                return 0
            optional = {
                "status": "'ok'",
                "reason": "''",
                "method": "'roofline'",
            }
            select_cols = [
                "tile_m", "tile_n", "tile_k", "num_warps", "num_stages",
                "latency_ms", "tflops", "sampled_at", "trial_id",
                *[name if name in columns else fallback for name, fallback in optional.items()],
            ]
            filters = ["M=?", "N=?", "K=?", "dtype=?"]
            params: list[object] = [self.workload.M, self.workload.N, self.workload.K, self.workload.dtype]
            if "arch" in columns:
                filters.append("arch=?")
                params.append(self.workload.arch)
            if "layout" in columns:
                filters.append("layout=?")
                params.append(self.workload.layout)
            cur = conn.execute(
                f"SELECT {', '.join(select_cols)} FROM tuning_results WHERE {' AND '.join(filters)}",
                tuple(params),
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
                        status=str(row[9]),
                        reason=str(row[10]),
                        method=str(row[11]),
                    )
                    self._results.append(res)
                    if self._best is None or res.tflops > self._best.tflops:
                        self._best = res
                    count += 1
                except (ValueError, TypeError):
                    continue  # skip corrupted rows
            return count
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return 0
        finally:
            if conn is not None:
                conn.close()

    def save_to_cache(self, db_path: str) -> None:
        """Persist all evaluated results to SQLite cache (upsert-style)."""
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            _ensure_cache_schema(conn)
            for r in self._results:
                conn.execute(
                    """
                    INSERT INTO tuning_results (
                        M, N, K, dtype, arch, layout, movement_json,
                        tile_m, tile_n, tile_k, num_warps, num_stages,
                        latency_ms, tflops, sampled_at, trial_id, status, reason, method
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        self.workload.M, self.workload.N,
                        self.workload.K, self.workload.dtype,
                        self.workload.arch, self.workload.layout,
                        json.dumps(dict(self.workload.movement), sort_keys=True),
                        r.config.tile_m, r.config.tile_n, r.config.tile_k,
                        r.config.num_warps, r.config.num_stages,
                        r.latency_ms, r.tflops,
                        r.sampled_at, r.trial_id, r.status, r.reason, r.method,
                    ),
                )
            conn.commit()

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

    def _evaluate(self, cfg: TuningConfig, trial_id: int, *, method: str = "roofline") -> TuningResult:
        latency_ms = self._mock_latency(cfg)
        tflops = self.workload.tflops_at(latency_ms)
        status = "unmeasured" if method == "on_device" else "ok"
        reason = "runtime device timers are not wired; using deterministic roofline estimate" if method == "on_device" else ""
        return TuningResult(cfg, latency_ms, tflops, time.time(), trial_id, status, reason, method)

    def _is_feasible(self, cfg: TuningConfig) -> bool:
        return self._candidate_generator().rejection_reason(cfg) == ""

    def legal_candidates(self) -> list[TuningConfig]:
        generator = self._candidate_generator()
        candidates = generator.candidates()
        self._rejections = list(generator.rejections)
        return candidates

    @property
    def rejected_candidates(self) -> List[CandidateRejection]:
        return list(self._rejections)

    def _candidate_generator(self) -> LegalGEMMCandidateGenerator:
        return LegalGEMMCandidateGenerator(
            self.workload,
            tile_choices=self._TILE_CHOICES,
            warp_choices=self._WARP_CHOICES,
            stage_choices=self._STAGE_CHOICES,
            smem_budget_bytes=self.smem_budget_bytes,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def tune(self, max_trials: int = 50, *, method: str = "roofline") -> TuningResult:
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
        if method in {"on_device", "grid", "roofline"}:
            return self._tune_grid(max_trials, method=method)
        try:
            import optuna  # noqa: F401
            return self._tune_optuna(max_trials, method=method)
        except ImportError:
            return self._tune_grid(max_trials, method=method)

    def _tune_optuna(self, max_trials: int, *, method: str = "roofline") -> TuningResult:
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
            reason = self._candidate_generator().rejection_reason(cfg)
            if reason:
                self._rejections.append(CandidateRejection(cfg, reason))
                raise optuna.TrialPruned()
            res = self._evaluate(cfg, trial.number, method=method)
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
        if self._best is None:
            raise ValueError("no legal GEMM tuning candidates were available")
        return self._best

    def _tune_grid(self, max_trials: int, *, method: str = "roofline") -> TuningResult:
        """Deterministic grid search fallback."""
        trial_id = len(self._results)
        candidates = self.legal_candidates()
        if not candidates:
            raise ValueError("no legal GEMM tuning candidates were available")

        for cfg in candidates[:max_trials]:
            res = self._evaluate(cfg, trial_id, method=method)
            self._results.append(res)
            if self._best is None or res.tflops > self._best.tflops:
                self._best = res
            trial_id += 1

        # By contract, the loop above sets ``self._best`` at least once
        # because each trial produces a ``TuningResult``; assert for mypy.
        assert self._best is not None
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

    def schedule_artifact(self, arch: str = "generic") -> Dict[str, Any]:
        """Return a reproducible schedule artifact for deployment bundles."""
        tile = self._best.config.to_dict() if self._best is not None else {}
        latency_ms = self._best.latency_ms if self._best is not None else None
        tflops = self._best.tflops if self._best is not None else None
        status = self._best.status if self._best is not None else "unmeasured"
        reason = self._best.reason if self._best is not None else "no tuning result has been selected"
        movement = dict(self.workload.movement)
        movement.setdefault("stages", tile.get("num_stages", 0))
        artifact = {
            "version": 1,
            "kind": "tessera.schedule_artifact",
            "arch": arch,
            "cache_key": self.cache_key(arch=arch),
            "shape": {
                "M": self.workload.M,
                "N": self.workload.N,
                "K": self.workload.K,
                "dtype": self.workload.dtype,
            },
            "layout": self.workload.layout,
            "numeric_policy": {
                "storage": self.workload.dtype,
                "accum": "f32" if self.workload.dtype != "int8" else "s32",
                "rounding": RTNE,
                "deterministic": True,
            },
            "movement": movement,
            "target_features": self.target_features(arch),
            "measurements": {
                "status": status,
                "reason": reason,
                "latency_ms": latency_ms,
                "tflops": tflops,
                "trial_count": len(self._results),
                "rejected_count": len(self._rejections),
            },
            "tile": tile,
        }
        artifact["hash"] = self._hash_payload(artifact)
        artifact["telemetry"] = make_event(
            "autotune.best",
            source="autotune",
            op="matmul",
            shape={"M": self.workload.M, "N": self.workload.N, "K": self.workload.K},
            dtype=self.workload.dtype,
            arch=arch,
            schedule_hash=str(artifact["hash"]),
            kernel_id="gemm",
            latency_ms=latency_ms,
            tflops=tflops,
            status=status,
            metadata={"tile": tile, "reason": reason, "target_features": artifact["target_features"]},
        )
        return artifact

    def cache_key(self, arch: str = "generic") -> Dict[str, object]:
        return {
            "op": "matmul",
            "shape": {"M": self.workload.M, "N": self.workload.N, "K": self.workload.K},
            "dtype": self.workload.dtype,
            "arch": arch,
            "layout": self.workload.layout,
            "numeric_policy": {
                "storage": self.workload.dtype,
                "accum": "f32" if self.workload.dtype != "int8" else "s32",
            },
            "movement": dict(self.workload.movement),
        }

    @staticmethod
    def target_features(arch: str = "generic") -> Dict[str, object]:
        if arch.startswith("sm") or arch.startswith("nvidia"):
            return {"family": "nvidia", "tensor_cores": True, "device_timers": False}
        if arch.startswith("gfx") or arch == "rocm":
            return {"family": "rocm", "mfma": True, "device_timers": False}
        if arch.startswith("apple") or arch == "apple_cpu":
            return {"family": "apple", "accelerate": True, "device_timers": False}
        if arch in {"cpu", "x86_64"}:
            return {"family": "cpu", "wall_clock": True, "device_timers": False}
        return {"family": "generic", "device_timers": False}

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

    def cost_measurements(self) -> List[Dict[str, Any]]:
        """Return autotuner measurements for learned surrogate training."""
        rows: List[Dict[str, Any]] = []
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
                "status": r.status,
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


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.DatabaseError:
        return set()


def _ensure_cache_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tuning_results (
            M INT, N INT, K INT, dtype TEXT,
            arch TEXT DEFAULT 'generic',
            layout TEXT DEFAULT 'row_major',
            movement_json TEXT DEFAULT '{}',
            tile_m INT, tile_n INT, tile_k INT,
            num_warps INT, num_stages INT,
            latency_ms REAL, tflops REAL,
            sampled_at REAL, trial_id INT,
            status TEXT DEFAULT 'ok',
            reason TEXT DEFAULT '',
            method TEXT DEFAULT 'roofline'
        )
        """
    )
    columns = _table_columns(conn, "tuning_results")
    additions = {
        "arch": "TEXT DEFAULT 'generic'",
        "layout": "TEXT DEFAULT 'row_major'",
        "movement_json": "TEXT DEFAULT '{}'",
        "status": "TEXT DEFAULT 'ok'",
        "reason": "TEXT DEFAULT ''",
        "method": "TEXT DEFAULT 'roofline'",
    }
    for name, ddl in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE tuning_results ADD COLUMN {name} {ddl}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tuning_lookup ON tuning_results(M, N, K, dtype, arch, layout)"
    )
