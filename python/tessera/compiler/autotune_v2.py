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
from .reuse_distance_cost import estimate_gemm_reuse_distance
from .rounding import RTNE
from .tile_rasterization import RASTER_ORDER_CHOICES


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
    #: Block rasterization order — how the 1-D block id walks the 2-D tile grid
    #: (``tile_rasterization.RasterOrder``). A permutation of block ids, so it is
    #: semantics-preserving by construction and provable hardware-free. Defaults
    #: to ``"row_major"``, which is the mapping every emitter already uses, so an
    #: unset config is byte-identical to pre-knob behaviour.
    #:
    #: **Carried, not analytically promoted.** The reuse-distance model can score
    #: this axis, but ROCM-CALIB-1 proved that an unvalidated locality metric must
    #: not change a production raster choice. The field remains available to
    #: exact-device sweeps and measured records; automatic enumeration waits for
    #: an architecture-owned correlation/retain verdict.
    raster_order: str = "row_major"
    #: Panel height for the grouped orders. 1 makes them degenerate to
    #: row/column-major, keeping the identity reachable from anywhere in the
    #: search space. Ignored by the non-grouped orders. Carried, not swept — see
    #: :attr:`raster_order`.
    raster_group: int = 1

    def __post_init__(self) -> None:
        for name, val in [("tile_m", self.tile_m), ("tile_n", self.tile_n),
                          ("tile_k", self.tile_k)]:
            if val <= 0 or (val & (val - 1)) != 0:
                raise ValueError(f"{name}={val} must be a positive power of 2")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError(f"num_warps={self.num_warps} must be in {{1,2,4,8}}")
        if self.num_stages < 1:
            raise ValueError(f"num_stages={self.num_stages} must be >= 1")
        if self.raster_order not in RASTER_ORDER_CHOICES:
            raise ValueError(
                f"raster_order={self.raster_order!r} must be one of "
                f"{list(RASTER_ORDER_CHOICES)}")
        if self.raster_group < 1:
            raise ValueError(
                f"raster_group={self.raster_group} must be >= 1")

    def smem_bytes(self) -> int:
        """Approximate shared-memory footprint (bytes, BF16)."""
        return 2 * (self.tile_m * self.tile_k + self.tile_k * self.tile_n) * self.num_stages

    def to_ir_attr(self) -> str:
        return (
            f"{{tessera.tile_config = {{tile_m = {self.tile_m}, "
            f"tile_n = {self.tile_n}, tile_k = {self.tile_k}, "
            f"num_warps = {self.num_warps}, num_stages = {self.num_stages}, "
            f'raster_order = "{self.raster_order}", '
            f"raster_group = {self.raster_group}}}}}"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "raster_order": self.raster_order,
            "raster_group": self.raster_group,
        }

    def __repr__(self) -> str:
        return (f"TuningConfig(tile_m={self.tile_m}, tile_n={self.tile_n}, "
                f"tile_k={self.tile_k}, num_warps={self.num_warps}, "
                f"num_stages={self.num_stages}, "
                f"raster_order={self.raster_order!r}, "
                f"raster_group={self.raster_group})")


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
    method: str = "reuse_distance"

    def __repr__(self) -> str:
        suffix = f", status={self.status!r}" if self.status != "ok" else ""
        return (f"TuningResult(tflops={self.tflops:.2f}, "
                f"latency_ms={self.latency_ms:.4f}, config={self.config!r}{suffix})")

    def to_measured_schedule(self, *, target: str) -> Dict[str, Any]:
        """Return the lowering input only for a successful measured trial."""
        if self.status != "ok":
            raise ValueError(
                f"cannot apply tuning result with status {self.status!r}"
            )
        if self.method != "measured":
            raise ValueError(
                "only device-measured tuning results may drive lowering"
            )
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0.0:
            raise ValueError(
                "measured tuning result requires positive finite latency"
            )
        return {
            "method": "measured",
            "target": str(target),
            "latency_ms": float(self.latency_ms),
            "config": self.config.to_dict(),
        }


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
        Hardware peak (TFLOPs/s); used by the analytical compute bound.
        Default 312 = NVIDIA A100 BF16.
    dram_bw_gbps:
        Sustained or specification DRAM bandwidth in GB/s.  The provenance is
        owned by ``TargetPerf`` at production call sites.
    cache_bytes:
        Capacity of the cache represented by the tile-reuse simulation.  Zero
        is an explicit conservative no-cache boundary, not a guessed fallback.
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
        dram_bw_gbps: float = 2039.0,
        cache_bytes: int = 40 * 1024 * 1024,
        seed: int = 42,
        smem_budget_bytes: int = 98_304,
    ) -> None:
        if peak_tflops <= 0:
            raise ValueError("peak_tflops must be positive")
        if dram_bw_gbps <= 0:
            raise ValueError("dram_bw_gbps must be positive")
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be non-negative")
        if smem_budget_bytes <= 0:
            raise ValueError("smem_budget_bytes must be positive")
        self.workload = workload
        self.peak_tflops = peak_tflops
        self.dram_bw_gbps = dram_bw_gbps
        self.cache_bytes = cache_bytes
        self.seed = seed
        self.smem_budget_bytes = smem_budget_bytes
        self._results: List[TuningResult] = []
        self._best: Optional[TuningResult] = None
        self._rejections: List[CandidateRejection] = []
        self._warm_start_skipped: List[str] = []

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def warm_start_from_cache(self, db_path: str) -> int:
        """
        Load prior results from SQLite cache.

        Compatible with the v1 schema; a missing table or column falls back to
        that column's default rather than failing.

        Returns the number of results loaded. Rows that could **not** be loaded
        are recorded with a reason in :attr:`warm_start_skipped` — check it when
        the count is lower than expected, rather than assuming a cold cache.
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
                "raster_order": "'row_major'",
                "raster_group": "1",
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
            if "movement_json" in columns:
                filters.append("movement_json=?")
                params.append(json.dumps(dict(self.workload.movement), sort_keys=True))
            cur = conn.execute(
                f"SELECT {', '.join(select_cols)} FROM tuning_results WHERE {' AND '.join(filters)}",
                tuple(params),
            )
            count = 0
            for row in cur.fetchall():
                # A row written by a newer Tessera may name a rasterization order
                # this build does not know. Its latency was measured *with* that
                # order, so it cannot be re-labelled row_major — but dropping it
                # into the generic "corrupted row" path below would make a valid
                # measurement vanish with no diagnostic, which is the silent
                # degrade the raster columns exist to prevent. Record it.
                raster_order = str(row[12])
                if raster_order not in RASTER_ORDER_CHOICES:
                    self._warm_start_skipped.append(
                        f"unknown raster_order {raster_order!r} (tile "
                        f"{row[0]}x{row[1]}x{row[2]}) — row written by a build "
                        f"with a rasterization order this one does not have")
                    continue
                try:
                    cfg = TuningConfig(
                        tile_m=int(row[0]), tile_n=int(row[1]),
                        tile_k=int(row[2]), num_warps=int(row[3]),
                        num_stages=int(row[4]),
                        raster_order=raster_order, raster_group=int(row[13]),
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
                    self._consider_best(res)
                    count += 1
                except (ValueError, TypeError) as exc:
                    self._warm_start_skipped.append(
                        f"unusable cache row (tile {row[0]}x{row[1]}x{row[2]}): "
                        f"{exc}")
                    continue
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
                        latency_ms, tflops, sampled_at, trial_id, status, reason, method,
                        raster_order, raster_group
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                        r.config.raster_order, r.config.raster_group,
                    ),
                )
            conn.commit()

    # ------------------------------------------------------------------
    # Hardware-free reuse-distance model
    # ------------------------------------------------------------------

    def _analytical_latency(self, cfg: TuningConfig) -> float:
        return estimate_gemm_reuse_distance(
            self.workload,
            cfg,
            peak_tflops=self.peak_tflops,
            dram_bw_gbps=self.dram_bw_gbps,
            cache_bytes=self.cache_bytes,
        ).latency_ms

    def _evaluate(
        self,
        cfg: TuningConfig,
        trial_id: int,
        *,
        method: str = "reuse_distance",
    ) -> TuningResult:
        latency_ms = self._analytical_latency(cfg)
        tflops = self.workload.tflops_at(latency_ms)
        status = "unmeasured" if method == "on_device" else "ok"
        reason = (
            "runtime device timers are not wired; reuse-distance estimate is "
            "reported but cannot drive lowering"
            if method == "on_device"
            else ""
        )
        actual_method = "on_device" if method == "on_device" else "reuse_distance"
        return TuningResult(
            cfg,
            latency_ms,
            tflops,
            time.time(),
            trial_id,
            status,
            reason,
            actual_method,
        )

    @staticmethod
    def _result_rank(result: TuningResult) -> tuple[int, int, float]:
        """Measured exact-workload rows always outrank analytical estimates."""

        return (
            int(result.status == "ok" and result.method == "measured"),
            int(result.status == "ok"),
            result.tflops,
        )

    def _consider_best(self, result: TuningResult) -> None:
        if (
            self._best is None
            or self._result_rank(result) > self._result_rank(self._best)
        ):
            self._best = result

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

    @property
    def warm_start_skipped(self) -> List[str]:
        """Why each cache row was skipped during :meth:`warm_start_from_cache`.

        A warm start that silently returns fewer rows than the cache holds is
        indistinguishable from a cold one. Every skip lands here with a reason —
        an unknown rasterization order (version skew) or an unusable row — so the
        loss is visible rather than inferred from a low hit count.
        """
        return list(self._warm_start_skipped)

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

    def tune(
        self, max_trials: int = 50, *, method: str = "reuse_distance"
    ) -> TuningResult:
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
        if method in {"on_device", "grid", "roofline", "reuse_distance"}:
            return self._tune_grid(max_trials, method=method)
        try:
            import optuna  # noqa: F401
            return self._tune_optuna(max_trials, method=method)
        except ImportError:
            return self._tune_grid(max_trials, method=method)

    def _tune_optuna(
        self, max_trials: int, *, method: str = "reuse_distance"
    ) -> TuningResult:
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
            self._consider_best(res)
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

    def _tune_grid(
        self, max_trials: int, *, method: str = "reuse_distance"
    ) -> TuningResult:
        """Deterministic grid search fallback."""
        trial_id = len(self._results)
        candidates = self.legal_candidates()
        if not candidates:
            raise ValueError("no legal GEMM tuning candidates were available")

        for cfg in candidates[:max_trials]:
            res = self._evaluate(cfg, trial_id, method=method)
            self._results.append(res)
            self._consider_best(res)
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
            method TEXT DEFAULT 'roofline',
            raster_order TEXT DEFAULT 'row_major',
            raster_group INT DEFAULT 1
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
        # Without these the cache would round-trip a tuned swizzle back to
        # row-major: the config would reload as the identity and look like the
        # swizzle simply did not help. Pre-existing rows migrate to the identity,
        # which is what they actually measured.
        "raster_order": "TEXT DEFAULT 'row_major'",
        "raster_group": "INT DEFAULT 1",
    }
    for name, ddl in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE tuning_results ADD COLUMN {name} {ddl}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tuning_lookup ON tuning_results(M, N, K, dtype, arch, layout)"
    )
