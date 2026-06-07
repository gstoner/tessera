"""Reference LDT/model-primitive benchmark kernels.

The package intentionally starts as a reference plus artifact-visibility
benchmark.  It exercises the state shapes and primitive families needed by
Lattice Deduction Transformers, MOPD/OPD, Mamba-2, GQA, and Latent MoE without
claiming native compiler execution before oracle-backed fixtures exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmarks.common import (
    ArtifactLevels,
    BenchmarkOperator,
    BenchmarkRow,
    CompilerPath,
    Correctness,
    ExecutionKind,
    Profile,
    RuntimeStatus,
    telemetry_for_row,
)


LDT_PRIMITIVE_GAPS: tuple[str, ...] = (
    "candidate_axis_count_nonzero",
    "empty_cell_conflict",
    "singleton_cell_detection",
    "lattice_meet_bool_mask",
    "lattice_join_solution_alpha_or",
    "threshold_eliminate",
    "masked_categorical_sample",
    "branch_pin_update",
    "asymmetric_bce_per_cell_ce_loss",
)

MODEL_PRIMITIVE_GAPS: dict[str, tuple[str, ...]] = {
    "mopd_policy_loss_core": (
        "multi_teacher_weighted_kl",
        "importance_weight_clip",
        "rollout_batch_metadata",
    ),
    "mamba2_ssd_core": (
        "chunked_selective_scan",
        "ssd_state_carry",
        "causal_conv1d_lowering",
    ),
    "gqa_decode_core": (
        "repeat_free_grouped_kv_index",
        "kv_cache_append_read",
        "rope_grouped_attention_fusion",
    ),
    "latent_moe_core": (
        "latent_shared_down_up_projection",
        "topk_router_dispatch_combine",
        "grouped_gemm_or_segment_reduce",
    ),
}


@dataclass(frozen=True)
class LatticeReasoningConfig:
    """Shape and threshold configuration for one lattice benchmark run."""

    B: int = 1
    H: int = 4
    W: int = 4
    V: int = 4
    K: int = 3
    theta_elim: float = 0.1
    theta_cls: float = 0.6
    decide_temperature: float = 1.5
    seed: int = 20260607

    @property
    def shape_signature(self) -> str:
        return f"B{self.B}_H{self.H}_W{self.W}_V{self.V}_K{self.K}"


@dataclass(frozen=True)
class StepResult:
    lattice: np.ndarray
    conflict: bool
    solved: bool
    loss: float
    branch_cell: tuple[int, int, int] | None
    pinned_candidate: int | None
    eliminated_count: int


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-arr))


def masked_softmax(
    logits: np.ndarray,
    mask: np.ndarray,
    *,
    axis: int = -1,
    temperature: float = 1.0,
) -> np.ndarray:
    """Numerically stable softmax that returns zero outside ``mask``."""

    z = np.asarray(logits, dtype=np.float32) / float(temperature)
    m = np.asarray(mask, dtype=bool)
    if z.shape != m.shape:
        raise ValueError(f"logits shape {z.shape} does not match mask {m.shape}")
    neg_inf = np.full_like(z, -np.inf)
    masked = np.where(m, z, neg_inf)
    max_z = np.max(masked, axis=axis, keepdims=True)
    max_z = np.where(np.isfinite(max_z), max_z, 0.0)
    exp_z = np.where(m, np.exp(masked - max_z), 0.0)
    denom = np.sum(exp_z, axis=axis, keepdims=True)
    return np.divide(exp_z, denom, out=np.zeros_like(exp_z), where=denom > 0)


def candidate_counts(lattice: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    state = np.asarray(lattice, dtype=bool)
    counts = np.count_nonzero(state, axis=-1)
    if mask is None:
        return counts
    return np.where(np.asarray(mask, dtype=bool), counts, 0)


def lattice_meet(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    left = np.asarray(a, dtype=bool)
    right = np.asarray(b, dtype=bool)
    if left.shape != right.shape:
        raise ValueError(f"lattice shapes must match, got {left.shape} and {right.shape}")
    return np.logical_and(left, right)


def lattice_alpha(solutions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Return OR abstraction of solutions still consistent with ``state``."""

    sols = np.asarray(solutions, dtype=bool)
    st = np.asarray(state, dtype=bool)
    if sols.ndim != st.ndim + 1:
        raise ValueError("solutions must have leading K axis before lattice shape")
    if sols.shape[1:] != st.shape:
        raise ValueError(f"solution shape {sols.shape[1:]} does not match state {st.shape}")
    consistent = np.all(np.logical_or(~sols, st), axis=tuple(range(1, sols.ndim)))
    if not np.any(consistent):
        return np.zeros_like(st, dtype=bool)
    return np.any(sols[consistent], axis=0)


def threshold_eliminate(
    lattice: np.ndarray,
    candidate_logits: np.ndarray,
    *,
    theta_elim: float,
) -> np.ndarray:
    state = np.asarray(lattice, dtype=bool)
    probs = sigmoid(candidate_logits)
    if state.shape != probs.shape:
        raise ValueError(f"logits shape {probs.shape} does not match lattice {state.shape}")
    return np.logical_and(state, probs >= float(theta_elim))


def asymmetric_bce_loss(
    logits: np.ndarray,
    target: np.ndarray,
    *,
    w_pos: float = 8.0,
    w_neg: float = 1.0,
) -> float:
    probs = np.clip(sigmoid(logits), 1.0e-6, 1.0 - 1.0e-6)
    y = np.asarray(target, dtype=np.float32)
    loss = -(w_pos * y * np.log(probs) + w_neg * (1.0 - y) * np.log(1.0 - probs))
    return float(np.mean(loss))


def _per_cell_ce_loss(logits: np.ndarray, target: np.ndarray) -> float:
    one_hot_cells = np.count_nonzero(target, axis=-1) == 1
    if not np.any(one_hot_cells):
        return 0.0
    probs = masked_softmax(logits, np.ones_like(target, dtype=bool), axis=-1)
    chosen = np.sum(probs * target.astype(np.float32), axis=-1)
    return float(-np.mean(np.log(np.clip(chosen[one_hot_cells], 1.0e-6, 1.0))))


def _first_conflict_or_solved(state: np.ndarray, cls_logit: float, mask: np.ndarray, theta_cls: float) -> tuple[bool, bool]:
    counts = candidate_counts(state, mask)
    active = np.asarray(mask, dtype=bool)
    empty_conflict = bool(np.any((counts == 0) & active))
    cls_conflict = bool(float(sigmoid(cls_logit)) > float(theta_cls))
    conflict = empty_conflict or cls_conflict
    solved = bool(np.all((counts == 1) | ~active) and not conflict)
    return conflict, solved


def _branch_pin(
    state: np.ndarray,
    logits: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float,
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int] | None, int | None]:
    counts = candidate_counts(state, mask)
    cells = np.argwhere((counts >= 2) & np.asarray(mask, dtype=bool))
    if len(cells) == 0:
        return state, None, None
    cell = tuple(int(v) for v in cells[int(rng.integers(0, len(cells)))])
    alive = state[cell]
    probs = masked_softmax(logits[cell], alive, temperature=temperature)
    candidate = int(rng.choice(np.arange(state.shape[-1]), p=probs))
    out = state.copy()
    out[cell] = False
    out[cell + (candidate,)] = True
    return out, cell, candidate


def ldt_step(
    lattice: np.ndarray,
    candidate_logits: np.ndarray,
    cls_logit: float,
    *,
    solutions: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    theta_elim: float = 0.1,
    theta_cls: float = 0.6,
    decide_temperature: float = 1.5,
    rng: np.random.Generator | None = None,
) -> StepResult:
    """Run one deterministic LDT-style reference step."""

    state = np.asarray(lattice, dtype=bool)
    active_mask = np.ones(state.shape[:-1], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if active_mask.shape != state.shape[:-1]:
        raise ValueError(f"mask shape {active_mask.shape} does not match lattice cells {state.shape[:-1]}")
    eliminated = threshold_eliminate(state, candidate_logits, theta_elim=theta_elim)
    eliminated_count = int(np.count_nonzero(state) - np.count_nonzero(eliminated))
    conflict, solved = _first_conflict_or_solved(eliminated, cls_logit, active_mask, theta_cls)

    loss = 0.0
    if solutions is not None:
        alpha = lattice_alpha(solutions, state)
        if not np.any(alpha):
            conflict = True
        else:
            target = lattice_meet(state, alpha)
            loss = (
                asymmetric_bce_loss(candidate_logits, target)
                + 0.2 * _per_cell_ce_loss(candidate_logits, target)
            )

    branch_cell = None
    pinned_candidate = None
    out = eliminated
    if not conflict and not solved:
        out, branch_cell, pinned_candidate = _branch_pin(
            out,
            np.asarray(candidate_logits, dtype=np.float32),
            rng or np.random.default_rng(0),
            temperature=decide_temperature,
            mask=active_mask,
        )
    return StepResult(
        lattice=out,
        conflict=conflict,
        solved=solved,
        loss=loss,
        branch_cell=branch_cell,
        pinned_candidate=pinned_candidate,
        eliminated_count=eliminated_count,
    )


def _make_lattice_case(cfg: LatticeReasoningConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    lattice = np.ones((cfg.B, cfg.H, cfg.W, cfg.V), dtype=bool)
    mask = np.ones((cfg.B, cfg.H, cfg.W), dtype=bool)
    solutions = np.zeros((cfg.K, cfg.B, cfg.H, cfg.W, cfg.V), dtype=bool)
    for k in range(cfg.K):
        digits = (np.indices((cfg.B, cfg.H, cfg.W)).sum(axis=0) + k) % cfg.V
        solutions[k] = np.eye(cfg.V, dtype=bool)[digits]
    logits = np.full(lattice.shape, 2.5, dtype=np.float32)
    target = lattice_alpha(solutions, lattice)
    logits = np.where(target, logits, rng.normal(-2.0, 0.1, size=logits.shape)).astype(np.float32)
    return lattice, logits, solutions, mask


def mopd_policy_loss_core(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    teacher_weights: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    temperature: float = 1.0,
    clip_weight: float = 2.0,
) -> float:
    """Multi-teacher on-policy distillation loss over student samples."""

    student = np.asarray(student_logits, dtype=np.float32)
    teachers = np.asarray(teacher_logits, dtype=np.float32)
    weights = np.asarray(teacher_weights, dtype=np.float32)
    if teachers.shape[1:] != student.shape:
        raise ValueError("teacher logits must have leading teacher axis")
    weights = np.clip(weights / np.sum(weights), 0.0, float(clip_weight))
    teacher_probs = masked_softmax(teachers, np.ones_like(teachers, dtype=bool), temperature=temperature)
    mix = np.tensordot(weights, teacher_probs, axes=(0, 0))
    student_probs = np.clip(masked_softmax(student, np.ones_like(student, dtype=bool), temperature=temperature), 1.0e-6, 1.0)
    kl = mix * (np.log(np.clip(mix, 1.0e-6, 1.0)) - np.log(student_probs))
    per_token = np.sum(kl, axis=-1)
    if mask is None:
        return float(np.mean(per_token))
    m = np.asarray(mask, dtype=np.float32)
    return float(np.sum(per_token * m) / max(float(np.sum(m)), 1.0))


def mamba2_ssd_core(x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, gate: np.ndarray, *, chunk: int = 4) -> np.ndarray:
    """Small chunked selective-SSM/SSD recurrence reference."""

    xs = np.asarray(x, dtype=np.float32)
    A = np.asarray(a, dtype=np.float32)
    B = np.asarray(b, dtype=np.float32)
    C = np.asarray(c, dtype=np.float32)
    G = sigmoid(gate)
    if xs.shape != A.shape or xs.shape != B.shape or xs.shape != C.shape or xs.shape != G.shape:
        raise ValueError("x, a, b, c, and gate must share shape")
    h = np.zeros((xs.shape[0], xs.shape[2]), dtype=np.float32)
    ys = np.zeros_like(xs)
    for start in range(0, xs.shape[1], int(chunk)):
        stop = min(start + int(chunk), xs.shape[1])
        for t in range(start, stop):
            h = np.exp(np.clip(A[:, t, :], -8.0, 2.0)) * h + B[:, t, :] * xs[:, t, :]
            ys[:, t, :] = G[:, t, :] * C[:, t, :] * h
    return ys


def gqa_decode_core(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray) -> np.ndarray:
    """Repeat-free grouped-query decode attention."""

    qv = np.asarray(q, dtype=np.float32)
    k = np.asarray(k_cache, dtype=np.float32)
    v = np.asarray(v_cache, dtype=np.float32)
    if k.shape != v.shape:
        raise ValueError("k_cache and v_cache must share shape")
    B, Hq, D = qv.shape
    if k.shape[0] != B or k.shape[-1] != D or Hq % k.shape[2] != 0:
        raise ValueError("q heads must be divisible by kv heads")
    out = np.zeros_like(qv)
    scale = 1.0 / np.sqrt(float(D))
    group = Hq // k.shape[2]
    for h in range(Hq):
        kvh = h // group
        scores = np.einsum("bd,btd->bt", qv[:, h, :], k[:, :, kvh, :]) * scale
        probs = masked_softmax(scores, np.ones_like(scores, dtype=bool), axis=-1)
        out[:, h, :] = np.einsum("bt,btd->bd", probs, v[:, :, kvh, :])
    return out


def latent_moe_core(
    x: np.ndarray,
    down: np.ndarray,
    router: np.ndarray,
    experts: np.ndarray,
    up: np.ndarray,
    *,
    top_k: int = 2,
) -> np.ndarray:
    """Latent expert dispatch/combine with shared down/up projections."""

    tokens = np.asarray(x, dtype=np.float32)
    latent = tokens @ np.asarray(down, dtype=np.float32)
    scores = latent @ np.asarray(router, dtype=np.float32)
    k = min(int(top_k), scores.shape[-1])
    top = np.argpartition(scores, -k, axis=-1)[:, -k:]
    weights = np.take_along_axis(masked_softmax(scores, np.ones_like(scores, dtype=bool)), top, axis=-1)
    expert_w = np.asarray(experts, dtype=np.float32)
    combined = np.zeros_like(latent)
    for i in range(tokens.shape[0]):
        for slot, expert_id in enumerate(top[i]):
            combined[i] += weights[i, slot] * (latent[i] @ expert_w[int(expert_id)])
    return combined @ np.asarray(up, dtype=np.float32)


def _artifact_texts() -> dict[str, str]:
    graph = "\n".join((
        "func @lattice_reasoning_core(%state, %candidate_logits, %cls) {",
        "  %p = tessera.sigmoid %candidate_logits",
        "  %elim = tessera.where %p >= theta_elim, %state, false",
        "  %counts = tessera.count_nonzero %elim axis = candidate",
        "  %conflict = tessera.any %counts == 0 or %cls > theta_cls",
        "  %branch = tessera.masked_categorical_sample %candidate_logits, %elim",
        "}",
    ))
    schedule = "schedule @lattice_reasoning_core { recurrent_iterations = 16, tile = [4, 4] }"
    tile = "tile @lattice_reasoning_core { cell_tile = [4, 4], candidate_axis = 4 }"
    target = "target @lattice_reasoning_core { status = artifact_only, backend = graph_ir_only }"
    return {"graph_ir": graph, "schedule_ir": schedule, "tile_ir": tile, "target_ir": target}


def _artifact_hash(texts: dict[str, str]) -> str:
    return hashlib.sha256("\n".join(texts[k] for k in ("graph_ir", "schedule_ir", "tile_ir", "target_ir")).encode("utf-8")).hexdigest()


class LatticeReasoningBenchmark:
    """Reference benchmark harness for reasoning-model compiler primitives."""

    def __init__(self, *, warmup: int = 1, reps: int = 3):
        self.warmup = int(warmup)
        self.reps = int(reps)

    def run_lattice_step(self, cfg: LatticeReasoningConfig) -> tuple[StepResult, float]:
        lattice, logits, solutions, mask = _make_lattice_case(cfg)
        rng = np.random.default_rng(cfg.seed)
        for _ in range(self.warmup):
            ldt_step(
                lattice,
                logits,
                -4.0,
                solutions=solutions,
                mask=mask,
                theta_elim=cfg.theta_elim,
                theta_cls=cfg.theta_cls,
                decide_temperature=cfg.decide_temperature,
                rng=rng,
            )
        start = time.perf_counter()
        result = None
        for _ in range(max(self.reps, 1)):
            result = ldt_step(
                lattice,
                logits,
                -4.0,
                solutions=solutions,
                mask=mask,
                theta_elim=cfg.theta_elim,
                theta_cls=cfg.theta_cls,
                decide_temperature=cfg.decide_temperature,
                rng=rng,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(self.reps, 1)
        assert result is not None
        return result, elapsed_ms

    def rows(self, cfg: LatticeReasoningConfig) -> list[BenchmarkRow]:
        rows: list[BenchmarkRow] = []
        step, latency_ms = self.run_lattice_step(cfg)
        cells = cfg.B * cfg.H * cfg.W
        rows.append(BenchmarkRow(
            operator=BenchmarkOperator("lattice_reasoning_step", "bool/f32", cfg.shape_signature),
            compiler_path=CompilerPath.REFERENCE,
            runtime_status=RuntimeStatus.EXECUTABLE,
            correctness=Correctness(passed=bool(not step.conflict), tolerance=0.0),
            profile=Profile(cpu_wall_ms=latency_ms, memory_bytes=cfg.B * cfg.H * cfg.W * cfg.V),
            metrics={
                "cells": cells,
                "candidates": cfg.V,
                "eliminated_count": step.eliminated_count,
                "loss": step.loss,
                "branch_pinned": step.pinned_candidate is not None,
            },
            reason="numpy oracle reference for one LDT-style step",
            execution_kind=ExecutionKind.REFERENCE,
        ))

        rng = np.random.default_rng(cfg.seed + 17)
        rows.extend(self._model_primitive_rows(cfg, rng))

        texts = _artifact_texts()
        artifact_hash = _artifact_hash(texts)
        rows.append(BenchmarkRow(
            operator=BenchmarkOperator("lattice_reasoning_compiler_artifact", "bool/f32", cfg.shape_signature),
            compiler_path=CompilerPath.ARTIFACT_ONLY,
            runtime_status=RuntimeStatus.ARTIFACT_ONLY,
            artifact_levels=ArtifactLevels(
                graph=True,
                schedule=True,
                tile=True,
                target=True,
                artifact_hash=artifact_hash,
            ),
            metrics={
                "primitive_gaps": list(LDT_PRIMITIVE_GAPS),
                "artifact_hash_inputs": sorted(texts),
            },
            reason="compiler primitive targets are visible, but native lowering/execution is not claimed",
            execution_kind=ExecutionKind.ARTIFACT_ONLY,
        ))

        out: list[BenchmarkRow] = []
        for row in rows:
            out.append(row if row.telemetry else _with_telemetry(row, cfg))
        return out

    def _model_primitive_rows(self, cfg: LatticeReasoningConfig, rng: np.random.Generator) -> list[BenchmarkRow]:
        B, T, V = cfg.B, max(cfg.H, 4), cfg.V
        rows: list[BenchmarkRow] = []

        student = rng.normal(size=(B, T, V)).astype(np.float32)
        teachers = rng.normal(size=(3, B, T, V)).astype(np.float32)
        mask = (rng.random(size=(B, T)) > 0.1)
        teacher_weights = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        rows.append(_reference_metric_row(
            "mopd_policy_loss_core",
            "f32",
            cfg.shape_signature,
            lambda: mopd_policy_loss_core(student, teachers, teacher_weights, mask=mask),
            {"primitive_gaps": list(MODEL_PRIMITIVE_GAPS["mopd_policy_loss_core"])},
        ))

        x = rng.normal(size=(B, T, V)).astype(np.float32)
        rows.append(_reference_metric_row(
            "mamba2_ssd_core",
            "f32",
            cfg.shape_signature,
            lambda: mamba2_ssd_core(
                x,
                rng.normal(-0.2, 0.1, size=x.shape).astype(np.float32),
                rng.normal(size=x.shape).astype(np.float32),
                rng.normal(size=x.shape).astype(np.float32),
                rng.normal(size=x.shape).astype(np.float32),
            ),
            {"primitive_gaps": list(MODEL_PRIMITIVE_GAPS["mamba2_ssd_core"])},
        ))

        Hq, Hkv, D, S = 4, 2, max(V, 4), max(T, 4)
        rows.append(_reference_metric_row(
            "gqa_decode_core",
            "f32",
            f"B{B}_Hq{Hq}_Hkv{Hkv}_S{S}_D{D}",
            lambda: gqa_decode_core(
                rng.normal(size=(B, Hq, D)).astype(np.float32),
                rng.normal(size=(B, S, Hkv, D)).astype(np.float32),
                rng.normal(size=(B, S, Hkv, D)).astype(np.float32),
            ),
            {"primitive_gaps": list(MODEL_PRIMITIVE_GAPS["gqa_decode_core"])},
        ))

        tokens, hidden, latent, experts = max(cells := cfg.H * cfg.W, 4), max(V * 2, 8), max(V, 4), 4
        rows.append(_reference_metric_row(
            "latent_moe_core",
            "f32",
            f"T{tokens}_D{hidden}_L{latent}_E{experts}",
            lambda: latent_moe_core(
                rng.normal(size=(tokens, hidden)).astype(np.float32),
                rng.normal(size=(hidden, latent)).astype(np.float32),
                rng.normal(size=(latent, experts)).astype(np.float32),
                rng.normal(size=(experts, latent, latent)).astype(np.float32),
                rng.normal(size=(latent, hidden)).astype(np.float32),
            ),
            {
                "cells": cells,
                "primitive_gaps": list(MODEL_PRIMITIVE_GAPS["latent_moe_core"]),
            },
        ))
        return rows

    def to_json(self, rows: Sequence[BenchmarkRow], path: str | Path) -> None:
        Path(path).write_text(json.dumps([row.to_dict() for row in rows], indent=2) + "\n")


def _reference_metric_row(
    name: str,
    dtype: str,
    shape: str,
    fn,
    metrics: dict[str, Any],
) -> BenchmarkRow:
    start = time.perf_counter()
    out = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    arr = np.asarray(out)
    return BenchmarkRow(
        operator=BenchmarkOperator(name, dtype, shape),
        compiler_path=CompilerPath.REFERENCE,
        runtime_status=RuntimeStatus.EXECUTABLE,
        correctness=Correctness(passed=bool(np.all(np.isfinite(arr))), tolerance=0.0),
        profile=Profile(cpu_wall_ms=elapsed_ms, memory_bytes=int(arr.size * max(arr.itemsize, 1))),
        metrics={
            "output_shape": list(arr.shape),
            "output_checksum": float(np.sum(arr, dtype=np.float64)),
            **metrics,
        },
        reason="numpy oracle reference for model primitive core",
        execution_kind=ExecutionKind.REFERENCE,
    )


def _with_telemetry(row: BenchmarkRow, cfg: LatticeReasoningConfig) -> BenchmarkRow:
    telemetry = telemetry_for_row(
        row,
        source="lattice_reasoning_core",
        metadata={"benchmark_shape": cfg.shape_signature},
    )
    return BenchmarkRow(
        operator=row.operator,
        compiler_path=row.compiler_path,
        runtime_status=row.runtime_status,
        artifact_levels=row.artifact_levels,
        correctness=row.correctness,
        profile=row.profile,
        metrics=row.metrics,
        telemetry=telemetry,
        reason=row.reason,
        execution_kind=row.execution_kind,
    )


def build_report(*, smoke: bool = False, reps: int = 3, seed: int = 20260607) -> dict[str, Any]:
    cfg = LatticeReasoningConfig(
        H=4 if smoke else 9,
        W=4 if smoke else 9,
        V=4 if smoke else 9,
        K=3 if smoke else 1,
        seed=seed,
    )
    bench = LatticeReasoningBenchmark(warmup=0 if smoke else 1, reps=reps)
    rows = bench.rows(cfg)
    return {
        "benchmark": "lattice_reasoning_core",
        "mode": "smoke" if smoke else "main",
        "config": {
            "B": cfg.B,
            "H": cfg.H,
            "W": cfg.W,
            "V": cfg.V,
            "K": cfg.K,
            "theta_elim": cfg.theta_elim,
            "theta_cls": cfg.theta_cls,
            "decide_temperature": cfg.decide_temperature,
            "seed": cfg.seed,
        },
        "rows": [row.to_dict() for row in rows],
    }
