"""Current-compiler LDT/model-primitive benchmark kernels.

The package emits three proof levels:

* NumPy reference rows for correctness-bearing oracle timing.
* Tessera primitive rows that call the public compiler/runtime API.
* Apple GPU executable rows when `@jit(target="apple_gpu")` reaches the Metal
  runtime and numerically agrees with the oracle.
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

LANDED_LDT_PRIMITIVES: tuple[str, ...] = (
    "count_nonzero",
    "popcount",
    "masked_categorical",
    "asymmetric_bce",
)

APPLE_GPU_EXECUTABLE_MODEL_PRIMITIVES: tuple[str, ...] = (
    "count_nonzero",
    "popcount",
    "masked_categorical",
    "asymmetric_bce",
    "selective_ssm_scalar_A",
    "grouped_gemm_fused",
    "z_loss",
    "load_balance_loss",
)

REGISTRY_MISMATCH_NOTES: tuple[str, ...] = (
    "capabilities.supports_op currently reports artifact_only for several Apple GPU ops that file-based tests execute through metal_runtime",
    "benchmark rows treat observed metal_runtime execution as stronger evidence than the conservative registry status",
)


def _import_tessera():
    try:
        import tessera as ts  # type: ignore
        import tessera.losses as losses  # type: ignore
    except Exception:
        return None, None
    return ts, losses


def _apple_gpu_available() -> bool:
    try:
        from tessera import _apple_gpu_backend as agb  # type: ignore
        from tessera import _jit_boundary as jb  # type: ignore
    except Exception:
        return False
    return bool(agb.is_available() and jb.is_available())


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
        "  %branch = tessera.masked_categorical %candidate_logits, %elim",
        "  %loss = tessera.loss.asymmetric_bce %candidate_logits, %target",
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
        lattice, logits, solutions, mask = _make_lattice_case(cfg)
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
        rows.extend(self._ldt_primitive_rows(cfg, lattice, logits, solutions, mask))
        rows.extend(self._model_primitive_rows(cfg, rng))
        rows.extend(self._apple_gpu_executable_rows(cfg, rng))

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
                "landed_primitives": list(LANDED_LDT_PRIMITIVES),
                "remaining_integrated_step_work": list(LDT_PRIMITIVE_GAPS),
                "registry_mismatch_notes": list(REGISTRY_MISMATCH_NOTES),
                "artifact_hash_inputs": sorted(texts),
            },
            reason="integrated LDT step remains artifact-only; individual primitives have stronger rows",
            execution_kind=ExecutionKind.ARTIFACT_ONLY,
        ))

        out: list[BenchmarkRow] = []
        for row in rows:
            out.append(row if row.telemetry else _with_telemetry(row, cfg))
        return out

    def _ldt_primitive_rows(
        self,
        cfg: LatticeReasoningConfig,
        lattice: np.ndarray,
        logits: np.ndarray,
        solutions: np.ndarray,
        mask: np.ndarray,
    ) -> list[BenchmarkRow]:
        target = lattice_meet(lattice, lattice_alpha(solutions, lattice)).astype(np.float32)
        live = lattice.astype(np.float32)
        bitmask = np.packbits(lattice.astype(np.uint8), axis=-1).astype(np.int64)
        return [
            _tessera_metric_row(
                "ldt_count_nonzero_tessera",
                "f32",
                cfg.shape_signature,
                lambda: _tessera_count_nonzero(live),
                candidate_counts(lattice, mask),
                {"primitive": "count_nonzero", "axis": -1},
            ),
            _tessera_metric_row(
                "ldt_popcount_tessera",
                "i64",
                cfg.shape_signature,
                lambda: _tessera_popcount(bitmask),
                _popcount_ref(bitmask),
                {"primitive": "popcount", "encoding": "packed_candidate_bits"},
            ),
            _tessera_metric_row(
                "ldt_masked_categorical_tessera",
                "f32",
                cfg.shape_signature,
                lambda: _tessera_masked_categorical(logits, lattice.astype(np.int32)),
                np.argmax(np.where(lattice, logits, -np.inf), axis=-1),
                {"primitive": "masked_categorical", "mode": "greedy"},
            ),
            _tessera_metric_row(
                "ldt_asymmetric_bce_tessera",
                "f32",
                cfg.shape_signature,
                lambda: _tessera_asymmetric_bce(logits, target),
                asymmetric_bce_loss(logits, target, w_pos=8.0, w_neg=1.0),
                {"primitive": "asymmetric_bce", "pos_weight": 8.0, "neg_weight": 1.0},
            ),
        ]

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

    def _apple_gpu_executable_rows(self, cfg: LatticeReasoningConfig, rng: np.random.Generator) -> list[BenchmarkRow]:
        if not _apple_gpu_available():
            return [_skipped_row(
                "apple_gpu_current_compiler_primitives",
                "f32",
                cfg.shape_signature,
                reason="apple_gpu runtime or libtessera_jit unavailable",
            )]

        rows: list[BenchmarkRow] = []
        lattice, logits, solutions, _mask = _make_lattice_case(cfg)
        target = lattice_meet(lattice, lattice_alpha(solutions, lattice)).astype(np.float32)
        live = lattice.astype(np.float32)
        bitmask = np.packbits(lattice.astype(np.uint8), axis=-1).astype(np.int64)
        rows.extend([
            _apple_gpu_metric_row(
                "apple_gpu_ldt_count_nonzero",
                "f32",
                cfg.shape_signature,
                lambda: _apple_gpu_count_nonzero(live),
                candidate_counts(lattice),
                {"primitive": "count_nonzero"},
            ),
            _apple_gpu_metric_row(
                "apple_gpu_ldt_popcount",
                "i64",
                cfg.shape_signature,
                lambda: _apple_gpu_popcount(bitmask),
                _popcount_ref(bitmask),
                {"primitive": "popcount"},
            ),
            _apple_gpu_metric_row(
                "apple_gpu_ldt_masked_categorical",
                "f32",
                cfg.shape_signature,
                lambda: _apple_gpu_masked_categorical(logits, lattice.astype(np.int32)),
                np.argmax(np.where(lattice, logits, -np.inf), axis=-1),
                {"primitive": "masked_categorical", "mode": "greedy"},
            ),
            _apple_gpu_metric_row(
                "apple_gpu_ldt_asymmetric_bce",
                "f32",
                cfg.shape_signature,
                lambda: _apple_gpu_asymmetric_bce(logits, target),
                asymmetric_bce_loss(logits, target, w_pos=8.0, w_neg=1.0),
                {"primitive": "asymmetric_bce"},
            ),
        ])

        x, A, Bm, C, delta = _selective_ssm_inputs(rng, B=cfg.B, S=max(cfg.H * 2, 8), D=max(cfg.V, 4), N=3)
        ts, _losses = _import_tessera()
        ssm_ref = np.asarray(ts.ops.selective_ssm(x, A, Bm, C, delta)) if ts is not None else np.zeros_like(x)
        rows.append(_apple_gpu_metric_row(
            "apple_gpu_mamba2_selective_ssm",
            "f32",
            f"B{cfg.B}_S{max(cfg.H * 2, 8)}_D{max(cfg.V, 4)}_N3",
            lambda: _apple_gpu_selective_ssm(x, A, Bm, C, delta),
            ssm_ref,
            {"primitive": "selective_ssm", "scope": "scalar_A", "execution_mode_expected": "metal_runtime"},
        ))

        gx, gw, group_sizes = _grouped_gemm_inputs(rng)
        grouped_ref = _grouped_gemm_ref(gx, gw, group_sizes)
        rows.append(_apple_gpu_metric_row(
            "apple_gpu_grouped_gemm_fused",
            "f32",
            f"T{gx.shape[0]}_K{gx.shape[1]}_N{gw.shape[2]}_E{gw.shape[0]}",
            lambda: _apple_gpu_grouped_gemm(gx, gw, group_sizes),
            grouped_ref,
            {"primitive": "grouped_gemm", "path": "fused_msl_one_dispatch"},
        ))

        router_logits = rng.normal(size=(32, 4)).astype(np.float32)
        router_probs = np.exp(router_logits - router_logits.max(axis=-1, keepdims=True))
        router_probs /= router_probs.sum(axis=-1, keepdims=True)
        if _losses is not None:
            rows.append(_apple_gpu_metric_row(
                "apple_gpu_moe_z_loss",
                "f32",
                "T32_E4",
                lambda: _apple_gpu_z_loss(router_logits),
                _losses.z_loss(router_logits),
                {"primitive": "z_loss"},
            ))
            rows.append(_apple_gpu_metric_row(
                "apple_gpu_moe_load_balance_loss",
                "f32",
                "T32_E4",
                lambda: _apple_gpu_load_balance_loss(router_probs.astype(np.float32)),
                _losses.load_balance_loss(router_probs),
                {"primitive": "load_balance_loss"},
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


def _popcount_ref(x: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda v: int(v).bit_count())(np.asarray(x))


def _tessera_count_nonzero(x: np.ndarray) -> np.ndarray:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")
    return np.asarray(ts.ops.count_nonzero(x, axis=-1))


def _tessera_popcount(x: np.ndarray) -> np.ndarray:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")
    return np.asarray(ts.ops.popcount(x))


def _tessera_masked_categorical(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")
    return np.asarray(ts.ops.masked_categorical(logits, mask))


def _tessera_asymmetric_bce(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")
    return np.asarray(ts.ops.asymmetric_bce(logits, target, pos_weight=8.0, neg_weight=1.0))


def _selective_ssm_inputs(
    rng: np.random.Generator,
    *,
    B: int,
    S: int,
    D: int,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(size=(B, S, D)).astype(np.float32)
    A = (-np.abs(rng.normal(size=D)) - 0.1).astype(np.float32)
    Bm = rng.normal(size=(B, S, N)).astype(np.float32)
    C = rng.normal(size=(B, S, N)).astype(np.float32)
    delta = (np.abs(rng.normal(size=(B, S, D))) * 0.25).astype(np.float32)
    return x, A, Bm, C, delta


def _grouped_gemm_inputs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_sizes = np.array([5, 3, 4], dtype=np.int64)
    x = rng.normal(size=(int(group_sizes.sum()), 8)).astype(np.float32)
    weights = rng.normal(size=(len(group_sizes), 8, 6)).astype(np.float32)
    return x, weights, group_sizes


def _grouped_gemm_ref(x: np.ndarray, weights: np.ndarray, group_sizes: np.ndarray) -> np.ndarray:
    out = np.zeros((x.shape[0], weights.shape[2]), dtype=np.float32)
    offset = 0
    for expert in range(weights.shape[0]):
        n = int(group_sizes[expert])
        out[offset:offset + n] = x[offset:offset + n] @ weights[expert]
        offset += n
    return out


def _apple_gpu_count_nonzero(x: np.ndarray) -> tuple[np.ndarray, str]:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")

    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.count_nonzero(x, axis=-1)

    out = np.asarray(f(x))
    return out, str(f.runtime_artifact().metadata.get("execution_mode"))


def _apple_gpu_popcount(x: np.ndarray) -> tuple[np.ndarray, str]:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")

    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.popcount(x)

    out = np.asarray(f(x))
    return out, str(f.runtime_artifact().metadata.get("execution_mode"))


def _apple_gpu_masked_categorical(logits: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, str]:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")

    @ts.jit(target="apple_gpu")
    def f(logits, mask):
        return ts.ops.masked_categorical(logits, mask)

    out = np.asarray(f(logits, mask))
    return out, str(f.runtime_artifact().metadata.get("execution_mode"))


def _apple_gpu_asymmetric_bce(logits: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, str]:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")

    @ts.jit(target="apple_gpu")
    def f(logits, target):
        return ts.ops.asymmetric_bce(logits, target, pos_weight=8.0, neg_weight=1.0)

    out = np.asarray(f(logits, target))
    return out, str(f.runtime_artifact().metadata.get("execution_mode"))


def _apple_gpu_selective_ssm(
    x: np.ndarray,
    A: np.ndarray,
    Bm: np.ndarray,
    C: np.ndarray,
    delta: np.ndarray,
) -> tuple[np.ndarray, str]:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")

    @ts.jit(target="apple_gpu")
    def f(x, A, Bm, C, delta):
        return ts.ops.selective_ssm(x, A, Bm, C, delta)

    out = np.asarray(f(x, A, Bm, C, delta))
    return out, str(f.runtime_artifact().metadata.get("execution_mode"))


def _apple_gpu_grouped_gemm(
    x: np.ndarray,
    weights: np.ndarray,
    group_sizes: np.ndarray,
) -> tuple[np.ndarray, str]:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")

    @ts.jit(target="apple_gpu")
    def f(x, weights, group_sizes):
        return ts.ops.grouped_gemm(x, weights, group_sizes)

    out = np.asarray(f(x, weights, group_sizes))
    return out, str(f.runtime_artifact().metadata.get("execution_mode"))


def _apple_gpu_z_loss(router_logits: np.ndarray) -> tuple[np.ndarray, str]:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")

    @ts.jit(target="apple_gpu")
    def f(router_logits):
        return ts.ops.z_loss(router_logits)

    out = np.asarray(f(router_logits))
    return out, str(f.runtime_artifact().metadata.get("execution_mode"))


def _apple_gpu_load_balance_loss(router_probs: np.ndarray) -> tuple[np.ndarray, str]:
    ts, _losses = _import_tessera()
    if ts is None:
        raise RuntimeError("tessera import failed")

    @ts.jit(target="apple_gpu")
    def f(router_probs):
        return ts.ops.load_balance_loss(router_probs)

    out = np.asarray(f(router_probs))
    return out, str(f.runtime_artifact().metadata.get("execution_mode"))


def _tessera_metric_row(
    name: str,
    dtype: str,
    shape: str,
    fn,
    expected: np.ndarray | float,
    metrics: dict[str, Any],
) -> BenchmarkRow:
    start = time.perf_counter()
    out = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    arr = np.asarray(out)
    exp = np.asarray(expected)
    passed = bool(np.allclose(arr, exp, rtol=1e-5, atol=1e-6))
    return BenchmarkRow(
        operator=BenchmarkOperator(name, dtype, shape),
        compiler_path=CompilerPath.REFERENCE,
        runtime_status=RuntimeStatus.EXECUTABLE,
        correctness=Correctness(
            max_error=float(np.max(np.abs(arr.astype(np.float64) - exp.astype(np.float64)))) if arr.size and exp.size else 0.0,
            tolerance=1.0e-5,
            passed=passed,
        ),
        profile=Profile(cpu_wall_ms=elapsed_ms, memory_bytes=int(arr.size * max(arr.itemsize, 1))),
        metrics={
            "primitive_api": "tessera.ops",
            "output_shape": list(arr.shape),
            "output_checksum": float(np.sum(arr, dtype=np.float64)),
            **metrics,
        },
        reason="public Tessera primitive API reference/runtime row",
        execution_kind=ExecutionKind.REFERENCE,
    )


def _apple_gpu_metric_row(
    name: str,
    dtype: str,
    shape: str,
    fn,
    expected: np.ndarray | float,
    metrics: dict[str, Any],
) -> BenchmarkRow:
    try:
        start = time.perf_counter()
        raw = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if isinstance(raw, tuple) and len(raw) == 2:
            out, execution_mode = raw
        else:
            out, execution_mode = raw, "unknown"
        arr = np.asarray(out)
        exp = np.asarray(expected)
        max_error = float(np.max(np.abs(arr.astype(np.float64) - exp.astype(np.float64)))) if arr.size and exp.size else 0.0
        passed = bool(np.allclose(arr, exp, rtol=1e-3, atol=1e-4))
        executable = execution_mode == "metal_runtime" and passed
        return BenchmarkRow(
            operator=BenchmarkOperator(name, dtype, shape, target="apple_gpu"),
            compiler_path=CompilerPath.TESSERA_JIT_APPLE_GPU,
            runtime_status=RuntimeStatus.EXECUTABLE if executable else RuntimeStatus.SKIPPED,
            correctness=Correctness(max_error=max_error, tolerance=1.0e-3, passed=passed),
            profile=Profile(kernel_elapsed_ms=elapsed_ms, memory_bytes=int(arr.size * max(arr.itemsize, 1))),
            metrics={
                "execution_mode": execution_mode,
                "observed_native_execution": executable,
                "output_shape": list(arr.shape),
                "output_checksum": float(np.sum(arr, dtype=np.float64)),
                **metrics,
            },
            reason="Apple GPU @jit row reached metal_runtime" if executable else "Apple GPU row did not prove metal_runtime execution",
            execution_kind=ExecutionKind.OPTIMIZED_NATIVE if executable else ExecutionKind.ARTIFACT_ONLY,
        )
    except Exception as exc:
        return _skipped_row(name, dtype, shape, reason=str(exc), metrics=metrics)


def _skipped_row(
    name: str,
    dtype: str,
    shape: str,
    *,
    reason: str,
    metrics: dict[str, Any] | None = None,
) -> BenchmarkRow:
    return BenchmarkRow(
        operator=BenchmarkOperator(name, dtype, shape),
        compiler_path=CompilerPath.RUNTIME_UNAVAILABLE,
        runtime_status=RuntimeStatus.SKIPPED,
        correctness=Correctness(passed=None),
        metrics=dict(metrics or {}),
        reason=reason,
        execution_kind=ExecutionKind.ARTIFACT_ONLY,
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
