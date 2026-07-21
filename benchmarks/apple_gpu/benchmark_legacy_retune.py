"""Paired per-primitive Apple retuning corpus for APPLE-RETUNE-1.

Every family uses the same interleaved-trial schema. End-to-end and complete
Metal command-buffer intervals remain distinct; a multi-dispatch or mapped-
memory route leaves device timing explicitly unavailable instead of timing only
its final subdispatch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np

from tessera import runtime as rt
from tessera import _apple_gpu_backend as agb
from tessera._apple_gpu_dispatch import (
    clear_dispatch_telemetry,
    read_dispatch_telemetry,
    set_dispatch_telemetry_enabled,
)
from tessera.cache import SSMStateHandle
from tessera.cache.resident_kv import ResidentLatentKVCache
from tessera.compiler.apple_route_selector import (
    ROUTE_REPORT_SCHEMA_VERSION,
    aggregate_stable_route_reports,
    live_apple_route_context,
    seal_strict_route_ledger,
)


@dataclass(frozen=True)
class Route:
    name: str
    call: Callable[[], Any]
    native: bool
    complete_device_scope: bool
    api: str


@dataclass(frozen=True)
class Case:
    family: str
    op: str
    shape: str
    dtype: str
    incumbent: Route
    candidate: Route
    oracle: Any
    logical_input_bytes: int = 0
    logical_output_bytes: int = 0
    transport_class: str | None = None
    rtol: float = 5e-4
    atol: float = 5e-4


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _bytes(*values: np.ndarray) -> int:
    """Logical payload bytes, never a claim about private Metal allocation."""
    return sum(int(value.nbytes) for value in values)


def _cases(seed: int = 1701, *, scale: int = 1) -> list[Case]:
    if scale < 1:
        raise ValueError("retune shape scale must be positive")
    rng = np.random.default_rng(seed)

    def f(shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(rng.standard_normal(shape), dtype=np.float32)
    cases: list[Case] = []

    # Grouped GEMM: one routing-aware MSL dispatch versus per-expert MPS GEMMs.
    T, K, N, E = 32 * scale, 64 * scale, 64 * scale, 4
    x, w = f((T, K)), f((E, K, N))
    gs = np.full(E, T // E, np.int64)
    expert_ids = np.repeat(np.arange(E, dtype=np.int32), gs)
    grouped_oracle = np.concatenate([
        x[i * (T // E):(i + 1) * (T // E)] @ w[i] for i in range(E)
    ])

    def grouped_fused(gx=x, gw=w, gids=expert_ids) -> Any:
        return agb.gpu_grouped_gemm(gx, gw, gids)

    def grouped_per_expert(gx=x, gw=w, group_size=T // E, experts=E) -> Any:
        return np.concatenate([
            agb.gpu_matmul(
                np.ascontiguousarray(gx[i * group_size:(i + 1) * group_size]),
                np.ascontiguousarray(gw[i]),
            )
            for i in range(experts)
        ])

    cases.append(Case(
        "grouped_gemm", "retune_grouped_gemm", f"{T}x{K}x{N}_e{E}", "f32",
        Route("grouped_fused", grouped_fused, True, True, "MSL.grouped_gemm"),
        Route("per_expert_mps", grouped_per_expert, True, False, "MPS.matmul_loop"),
        grouped_oracle, _bytes(x, w, expert_ids), int(grouped_oracle.nbytes),
        "grouped_gemm_logical_io",
    ))

    # MoE SwiGLU: shipped composed route versus the opt-in single MSL kernel.
    T, K, H, M, E = 16 * scale, 32 * scale, 64 * scale, 32 * scale, 4
    x, wg, wu, wd = f((T, K)), f((E, K, H)), f((E, K, H)), f((E, H, M))
    gs = np.full(E, T // E, np.int64)
    expert_ids = np.repeat(np.arange(E, dtype=np.int32), gs)
    moe_parts = []
    for i in range(E):
        block = x[i * (T // E):(i + 1) * (T // E)]
        moe_parts.append((_silu(block @ wg[i]) * (block @ wu[i])) @ wd[i])
    moe_oracle = np.concatenate(moe_parts)

    def moe_composed(mx=x, mwg=wg, mwu=wu, mwd=wd, mgs=gs) -> Any:
        return rt._apple_gpu_dispatch_moe_swiglu_block(
            (mx, mwg, mwu, mwd, mgs), {"grouped_kind": "contiguous"}, np)

    def moe_fused(mx=x, mwg=wg, mwu=wu, mwd=wd, mids=expert_ids) -> Any:
        return agb.gpu_moe_swiglu_block(mx, mwg, mwu, mwd, mids)

    cases.append(Case(
        "moe", "retune_moe_swiglu", f"{T}x{K}x{H}x{M}_e{E}", "f32",
        Route("composed", moe_composed, True, False, "3xMSL.grouped+MSL.silu"),
        Route("single_fused", moe_fused, True, True, "MSL.moe_swiglu"),
        moe_oracle, _bytes(x, wg, wu, wd, gs), int(moe_oracle.nbytes),
        "grouped_swiglu_logical_io",
    ))

    # Reduction: records the native MPSGraph incumbent and the reference peer;
    # reference evidence can validate numerics but can never select production.
    reduce_x = f((64 * scale, 257 * scale))
    cases.append(Case(
        "reduction", "retune_reduce_sum", f"{64 * scale}x{257 * scale}_axis1", "f32",
        Route(
            "mpsgraph", lambda: rt._apple_gpu_dispatch_reduce(
                "tessera.reduce", [reduce_x], {"axis": 1}, np),
            True, True, "MPSGraph.reductionSum"),
        Route("numpy_reference", lambda: np.sum(reduce_x, axis=1),
              False, False, "reference_cpu"),
        np.sum(reduce_x, axis=1), int(reduce_x.nbytes),
        int(np.sum(reduce_x, axis=1).nbytes), "reduction_logical_io",
    ))

    # Contiguous KV movement is unified-memory residency, not a GPU dispatch.
    kv_seq, kv_latent, kv_rope = 128 * scale, 32 * scale, 8 * scale
    cache = ResidentLatentKVCache(latent_dim=kv_latent, rope_dim=kv_rope, max_seq=kv_seq)
    cache.append(f((kv_seq, kv_latent)), f((kv_seq, kv_rope)))
    cases.append(Case(
        "kv_movement", "retune_resident_kv_read", f"{kv_seq}x{kv_latent}x{kv_rope}", "f32",
        Route("resident_view", lambda: cache.latent_window().numpy(),
              False, False, "Metal.shared_buffer_view"),
        Route("host_copy", lambda: cache.latent_numpy().copy(),
              False, False, "reference_cpu_copy"),
        cache.latent_numpy().copy(), 0, int(cache.latent_numpy().nbytes),
        "resident_mapped_view",
    ))

    # MLA: latent absorbed route versus explicit K/V materialization.
    B, heads, sq, skv, dn, dr, dv, dl = 1, 4, 1, 64 * scale, 16 * scale, 8 * scale, 16 * scale, 32 * scale
    qn, qr = f((B, heads, sq, dn)), f((B, heads, sq, dr))
    ckv, kr = f((B, skv, dl)), f((B, skv, dr))
    wuk, wuv = f((heads, dl, dn)), f((heads, dl, dv))
    wuk_t = np.ascontiguousarray(np.transpose(wuk, (0, 2, 1)))
    positions_q = np.arange(sq, dtype=np.float32)[:, None]
    positions_k = np.arange(skv, dtype=np.float32)[:, None]
    inv = 1.0 / (10000.0 ** (np.arange(0, dr, 2, dtype=np.float32) / dr))
    cq, sqv = np.cos(positions_q * inv), np.sin(positions_q * inv)
    ck, sk = np.cos(positions_k * inv), np.sin(positions_k * inv)
    kn = np.einsum("bsl,hld->bhsd", ckv, wuk).astype(np.float32)
    values = np.einsum("bsl,hld->bhsd", ckv, wuv).astype(np.float32)

    def mla_absorbed() -> Any:
        return rt._apple_gpu_mla_absorb_decode(
            qn, qr, ckv, kr, wuk_t, wuv, cq, sqv, ck, sk, np)

    def mla_explicit() -> Any:
        return rt._apple_gpu_mla_decode_rope(
            qn, qr, kn, kr, values, cq, sqv, ck, sk, np)

    def rotate_interleaved(value: np.ndarray, cos: np.ndarray,
                           sin: np.ndarray) -> np.ndarray:
        out = np.empty_like(value)
        first, second = value[..., 0::2], value[..., 1::2]
        out[..., 0::2] = first * cos - second * sin
        out[..., 1::2] = first * sin + second * cos
        return out

    qrr = rotate_interleaved(qr, cq[None, None], sqv[None, None])
    krr = rotate_interleaved(kr, ck[None], sk[None])
    qfull = np.concatenate((qn, qrr), axis=-1)
    kfull = np.concatenate((
        kn, np.broadcast_to(krr[:, None], (B, heads, skv, dr))), axis=-1)
    scores = np.einsum("bhqd,bhkd->bhqk", qfull, kfull) / math.sqrt(dn + dr)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    mla_oracle = np.einsum("bhqk,bhkd->bhqd", weights, values)
    cases.append(Case(
        "mla_decode", "retune_mla_decode", f"{B}x{heads}x{sq}x{skv}x{dn}x{dr}x{dv}x{dl}", "f32",
        Route("explicit", mla_explicit, True, True, "MSL.mla_decode_rope"),
        Route("absorbed", mla_absorbed, True, True, "MSL.mla_absorb_decode"),
        mla_oracle, _bytes(qn, qr, ckv, kr, wuk_t, wuv, cq, sqv, ck, sk),
        int(mla_oracle.nbytes), "mla_logical_io",
    ))

    # Replay decode: block dispatch versus token-at-a-time output reconstruction.
    B, D, N, tokens = 1, 32 * scale, 16 * scale, 8 * scale
    a = -np.abs(f((D,)))
    delta = np.abs(f((tokens, B, D))) * 0.1
    xin, binp, cinp = f((tokens, B, D)), f((tokens, B, N)), f((tokens, B, N))
    reference = SSMStateHandle(B, D, N, a, capacity=16)
    decode_oracle = np.stack([
        reference.step(delta[i], xin[i], binp[i], cinp[i])
        for i in range(tokens)
    ])

    def fused_block() -> Any:
        handle = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=16)
        return handle.decode_block(delta, xin, binp, cinp)

    def token_loop() -> Any:
        handle = rt.apple_gpu_fused_ssm_state_handle(B, D, N, a, capacity=16)
        return np.stack([
            handle.step(delta[i], xin[i], binp[i], cinp[i])
            for i in range(tokens)
        ])

    cases.append(Case(
        "decode", "retune_replay_decode", f"{B}x{D}x{N}_t{tokens}", "f32",
        Route("fused_block", fused_block, True, True, "MSL.ssm_block_decode"),
        Route("token_loop", token_loop, True, False, "MSL.ssm_replay_loop"),
        decode_oracle, _bytes(delta, xin, binp, cinp, a), int(decode_oracle.nbytes),
        "replay_logical_io",
    ))
    return cases


def _low_precision_moe_cases(seed: int = 1701, *, scale: int = 1) -> list[Case]:
    """Owned raw-storage MoE route versus its storage-faithful CPU oracle.

    There is deliberately no invented multi-dispatch low-precision peer.  The
    raw f16/bf16 ABI is the production incumbent and owns one complete command
    buffer; the reference peer proves its numerical contract but is ineligible
    for selector admission.  This lets strict-v2 retain the real native route
    only after fresh paired evidence, rather than treating an f32 conversion as
    a competing low-precision implementation.
    """
    if scale < 1:
        raise ValueError("retune shape scale must be positive")
    rng = np.random.default_rng(seed)
    T, K, H, M, E = 16 * scale, 32 * scale, 64 * scale, 32 * scale, 4
    f32 = lambda shape: np.asarray(rng.standard_normal(shape), dtype=np.float32)
    base_x, base_wg = f32((T, K)), f32((E, K, H))
    base_wu, base_wd = f32((E, K, H)), f32((E, H, M))
    groups = np.full(E, T // E, np.int64)
    expert_ids = np.repeat(np.arange(E, dtype=np.int32), groups)
    dtypes: list[tuple[str, Any]] = [("f16", np.float16)]
    bf16 = rt._bfloat16_dtype()
    if bf16 is not None:
        dtypes.append(("bf16", bf16))
    cases: list[Case] = []
    for dtype_name, dtype in dtypes:
        x = np.ascontiguousarray(base_x.astype(dtype))
        wg = np.ascontiguousarray(base_wg.astype(dtype))
        wu = np.ascontiguousarray(base_wu.astype(dtype))
        wd = np.ascontiguousarray(base_wd.astype(dtype))

        def reference(mx=x, mwg=wg, mwu=wu, mwd=wd, mdtype=dtype) -> Any:
            parts = []
            for expert in range(E):
                start, stop = expert * (T // E), (expert + 1) * (T // E)
                block = mx[start:stop].astype(np.float32)
                gate = block @ mwg[expert].astype(np.float32)
                up = block @ mwu[expert].astype(np.float32)
                parts.append((_silu(gate) * up) @ mwd[expert].astype(np.float32))
            return np.ascontiguousarray(np.concatenate(parts).astype(mdtype))

        oracle = reference()

        def native(mx=x, mwg=wg, mwu=wu, mwd=wd, mids=expert_ids) -> Any:
            return agb.gpu_moe_swiglu_block_lowp(mx, mwg, mwu, mwd, mids)

        cases.append(Case(
            "moe", "retune_moe_swiglu_lowp", f"{T}x{K}x{H}x{M}_e{E}", dtype_name,
            Route("single_fused_lowp", native, True, True,
                  "MSL.moe_swiglu.raw16.complete_command_buffer"),
            Route("reference_cpu", reference, False, False,
                  "reference_cpu.storage_faithful"),
            oracle, _bytes(x, wg, wu, wd, groups), int(oracle.nbytes),
            "grouped_swiglu_logical_io", rtol=6e-2, atol=6e-2,
        ))
    return cases


def _measure(route: Route, oracle: Any, *, reps: int,
             rtol: float = 5e-4, atol: float = 5e-4) -> dict[str, Any]:
    wall: list[int] = []
    device: list[int] = []
    correct = True
    resources: Any = None
    for _ in range(reps):
        clear_dispatch_telemetry()
        started = time.perf_counter_ns()
        output = route.call()
        wall.append(time.perf_counter_ns() - started)
        correct = correct and bool(np.allclose(output, oracle, rtol=rtol, atol=atol))
        telemetry = read_dispatch_telemetry()
        device_time = telemetry.get("device_time_ns")
        if route.complete_device_scope and isinstance(device_time, int):
            device.append(device_time)
        if telemetry.get("resources") is not None:
            resources = telemetry["resources"]
    return {
        "wall_ns": int(statistics.median(wall)),
        "device_ns": int(statistics.median(device)) if len(device) == reps else None,
        "device_coverage": len(device) / reps,
        "correct": correct,
        "resources": resources or {"api": route.api},
    }


def _row(case: Case, route: Route, trials: list[dict[str, Any]], *,
         device: str, reps: int) -> dict[str, Any]:
    native = (route.native and rt.DeviceTensor.is_metal()
              and all(item["correct"] for item in trials))
    device_trials = [item["device_ns"] for item in trials]
    complete_device = all(value is not None for value in device_trials)
    end_to_end_ns = int(statistics.median(item["wall_ns"] for item in trials))
    logical_bytes = case.logical_input_bytes + case.logical_output_bytes
    return {
        "family": case.family,
        "op": case.op,
        "shape": case.shape,
        "dtype": case.dtype,
        "device": device,
        "route": route.name,
        "reps": reps,
        "native_dispatched": native,
        "numerically_validated": all(item["correct"] for item in trials),
        "execution_kind": "native_gpu" if native else "reference_cpu",
        "telemetry": {
            "end_to_end_median_ns": end_to_end_ns,
            "device_time_median_ns": (
                int(statistics.median(device_trials)) if complete_device else None),
            "paired_trial_end_to_end_medians_ns": [
                item["wall_ns"] for item in trials],
            "paired_trial_device_medians_ns": (
                device_trials if complete_device else None),
            "device_time_coverage": min(item["device_coverage"] for item in trials),
            "device_time_scope": (
                "complete_route_command_buffer" if route.complete_device_scope
                else "unavailable_multi_dispatch_or_mapped_memory"),
            "resources": trials[-1]["resources"],
            "transport": ({
                "class": case.transport_class,
                "logical_input_bytes": case.logical_input_bytes,
                "logical_output_bytes": case.logical_output_bytes,
                "logical_total_bytes": logical_bytes,
                "end_to_end_logical_bandwidth_gb_s": (
                    logical_bytes / end_to_end_ns if end_to_end_ns > 0 else None),
                "scope": "logical_host_visible_io_not_device_bandwidth",
            } if case.transport_class is not None else None),
        },
    }


def low_precision_candidate_status() -> list[dict[str, str]]:
    """Record the distinct grouped and MoE low-precision route states."""
    return [
        {"family": "grouped_gemm", "dtype": dtype,
         "status": "unsupported_no_owned_same_abi",
         "reason": "only tessera_apple_gpu_grouped_gemm_f32 is registered"}
        for dtype in ("f16", "bf16")
    ] + [
        {"family": "moe_swiglu", "dtype": dtype,
         "status": "owned_same_abi_ready_for_measurement",
         "reason": ("tessera_apple_gpu_moe_swiglu_" + dtype +
                    " owns raw 16-bit storage and one complete command buffer")}
        for dtype in ("f16", "bf16")
    ]


def run_report(*, reps: int, trials: int, seed: int = 1701,
               profile: str = "core") -> dict[str, Any]:
    if reps < 1 or trials < 3:
        raise ValueError("retune requires reps >= 1 and at least three paired trials")
    context = live_apple_route_context()
    rows: list[dict[str, Any]] = []
    set_dispatch_telemetry_enabled(True)
    try:
        if profile == "core":
            cases = _cases(seed)
        elif profile == "extended":
            # A second, independently seeded larger-shape pass.  It is not a
            # dtype claim: f16/bf16 are measured by ``low_precision`` below.
            cases = _cases(seed) + _cases(seed + 10_000, scale=2)
        elif profile == "low_precision":
            cases = (_low_precision_moe_cases(seed) +
                     _low_precision_moe_cases(seed + 10_000, scale=2))
        else:
            raise ValueError(f"unknown retune profile {profile!r}")
        for case in cases:
            case.incumbent.call()
            case.candidate.call()
            measured: dict[str, list[dict[str, Any]]] = {
                case.incumbent.name: [], case.candidate.name: []}
            for trial in range(trials):
                order: tuple[Route, ...] = (case.incumbent, case.candidate)
                if trial % 2:
                    order = tuple(reversed(order))
                for route in order:
                    measured[route.name].append(_measure(
                        route, case.oracle, reps=reps, rtol=case.rtol, atol=case.atol))
            rows.extend([
                _row(case, case.incumbent, measured[case.incumbent.name],
                     device=context.device, reps=reps),
                _row(case, case.candidate, measured[case.candidate.name],
                     device=context.device, reps=reps),
            ])
    finally:
        set_dispatch_telemetry_enabled(False)
    return {
        "schema_version": ROUTE_REPORT_SCHEMA_VERSION,
        "context": context.as_mapping(),
        "runs": rows,
        "low_precision_candidates": low_precision_candidate_status(),
    }


def build_strict_ledger(reports: list[dict[str, Any]], *, valid_days: int = 30) -> dict[str, Any]:
    incumbents = {
        "retune_grouped_gemm": "grouped_fused",
        "retune_moe_swiglu": "composed",
        "retune_moe_swiglu_lowp": "single_fused_lowp",
        "retune_reduce_sum": "mpsgraph",
        "retune_resident_kv_read": "resident_view",
        "retune_mla_decode": "explicit",
        "retune_replay_decode": "fused_block",
    }
    stable = aggregate_stable_route_reports(reports, incumbent_routes=incumbents)
    return seal_strict_route_ledger(stable, reports, valid_days=valid_days)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--profile", choices=("core", "extended", "low_precision"),
                        default="core")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    reports = [run_report(reps=args.reps, trials=args.trials, seed=1701 + run,
                          profile=args.profile)
               for run in range(args.runs)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": "tessera.apple.legacy-retune.v1",
        "reports": reports,
    }, indent=2) + "\n", encoding="utf-8")
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(json.dumps(build_strict_ledger(reports), indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
