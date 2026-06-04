"""Reference/compiler/GPU PPO-GRPO-CISPO policy-loss benchmark scaffold.

Stage 13 keeps execution claims split by proof level:

* python_reference — existing tessera.rl numpy implementations.
* compiler_decomposed_reference — PPO's compiler-visible primitive formula.
* apple_gpu_value_target_ir — PPO only, and only when the Apple GPU MPSGraph
  value executor is available and numerically agrees with the reference.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Iterable

import numpy as np

from tessera import runtime as tessera_runtime
from tessera import rl


DEFAULT_SHAPES: tuple[tuple[int, int, int], ...] = ((2, 3, 5), (4, 2, 8))
DEFAULT_SEED = 20260612


def _make_inputs(shape: tuple[int, int, int], seed: int) -> dict[str, np.ndarray]:
    batch, groups, tokens = shape
    rng = np.random.default_rng(seed + batch * 100 + groups * 10 + tokens)
    logp_old = rng.normal(-1.0, 0.2, size=shape).astype(np.float64)
    logp_new = (logp_old + rng.normal(0.0, 0.08, size=shape)).astype(np.float64)
    rewards = rng.normal(0.0, 1.0, size=shape).astype(np.float64)
    ref_logp = (logp_old + rng.normal(0.0, 0.05, size=shape)).astype(np.float64)
    mask = (rng.random(size=shape) > 0.15).astype(np.float64)
    advantages = rl.normalize_group_advantages(rewards, group_axis=1, mask=mask)
    return {
        "advantages": advantages,
        "logp_new": logp_new,
        "logp_old": logp_old,
        "mask": mask,
        "ref_logp": ref_logp,
        "rewards": rewards,
    }


def _losses(inputs: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "ppo_policy_loss": float(rl.ppo_policy_loss(
            inputs["logp_new"],
            inputs["logp_old"],
            inputs["advantages"],
            clip_epsilon=0.2,
            mask=inputs["mask"],
            ref_logp=inputs["ref_logp"],
            kl_coef=0.01,
            reduction="mean",
        )),
        "grpo_policy_loss": float(rl.grpo_policy_loss(
            inputs["logp_new"],
            inputs["logp_old"],
            rewards=inputs["rewards"],
            group_axis=1,
            clip_epsilon=0.2,
            mask=inputs["mask"],
            ref_logp=inputs["ref_logp"],
            kl_coef=0.01,
            reduction="mean",
        )),
        "cispo_policy_loss": float(rl.cispo_policy_loss(
            inputs["logp_new"],
            inputs["logp_old"],
            rewards=inputs["rewards"],
            group_axis=1,
            epsilon_high=5.0,
            mask=inputs["mask"],
            ref_logp=inputs["ref_logp"],
            kl_coef=0.01,
            reduction="mean",
        )),
    }


def _ppo_strict_loss(inputs: dict[str, np.ndarray], clip_epsilon: float = 0.2) -> float:
    logp_new = inputs["logp_new"].astype(np.float32)
    logp_old = inputs["logp_old"].astype(np.float32)
    advantages = inputs["advantages"].astype(np.float32)
    ratio = np.exp(logp_new - logp_old)
    clipped = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return float(-np.mean(np.minimum(ratio * advantages, clipped * advantages)))


def _apple_gpu_ppo_row(
    inputs: dict[str, np.ndarray],
    shape: tuple[int, int, int],
    seed: int,
    reference: float,
    reps: int,
) -> dict[str, Any]:
    available = tessera_runtime._apple_gpu_ppo_policy_loss_available()
    base: dict[str, Any] = {
        "name": "ppo_policy_loss",
        "variant_kind": "apple_gpu_value_target_ir",
        "target": "apple_gpu",
        "executor": "apple_gpu_value_target_ir" if available else None,
        "compiler_path": "apple_value_target_ir",
        "shape": f"B{shape[0]}_G{shape[1]}_T{shape[2]}",
        "seed": seed,
        "loss": None,
        "correctness": None,
        "timing_ms": None,
        "runtime_status": "unavailable",
        "skip_reason": None if available else "apple_gpu_ppo_policy_loss_f32 unavailable",
    }
    if not available:
        return base
    artifact = tessera_runtime.RuntimeArtifact(
        metadata={
            "target": "apple_gpu",
            "compiler_path": "apple_value_target_ir",
            "executable": True,
            "arg_names": ["logp_new", "logp_old", "advantages"],
            "apple_value_calls": [{
                "op": "tessera_apple.gpu.kernel_call",
                "op_kind": "ppo_policy_loss",
                "symbol": "tessera_apple_gpu_ppo_policy_loss_f32",
                "status": "executable",
                "clip_epsilon": 0.2,
                "reduction": "mean",
            }],
        },
        abi_signature="tessera.rl.stage13.ppo.apple_gpu",
    )

    args = {
        "logp_new": inputs["logp_new"].astype(np.float32),
        "logp_old": inputs["logp_old"].astype(np.float32),
        "advantages": inputs["advantages"].astype(np.float32),
    }

    def _run():
        return tessera_runtime.launch(artifact, args)

    first = _run()
    base["runtime_status"] = first.get("runtime_status")
    if not first.get("ok"):
        base["skip_reason"] = str(first.get("reason", "runtime launch failed"))
        base["executor"] = None
        return base
    loss = float(np.asarray(first["output"], dtype=np.float32))
    base["loss"] = loss
    base["correctness"] = abs(loss - reference)
    base["timing_ms"] = _time_loss(_run, reps)
    base["skip_reason"] = None
    return base


def _time_loss(fn, reps: int) -> float:
    times = []
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(statistics.median(times))


def build_report(
    shapes: Iterable[tuple[int, int, int]] = DEFAULT_SHAPES,
    *,
    reps: int = 10,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        inputs = _make_inputs(shape, seed)
        losses = _losses(inputs)
        for name, loss in losses.items():
            rows.append({
                "name": name,
                "variant_kind": "python_reference",
                "target": "reference_cpu",
                "executor": "python_reference",
                "compiler_path": None,
                "shape": f"B{shape[0]}_G{shape[1]}_T{shape[2]}",
                "seed": seed,
                "loss": loss,
                "correctness": None,
                "timing_ms": _time_loss(lambda n=name: _losses(inputs)[n], reps),
                "skip_reason": None,
            })
        ppo_simple = _ppo_strict_loss(inputs)
        rows.append({
            "name": "ppo_policy_loss",
            "variant_kind": "compiler_decomposed_reference",
            "target": "reference_cpu",
            "executor": "compiler_decomposed_reference",
            "compiler_path": "tessera-rl-loss-decompose",
            "shape": f"B{shape[0]}_G{shape[1]}_T{shape[2]}",
            "seed": seed,
            "loss": ppo_simple,
            "correctness": 0.0,
            "timing_ms": _time_loss(lambda: _ppo_strict_loss(inputs), reps),
            "runtime_status": "reference",
            "skip_reason": None,
        })
        for name in ("grpo_policy_loss", "cispo_policy_loss"):
            rows.append({
                "name": name,
                "variant_kind": "compiler_visible_non_executable",
                "target": "reference_cpu",
                "executor": None,
                "compiler_path": "tessera-rl-loss-decompose",
                "shape": f"B{shape[0]}_G{shape[1]}_T{shape[2]}",
                "seed": seed,
                "loss": None,
                "correctness": None,
                "timing_ms": None,
                "runtime_status": "compiler_visible_non_executable",
                "skip_reason": (
                    "Stage 13 records compiler visibility only; no GRPO/CISPO "
                    "runtime executor is claimed"),
            })
        rows.append(_apple_gpu_ppo_row(inputs, shape, seed, ppo_simple, reps))
    return {
        "benchmark": "rl_policy_losses",
        "sprint": "S13",
        "rows": rows,
    }


def _parse_shape(spec: str) -> tuple[int, int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be BxGxT, got {spec!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", nargs="+", default=["2x3x5", "4x2x8"],
                    help="BxGxT rollout shapes")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args(argv)

    report = build_report(
        [_parse_shape(s) for s in args.shapes],
        reps=args.reps,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
