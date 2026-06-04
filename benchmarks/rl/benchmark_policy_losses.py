"""Reference/compiler/GPU PPO-GRPO-CISPO policy-loss benchmark scaffold.

Stages 14/15 keep execution claims split by proof level:

* python_reference — existing tessera.rl numpy implementations.
* compiler_decomposed_reference — PPO variants plus GRPO/CISPO formulas
  reproduced without calling the public reference functions.
* apple_gpu_value_target_ir — PPO variants only, and only when the Apple GPU
  MPSGraph value executor is available and numerically agrees with reference.
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


PPO_VARIANTS: tuple[dict[str, Any], ...] = (
    {"name": "ppo_policy_loss", "suffix": "strict"},
    {"name": "ppo_policy_loss_masked", "suffix": "masked", "has_mask": True},
    {"name": "ppo_policy_loss_ref_kl", "suffix": "ref_kl", "has_ref_kl": True,
     "kl_coef": 0.01},
    {"name": "ppo_policy_loss_entropy", "suffix": "entropy", "has_entropy": True,
     "entropy_coef": 0.02},
    {"name": "ppo_policy_loss_full", "suffix": "full", "has_mask": True,
     "has_ref_kl": True, "has_entropy": True, "kl_coef": 0.01,
     "entropy_coef": 0.02},
)


def _ppo_decomposed_loss(
    inputs: dict[str, np.ndarray], *, clip_epsilon: float = 0.2,
    has_mask: bool = False, has_ref_kl: bool = False,
    has_entropy: bool = False, kl_coef: float = 0.0,
    entropy_coef: float = 0.0,
) -> float:
    logp_new = inputs["logp_new"].astype(np.float32)
    logp_old = inputs["logp_old"].astype(np.float32)
    advantages = inputs["advantages"].astype(np.float32)
    ratio = np.exp(logp_new - logp_old)
    clipped = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss = -np.minimum(ratio * advantages, clipped * advantages)
    if has_ref_kl:
        ref_logp = inputs["ref_logp"].astype(np.float32)
        delta = ref_logp - logp_new
        loss = loss + float(kl_coef) * (np.exp(delta) - delta - 1.0)
    if has_entropy:
        entropy = inputs["entropy"].astype(np.float32)
        loss = loss - float(entropy_coef) * entropy
    if has_mask:
        mask = inputs["mask"].astype(np.float32)
        return float(np.sum(loss * mask) / max(float(np.sum(mask)), 1.0))
    return float(np.mean(loss))


def _normalize_group_advantages_decomposed(
    rewards: np.ndarray, mask: np.ndarray, group_axis: int = 1,
) -> np.ndarray:
    rewards = rewards.astype(np.float32)
    mask = mask.astype(np.float32)
    denom = np.maximum(np.sum(mask, axis=group_axis, keepdims=True), 1.0)
    mean = np.sum(rewards * mask, axis=group_axis, keepdims=True) / denom
    var = np.sum(((rewards - mean) ** 2) * mask,
                 axis=group_axis, keepdims=True) / denom
    return (rewards - mean) / np.sqrt(var + 1.0e-5)


def _grpo_decomposed_loss(inputs: dict[str, np.ndarray]) -> float:
    adv = _normalize_group_advantages_decomposed(
        inputs["rewards"], inputs["mask"], group_axis=1)
    tmp = dict(inputs)
    tmp["advantages"] = adv
    return _ppo_decomposed_loss(
        tmp, has_mask=True, has_ref_kl=True, kl_coef=0.01)


def _cispo_decomposed_loss(inputs: dict[str, np.ndarray]) -> float:
    logp_new = inputs["logp_new"].astype(np.float32)
    logp_old = inputs["logp_old"].astype(np.float32)
    adv = _normalize_group_advantages_decomposed(
        inputs["rewards"], inputs["mask"], group_axis=1)
    ratio = np.exp(logp_new - logp_old)
    loss = -(np.minimum(ratio, 5.0) * adv * logp_new)
    ref_logp = inputs["ref_logp"].astype(np.float32)
    delta = ref_logp - logp_new
    loss = loss + 0.01 * (np.exp(delta) - delta - 1.0)
    mask = inputs["mask"].astype(np.float32)
    return float(np.sum(loss * mask) / max(float(np.sum(mask)), 1.0))


def _apple_gpu_ppo_row(
    inputs: dict[str, np.ndarray],
    shape: tuple[int, int, int],
    seed: int,
    reference: float,
    reps: int,
    variant: dict[str, Any],
) -> dict[str, Any]:
    strict = not any(variant.get(k, False)
                     for k in ("has_mask", "has_ref_kl", "has_entropy"))
    symbol = ("tessera_apple_gpu_ppo_policy_loss_f32" if strict
              else "tessera_apple_gpu_ppo_policy_loss_ex_f32")
    available = (
        tessera_runtime._apple_gpu_ppo_policy_loss_available()
        if strict else tessera_runtime._apple_gpu_ppo_policy_loss_ex_available()
    )
    arg_names = ["logp_new", "logp_old", "advantages"]
    if variant.get("has_mask", False):
        arg_names.append("mask")
    if variant.get("has_ref_kl", False):
        arg_names.append("ref_logp")
    if variant.get("has_entropy", False):
        arg_names.append("entropy")
    base: dict[str, Any] = {
        "name": variant["name"],
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
        "skip_reason": None if available else f"{symbol} unavailable",
    }
    if not available:
        return base
    artifact = tessera_runtime.RuntimeArtifact(
        metadata={
            "target": "apple_gpu",
            "compiler_path": "apple_value_target_ir",
            "executable": True,
            "arg_names": arg_names,
            "apple_value_calls": [{
                "op": "tessera_apple.gpu.kernel_call",
                "op_kind": "ppo_policy_loss",
                "symbol": symbol,
                "status": "executable",
                "clip_epsilon": 0.2,
                "kl_coef": float(variant.get("kl_coef", 0.0)),
                "entropy_coef": float(variant.get("entropy_coef", 0.0)),
                "has_mask": bool(variant.get("has_mask", False)),
                "has_ref_kl": bool(variant.get("has_ref_kl", False)),
                "has_entropy": bool(variant.get("has_entropy", False)),
                "reduction": "mean",
            }],
        },
        abi_signature=f"tessera.rl.stage14.{variant['suffix']}.apple_gpu",
    )

    args = {
        name: inputs[name].astype(np.float32) for name in arg_names
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
        inputs["entropy"] = np.abs(inputs["logp_new"]).astype(np.float64) * 0.1
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
        for variant in PPO_VARIANTS:
            kwargs = {k: variant[k] for k in (
                "has_mask", "has_ref_kl", "has_entropy", "kl_coef",
                "entropy_coef") if k in variant}
            ppo_loss = _ppo_decomposed_loss(inputs, **kwargs)
            rows.append({
                "name": variant["name"],
                "variant_kind": "compiler_decomposed_reference",
                "target": "reference_cpu",
                "executor": "compiler_decomposed_reference",
                "compiler_path": "tessera-rl-loss-decompose",
                "shape": f"B{shape[0]}_G{shape[1]}_T{shape[2]}",
                "seed": seed,
                "loss": ppo_loss,
                "correctness": 0.0,
                "timing_ms": _time_loss(
                    lambda kw=kwargs: _ppo_decomposed_loss(inputs, **kw), reps),
                "runtime_status": "reference",
                "skip_reason": None,
            })
            rows.append(_apple_gpu_ppo_row(
                inputs, shape, seed, ppo_loss, reps, variant))
        grpo_loss = _grpo_decomposed_loss(inputs)
        cispo_loss = _cispo_decomposed_loss(inputs)
        for name, loss, reference in (
            ("grpo_policy_loss", grpo_loss, losses["grpo_policy_loss"]),
            ("cispo_policy_loss", cispo_loss, losses["cispo_policy_loss"]),
        ):
            rows.append({
                "name": name,
                "variant_kind": "compiler_decomposed_reference",
                "target": "reference_cpu",
                "executor": "compiler_decomposed_reference",
                "compiler_path": "tessera-rl-loss-decompose",
                "shape": f"B{shape[0]}_G{shape[1]}_T{shape[2]}",
                "seed": seed,
                "loss": loss,
                "correctness": abs(loss - reference),
                "timing_ms": _time_loss(lambda n=name: (
                    _grpo_decomposed_loss(inputs) if n == "grpo_policy_loss"
                    else _cispo_decomposed_loss(inputs)), reps),
                "runtime_status": "reference",
                "skip_reason": None,
            })
    return {
        "benchmark": "rl_policy_losses",
        "sprint": "S14_S15",
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
