"""Reference PPO / GRPO / CISPO policy-loss benchmark scaffold.

Stage 12 keeps RL post-training losses honest: these rows measure the existing
Python/numpy reference implementations only. They do not claim a compiler path,
Apple executor, GPU executor, or independent correctness oracle.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Iterable

import numpy as np

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
    return {
        "benchmark": "rl_policy_losses",
        "sprint": "S12",
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
