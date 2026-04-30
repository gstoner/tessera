from __future__ import annotations

import argparse
from pathlib import Path

from rlvr_suite.dataset import make_tasks
from rlvr_suite.rollout import ToyPolicy, collect_grouped_rollouts
from rlvr_suite.telemetry import JsonlLogger
from rlvr_suite.trainer import GRPOConfig, grpo_accounting_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--log", type=Path, default=Path("examples/advanced/rlvr_reasoning_suite/runs/rewards.jsonl"))
    args = parser.parse_args()

    tasks = make_tasks(seed=7, count=max(args.steps, 1))
    policy = ToyPolicy(seed=11)
    logger = JsonlLogger(args.log)
    cfg = GRPOConfig(group_size=args.group_size, clip_low=0.2, clip_high=0.28, resample_on_correct=True)

    for step in range(args.steps):
        task = tasks[step % len(tasks)]
        group = collect_grouped_rollouts(policy, task, cfg.group_size)
        report = grpo_accounting_step(group, cfg)
        logger.write({"step": step, "task_id": task.task_id, **report})
        print(
            f"step={step} task={task.task_id} reward_mean={report['reward_mean']:.3f} "
            f"kept={report['kept_rollouts']}/{report['total_rollouts']} best={report['best_answer']}"
        )

    logger.close()
    print(f"wrote {args.log}")


if __name__ == "__main__":
    main()
