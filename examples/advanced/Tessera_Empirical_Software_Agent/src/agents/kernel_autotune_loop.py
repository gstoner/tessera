from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .sandbox_executor import run_candidate
from .scoring_api import ScoreResult


TILE_CANDIDATES = [
    {"block_m": 16, "block_n": 16, "unroll": 1},
    {"block_m": 16, "block_n": 32, "unroll": 2},
    {"block_m": 32, "block_n": 32, "unroll": 2},
    {"block_m": 32, "block_n": 64, "unroll": 4},
]


def write_candidate_config(workspace: Path, cfg: dict[str, int]) -> None:
    (workspace / "candidate_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def score_kernel_workspace(workspace: str) -> ScoreResult:
    logs = run_candidate(workspace, "python3 benchmark_kernel.py", timeout_s=30)
    value = 0.0
    if logs.get("ok"):
        try:
            payload = json.loads(logs["stdout"].strip().splitlines()[-1])
            value = payload["score"]
            logs["metrics"] = payload
        except (KeyError, json.JSONDecodeError, IndexError):
            logs["parse_error"] = True
    return ScoreResult(value=value, metric="correctness_weighted_throughput", metadata=logs)


def run_kernel_autotune(task_dir: Path, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    best: tuple[float, Path | None] = (-1.0, None)
    reports = []

    for idx, cfg in enumerate(TILE_CANDIDATES):
        workspace = out_dir / f"candidate_{idx:02d}"
        shutil.copytree(task_dir, workspace, dirs_exist_ok=True)
        write_candidate_config(workspace, cfg)
        score = score_kernel_workspace(str(workspace))
        report = {"candidate": idx, "config": cfg, "score": score.value, "metadata": score.metadata}
        reports.append(report)
        (workspace / "score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if score.value > best[0]:
            best = (score.value, workspace)

    summary = {"best_score": best[0], "best_path": str(best[1]), "reports": reports}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, default=Path("examples/advanced/Tessera_Empirical_Software_Agent/examples/kernel_autotuning"))
    parser.add_argument("--out", type=Path, default=Path("examples/advanced/Tessera_Empirical_Software_Agent/runs/kernel_autotune"))
    args = parser.parse_args()
    print(json.dumps(run_kernel_autotune(args.task, args.out), indent=2))


if __name__ == "__main__":
    main()
