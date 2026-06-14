"""Small deterministic CGG benchmark/report harness.

Runs the adapter-style DiffusionGemma CGG example across a grid of guidance
scales and adapter strengths, then emits a compact JSON artifact suitable for
regression tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
EXAMPLE_DIR = Path(__file__).resolve().parent
for path in (PYTHON, EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cgg_diffusion_gemma import run_demo  # noqa: E402


def _grid(smoke: bool) -> list[dict[str, float]]:
    gammas = [0.0, 0.75] if smoke else [0.0, 0.5, 0.75, 1.0]
    scales = [0.08] if smoke else [0.04, 0.08, 0.12]
    cases: list[dict[str, float]] = []
    for gamma in gammas:
        for scale in scales:
            cases.append({
                "quality_gamma": gamma,
                "safety_gamma": 0.25 if gamma else 0.0,
                "quality_scale": scale,
                "unfavored_scale": -0.75 * scale,
                "safety_scale": 0.5625 * scale,
            })
    return cases


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.fmean(float(row[key]) for row in rows)) if rows else 0.0


def run_benchmark(*, smoke: bool = False, seed: int = 123) -> dict[str, Any]:
    runs = []
    for case_id, case in enumerate(_grid(smoke)):
        report = run_demo(seed=seed, **case)
        delta = report["guided_vs_unguided_deltas"]
        runs.append({
            "case_id": case_id,
            "config": report["config"],
            "canvas_l2_delta": report["canvas_l2_delta"],
            "quality_proxy_delta": delta["quality_proxy"],
            "safety_proxy_delta": delta["safety_proxy"],
            "entropy_mean_delta": delta["entropy_mean"],
            "score_norm_last_delta": delta["score_norm_last"],
            "token_delta_count": delta["token_delta_count"],
            "accepted_tokens_delta": delta["accepted_tokens"],
            "per_step": report["per_step"],
        })
    return {
        "benchmark": "cgg_adapter_guidance",
        "mode": "smoke" if smoke else "full",
        "seed": seed,
        "num_runs": len(runs),
        "aggregate": {
            "mean_canvas_l2_delta": round(_mean(runs, "canvas_l2_delta"), 6),
            "mean_quality_proxy_delta": round(_mean(runs, "quality_proxy_delta"), 6),
            "mean_safety_proxy_delta": round(_mean(runs, "safety_proxy_delta"), 6),
            "mean_entropy_delta": round(_mean(runs, "entropy_mean_delta"), 6),
            "mean_token_delta_count": round(_mean(runs, "token_delta_count"), 6),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a CI-friendly two-case sweep.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    report = run_benchmark(smoke=args.smoke, seed=args.seed)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
