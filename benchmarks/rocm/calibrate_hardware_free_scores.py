#!/usr/bin/env python3
"""ROCM-CALIB-1: frozen gfx1151 score calibration and terminal verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera.compiler.hardware_free_scores import (  # noqa: E402
    GFX1151_BANK_GEOMETRY,
    step_distance_histogram,
)

DEFAULT_RETUNE = (
    ROOT / "benchmarks/baselines/"
    "rocm_gfx1151_compiler_retune_2026_07_15.json"
)
DEFAULT_HOT_PATHS = (
    ROOT / "benchmarks/baselines/rocm_gfx1151_hot_paths.json"
)

# Frozen before measurement. A home-architecture failure rejects the score;
# this script intentionally has no coefficient fitting or retuning path.
MIN_SHAPES = 4
MIN_MEDIAN_SPEARMAN = 0.50
MIN_POSITIVE_FRACTION = 0.75
F32_TILE_CANDIDATES = (
    (1, 1), (2, 2), (2, 4), (3, 4), (4, 4),
    (4, 6), (6, 4), (4, 8), (8, 4),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x, y = _rank(left), _rank(right)
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return None if dx == 0 or dy == 0 else numerator / (dx * dy)


def _f32_locality(shape: list[int], tile: list[int]) -> dict[str, object]:
    """Exact register-blocked load order from GenerateROCMGemmF32Kernel.cpp."""
    _, _, k_extent = shape
    tm, tn = tile
    a_accesses = [
        (row, k) for k in range(k_extent) for row in range(tm)
    ]
    b_accesses = [
        (k, column) for k in range(k_extent) for column in range(tn)
    ]
    a = step_distance_histogram(a_accesses)
    b = step_distance_histogram(b_accesses)
    transitions = a.total + b.total
    combined = (
        (a.locality_score * a.total + b.locality_score * b.total) / transitions
        if transitions else 1.0
    )
    return {
        "score": combined,
        "a": vars(a),
        "b": vars(b),
        "definition": (
            "transition-weighted mean of A/B Manhattan step-distance scores; "
            "<=1 scores 1, <=16 scores 0.5, larger scores 0"
        ),
    }


def analyze(measurement: Path, retune: Path, hot_paths: Path) -> dict:
    raw = json.loads(measurement.read_text())
    committed = json.loads(retune.read_text())
    hot = json.loads(hot_paths.read_text())
    if raw.get("evidence_arch") != "gfx1151" or not raw.get("all_correct"):
        raise ValueError("measurement must be a correct gfx1151 f32 retune run")

    rows = []
    by_shape: dict[tuple[int, ...], list[dict]] = {}
    for source in raw["rows"]:
        locality = _f32_locality(source["shape"], source["tile"])
        row = {
            "shape": source["shape"],
            "tile": source["tile"],
            "device_median_ms": source["device_median_ms"],
            "tflops": source["tflops"],
            "locality": locality,
            "bank_conflict": None,
            "bank_reason": (
                "not applicable: this production f32 retune kernel has no LDS "
                "loads/stores, so there is no bank-address trace"
            ),
        }
        rows.append(row)
        by_shape.setdefault(tuple(source["shape"]), []).append(row)

    correlations = []
    for shape, candidates in sorted(by_shape.items()):
        rho = _spearman(
            [candidate["locality"]["score"] for candidate in candidates],
            [candidate["tflops"] for candidate in candidates],
        )
        correlations.append({
            "shape": list(shape),
            "candidate_count": len(candidates),
            "spearman_locality_vs_tflops": rho,
        })

    finite = [
        row["spearman_locality_vs_tflops"]
        for row in correlations
        if row["spearman_locality_vs_tflops"] is not None
    ]
    median_rho = statistics.median(finite) if finite else None
    positive_fraction = (
        sum(value > 0 for value in finite) / len(finite) if finite else 0.0
    )
    locality_pass = (
        len(finite) >= MIN_SHAPES
        and median_rho is not None
        and median_rho >= MIN_MEDIAN_SPEARMAN
        and positive_fraction >= MIN_POSITIVE_FRACTION
    )

    committed_rows = committed["f32_gemm"]["rows"]
    fresh_winners = {
        shape: min(candidates, key=lambda row: row["device_median_ms"])["tile"]
        for shape, candidates in by_shape.items()
    }
    winner_checks = [{
        "shape": row["shape"],
        "committed_tile": row["tile"],
        "fresh_tile": fresh_winners.get(tuple(row["shape"])),
        "match": fresh_winners.get(tuple(row["shape"])) == row["tile"],
    } for row in committed_rows]
    hot_f32 = [
        row for row in hot["rows"]
        if row.get("op") == "gemm_f32"
    ]

    return {
        "schema": "tessera.rocm.hardware_free_calibration.v1",
        "sync_key": "COSTMODEL-CALIB-2026-07-29",
        "architecture": "gfx1151",
        "constants": {
            "wave_size": GFX1151_BANK_GEOMETRY.wave_size,
            "bank_count_per_wave32_domain": GFX1151_BANK_GEOMETRY.bank_count,
            "bank_width_bytes": GFX1151_BANK_GEOMETRY.bank_width_bytes,
            "phases": [list(phase) for phase in GFX1151_BANK_GEOMETRY.phases],
            "source": GFX1151_BANK_GEOMETRY.source,
        },
        "inputs": {
            "measurement": str(measurement),
            "measurement_sha256": _sha256(measurement),
            "committed_retune": str(retune),
            "committed_retune_sha256": _sha256(retune),
            "committed_hot_paths": str(hot_paths),
            "committed_hot_paths_sha256": _sha256(hot_paths),
        },
        "quality": {
            "comparison_grain": "within shape across tile candidates",
            "candidate_rows": len(rows),
            "shapes": len(by_shape),
            "all_correct": raw["all_correct"],
            "committed_winner_checks": winner_checks,
            "hot_path_rows": len(hot_f32),
            "hot_path_identifiability": (
                "summary-only: rows have one production latency per shape and "
                "no candidate access descriptor, so no rank correlation can "
                "be computed from the hot-path file alone"
            ),
        },
        "gate": {
            "minimum_shapes": MIN_SHAPES,
            "minimum_median_spearman": MIN_MEDIAN_SPEARMAN,
            "minimum_positive_fraction": MIN_POSITIVE_FRACTION,
            "observed_shapes": len(finite),
            "median_spearman": median_rho,
            "positive_fraction": positive_fraction,
        },
        "correlations": correlations,
        "rows": rows,
        "verdict": {
            "locality": "retain" if locality_pass else "reject",
            "bank_conflict": "reject_as_latency_ranker",
            "bank_conflict_scope": (
                "retain only as a structural LDS-layout diagnostic when an "
                "explicit lane-address trace exists"
            ),
            "terminal": not locality_pass,
            "policy": (
                "home-architecture gate; a failed locality rank is rejected "
                "without coefficient retuning"
            ),
        },
    }


def analyze_committed(retune: Path, hot_paths: Path) -> dict:
    """Use the committed winner partial order when raw candidate rows are absent."""
    committed = json.loads(retune.read_text())
    hot = json.loads(hot_paths.read_text())
    rows = []
    correlations = []
    for measured in committed["f32_gemm"]["rows"]:
        shape = measured["shape"]
        winner = measured["tile"]
        candidates = []
        for tile in F32_TILE_CANDIDATES:
            locality = _f32_locality(shape, list(tile))
            row = {
                "shape": shape,
                "tile": list(tile),
                "locality": locality,
                "committed_winner": list(tile) == winner,
                "bank_conflict": None,
                "bank_reason": (
                    "not applicable: the production f32 retune kernel has no "
                    "LDS loads/stores, so there is no bank-address trace"
                ),
            }
            rows.append(row)
            candidates.append(row)
        rho = _spearman(
            [row["locality"]["score"] for row in candidates],
            [float(row["committed_winner"]) for row in candidates],
        )
        top_score = max(
            candidates, key=lambda row: row["locality"]["score"])["tile"]
        correlations.append({
            "shape": shape,
            "candidate_count": len(candidates),
            "committed_winner": winner,
            "score_top_tile": top_score,
            "score_reproduces_winner": top_score == winner,
            "spearman_locality_vs_committed_winner_partial_order": rho,
        })

    finite = [
        row["spearman_locality_vs_committed_winner_partial_order"]
        for row in correlations
        if row["spearman_locality_vs_committed_winner_partial_order"] is not None
    ]
    median_rho = statistics.median(finite) if finite else None
    positive_fraction = (
        sum(value > 0 for value in finite) / len(finite) if finite else 0.0
    )
    locality_pass = (
        len(finite) >= MIN_SHAPES
        and median_rho is not None
        and median_rho >= MIN_MEDIAN_SPEARMAN
        and positive_fraction >= MIN_POSITIVE_FRACTION
    )
    hot_f32 = [row for row in hot["rows"] if row.get("op") == "gemm_f32"]
    return {
        "schema": "tessera.rocm.hardware_free_calibration.v1",
        "sync_key": "COSTMODEL-CALIB-2026-07-29",
        "architecture": "gfx1151",
        "constants": {
            "wave_size": GFX1151_BANK_GEOMETRY.wave_size,
            "bank_count_per_wave32_domain": GFX1151_BANK_GEOMETRY.bank_count,
            "bank_width_bytes": GFX1151_BANK_GEOMETRY.bank_width_bytes,
            "phases": [list(phase) for phase in GFX1151_BANK_GEOMETRY.phases],
            "source": GFX1151_BANK_GEOMETRY.source,
        },
        "inputs": {
            "measurement": None,
            "measurement_status": (
                "unavailable: HIP event elapsed time returned zero on the live "
                "WSL gfx1151; no host-wall substitute was relabeled as device time"
            ),
            "committed_retune": str(retune),
            "committed_retune_sha256": _sha256(retune),
            "committed_hot_paths": str(hot_paths),
            "committed_hot_paths_sha256": _sha256(hot_paths),
        },
        "quality": {
            "comparison_grain": (
                "within shape across the nine committed retune candidates; "
                "performance evidence is the committed winner partial order"
            ),
            "candidate_rows": len(rows),
            "shapes": len(correlations),
            "winner_reproduction_count": sum(
                row["score_reproduces_winner"] for row in correlations),
            "hot_path_rows": len(hot_f32),
            "hot_path_identifiability": (
                "summary-only: one production latency per shape and no "
                "alternative candidate descriptor; useful as a corpus identity "
                "check, not an independent within-shape rank"
            ),
            "limitation": (
                "non-winning pairwise latency order is unavailable in the "
                "committed summary, so winner-vs-rest is the only identifiable "
                "rank relation"
            ),
        },
        "gate": {
            "minimum_shapes": MIN_SHAPES,
            "minimum_median_spearman": MIN_MEDIAN_SPEARMAN,
            "minimum_positive_fraction": MIN_POSITIVE_FRACTION,
            "observed_shapes": len(finite),
            "median_spearman": median_rho,
            "positive_fraction": positive_fraction,
        },
        "correlations": correlations,
        "rows": rows,
        "verdict": {
            "locality": "retain" if locality_pass else "reject",
            "bank_conflict": "reject_as_latency_ranker",
            "bank_conflict_scope": (
                "retain only as a structural LDS-layout diagnostic when an "
                "explicit lane-address trace exists"
            ),
            "terminal": not locality_pass,
            "policy": (
                "home-architecture gate; a failed locality rank is rejected "
                "without coefficient retuning"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement", type=Path)
    parser.add_argument("--retune", type=Path, default=DEFAULT_RETUNE)
    parser.add_argument("--hot-paths", type=Path, default=DEFAULT_HOT_PATHS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        analyze(args.measurement, args.retune, args.hot_paths)
        if args.measurement
        else analyze_committed(args.retune, args.hot_paths)
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["verdict"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
