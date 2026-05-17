"""Visible GA + EBM demos for the Apple-path story.

These demos are intentionally tiny, deterministic, and CI-runnable. They use
the Python reference surfaces as the always-available execution path while also
surfacing the Apple GPU manifest status for the GA kernels that have native MSL
coverage.

Run:

    .venv/bin/python examples/conformance/apple_path_ga_ebm_demos.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
for path in (str(ROOT), str(PYTHON)):
    if path not in sys.path:
        sys.path.insert(0, path)

from tessera.compiler.backend_manifest import manifest_for
from tessera.ebm import inner_step, self_verify
from tessera.ga import Cl, Multivector, inner, rotor_from_axis, rotor_sandwich


def _so3_rotate(axis: np.ndarray, angle: float, v: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)) @ v


def _point_cloud_feature(points: np.ndarray) -> float:
    """Rotation-invariant scalar feature: sum of pairwise GA inner products."""
    algebra = Cl(3, 0)
    mvs = [Multivector.from_vector(p, algebra, dtype=np.float64) for p in points]
    total = 0.0
    for i in range(len(mvs)):
        for j in range(i + 1, len(mvs)):
            total += float(inner(mvs[i], mvs[j]))
    return total


def _apple_gpu_status(op_name: str) -> dict[str, Any]:
    for entry in manifest_for(op_name):
        if entry.target == "apple_gpu":
            return {
                "target": entry.target,
                "status": entry.status,
                "dtypes": list(entry.dtypes),
                "feature_flags": list(entry.feature_flags),
                "notes": entry.notes,
            }
    return {"target": "apple_gpu", "status": "missing", "dtypes": []}


def rotation_invariant_point_cloud_demo(seed: int = 17) -> dict[str, Any]:
    """Demo 1: GA rotation-invariant point-cloud features.

    The feature is intentionally simple: sum_{i<j} <p_i, p_j>. Since the GA
    inner product returns a scalar in Cl(3,0), this feature is invariant under
    rotations. The demo verifies the invariant twice:

    - through an independent Rodrigues SO(3) reference;
    - through Tessera's own rotor_sandwich path, the GA op with Apple GPU MSL
      coverage in the backend manifest.
    """
    rng = np.random.RandomState(seed)
    algebra = Cl(3, 0)
    points = rng.randn(9, 3).astype(np.float64)

    axis = rng.randn(3).astype(np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = float(rng.uniform(-math.pi, math.pi))

    rotated_ref = np.stack([_so3_rotate(axis, angle, p) for p in points], axis=0)

    # Rotor path: pick the e12 plane for a deterministic visible Apple-path op.
    bivector = Multivector.from_blade(algebra.blade("e12"), algebra, dtype=np.float64)
    rotor = rotor_from_axis(bivector, math.pi / 5.0)
    rotated_rotor = []
    for p in points:
        v = Multivector.from_vector(p, algebra, dtype=np.float64)
        out = rotor_sandwich(rotor, v)
        rotated_rotor.append(
            np.array(
                [
                    out.coefficients[algebra.blade("e1").mask],
                    out.coefficients[algebra.blade("e2").mask],
                    out.coefficients[algebra.blade("e3").mask],
                ],
                dtype=np.float64,
            )
        )
    rotated_rotor = np.stack(rotated_rotor, axis=0)

    original = _point_cloud_feature(points)
    rodrigues = _point_cloud_feature(rotated_ref)
    rotor_path = _point_cloud_feature(rotated_rotor)

    return {
        "name": "ga_rotation_invariant_point_cloud",
        "seed": seed,
        "n_points": int(points.shape[0]),
        "feature_original": original,
        "feature_rodrigues": rodrigues,
        "feature_rotor_path": rotor_path,
        "max_abs_drift": max(abs(original - rodrigues), abs(original - rotor_path)),
        "apple_gpu_manifest": {
            "clifford_inner": _apple_gpu_status("clifford_inner"),
            "clifford_rotor_sandwich": _apple_gpu_status("clifford_rotor_sandwich"),
        },
    }


def ebt_tiny_inner_loop_demo(seed: int = 23) -> dict[str, Any]:
    """Demo 2: EBT-tiny inner-loop refinement.

    This is the smallest useful Energy-Based Transformer-style pattern:
    initialize K noisy candidates, run T inner energy-gradient steps, then
    self-verify by choosing the lowest final energy. The energy is a quadratic
    denoising objective so the expected improvement is deterministic and large.
    """
    rng = np.random.RandomState(seed)
    batch = 4
    candidates = 3
    width = 5
    steps = 4
    eta = 0.35

    target = rng.randn(batch, width).astype(np.float64)
    y0 = target[:, None, :] + rng.randn(batch, candidates, width).astype(np.float64)
    zero_shot_energy = 0.5 * np.sum((y0 - target[:, None, :]) ** 2, axis=-1)

    y = y0.copy()
    for _ in range(steps):
        grad = y - target[:, None, :]
        y = inner_step(y, grad, eta=eta)

    final_energy = 0.5 * np.sum((y - target[:, None, :]) ** 2, axis=-1)
    chosen = self_verify(final_energy, y)
    chosen_energy = 0.5 * np.sum((chosen - target) ** 2, axis=-1)

    return {
        "name": "ebt_tiny_inner_loop_refinement",
        "seed": seed,
        "batch": batch,
        "candidates": candidates,
        "steps": steps,
        "eta": eta,
        "zero_shot_mean_energy": float(zero_shot_energy.mean()),
        "final_mean_energy": float(final_energy.mean()),
        "chosen_mean_energy": float(chosen_energy.mean()),
        "improvement_ratio": float(final_energy.mean() / zero_shot_energy.mean()),
        "self_verify_indices": np.argmin(final_energy, axis=1).astype(int).tolist(),
        "execution_note": (
            "Python reference path; EBM dialect annotation passes are lit-testable, "
            "native Apple GPU EBM fusion is follow-up backend work."
        ),
    }


def run_all_demos() -> dict[str, Any]:
    ga = rotation_invariant_point_cloud_demo()
    ebm = ebt_tiny_inner_loop_demo()
    return {
        "ga": ga,
        "ebm": ebm,
        "ok": bool(ga["max_abs_drift"] < 1e-9 and ebm["improvement_ratio"] < 0.25),
    }


def main() -> None:
    print(json.dumps(run_all_demos(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
