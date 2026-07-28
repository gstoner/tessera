"""Record resident AVX-512 DeltaNet-backward chunk/worker selector evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402
from tessera.autodiff.vjp import get_vjp  # noqa: E402


def _inputs(shape: tuple[int, int, int, int]) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(20260728)
    q = (rng.standard_normal(shape) * 0.2).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True) + 1e-6
    v = (rng.standard_normal(shape) * 0.2).astype(np.float32)
    gate = rng.standard_normal(shape).astype(np.float32)
    beta = rng.uniform(0.2, 0.8, shape[:-1]).astype(np.float32)
    decay = rng.uniform(0.7, 0.95, shape[:-1]).astype(np.float32)
    dy = rng.standard_normal(shape).astype(np.float32)
    return q, k, v, gate, beta, decay, dy


class ResidentSession:
    def __init__(
        self,
        values: tuple[np.ndarray, ...],
        *,
        chunk_size: int,
        workers: int,
        modified: bool,
    ) -> None:
        library = rt._load_x86_elementwise()
        if library is None:
            raise RuntimeError("libtessera_x86_elementwise.so is unavailable")
        self.function = library.tessera_x86_deltanet_bwd_f32
        cf = ctypes.POINTER(ctypes.c_float)
        self.function.restype = None
        self.function.argtypes = (
            [cf] * 19
            + [ctypes.c_int64] * 5
            + [ctypes.c_int32] * 5
            + [ctypes.c_int64, ctypes.c_int32]
        )
        self.values = tuple(np.ascontiguousarray(value, np.float32) for value in values)
        q, k, v, gate, beta, decay, dy = self.values
        self.B, self.H, self.S, self.Dqk = q.shape
        self.Dv = v.shape[-1]
        self.chunk_size = chunk_size
        self.workers = workers
        self.modified = modified
        bh = self.B * self.H
        state = self.Dqk * self.Dv
        chunks = (self.S + chunk_size - 1) // chunk_size
        self.arrays = {
            "dq": np.zeros_like(q), "dk": np.zeros_like(k),
            "dv": np.zeros_like(v), "dg": np.zeros_like(v),
            "db": np.zeros_like(beta), "da": np.zeros_like(decay),
            "hist": np.zeros(bh * (self.S + 1) * state, np.float32),
            "ds": np.zeros(bh * state, np.float32),
            "dn": np.zeros(bh * state, np.float32),
            "du": np.zeros(bh * state, np.float32),
            "chunk_scale": np.zeros(bh * chunks, np.float32),
            "chunk_update": np.zeros(bh * chunks * state, np.float32),
        }
        self.workspace_bytes = sum(array.nbytes for array in self.arrays.values())

    @staticmethod
    def _pointer(array: np.ndarray) -> Any:
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def launch(self) -> None:
        q, k, v, gate, beta, decay, dy = self.values
        a = self.arrays
        self.function(
            *(self._pointer(value) for value in (q, k, v, gate, beta, decay, dy)),
            *(self._pointer(a[name]) for name in (
                "dq", "dk", "dv", "dg", "db", "da", "hist", "ds", "dn",
                "du", "chunk_scale", "chunk_update",
            )),
            self.B, self.H, self.S, self.Dqk, self.Dv,
            0, int(self.modified), 1, 1, 1,
            self.chunk_size, self.workers,
        )

    def gradients(self) -> tuple[np.ndarray, ...]:
        return tuple(
            self.arrays[name].copy() for name in ("dq", "dk", "dv", "dg", "db", "da")
        )


def _samples(session: ResidentSession, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        session.launch()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        session.launch()
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return samples


def record(
    *,
    shape: tuple[int, int, int, int] = (4, 8, 128, 16),
    warmup: int = 5,
    iterations: int = 21,
) -> dict[str, Any]:
    values = _inputs(shape)
    candidates = [
        ("serial_c64_w1", 64, 1),
        ("parallel_c16_w8", 16, 8),
        ("parallel_c32_w8", 32, 8),
        ("parallel_c64_w8", 64, 8),
    ]
    cohorts: list[dict[str, Any]] = []
    sessions: dict[str, ResidentSession] = {}
    for run in range(2):
        rows: list[dict[str, Any]] = []
        for name, chunk, workers in candidates:
            session = sessions.setdefault(
                name,
                ResidentSession(
                    values, chunk_size=chunk, workers=workers, modified=True
                ),
            )
            samples = _samples(session, warmup, iterations)
            rows.append({
                "candidate": name,
                "chunk_size": chunk,
                "workers": workers,
                "median_ms": statistics.median(samples),
                "p95_ms": sorted(samples)[
                    min(len(samples) - 1, int(0.95 * len(samples)))
                ],
                "samples_ms": samples,
                "workspace_bytes": session.workspace_bytes,
            })
        winner = min(rows, key=lambda row: row["median_ms"])["candidate"]
        cohorts.append({"run": run + 1, "winner": winner, "candidates": rows})
    winner_consensus = len({cohort["winner"] for cohort in cohorts}) == 1
    winner = cohorts[0]["winner"] if winner_consensus else ""
    medians = [
        next(
            row["median_ms"] for row in cohort["candidates"]
            if row["candidate"] == winner
        )
        for cohort in cohorts
    ] if winner else []
    stability = (
        abs(medians[0] - medians[1]) / max(medians) if medians else 1.0
    )
    session = sessions[winner or candidates[0][0]]
    session.launch()
    expected = get_vjp("modified_delta_attention")(
        values[-1], *values[:-1], erase=False
    )
    max_error = max(
        float(np.max(np.abs(actual - reference)))
        for actual, reference in zip(session.gradients(), expected)
    )
    flags = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    avx512 = "avx512f" in flags
    library_path = Path(rt._load_x86_elementwise()._name)
    fingerprint = "sha256:" + hashlib.sha256(library_path.read_bytes()).hexdigest()
    return {
        "schema": "tessera.sequence_mixer.selector_packet.v1",
        "target": "x86_avx512",
        "device": platform.processor() or platform.machine(),
        "timing_scope": "resident_program_wall_ms",
        "shape": list(shape),
        "dtype": "f32",
        "operation": "modified_delta_attention_backward",
        "erase": False,
        "compiler_fingerprint": fingerprint,
        "warmup": warmup,
        "iterations": iterations,
        "cohorts": cohorts,
        "winner": winner,
        "winner_consensus": winner_consensus,
        "stability_fraction": stability,
        "max_abs_error": max_error,
        "selector_eligible": bool(
            avx512 and winner_consensus and stability <= 0.05 and max_error <= 3e-3
        ),
        "host": {
            "avx512f": avx512,
            "logical_cpus": os.cpu_count(),
            "platform": platform.platform(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=21)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = record(warmup=args.warmup, iterations=args.iterations)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
