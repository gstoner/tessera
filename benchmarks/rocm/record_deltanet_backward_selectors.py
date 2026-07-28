"""Record resident gfx1151 DeltaNet-backward chunk selector evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
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
        parallel_chunks: bool,
    ) -> None:
        self.values = tuple(np.ascontiguousarray(value, np.float32) for value in values)
        q, k, v, gate, beta, decay, dy = self.values
        self.B, self.H, self.S, self.Dqk = q.shape
        self.Dv = v.shape[-1]
        self.bh = self.B * self.H
        self.chunk_size = chunk_size
        self.parallel_chunks = parallel_chunks
        self.chunks = (self.S + chunk_size - 1) // chunk_size
        self.hsaco = rt._build_compiled_deltanet_bwd_hsaco(
            self.Dqk, self.Dv, False, True, True, True, True, chunk_size
        )
        self.hip = rt._load_hip_for_launch()
        if self.hip is None or self.hip.hipInit(0) != 0:
            raise RuntimeError("HIP is unavailable")
        self.module = ctypes.c_void_p()
        if self.hip.hipModuleLoadData(ctypes.byref(self.module), self.hsaco):
            raise RuntimeError("DeltaNet backward HSACO load failed")
        self.functions: dict[str, ctypes.c_void_p] = {}
        for name in (
            "dn_bwd_checkpoint", "dn_bwd_chunk_summary",
            "dn_bwd_chunk_prefix", "dn_bwd_chunk_fill", "dn_bwd_reverse",
        ):
            function = ctypes.c_void_p()
            if self.hip.hipModuleGetFunction(
                ctypes.byref(function), self.module, name.encode()
            ):
                raise RuntimeError(f"missing kernel {name}")
            self.functions[name] = function
        state = self.Dqk * self.Dv
        scalar = self.bh * self.S
        self.host = {
            "q": q.reshape(-1), "k": k.reshape(-1), "v": v.reshape(-1),
            "gate": gate.reshape(-1), "beta": beta.reshape(-1),
            "decay": decay.reshape(-1), "dy": dy.reshape(-1),
            "hist": np.zeros(self.bh * (self.S + 1) * state, np.float32),
            "ds": np.zeros(self.bh * state, np.float32),
            "dn": np.zeros(self.bh * state, np.float32),
            "du": np.zeros(self.bh * state, np.float32),
            "chunk_scale": np.zeros(self.bh * self.chunks, np.float32),
            "chunk_update": np.zeros(
                self.bh * self.chunks * state, np.float32
            ),
            "dq": np.zeros(q.size, np.float32),
            "dk": np.zeros(k.size, np.float32),
            "dv": np.zeros(v.size, np.float32),
            "dg": np.zeros(v.size, np.float32),
            "db": np.zeros(scalar, np.float32),
            "da": np.zeros(scalar, np.float32),
        }
        self.device = {name: ctypes.c_void_p() for name in self.host}
        for name, array in self.host.items():
            if self.hip.hipMalloc(ctypes.byref(self.device[name]), array.nbytes):
                self.close()
                raise RuntimeError(f"HIP allocation failed for {name}")
            self.hip.hipMemset(self.device[name], 0, array.nbytes)
        for name in ("q", "k", "v", "gate", "beta", "decay", "dy"):
            array = self.host[name]
            self.hip.hipMemcpy(
                self.device[name], array.ctypes.data_as(ctypes.c_void_p),
                array.nbytes, 1,
            )
        self.workspace_bytes = sum(
            array.nbytes for name, array in self.host.items()
            if name not in {"q", "k", "v", "gate", "beta", "decay", "dy"}
        )

    def _memref(self, name: str) -> list[Any]:
        pointer = self.device[name]
        return [
            ctypes.c_void_p(pointer.value), ctypes.c_void_p(pointer.value),
            ctypes.c_int64(0), ctypes.c_int64(self.host[name].size),
            ctypes.c_int64(1),
        ]

    def _launch(
        self,
        name: str,
        arrays: list[str],
        *,
        dimensions: tuple[int, ...] = (),
        grid_x: int | None = None,
    ) -> None:
        arguments: list[Any] = []
        for array in arrays:
            arguments.extend(self._memref(array))
        arguments.extend(ctypes.c_int64(value) for value in dimensions)
        pointers = (ctypes.c_void_p * len(arguments))()
        for index, value in enumerate(arguments):
            pointers[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
        rc = self.hip.hipModuleLaunchKernel(
            self.functions[name], grid_x or self.bh, 1, 1, 1, 1, 1,
            0, None, pointers, None,
        )
        if rc:
            raise RuntimeError(f"{name} launch failed rc={rc}")

    def launch(self) -> None:
        if self.parallel_chunks and self.chunks > 1:
            self._launch(
                "dn_bwd_chunk_summary",
                ["k", "v", "beta", "decay", "chunk_scale", "chunk_update"],
                dimensions=(self.S, self.chunks),
                grid_x=self.bh * self.chunks,
            )
            self._launch(
                "dn_bwd_chunk_prefix",
                ["chunk_scale", "chunk_update", "hist"],
                dimensions=(self.S, self.chunks),
            )
            self._launch(
                "dn_bwd_chunk_fill",
                ["k", "v", "beta", "decay", "hist", "chunk_update"],
                dimensions=(self.S, self.chunks),
                grid_x=self.bh * self.chunks,
            )
        else:
            self._launch(
                "dn_bwd_checkpoint",
                ["k", "v", "beta", "decay", "hist"],
                dimensions=(self.S,),
            )
        self._launch(
            "dn_bwd_reverse",
            [
                "q", "k", "v", "gate", "beta", "decay", "dy", "hist",
                "ds", "dn", "du", "dq", "dk", "dv", "dg", "db", "da",
            ],
            dimensions=(self.S,),
        )

    def synchronize(self) -> None:
        if self.hip.hipDeviceSynchronize():
            raise RuntimeError("HIP synchronization failed")

    def gradients(self) -> tuple[np.ndarray, ...]:
        outputs = []
        shapes = (
            self.values[0].shape, self.values[1].shape, self.values[2].shape,
            self.values[3].shape, self.values[4].shape, self.values[5].shape,
        )
        for name, shape in zip(("dq", "dk", "dv", "dg", "db", "da"), shapes):
            array = self.host[name]
            self.hip.hipMemcpy(
                array.ctypes.data_as(ctypes.c_void_p), self.device[name],
                array.nbytes, 2,
            )
            outputs.append(array.reshape(shape).copy())
        return tuple(outputs)

    def close(self) -> None:
        for pointer in getattr(self, "device", {}).values():
            if pointer.value:
                self.hip.hipFree(pointer)
                pointer.value = None


def _samples(session: ResidentSession, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        session.launch()
        session.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        session.launch()
        session.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return samples


def record(
    *,
    shape: tuple[int, int, int, int] = (2, 8, 128, 16),
    warmup: int = 5,
    iterations: int = 21,
) -> dict[str, Any]:
    values = _inputs(shape)
    configs = [
        ("serial_c64", 64, False),
        ("parallel_c16", 16, True),
        ("parallel_c32", 32, True),
        ("parallel_c64", 64, True),
    ]
    sessions = {
        name: ResidentSession(values, chunk_size=chunk, parallel_chunks=parallel)
        for name, chunk, parallel in configs
    }
    try:
        cohorts = []
        for run in range(2):
            rows = []
            for name, chunk, parallel in configs:
                session = sessions[name]
                samples = _samples(session, warmup, iterations)
                rows.append({
                    "candidate": name, "chunk_size": chunk,
                    "parallel_chunks": parallel,
                    "median_ms": statistics.median(samples),
                    "p95_ms": sorted(samples)[
                        min(len(samples) - 1, int(0.95 * len(samples)))
                    ],
                    "samples_ms": samples,
                    "workspace_bytes": session.workspace_bytes,
                    "image_digest": "sha256:" + hashlib.sha256(
                        session.hsaco
                    ).hexdigest(),
                })
            winner = min(rows, key=lambda row: row["median_ms"])["candidate"]
            cohorts.append({"run": run + 1, "winner": winner, "candidates": rows})
        consensus = len({cohort["winner"] for cohort in cohorts}) == 1
        winner = cohorts[0]["winner"] if consensus else ""
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
        session = sessions[winner or "serial_c64"]
        session.launch()
        session.synchronize()
        expected = get_vjp("modified_delta_attention")(
            values[-1], *values[:-1], erase=False
        )
        max_error = max(
            float(np.max(np.abs(actual - reference)))
            for actual, reference in zip(session.gradients(), expected)
        )
        version = Path("/proc/version").read_text(
            encoding="utf-8", errors="ignore"
        )
        wsl = "microsoft" in version.lower()
        return {
            "schema": "tessera.sequence_mixer.selector_packet.v1",
            "target": "rocm_gfx1151",
            "device": rt._rocm_live_arch(),
            "timing_scope": "resident_program_wall_ms",
            "shape": list(shape), "dtype": "f32",
            "operation": "modified_delta_attention_backward",
            "erase": False, "warmup": warmup, "iterations": iterations,
            "cohorts": cohorts, "winner": winner,
            "winner_consensus": consensus,
            "stability_fraction": stability,
            "max_abs_error": max_error,
            "selector_eligible": bool(
                not wsl and consensus and stability <= 0.05
                and max_error <= 3e-3
            ),
            "host": {
                "wsl": wsl, "platform": platform.platform(),
                "rocm_version": "7.14.60850",
            },
        }
    finally:
        for session in sessions.values():
            session.close()


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
