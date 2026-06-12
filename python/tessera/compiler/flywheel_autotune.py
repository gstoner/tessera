"""Phase E3 / Pillar 2 — bridge the flywheel corpus to ``autotune_v2``
(the BaCO-style Bayesian search). See docs/audit/compiler/EVALUATOR_PLAN.md §6.

The flywheel measures real latency; ``autotune_v2`` generates BaCO-style **legal**
tile-config candidates and searches them. This bridge connects them:

  * ``gemm_workload_for`` / ``measured_tflops`` — convert a flywheel record into
    ``autotune_v2``'s ``GEMMWorkload`` and confirm the two modules agree on FLOP
    accounting (a measured record validates the analytical model);
  * ``autotuner_for`` — build a ``BayesianAutotuner`` for a record's workload,
    **device-grounded** with the flywheel's calibrated peak, exposing the legal
    constrained candidate set;
  * ``best_record`` — the measured selection criterion over a corpus.

Honest scope: ``autotune_v2``'s candidate space is NVIDIA-style tile/warp/stage
configs, which Apple's MPS matmul does not expose — so on Apple this bridge
connects measured FLOP-accounting + a device-calibrated roofline peak + best-by-
measurement. Full *measured tile-config* autotuning arrives with the NVIDIA
executable lane (the configs become measurable there).
"""

from __future__ import annotations

from typing import Any

from tessera.compiler.autotune_v2 import BayesianAutotuner, GEMMWorkload
from tessera.compiler.flywheel import AutotuneRecord, peak_for_device, record_matmul

# flywheel dtype names → GEMMWorkload's accepted names.
_DTYPE_MAP = {"f32": "fp32", "f16": "fp16", "bf16": "bf16", "fp32": "fp32", "fp16": "fp16"}


def gemm_workload_for(record: AutotuneRecord) -> GEMMWorkload:
    """The ``autotune_v2`` workload for a flywheel record's shape + dtype."""
    s = record.problem_shape
    return GEMMWorkload(
        M=s["M"], N=s["N"], K=s["K"],
        dtype=_DTYPE_MAP.get(record.dtype, "fp32"),
    )


def measured_tflops(record: AutotuneRecord) -> float | None:
    """The record's measured achieved TFLOP/s via ``autotune_v2``'s accounting.
    Equals ``record.achieved_tflops`` — the cross-check that the flywheel and the
    autotuner agree on FLOPs. ``None`` for a non-native record (no latency)."""
    if record.latency is None:
        return None
    return gemm_workload_for(record).tflops_at(record.latency.median_ms)


def autotuner_for(record: AutotuneRecord, *, seed: int = 42) -> BayesianAutotuner:
    """A ``BayesianAutotuner`` for the record's workload, grounded with the
    device-calibrated peak so its roofline view matches the real GPU. Use
    ``.legal_candidates()`` for the BaCO-style constrained search space."""
    peak = peak_for_device(record.device_id)
    return BayesianAutotuner(
        gemm_workload_for(record), peak_tflops=peak.peak_tflops, seed=seed
    )


def best_record(corpus: list[AutotuneRecord]) -> AutotuneRecord | None:
    """The empirically fastest native candidate in a corpus (the measured
    selection criterion), or ``None`` if nothing ran natively."""
    return pick_best(corpus, by="latency")


def pick_best(corpus: list[AutotuneRecord], *, by: str = "latency") -> AutotuneRecord | None:
    """The best native candidate by ``latency`` (lowest median) or ``tflops``
    (highest achieved). ``None`` if nothing ran natively."""
    native = [r for r in corpus if r.latency is not None]
    if not native:
        return None
    if by == "tflops":
        return max(native, key=lambda r: r.achieved_tflops or 0.0)
    if by == "latency":
        return min(native, key=lambda r: r.latency.median_ms)  # type: ignore[union-attr]
    raise ValueError(f"unknown selection objective {by!r}")


def autotune_matmul(
    target: str,
    fn: Any,
    m: int,
    n: int,
    k: int,
    rng: Any,
    *,
    dtypes: tuple[str, ...] = ("f32", "f16"),
    reps: int = 10,
) -> tuple[AutotuneRecord | None, list[AutotuneRecord]]:
    """Autotune one matmul over the measurable Apple knob space, returning
    ``(best, corpus)``.

    Realizes the *GPU-portability-needs-autotuning* thesis on Apple: autotune a
    single parametric program over the knobs the backend actually exposes.
    Apple's MPS matmul is a black box for tile/stage, so the measurable knob here
    is **dtype** (f32/f16) — each candidate is measured via the flywheel and the
    fastest wins. The richer tile/warp/stage candidate space (autotune_v2's
    ``LegalGEMMCandidateGenerator``) becomes measurable on the NVIDIA executable
    lane.
    """
    import numpy as np

    corpus: list[AutotuneRecord] = []
    for dt in dtypes:
        np_dtype = np.float16 if dt == "f16" else np.float32
        a = rng.standard_normal((m, k)).astype(np_dtype)
        b = rng.standard_normal((k, n)).astype(np_dtype)
        corpus.append(
            record_matmul(
                target, fn, (a, b), m=m, n=n, k=k, dtype=dt,
                schedule={"dtype": dt}, search_method="autotune", reps=reps,
            )
        )
    return pick_best(corpus, by="latency"), corpus
