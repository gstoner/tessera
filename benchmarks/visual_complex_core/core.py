"""Cross-lane (GA × EBM) library benchmark.

Sister of ``benchmarks/clifford_core`` and ``benchmarks/energy_core``;
composes their primitives in one flow to prove the lanes interoperate.

Forward path (deterministic from a seed):

  init multivector state x : (B, 8) in Cl(3, 0)
    -> apply rotor sandwich              — GA lane (n_rotors composed)
    -> clifford_energy(state, target)    — GA-norm-based EBM energy
    -> annealing schedule                — EBM lane
    -> langevin_step with analytic grad  — EBM lane, gradient in coefficient space
    -> grade_projection of final state   — GA lane
    -> partition_exact (stable logsumexp) — EBM lane scalar invariant

The matching IR fixture pins the cross-lane substrate: every GA op
neighbours every EBM op in the same module so a reordering refactor in
one lane that breaks the other surfaces immediately.

Oracle: independent recomposition of the same chain — sandwich + analytic
gradient + manual Langevin + analytic grade projection + numpy
logsumexp.  Any divergence is a *cross-lane* composition bug.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

import tessera.ga as ga
from tessera.ga.multivector import Multivector
from tessera.ebm.partition import partition_exact_from_energies
from tessera.rng import RNGKey, normal

# Reuse the building blocks from the sister benchmarks rather than
# re-deriving them.  This guarantees the visual-complex benchmark stays
# in lockstep with how the per-lane benchmarks treat each primitive.
from benchmarks.clifford_core.core import RotorSampler
from benchmarks.energy_core.core import annealing_schedule


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VisualComplexCoreConfig:
    """One visual-complex-core benchmark run.

    Cl(3, 0)'s 8 basis blades drive the model; ``n_rotors`` rotors are
    composed before sampling; ``n_steps`` annealed Langevin iterations
    run on the coefficient vector.
    """
    B: int = 4
    n_rotors: int = 2
    n_steps: int = 3
    eta: float = 0.05
    T_max: float = 1.0
    T_min: float = 0.1
    seed: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Cross-lane energy
# ─────────────────────────────────────────────────────────────────────────────


def clifford_energy(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Cl(3, 0) norm-squared-based EBM energy.

    ``E(state | target) = ½ Σ ||state_i - target||²`` per row, evaluated
    via ``tessera.ga.norm_squared(state - target)``.

    Returns: per-row energies, shape ``(B,)``, fp32.
    """
    algebra = ga.Cl(3, 0)
    s = np.asarray(state, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    diff_mv = Multivector(s - t[None, :], algebra)
    nsq = np.asarray(ga.norm_squared(diff_mv))
    return (0.5 * nsq).astype(np.float32, copy=False)


def _clifford_energy_grad(
    state: np.ndarray, target: np.ndarray,
) -> np.ndarray:
    """Analytic gradient of ``clifford_energy`` w.r.t. the per-row
    coefficient vector.

    For Cl(p, q, 0), ``norm_squared(mv) = Σ_i σ_i · c_i²`` where
    ``σ_i = signature(basis_i)``.  So ``∂E/∂c_i = σ_i · (c_i − t_i)``.
    The Cl(3, 0) blade signatures are all +1 (Euclidean), so the
    gradient reduces to ``(state - target)``.
    """
    return (state - target[None, :]).astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Oracle
# ─────────────────────────────────────────────────────────────────────────────


def composition_oracle(
    x_coeffs: np.ndarray, target_coeffs: np.ndarray,
    rotors: Sequence[Multivector],
    *, eta: float, schedule: np.ndarray, base_key: RNGKey,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Independent CPU oracle of the full GA × EBM chain.

    Reproduces:
      1. Sandwich application of each rotor.
      2. Langevin chain in coefficient space with analytic gradient.
      3. Grade-2 projection of the final state.
      4. logsumexp-based partition function from final per-row energies.

    Re-uses the exact RNG-threading convention the model uses
    (``base_key.fold_in(step).split(2)[0]``) so the oracle produces
    bit-identical noise samples.
    """
    algebra = ga.Cl(3, 0)
    # Sandwich.
    cur_mv = Multivector(x_coeffs.astype(np.float32, copy=True), algebra)
    for r in rotors:
        cur_mv = ga.rotor_sandwich(r, cur_mv)
    cur = np.asarray(cur_mv.coefficients, dtype=np.float32)
    # Langevin chain on coefficients, analytic gradient.
    target = target_coeffs.astype(np.float32, copy=False)
    for step, T in enumerate(schedule):
        grad = (cur - target[None, :])
        step_key = base_key.fold_in(f"step-{step}")
        sample_key, _ = step_key.split(2)
        xi = normal(sample_key, cur.shape, dtype="fp32")
        sigma = math.sqrt(2.0 * float(eta) * float(T))
        cur = (cur - float(eta) * grad + sigma * xi).astype(np.float32)
    # Grade-2 projection.
    bivec = ga.grade_projection(Multivector(cur, algebra), 2)
    # Final partition function (stable logsumexp on host).
    final_energies = clifford_energy(cur, target)
    Z = partition_exact_from_energies(final_energies, temperature=float(schedule[-1]))
    return (
        cur.astype(np.float32, copy=False),
        np.asarray(bivec.coefficients, dtype=np.float32),
        final_energies,
        float(Z),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


class VisualComplexCoreModel:
    """Library-layer composition of GA and EBM primitives in one flow.

    Forward (deterministic from cfg.seed):

      x_coeffs : (B, 8)         sampled via RNGKey.normal
      target   : (8,)           deterministic shift
      rotors   : n_rotors       from RotorSampler
      state    = sandwich(R_i, state) for each i  ── GA lane
      sched    = anneal(T_max → T_min, n_steps)   ── EBM lane
      for step, T in enumerate(sched):
          state = state − η · grad(clifford_energy) + √(2ηT) · ξ
      bivec    = grade_projection(state, grade=2)  ── GA lane
      Z        = partition_exact_from_energies(energies, T_min)  ── EBM lane
    """

    def __init__(self, cfg: VisualComplexCoreConfig):
        self.cfg = cfg
        self._algebra = ga.Cl(3, 0)
        root = RNGKey.from_seed(cfg.seed, name="visual_complex_core")
        self._init_key = root.fold_in("init")
        self._target_key = root.fold_in("target")
        self._sampler_key = root.fold_in("sampler")
        self._chain_key = root.fold_in("chain")

    @property
    def algebra(self) -> ga.Cl:
        return self._algebra

    @property
    def target_coeffs(self) -> np.ndarray:
        return normal(self._target_key, (self._algebra.dim,), dtype="fp32")

    @property
    def initial_coeffs(self) -> np.ndarray:
        return (normal(self._init_key,
                        (self.cfg.B, self._algebra.dim), dtype="fp32") * 0.5)

    @property
    def chain_key(self) -> RNGKey:
        return self._chain_key

    def rotors(self) -> list[Multivector]:
        sampler = RotorSampler(self._algebra, self._sampler_key)
        return [sampler.next_rotor() for _ in range(self.cfg.n_rotors)]

    def forward(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        cfg = self.cfg
        x_coeffs = self.initial_coeffs
        target = self.target_coeffs
        rotors = self.rotors()

        # GA lane: sandwich chain.
        cur_mv = Multivector(x_coeffs.astype(np.float32, copy=True), self._algebra)
        for r in rotors:
            cur_mv = ga.rotor_sandwich(r, cur_mv)
        cur = np.asarray(cur_mv.coefficients, dtype=np.float32, copy=True)

        # EBM lane: annealed Langevin chain on coefficient vector with
        # analytic gradient of clifford_energy.
        sched = annealing_schedule(cfg.n_steps, T_max=cfg.T_max, T_min=cfg.T_min)
        for step, T in enumerate(sched):
            grad = _clifford_energy_grad(cur, target)
            step_key = self._chain_key.fold_in(f"step-{step}")
            sample_key, _ = step_key.split(2)
            xi = normal(sample_key, cur.shape, dtype="fp32")
            sigma = math.sqrt(2.0 * float(cfg.eta) * float(T))
            cur = (cur - float(cfg.eta) * grad + sigma * xi).astype(np.float32)

        # GA lane: grade-2 projection of the post-anneal state.
        bivec = ga.grade_projection(Multivector(cur, self._algebra), 2)

        # EBM lane: final partition.
        final_energies = clifford_energy(cur, target)
        Z = partition_exact_from_energies(final_energies, temperature=float(sched[-1]))

        return (
            cur.astype(np.float32, copy=False),
            np.asarray(bivec.coefficients, dtype=np.float32),
            final_energies,
            float(Z),
        )

    __call__ = forward


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark harness
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VisualComplexCoreResult:
    backend: str
    op: str
    shape: dict[str, object]
    dtype: str
    latency_ms: float
    throughput_msps: float
    memory_bw_gb_s: float
    device: str
    tessera_version: str
    determinism_ok: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VisualComplexCoreBenchmark:
    """Small CPU/reference benchmark for the cross-lane composition."""

    BACKEND = "tessera-reference"
    OP = "visual_complex_core_forward"
    DEVICE = "cpu"
    VERSION = "pre-alpha"

    def __init__(self, *, warmup: int = 1, reps: int = 3):
        self.warmup = int(warmup)
        self.reps = int(reps)

    def run_one(self, cfg: VisualComplexCoreConfig) -> VisualComplexCoreResult:
        model = VisualComplexCoreModel(cfg)
        for _ in range(self.warmup):
            model()
        start = time.perf_counter()
        last = None
        for _ in range(self.reps):
            last = model()
        elapsed = (time.perf_counter() - start) / max(self.reps, 1)

        a = model()
        b = VisualComplexCoreModel(cfg)()
        determinism_ok = (
            np.array_equal(a[0], b[0])
            and np.array_equal(a[1], b[1])
            and np.array_equal(a[2], b[2])
            and a[3] == b[3]
        )
        samples = cfg.B
        throughput_msps = (samples / 1e6) / max(elapsed, 1e-12)
        bytes_per_step = cfg.B * model.algebra.dim * 4 * (cfg.n_steps + cfg.n_rotors + 1)
        memory_bw_gb_s = (bytes_per_step / 1e9) / max(elapsed, 1e-12)

        return VisualComplexCoreResult(
            backend=self.BACKEND,
            op=self.OP,
            shape={
                "B": cfg.B,
                "n_rotors": cfg.n_rotors,
                "n_steps": cfg.n_steps,
                "eta": cfg.eta,
                "T_max": cfg.T_max,
                "T_min": cfg.T_min,
                "algebra": "Cl(3,0)",
            },
            dtype="fp32",
            latency_ms=elapsed * 1000.0,
            throughput_msps=throughput_msps,
            memory_bw_gb_s=memory_bw_gb_s,
            device=self.DEVICE,
            tessera_version=self.VERSION,
            determinism_ok=determinism_ok,
        )

    def run(
        self, configs: Sequence[VisualComplexCoreConfig],
    ) -> list[VisualComplexCoreResult]:
        return [self.run_one(c) for c in configs]

    def report(self, results: Sequence[VisualComplexCoreResult]) -> None:
        if not results:
            return
        print(f"{'shape':<60} {'latency_ms':>12} det")
        for r in results:
            s = r.shape
            label = (
                f"B={s['B']} rotors={s['n_rotors']} steps={s['n_steps']} "
                f"η={s['eta']:.3f} T={s['T_max']:.2f}→{s['T_min']:.2f}"
            )
            det = "ok" if r.determinism_ok else "FAIL"
            print(f"{label:<60} {r.latency_ms:>12.2f} {det}")

    def to_json(
        self, results: Sequence[VisualComplexCoreResult], path: str,
    ) -> None:
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)


def default_sweep() -> list[VisualComplexCoreConfig]:
    return [
        VisualComplexCoreConfig(B=2, n_rotors=1, n_steps=2, seed=0),
        VisualComplexCoreConfig(B=4, n_rotors=2, n_steps=3, seed=0),
        VisualComplexCoreConfig(B=8, n_rotors=3, n_steps=4, seed=0),
    ]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps",   type=int, default=3)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    bench = VisualComplexCoreBenchmark(warmup=args.warmup, reps=args.reps)
    results = bench.run(default_sweep())
    bench.report(results)
    if args.output:
        bench.to_json(results, args.output)
        print(f"\nWrote {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
