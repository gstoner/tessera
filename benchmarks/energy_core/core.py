"""Generic energy-based-model (EBM) benchmark core.

Sister of ``benchmarks/grid_ai_core`` and ``benchmarks/clifford_core``:
stays above Tessera as library code, domain-neutral, exercises the EBM
primitives the Apple-GPU MSL kernels back.

Forward path (deterministic given a seed):

  init x : (B, D) from RNGKey.normal
    -> quadratic_energy(x, y)            — E(x | y) = ½ ||x − y||²
    -> annealing_schedule(n_steps)       — β: 0 → 1 linear
    -> for each β: langevin_step         — y' = y − η ∂E + √(2ηβ⁻¹) ξ
    -> partition_exact_from_energies     — stable logsumexp over final batch
    -> final_energy_mean                  — scalar reporting

The lit fixture in ``tests/tessera-ir/phase7`` pins the compiler-visible
substrate every EBM library needs:
  * tessera_ebm.energy_quadratic
  * tessera_ebm.langevin_step
  * tessera_ebm.annealing_schedule
  * tessera_ebm.partition_exact
  * tessera.logsumexp                    — the one real Graph IR op

Oracle: pure-numpy stable-logsumexp + central-diff gradient + manual
Langevin step, expressed inline so a regression in *composition* surfaces
as an oracle divergence rather than an oracle match.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from tessera.ebm.energy import energy_quadratic, langevin_step
from tessera.ebm.partition import partition_exact_from_energies
from tessera.rng import RNGKey, normal


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EnergyCoreConfig:
    """One Energy-core benchmark run.

    ``B`` parallel chains × ``D``-dimensional state × ``n_steps`` annealed
    Langevin iterations.  Numerical contract requires ``T_max > T_min > 0``.
    """
    B: int = 4
    D: int = 8
    n_steps: int = 4
    eta: float = 0.05
    T_max: float = 1.0
    T_min: float = 0.1
    seed: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — domain-neutral building blocks
# ─────────────────────────────────────────────────────────────────────────────


def annealing_schedule(n_steps: int, *, T_max: float, T_min: float) -> np.ndarray:
    """Linear temperature schedule from T_max down to T_min over n_steps."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")
    if T_max <= 0.0 or T_min <= 0.0 or T_max < T_min:
        raise ValueError(
            f"require T_max >= T_min > 0; got T_max={T_max}, T_min={T_min}"
        )
    if n_steps == 1:
        return np.array([T_max], dtype=np.float32)
    return np.linspace(T_max, T_min, n_steps, dtype=np.float32)


def quadratic_energy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Wrapper around ``tessera.ebm.energy.energy_quadratic`` returning
    fp32 (the kernel-fastpath dtype)."""
    return np.asarray(energy_quadratic(x, y), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Oracles
# ─────────────────────────────────────────────────────────────────────────────


def energy_grid_oracle(
    states: np.ndarray, target: np.ndarray, *, temperature: float,
) -> float:
    """Independent oracle for the partition function.

    Computes ``Z = Σ_i exp(-E(states_i) / T)`` via numpy's stable
    logsumexp.  Independent re-derivation: does not call
    ``partition_exact_from_energies`` so a regression there surfaces here.
    """
    # E(s_i) = ½ ||s_i − target||²  — flat-batch reduction.
    diff = (states - target[None, :]).astype(np.float64, copy=False)
    energies = 0.5 * np.sum(diff * diff, axis=-1)
    inv_t = 1.0 / float(temperature)
    neg = -energies * inv_t
    m = float(neg.max())
    return float(math.exp(m + math.log(float(np.exp(neg - m).sum()))))


def langevin_chain_oracle(
    y0: np.ndarray, target: np.ndarray,
    *, eta: float, schedule: np.ndarray, base_key: RNGKey,
) -> np.ndarray:
    """Independent numpy oracle for the annealed Langevin chain.

    Computes ``y_{t+1} = y_t − η · ∂E(y_t)/∂y + √(2 · η · T_t) · ξ_t``
    with analytic gradient for the quadratic energy (∂E/∂y = y − target).
    Uses the exact same RNGKey lineage convention as ``EnergyCoreModel``
    so the two paths produce bit-identical noise samples.
    """
    cur = y0.astype(np.float64, copy=True)
    key = base_key
    for step, T in enumerate(schedule):
        # Analytic gradient of ½||y − target||² is y − target.
        grad = cur - target.astype(np.float64, copy=False)
        # Match the model's RNG-threading exactly: langevin_step splits
        # its per-step key into (sample_key, next_key) and uses
        # ``sample_key`` for the noise.  The chain-orchestration layer
        # (EnergyCoreModel) re-derives the per-step key by fold_in.
        step_key = key.fold_in(f"step-{step}")
        sample_key, _ = step_key.split(2)
        xi = normal(sample_key, cur.shape, dtype="fp32").astype(np.float64)
        # Note: langevin_step performs everything at fp32 internally,
        # so we mirror that exactly to stay bit-comparable.
        sigma = math.sqrt(2.0 * float(eta) * float(T))
        cur = (cur - float(eta) * grad).astype(np.float32, copy=False)
        cur = (cur + (sigma * xi).astype(np.float32, copy=False)).astype(
            np.float64, copy=False,
        )
    return cur.astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


class EnergyCoreModel:
    """Library-layer composition over the EBM primitives.

    Forward (deterministic from cfg.seed):

      x_init  : (B, D) sampled via RNGKey.normal
      target  : (D,) deterministic shift
      sched   : (n_steps,) linear anneal T_max → T_min
      for step, T in enumerate(sched):
          y_{t+1} = y_t − η · ∂E + √(2·η·T) · ξ  via tessera.ebm.langevin_step
      Z = partition_exact_from_energies(energies(y_final), temperature=T_min)
      log_Z = log(Z)
      out   : (final_y, energies_final, log_Z)
    """

    def __init__(self, cfg: EnergyCoreConfig):
        if cfg.D <= 0:
            raise ValueError("D must be positive")
        self.cfg = cfg
        root = RNGKey.from_seed(cfg.seed, name="energy_core")
        self._init_key = root.fold_in("init")
        self._target_key = root.fold_in("target")
        self._chain_key = root.fold_in("chain")

    @property
    def target(self) -> np.ndarray:
        return normal(self._target_key, (self.cfg.D,), dtype="fp32")

    @property
    def initial_state(self) -> np.ndarray:
        return normal(self._init_key, (self.cfg.B, self.cfg.D), dtype="fp32") * 0.5

    @property
    def chain_key(self) -> RNGKey:
        return self._chain_key

    def forward(self) -> tuple[np.ndarray, np.ndarray, float]:
        cfg = self.cfg
        y = self.initial_state
        target = self.target
        sched = annealing_schedule(cfg.n_steps, T_max=cfg.T_max, T_min=cfg.T_min)

        # Analytic gradient for the quadratic energy keeps the chain
        # deterministic and decoupled from finite-difference jitter.
        def grad_fn(arr: np.ndarray) -> np.ndarray:
            return (arr - target).astype(np.float32, copy=False)

        cur = y
        for step, T in enumerate(sched):
            step_key = self._chain_key.fold_in(f"step-{step}")
            cur, _ = langevin_step(
                cur,
                lambda a: quadratic_energy(a, target),
                eta=float(cfg.eta),
                temperature=float(T),
                rng_key=step_key,
                grad_fn=grad_fn,
            )
        # Per-row energies of the final batch.
        energies_final = np.array(
            [float(quadratic_energy(cur[b], target).sum()) for b in range(cfg.B)],
            dtype=np.float32,
        )
        # Stable logsumexp partition at the final (coldest) temperature.
        Z = partition_exact_from_energies(energies_final, temperature=cfg.T_min)
        log_Z = math.log(max(Z, 1e-30))
        return cur.astype(np.float32, copy=False), energies_final, float(log_Z)

    __call__ = forward


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark harness
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EnergyCoreResult:
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


class EnergyCoreBenchmark:
    """Small reference benchmark for the EBM composition."""

    BACKEND = "tessera-reference"
    OP = "energy_core_forward"
    DEVICE = "cpu"
    VERSION = "pre-alpha"

    def __init__(self, *, warmup: int = 1, reps: int = 3):
        self.warmup = int(warmup)
        self.reps = int(reps)

    def run_one(self, cfg: EnergyCoreConfig) -> EnergyCoreResult:
        model = EnergyCoreModel(cfg)
        for _ in range(self.warmup):
            model()
        start = time.perf_counter()
        last = None
        for _ in range(self.reps):
            last = model()
        elapsed = (time.perf_counter() - start) / max(self.reps, 1)

        # Determinism: same cfg ⇒ identical outputs.
        a = model()
        b = EnergyCoreModel(cfg)()
        determinism_ok = (
            np.array_equal(a[0], b[0])
            and np.array_equal(a[1], b[1])
            and a[2] == b[2]
        )
        samples = cfg.B
        throughput_msps = (samples / 1e6) / max(elapsed, 1e-12)
        bytes_per_step = (cfg.B * cfg.D * 4 * (cfg.n_steps + 1))  # rough
        memory_bw_gb_s = (bytes_per_step / 1e9) / max(elapsed, 1e-12)

        return EnergyCoreResult(
            backend=self.BACKEND,
            op=self.OP,
            shape={
                "B": cfg.B,
                "D": cfg.D,
                "n_steps": cfg.n_steps,
                "eta": cfg.eta,
                "T_max": cfg.T_max,
                "T_min": cfg.T_min,
            },
            dtype="fp32",
            latency_ms=elapsed * 1000.0,
            throughput_msps=throughput_msps,
            memory_bw_gb_s=memory_bw_gb_s,
            device=self.DEVICE,
            tessera_version=self.VERSION,
            determinism_ok=determinism_ok,
        )

    def run(self, configs: Sequence[EnergyCoreConfig]) -> list[EnergyCoreResult]:
        return [self.run_one(c) for c in configs]

    def report(self, results: Sequence[EnergyCoreResult]) -> None:
        if not results:
            return
        print(f"{'shape':<54} {'latency_ms':>12} {'bw_gb/s':>10} det")
        for r in results:
            s = r.shape
            label = (
                f"B={s['B']} D={s['D']} n_steps={s['n_steps']} "
                f"η={s['eta']:.3f} T={s['T_max']:.2f}→{s['T_min']:.2f}"
            )
            det = "ok" if r.determinism_ok else "FAIL"
            print(
                f"{label:<54} {r.latency_ms:>12.2f} "
                f"{r.memory_bw_gb_s:>10.3f} {det}"
            )

    def to_json(self, results: Sequence[EnergyCoreResult], path: str) -> None:
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Default sweep + CLI
# ─────────────────────────────────────────────────────────────────────────────


def default_sweep() -> list[EnergyCoreConfig]:
    return [
        EnergyCoreConfig(B=4, D=4,  n_steps=3, seed=0),
        EnergyCoreConfig(B=4, D=8,  n_steps=4, seed=0),
        EnergyCoreConfig(B=8, D=16, n_steps=5, seed=0),
    ]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps",   type=int, default=3)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    bench = EnergyCoreBenchmark(warmup=args.warmup, reps=args.reps)
    results = bench.run(default_sweep())
    bench.report(results)
    if args.output:
        bench.to_json(results, args.output)
        print(f"\nWrote {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
