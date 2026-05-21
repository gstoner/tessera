"""Generic Clifford-algebra benchmark core.

Sister of ``benchmarks/grid_ai_core``: stays above Tessera as library code,
domain-neutral, exercises the GA primitives the Apple-GPU MSL kernels back.

Forward path (deterministic given a seed):

  tiled multivector input
    -> rotor sampler (axis-bivector + angle)
    -> rotor_from_axis
    -> rotor_sandwich (R x R†)
    -> grade_projection (bivector grade)
    -> geometric_product chain (R · x)
    -> norm-squared as scalar invariant

The matching IR fixture in ``tests/tessera-ir/phase7`` pins the compiler view:
every step shows up as a ``tessera.clifford.*`` op (generic form) so a
downstream lowering pass can pick them up.

Oracle: pure-numpy GA reference via ``tessera.ga.*``.  Because the same
numpy backend powers both the model forward AND the oracle, this proves
*composition* correctness (a chain of GA primitives in the canonical
order produces the canonical result) rather than per-op correctness
(which the GA unit tests already cover).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import tessera.ga as ga
from tessera.ga.multivector import Multivector
from tessera.rng import RNGKey, normal, uniform


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CliffordCoreConfig:
    """One Clifford-core benchmark run.

    ``algebra_signature`` is the Cl(p, q, r) tuple.  The default (3, 0, 0)
    matches the Apple GPU MSL kernel fast-path (8 basis blades, f32).
    """
    B: int = 8                           # batch count
    algebra_signature: tuple[int, int, int] = (3, 0, 0)
    tile: int = 4                        # multivector tile size
    n_rotors: int = 2                    # rotors composed per forward
    seed: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def tile_multivectors(
    coeffs: np.ndarray, tile: int,
) -> list[np.ndarray]:
    """Decompose a leading-batched multivector coefficient array into row-major
    tiles of size ``tile`` along axis 0.  Mirrors the tile_field pattern
    from grid_ai_core."""
    if coeffs.ndim < 2:
        raise ValueError(
            f"tile_multivectors expects rank>=2 (batch x dim); got {coeffs.ndim}"
        )
    B = coeffs.shape[0]
    if B % tile:
        raise ValueError(
            f"tile {tile} does not divide batch {B}"
        )
    return [coeffs[i:i + tile] for i in range(0, B, tile)]


class RotorSampler:
    """Deterministic rotor source.  Samples a bivector axis + angle from
    a fixed RNGKey lineage; converts to a rotor via ``ga.rotor_from_axis``.

    Use:
        sampler = RotorSampler(algebra, key)
        rotor = sampler.next_rotor()  # advances internal counter
    """

    def __init__(self, algebra: ga.Cl, key: RNGKey):
        self._algebra = algebra
        self._root_key = key
        self._counter = 0

    def next_rotor(self) -> Multivector:
        """Generate one rotor with deterministic axis + angle."""
        idx = self._counter
        self._counter += 1
        axis_key = self._root_key.fold_in(f"axis-{idx}")
        angle_key = self._root_key.fold_in(f"angle-{idx}")
        # Sample 3 bivector coefficients (e12, e13, e23) — for Cl(3, 0)
        # these live at coefficient indices 3, 5, 6.
        bvec_coeffs = np.zeros(self._algebra.dim, dtype=np.float32)
        biv_coeffs = normal(axis_key, (3,), dtype="fp32")
        # Cl(3, 0) blade order: 1, e1, e2, e12, e3, e13, e23, e123
        # Bivector grade-2 indices: 3 (e12), 5 (e13), 6 (e23)
        bvec_coeffs[3] = biv_coeffs[0]
        bvec_coeffs[5] = biv_coeffs[1]
        bvec_coeffs[6] = biv_coeffs[2]
        # Avoid zero-bivector by adding a small jitter.
        if float(np.linalg.norm(bvec_coeffs)) < 1e-4:
            bvec_coeffs[3] += 1e-2
        bvec = Multivector(bvec_coeffs, self._algebra)
        # Angle from uniform [0, 2π).
        angle_arr = uniform(
            angle_key, (1,), low=0.0, high=2.0 * np.pi, dtype="fp32",
        )
        angle = float(angle_arr.item())
        return ga.rotor_from_axis(bvec, angle)


def multivector_oracle_chain(
    x: Multivector, rotors: Sequence[Multivector],
) -> tuple[Multivector, np.ndarray, np.ndarray]:
    """Independent CPU oracle of the Clifford-core forward path.

    Returns (final_mv, bivector_part, norm_sq) using a hand-rolled
    composition of the same GA primitives — but expressed as a tight
    inline sequence so a regression in the forward model (e.g. wrong
    op order) surfaces as an oracle divergence, not an oracle match.
    """
    # Sequential sandwich applications.
    cur = x
    for r in rotors:
        cur = ga.rotor_sandwich(r, cur)
    # Geometric-product chain: R0 · R1 · x — the "rotor product first" path.
    rprod = rotors[0]
    for r in rotors[1:]:
        rprod = ga.geometric_product(rprod, r)
    composed = ga.geometric_product(rprod, x)
    # Grade-2 projection of the sandwiched result.
    bivec = ga.grade_projection(cur, 2)
    # Norm-squared as scalar invariant.
    norm_sq = ga.norm_squared(cur)
    return composed, bivec.coefficients, np.asarray(norm_sq)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


class CliffordCoreModel:
    """Tiny library-layer model composing GA primitives.

    Forward (deterministic from cfg.seed):

      x : (B, dim) multivector coefficients
      → tile_multivectors(coeffs)
      → RotorSampler ── n_rotors rotors
      → for each rotor: rotor_sandwich(R, x)
      → grade_projection(x_sandwiched, 2)
      → composed = R0 · R1 · ... · x (geometric_product chain)
      → norm_squared(composed) as scalar output

    Returns (final_multivector, bivector_part, norm_sq) — matching the
    oracle's tuple shape.
    """

    def __init__(self, cfg: CliffordCoreConfig):
        self.cfg = cfg
        self._algebra = ga.Cl(*cfg.algebra_signature)
        root = RNGKey.from_seed(cfg.seed, name="clifford_core")
        self._sampler_key = root.fold_in("sampler")

    @property
    def algebra(self) -> ga.Cl:
        return self._algebra

    def forward(self, x_coeffs: np.ndarray) -> tuple[Multivector, np.ndarray, np.ndarray]:
        cfg = self.cfg
        if x_coeffs.shape != (cfg.B, self._algebra.dim):
            raise ValueError(
                f"input coeffs {x_coeffs.shape} != (B={cfg.B}, dim={self._algebra.dim})"
            )
        # Tile pass-through (visible to compiler eventually).
        _ = tile_multivectors(x_coeffs, cfg.tile)
        x = Multivector(x_coeffs.astype(np.float32, copy=False), self._algebra)
        # Sample n_rotors rotors deterministically from the seed.
        sampler = RotorSampler(self._algebra, self._sampler_key)
        rotors = [sampler.next_rotor() for _ in range(cfg.n_rotors)]
        # Sandwich chain.
        cur = x
        for r in rotors:
            cur = ga.rotor_sandwich(r, cur)
        # Composed rotor-product applied to x.
        rprod = rotors[0]
        for r in rotors[1:]:
            rprod = ga.geometric_product(rprod, r)
        composed = ga.geometric_product(rprod, x)
        # Grade-2 projection of the sandwiched result.
        bivec = ga.grade_projection(cur, 2)
        # Scalar invariant.
        norm_sq = ga.norm_squared(cur)
        return composed, bivec.coefficients, np.asarray(norm_sq)

    __call__ = forward


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark harness
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CliffordCoreResult:
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


class CliffordCoreBenchmark:
    """Small reference benchmark for the GA composition."""

    BACKEND = "tessera-reference"
    OP = "clifford_core_forward"
    DEVICE = "cpu"
    VERSION = "pre-alpha"

    def __init__(self, *, warmup: int = 1, reps: int = 3):
        self.warmup = int(warmup)
        self.reps = int(reps)

    @staticmethod
    def make_input(cfg: CliffordCoreConfig) -> np.ndarray:
        rng = np.random.default_rng(cfg.seed ^ 0xCAFFE2)
        dim = 2 ** sum(cfg.algebra_signature)
        return rng.standard_normal((cfg.B, dim)).astype(np.float32)

    def run_one(self, cfg: CliffordCoreConfig) -> CliffordCoreResult:
        model = CliffordCoreModel(cfg)
        x = self.make_input(cfg)
        for _ in range(self.warmup):
            model(x)
        start = time.perf_counter()
        last = None
        for _ in range(self.reps):
            last = model(x)
        elapsed = (time.perf_counter() - start) / max(self.reps, 1)
        # Determinism: two model instances with same cfg ⇒ identical output.
        m2 = CliffordCoreModel(cfg)
        a_composed, a_bivec, a_nsq = model(x)
        b_composed, b_bivec, b_nsq = m2(x)
        determinism_ok = (
            np.array_equal(a_composed.coefficients, b_composed.coefficients)
            and np.array_equal(a_bivec, b_bivec)
            and np.array_equal(a_nsq, b_nsq)
        )
        samples = cfg.B
        throughput_msps = (samples / 1e6) / max(elapsed, 1e-12)
        bytes_per_step = (
            samples * model.algebra.dim * 4    # x
            + samples * model.algebra.dim * 4  # composed
            + samples * 4                       # norm_sq
        )
        memory_bw_gb_s = (bytes_per_step / 1e9) / max(elapsed, 1e-12)
        return CliffordCoreResult(
            backend=self.BACKEND,
            op=self.OP,
            shape={
                "B": cfg.B,
                "algebra_signature": list(cfg.algebra_signature),
                "tile": cfg.tile,
                "n_rotors": cfg.n_rotors,
            },
            dtype="fp32",
            latency_ms=elapsed * 1000.0,
            throughput_msps=throughput_msps,
            memory_bw_gb_s=memory_bw_gb_s,
            device=self.DEVICE,
            tessera_version=self.VERSION,
            determinism_ok=determinism_ok,
        )

    def run(self, configs: Sequence[CliffordCoreConfig]) -> list[CliffordCoreResult]:
        return [self.run_one(c) for c in configs]

    def report(self, results: Sequence[CliffordCoreResult]) -> None:
        if not results:
            return
        print(f"{'shape':<54} {'latency_ms':>12} {'thr_msps':>10} {'bw_gb/s':>10} det")
        for r in results:
            shape = r.shape
            label = (
                f"B={shape['B']} Cl{tuple(shape['algebra_signature'])} "
                f"tile={shape['tile']} rotors={shape['n_rotors']}"
            )
            det = "ok" if r.determinism_ok else "FAIL"
            print(
                f"{label:<54} {r.latency_ms:>12.2f} "
                f"{r.throughput_msps:>10.3f} {r.memory_bw_gb_s:>10.3f} {det}"
            )

    def to_json(self, results: Sequence[CliffordCoreResult], path: str) -> None:
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Default sweep + CLI
# ─────────────────────────────────────────────────────────────────────────────


def default_sweep() -> list[CliffordCoreConfig]:
    return [
        CliffordCoreConfig(B=4,  tile=2, n_rotors=1, seed=0),
        CliffordCoreConfig(B=8,  tile=4, n_rotors=2, seed=0),
        CliffordCoreConfig(B=16, tile=4, n_rotors=3, seed=0),
    ]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps",   type=int, default=3)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    bench = CliffordCoreBenchmark(warmup=args.warmup, reps=args.reps)
    results = bench.run(default_sweep())
    bench.report(results)
    if args.output:
        bench.to_json(results, args.output)
        print(f"\nWrote {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
