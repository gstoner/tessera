"""Target-explicit hardware-free locality and LDS bank-conflict facts.

These helpers intentionally report structural facts, not predicted latency.
Calibration decides whether either fact is useful for ranking kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


Coordinate = int | Sequence[int]


@dataclass(frozen=True)
class StepDistanceHistogram:
    sequential: int
    cache_line_jump: int
    large_jump: int

    @property
    def total(self) -> int:
        return self.sequential + self.cache_line_jump + self.large_jump

    @property
    def locality_score(self) -> float:
        """Predeclared ordinal score: sequential=1, line jump=1/2, large=0."""
        if self.total == 0:
            return 1.0
        return (self.sequential + 0.5 * self.cache_line_jump) / self.total


@dataclass(frozen=True)
class BankGeometry:
    architecture: str
    wave_size: int
    bank_count: int
    bank_width_bytes: int
    phases: tuple[tuple[int, ...], ...]
    source: str

    def __post_init__(self) -> None:
        lanes = tuple(lane for phase in self.phases for lane in phase)
        if sorted(lanes) != list(range(self.wave_size)):
            raise ValueError("bank phases must cover every wave lane exactly once")
        if self.bank_count < 1 or self.bank_width_bytes < 1:
            raise ValueError("bank count and width must be positive")


GFX1151_BANK_GEOMETRY = BankGeometry(
    architecture="gfx1151",
    wave_size=32,
    bank_count=32,
    bank_width_bytes=4,
    phases=(tuple(range(32)),),
    source=(
        "RDNA3.5 ISA 2.1 (wave32), 12.1 (64 WGP banks split into two "
        "32-bank SIMD32-affiliated sets), and 12.5 (wave32 one-cycle "
        "indexed access absent conflicts)"
    ),
)


def _as_coordinate(value: Coordinate) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    return tuple(int(component) for component in value)


def step_distance_histogram(accesses: Iterable[Coordinate]) -> StepDistanceHistogram:
    """Classify consecutive Manhattan distances using the AMD survey cutoffs."""
    values = [_as_coordinate(value) for value in accesses]
    sequential = cache_line = large = 0
    for left, right in zip(values, values[1:]):
        if len(left) != len(right):
            raise ValueError("all access coordinates must have equal rank")
        distance = sum(abs(a - b) for a, b in zip(left, right))
        if distance <= 1:
            sequential += 1
        elif distance <= 16:
            cache_line += 1
        else:
            large += 1
    return StepDistanceHistogram(sequential, cache_line, large)


def bank_conflict_degrees(
    byte_offsets: Sequence[int],
    geometry: BankGeometry = GFX1151_BANK_GEOMETRY,
) -> tuple[int, ...]:
    """Return the maximum same-bank lane multiplicity for every issue phase."""
    if len(byte_offsets) != geometry.wave_size:
        raise ValueError(
            f"expected {geometry.wave_size} lane offsets, got {len(byte_offsets)}")
    result: list[int] = []
    for phase in geometry.phases:
        counts: dict[int, int] = {}
        for lane in phase:
            bank = (
                int(byte_offsets[lane]) // geometry.bank_width_bytes
            ) % geometry.bank_count
            counts[bank] = counts.get(bank, 0) + 1
        result.append(max(counts.values(), default=0))
    return tuple(result)


__all__ = [
    "BankGeometry",
    "GFX1151_BANK_GEOMETRY",
    "StepDistanceHistogram",
    "bank_conflict_degrees",
    "step_distance_histogram",
]
