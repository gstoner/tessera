"""Host-side evaluators for ROCm dynamic-LDS launch expressions.

The current compiler records ``aligned_sum_of_slot_maxima``. SSA-interfering
arenas occupy distinct aligned slots; arenas with non-overlapping CFG lifetimes
share a slot sized to their maximum runtime byte count. The older
``max_of_aligned_sums`` evaluator remains for retained release packets.
"""

from __future__ import annotations

from collections.abc import Sequence


def align_up(value: int, alignment: int = 16) -> int:
    if value < 0:
        raise ValueError("dynamic LDS byte counts must be non-negative")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("dynamic LDS alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def packed_path_layout(
    arena_bytes: Sequence[int], alignment: int = 16
) -> tuple[tuple[int, ...], int]:
    """Return path-relative offsets and the aligned path total."""
    offsets: list[int] = []
    cursor = 0
    for size in arena_bytes:
        cursor = align_up(cursor, alignment)
        offsets.append(cursor)
        if size < 0:
            raise ValueError("dynamic LDS byte counts must be non-negative")
        cursor += int(size)
    return tuple(offsets), align_up(cursor, alignment)


def path_max_launch_bytes(
    paths: Sequence[Sequence[int]], alignment: int = 16
) -> int:
    """Evaluate ``max(path aligned-sum)`` for one HIP launch."""
    if not paths:
        return 0
    return max(packed_path_layout(path, alignment)[1] for path in paths)


def interference_slot_layout(
    slots: Sequence[Sequence[int]], alignment: int = 16
) -> tuple[tuple[int, ...], int]:
    """Return slot offsets and ``aligned_sum(max(slot alternatives))``."""
    slot_sizes: list[int] = []
    for alternatives in slots:
        if not alternatives:
            raise ValueError("dynamic LDS interference slots cannot be empty")
        if any(size < 0 for size in alternatives):
            raise ValueError("dynamic LDS byte counts must be non-negative")
        slot_sizes.append(max(int(size) for size in alternatives))
    return packed_path_layout(slot_sizes, alignment)


def interference_slot_launch_bytes(
    slots: Sequence[Sequence[int]], alignment: int = 16
) -> int:
    """Evaluate the current CFG-lifetime-aware HIP launch expression."""
    if not slots:
        return 0
    return interference_slot_layout(slots, alignment)[1]


__all__ = [
    "align_up",
    "interference_slot_launch_bytes",
    "interference_slot_layout",
    "packed_path_layout",
    "path_max_launch_bytes",
]
