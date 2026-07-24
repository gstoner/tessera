"""Host-side evaluator for ROCm dynamic-LDS launch expressions.

The compiler records ``max_of_aligned_sums``: arenas on one control-flow path
are packed, while mutually exclusive paths reuse offset zero.  Keeping this
small evaluator shared by runtime launchers and benchmarks prevents the host
from accidentally summing branch-local storage or disagreeing on alignment.
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


__all__ = ["align_up", "packed_path_layout", "path_max_launch_bytes"]
