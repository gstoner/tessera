"""Measured rematerialization-cost corpus consumed by Graph compilation.

The shared MLIR pass deliberately consumes an integer
``tessera.remat_cost_ns`` attribute rather than importing target policy. This
module is the target-aware boundary that resolves an exact measured row and
returns the value to stamp on the Graph producer.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


_DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/baselines/core_compiler_rematerialization_gfx1151_avx512.json"
)


@lru_cache(maxsize=1)
def load_rematerialization_cost_corpus() -> dict[str, Any]:
    path = Path(os.environ.get("TESSERA_REMAT_COST_CORPUS", _DEFAULT_CORPUS))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tessera.rematerialization.cross-target.v1":
        raise ValueError(f"unsupported rematerialization corpus schema: {path}")
    return payload


def measured_rematerialization_cost_ns(
    target: str,
    operation: str,
    shape: tuple[int, ...],
    *,
    consumer: str | None = None,
) -> int | None:
    """Return an exact-device, exact-shape recompute cost or ``None``.

    Deliberately do not extrapolate across shapes: the MLIR pass's analytical
    fallback is more honest than presenting a scaled host-wall measurement as
    exact target evidence.
    """
    normalized_target = "x86" if target in ("cpu", "x86", "avx512") else target
    wanted_shape = list(shape)
    matches = []
    for row in load_rematerialization_cost_corpus()["rows"]:
        if (
            row.get("target") == normalized_target
            and row.get("remat_attribute_operation") == operation
            and row.get("shape") == wanted_shape
            and (consumer is None or row.get("operation") == consumer)
        ):
            matches.append(int(row["recompute_cost_ns"]))
    # Once a shape has multiple fused consumers, a producer-only query is
    # ambiguous. Do not silently pick the first row and make corpus ordering a
    # cost-model policy. Callers must name the consumer chain.
    return matches[0] if len(matches) == 1 else None


__all__ = [
    "load_rematerialization_cost_corpus",
    "measured_rematerialization_cost_ns",
]
