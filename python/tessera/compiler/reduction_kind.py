"""The combiner of a ``tessera.reduce``, read the way the Graph IR states it.

``tessera.reduce`` carries its combiner as ``kind`` -- a required
``ReductionKindAttr`` in ``TesseraOps.td`` whose legal set is
:data:`REDUCTION_KINDS`. Both frontends canonicalize the public ``op=``
spelling into ``kind`` and drop ``op`` (``graph_ir.py`` and ``trace.py``), so a
consumer that reads ``op`` sees nothing and must not guess.

Found 2026-09-26 (NVIDIA pre-PR review): the CPU runtime executor and the CPU
compiler path both read ``kwargs.get("op", "sum")``, so
``tessera.ops.reduce(x, op="max", axis=1)`` under ``@tessera.jit`` emitted
``kind = "max"`` and returned the SUM. That is the fail-open default Decision
#21a forbids: a semantic key never defaults. Every consumer that executes a
reduce reads its combiner through :func:`reduction_kind`, which

* reads ``kind``; accepts the legacy ``op`` spelling only when ``kind`` is
  absent, and refuses the two disagreeing;
* fails closed with ``E_REDUCTION_KIND_MISSING`` when neither is present;
* refuses any value outside the ODS set with ``E_REDUCTION_KIND_UNSUPPORTED``
  (``amax``/``amin`` are op names that canonicalize to max/min before they
  reach a ``tessera.reduce``, not kinds).
"""

from __future__ import annotations

from typing import Any, Mapping

#: Exactly ``Tessera_ReductionKindAttr`` in ``src/compiler/ir/TesseraOps.td``;
#: ``tests/unit/test_reduction_kind.py`` pins the two against each other.
REDUCTION_KINDS: tuple[str, ...] = ("sum", "max", "min", "mean")


def _plain(value: Any) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1]
    return text


def reduction_kind(kwargs: Mapping[str, Any], *, where: str) -> str:
    """The reduction kind a ``tessera.reduce`` states, or a named refusal."""
    kind = kwargs.get("kind")
    legacy = kwargs.get("op")
    if kind is None and legacy is None:
        raise ValueError(
            f"E_REDUCTION_KIND_MISSING: {where}: tessera.reduce carries no 'kind' "
            f"(one of {', '.join(REDUCTION_KINDS)}); a reduction's combiner is a "
            "semantic key and is never defaulted (Decision #21a)")
    if kind is not None and legacy is not None and _plain(kind) != _plain(legacy):
        raise ValueError(
            f"E_REDUCTION_KIND_UNSUPPORTED: {where}: tessera.reduce states kind="
            f"{_plain(kind)!r} and op={_plain(legacy)!r}; refusing to pick one")
    value = _plain(kind if kind is not None else legacy)
    if value not in REDUCTION_KINDS:
        raise ValueError(
            f"E_REDUCTION_KIND_UNSUPPORTED: {where}: tessera.reduce kind {value!r} "
            f"is not one of {', '.join(REDUCTION_KINDS)}")
    return value


def apply_reduction(np: Any, kind: str, x: Any, *, axis: Any, keepdims: bool) -> Any:
    """Evaluate one ODS reduction kind with numpy (NaN propagates for max/min)."""
    fn = {"sum": np.sum, "max": np.max, "min": np.min, "mean": np.mean}[kind]
    return fn(x, axis=axis, keepdims=keepdims)


__all__ = ["REDUCTION_KINDS", "apply_reduction", "reduction_kind"]
