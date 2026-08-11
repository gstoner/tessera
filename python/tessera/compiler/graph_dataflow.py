"""Fail-closed Graph IR dataflow queries shared by compiler clients.

This is the Python query surface for the W2.1 contract implemented by the C++
``GraphDataflowAnalysis``.  It intentionally operates on emitted ``IROp``
records, not Python source names.  Every cached snapshot is tied to a structural
digest; mutating the graph invalidates the snapshot and queries return ``top``
until the caller explicitly recomputes it.

The C++ analysis is the production authority.  This module is useful to Python
pipeline construction, audit tooling, and parity tests before MLIR parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Optional

from .effects import Effect, registered_op_effect
from .graph_ir import GraphIRFunction, IROp, IRType
from .op_catalog import get_op_spec


def _ssa(name: str) -> str:
    return name.strip().lstrip("%")


def _shape(ir_type: Optional[IRType]) -> Optional[tuple[str, ...]]:
    if ir_type is None or ir_type.rank is None:
        return None
    return tuple(ir_type.shape)


@dataclass(frozen=True)
class ValueFact:
    """Independent shape and alias facts; ``None`` is conservative top."""

    shape: Optional[tuple[str, ...]] = None
    alias_roots: Optional[frozenset[str]] = None
    live: bool = True

    @classmethod
    def top(cls) -> "ValueFact":
        return cls()


class GraphDataflow:
    """Invalidatable product analysis for one straight-line Graph function."""

    schema_version = 1

    def __init__(self, function: GraphIRFunction):
        self.function = function
        self._digest = ""
        self._facts: dict[str, ValueFact] = {}
        self._producers: dict[str, IROp] = {}
        self._active: frozenset[int] = frozenset()
        self._valid = False

    @staticmethod
    def _structural_digest(function: GraphIRFunction) -> str:
        payload = {
            "args": [(a.name, str(a.ir_type)) for a in function.args],
            "returns": list(function.return_values),
            "ops": [
                {
                    "results": op.result_names,
                    "name": op.op_name,
                    "operands": list(op.operands),
                    "types": list(op.operand_types),
                    "result_type": op.result_type,
                    "attrs": op.attrs,
                    "kwargs": sorted((str(k), repr(v)) for k, v in op.kwargs.items()),
                    "inferred": [str(t) for t in op.inferred_types]
                    or ([str(op.inferred_type)] if op.inferred_type else []),
                }
                for op in function.body
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @property
    def valid(self) -> bool:
        if self._valid and self._digest != self._structural_digest(self.function):
            self.invalidate()
        return self._valid

    @property
    def digest(self) -> str:
        return self._digest if self.valid else ""

    def invalidate(self) -> None:
        self._valid = False
        self._facts.clear()
        self._producers.clear()
        self._active = frozenset()

    def recompute(self) -> "GraphDataflow":
        self.invalidate()
        facts: dict[str, ValueFact] = {}
        producers: dict[str, IROp] = {}
        for arg in self.function.args:
            name = _ssa(arg.name)
            facts[name] = ValueFact(_shape(arg.ir_type), frozenset({name}), False)

        for op in self.function.body:
            operand_facts = [facts.get(_ssa(name), ValueFact.top()) for name in op.operands]
            spec = get_op_spec(op.op_name)
            aliasing = op.kwargs.get("tessera.aliasing")
            if aliasing is None and spec is not None:
                aliasing = spec.aliasing
            effect = registered_op_effect(op.op_name, op.kwargs)
            result_types = list(op.inferred_types)
            if not result_types and op.inferred_type is not None:
                result_types = [op.inferred_type]

            for index, name in enumerate(op.result_names):
                result_name = _ssa(name)
                result_shape = _shape(result_types[index]) if index < len(result_types) else None
                if aliasing and aliasing != "none":
                    try:
                        operand_index = int(str(aliasing).rsplit("_", 1)[-1])
                        roots = operand_facts[operand_index].alias_roots
                    except (ValueError, IndexError):
                        roots = None
                elif effect == Effect.pure and spec is not None:
                    roots = frozenset({result_name})
                else:
                    # Unregistered/effectful producers may alias hidden state.
                    roots = None
                facts[result_name] = ValueFact(result_shape, roots, False)
                producers[result_name] = op

        live = {_ssa(name) for name in self.function.return_values}
        for op in self.function.body:
            if registered_op_effect(op.op_name, op.kwargs) > Effect.pure:
                live.update(_ssa(name) for name in op.operands)
        worklist = list(live)
        while worklist:
            producer = producers.get(worklist.pop())
            if producer is None:
                continue
            for operand in map(_ssa, producer.operands):
                if operand not in live:
                    live.add(operand)
                    worklist.append(operand)

        active_ops: set[int] = set()
        worklist = [_ssa(name) for name in self.function.return_values]
        visited: set[str] = set()
        while worklist:
            name = worklist.pop()
            if name in visited:
                continue
            visited.add(name)
            producer = producers.get(name)
            if producer is None:
                continue
            active_ops.add(id(producer))
            if producer.op_name != "tessera.stop_gradient":
                worklist.extend(map(_ssa, producer.operands))

        self._facts = {
            name: ValueFact(fact.shape, fact.alias_roots, name in live)
            for name, fact in facts.items()
        }
        self._producers = producers
        self._active = frozenset(active_ops)
        self._digest = self._structural_digest(self.function)
        self._valid = True
        return self

    def fact(self, value: str) -> ValueFact:
        if not self.valid:
            return ValueFact.top()
        return self._facts.get(_ssa(value), ValueFact.top())

    def may_alias(self, lhs: str, rhs: str) -> bool:
        if not self.valid:
            return True
        left = self.fact(lhs).alias_roots
        right = self.fact(rhs).alias_roots
        return left is None or right is None or bool(left & right)

    def is_active(self, op: IROp) -> bool:
        # Unknown/stale activity is conservatively active.
        return not self.valid or id(op) in self._active

    def has_memory_dependence(self, lhs: IROp, rhs: IROp) -> bool:
        if not self.valid:
            return True
        left = registered_op_effect(lhs.op_name, lhs.kwargs)
        right = registered_op_effect(rhs.op_name, rhs.kwargs)
        if left == Effect.pure and right == Effect.pure:
            return False
        # IROp currently has no value-scoped MemoryEffectOpInterface mirror.
        # Effectful pairs therefore remain top instead of guessing from names.
        return True

    def snapshot(self) -> Mapping[str, ValueFact]:
        return dict(self._facts) if self.valid else {}


def analyze_graph_dataflow(function: GraphIRFunction) -> GraphDataflow:
    """Build the canonical Python query object and run it to a fixed point."""

    return GraphDataflow(function).recompute()


__all__ = ["GraphDataflow", "ValueFact", "analyze_graph_dataflow"]
