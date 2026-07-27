"""Target-neutral dynamic local-memory launch expressions.

Backends own the physical address space and schedule, but the host ABI needs a
single deterministic way to serialize and evaluate the byte count passed at
launch.  Expressions are intentionally small, total, and signed-i64 checked so
the compiler, cache, runtime, and audit packets cannot disagree about sizing.
"""

from __future__ import annotations

from collections.abc import Mapping
import math


SIGNED_I64_MAX = (1 << 63) - 1
EXPRESSION_SCHEMA = "tessera.dynamic_local_memory.expression.v1"


def align_up(value: int, alignment: int = 16) -> int:
    if value < 0:
        raise ValueError("dynamic local-memory byte counts must be non-negative")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(
            "dynamic local-memory alignment must be a positive power of two"
        )
    result = (value + alignment - 1) & -alignment
    if result > SIGNED_I64_MAX:
        raise ValueError("dynamic local-memory expression is outside signed-i64")
    return result


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"dynamic local-memory expression {field} must be an integer"
        )
    return value


def referenced_arguments(expression: Mapping[str, object]) -> frozenset[str]:
    """Validate ``expression`` and return the scalar arguments it references."""

    kind = expression.get("kind")
    if not isinstance(kind, str):
        raise ValueError("dynamic local-memory expression kind must be a string")
    if kind == "argument":
        name = expression.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                "dynamic local-memory argument expression needs a name"
            )
        return frozenset((name,))
    if kind == "constant":
        value = _integer(expression.get("value"), "value")
        if value < 0:
            raise ValueError(
                "dynamic local-memory byte counts must be non-negative"
            )
        return frozenset()

    raw_operands = expression.get("operands")
    if not isinstance(raw_operands, list) or not raw_operands:
        raise ValueError("dynamic local-memory expression requires operands")
    references: set[str] = set()
    for operand in raw_operands:
        if not isinstance(operand, Mapping):
            raise ValueError(
                "dynamic local-memory expression operands must be mappings"
            )
        references.update(referenced_arguments(operand))
    if kind in {"add", "multiply", "max", "path_max"}:
        return frozenset(references)
    if kind == "align_up":
        if len(raw_operands) != 1:
            raise ValueError("dynamic local-memory align_up requires one operand")
        alignment = _integer(expression.get("alignment", 16), "alignment")
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(
                "dynamic local-memory alignment must be a positive power of two"
            )
        return frozenset(references)
    raise ValueError(f"unsupported dynamic local-memory expression {kind!r}")


def evaluate_launch_expression(
    expression: Mapping[str, object], arguments: Mapping[str, int]
) -> int:
    """Evaluate a validated expression with checked signed-i64 arithmetic."""

    referenced_arguments(expression)
    kind = str(expression["kind"])
    if kind == "argument":
        name = str(expression["name"])
        if name not in arguments:
            raise ValueError(
                f"dynamic local-memory expression lacks argument {name!r}"
            )
        value = _integer(arguments[name], name)
    elif kind == "constant":
        value = _integer(expression["value"], "value")
    else:
        raw_operands = expression["operands"]
        if not isinstance(raw_operands, list):
            raise ValueError("dynamic local-memory expression requires operands")
        operands: list[int] = []
        for operand in raw_operands:
            if not isinstance(operand, Mapping):
                raise ValueError(
                    "dynamic local-memory expression operands must be mappings"
                )
            operands.append(evaluate_launch_expression(operand, arguments))
        if kind == "add":
            value = sum(operands)
        elif kind == "multiply":
            value = math.prod(operands)
        elif kind in {"max", "path_max"}:
            value = max(operands)
        elif kind == "align_up":
            value = align_up(
                operands[0],
                _integer(expression.get("alignment", 16), "alignment"),
            )
        else:  # Validated above; retained for type-checker totality.
            raise ValueError(
                f"unsupported dynamic local-memory expression {kind!r}"
            )
    if value < 0:
        raise ValueError("dynamic local-memory byte counts must be non-negative")
    if value > SIGNED_I64_MAX:
        raise ValueError("dynamic local-memory expression is outside signed-i64")
    return value


__all__ = [
    "EXPRESSION_SCHEMA",
    "SIGNED_I64_MAX",
    "align_up",
    "evaluate_launch_expression",
    "referenced_arguments",
]
