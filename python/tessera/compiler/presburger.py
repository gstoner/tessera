"""Typed integer-affine constraints carried from Graph IR into MLIR.

The carrier is deliberately coefficient based.  It does not ask a C++ pass to
re-parse Python expression strings, and nonlinear products cannot accidentally
enter the Presburger proof domain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence


PRESBURGER_SCHEMA = "tessera.presburger_constraints.v1"


@dataclass(frozen=True)
class PresburgerConstraint:
    """An integer-affine relation ``sum(coeff[i] * symbol[i]) + c R 0``."""

    relation: Literal["eq", "ge", "mod"]
    coefficients: tuple[int, ...]
    constant: int = 0
    modulus: int | None = None
    remainder: int = 0

    def __post_init__(self) -> None:
        if self.relation not in {"eq", "ge", "mod"}:
            raise ValueError("Presburger relation must be 'eq', 'ge', or 'mod'")
        if not self.coefficients:
            raise ValueError("Presburger constraints require at least one coefficient")
        object.__setattr__(self, "coefficients", tuple(map(int, self.coefficients)))
        object.__setattr__(self, "constant", int(self.constant))
        object.__setattr__(self, "remainder", int(self.remainder))
        if self.relation == "mod":
            if not isinstance(self.modulus, int) or self.modulus < 2:
                raise ValueError("mod constraints require modulus >= 2")
            if not 0 <= self.remainder < self.modulus:
                raise ValueError("mod remainder must satisfy 0 <= remainder < modulus")
        elif self.modulus is not None or self.remainder:
            raise ValueError("eq/ge constraints cannot carry modulus or remainder")

    def evaluate(self, symbols: Sequence[str], bindings: Mapping[str, int]) -> bool | None:
        if len(symbols) != len(self.coefficients):
            raise ValueError("symbol/coefficient arity mismatch")
        if any(symbol not in bindings for symbol in symbols):
            return None
        value = self.constant + sum(
            coefficient * int(bindings[symbol])
            for symbol, coefficient in zip(symbols, self.coefficients)
        )
        if self.relation == "eq":
            return value == 0
        if self.relation == "ge":
            return value >= 0
        assert self.modulus is not None
        return value % self.modulus == self.remainder


@dataclass(frozen=True)
class PresburgerSystem:
    """Versioned, content-addressed conjunction of integer constraints."""

    symbols: tuple[str, ...]
    constraints: tuple[PresburgerConstraint, ...]

    def __post_init__(self) -> None:
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Presburger symbols must be non-empty and unique")
        if any(not symbol or any(char.isspace() for char in symbol) for symbol in self.symbols):
            raise ValueError("Presburger symbols cannot be empty or contain whitespace")
        if not self.constraints:
            raise ValueError("Presburger systems require at least one constraint")
        for constraint in self.constraints:
            if len(constraint.coefficients) != len(self.symbols):
                raise ValueError("every Presburger coefficient row must cover every symbol")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PRESBURGER_SCHEMA,
            "symbols": list(self.symbols),
            "constraints": [
                {
                    "relation": constraint.relation,
                    "coefficients": list(constraint.coefficients),
                    "constant": constraint.constant,
                    "modulus": constraint.modulus,
                    "remainder": constraint.remainder,
                }
                for constraint in self.constraints
            ],
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_mlir_attr(self) -> str:
        constraints = ", ".join(
            f'{{relation = "{constraint.relation}", coefficients = array<i64: '
            f'{", ".join(map(str, constraint.coefficients))}>, '
            f"constant = {constraint.constant} : i64"
            + (
                f", modulus = {constraint.modulus} : i64, "
                f"remainder = {constraint.remainder} : i64"
                if constraint.relation == "mod"
                else ""
            )
            + "}"
            for constraint in self.constraints
        )
        symbols = ", ".join(json.dumps(symbol) for symbol in self.symbols)
        return (
            "{version = 1 : i64, symbols = ["
            + symbols
            + "], constraints = ["
            + constraints
            + "]}"
        )

    def check_witness(self, bindings: Mapping[str, int]) -> bool | None:
        results = [constraint.evaluate(self.symbols, bindings) for constraint in self.constraints]
        if any(result is False for result in results):
            return False
        return True if all(result is True for result in results) else None


def attach_presburger_system(function: Any, system: PresburgerSystem) -> None:
    """Attach the typed carrier and its content digest to a Graph function."""

    function.fn_attrs["tessera.presburger_constraints"] = system.to_mlir_attr()
    function.fn_attrs["tessera.presburger_digest"] = json.dumps(system.digest)


def presburger_system_from_constraints(
    constraints: Iterable[Any],
) -> PresburgerSystem | None:
    """Translate the integer-affine subset of public frontend constraints.

    Divisibility is represented exactly as a modular row.  The C++ consumer
    introduces one existential quotient per row; no affine approximation is
    used.
    """

    from .constraints import Divisible, Equal, Range

    affine = tuple(
        item for item in constraints if isinstance(item, (Divisible, Equal, Range))
    )
    if not affine:
        return None
    symbols = tuple(sorted({name for item in affine for name in item.dim_names()}))
    positions = {name: index for index, name in enumerate(symbols)}

    def row(coefficients: Iterable[tuple[str, int]]) -> tuple[int, ...]:
        values = [0] * len(symbols)
        for name, coefficient in coefficients:
            values[positions[name]] += coefficient
        return tuple(values)

    rows: list[PresburgerConstraint] = []
    for item in affine:
        if isinstance(item, Equal):
            rows.append(
                PresburgerConstraint(
                    "eq", row(((item.dim_a, 1), (item.dim_b, -1))), 0
                )
            )
        elif isinstance(item, Range):
            rows.extend(
                (
                    PresburgerConstraint("ge", row(((item.dim, 1),)), -item.lo),
                    PresburgerConstraint("ge", row(((item.dim, -1),)), item.hi),
                )
            )
        else:
            rows.append(
                PresburgerConstraint(
                    "mod", row(((item.dim, 1),)), modulus=item.divisor
                )
            )
    return PresburgerSystem(symbols, tuple(rows))


__all__ = [
    "PRESBURGER_SCHEMA",
    "PresburgerConstraint",
    "PresburgerSystem",
    "attach_presburger_system",
    "presburger_system_from_constraints",
]
