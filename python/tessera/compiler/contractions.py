"""MSW-7 Python contraction normal form and declared identity operands.

This is reference algebra, not an IR lowering. Ordinary einsums keep NumPy's
execution/AD authority; the canonical spelling is consumed by that surface.
Only explicit Delta/Epsilon operands authorize identity rewrites: dense arrays
are never guessed to be identities. Operand order and output axis order remain
significant. Floating-point reassociation is confined to the opt-in identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import string
import numpy as np


@dataclass(frozen=True)
class Contraction:
    inputs: tuple[str, ...]
    output: str

    @property
    def key(self) -> str:
        return ",".join(self.inputs) + "->" + self.output


def normal_form(spec: str) -> Contraction:
    """Alpha-normalize indices, resolving implicit output BEFORE renaming."""
    spec = "".join(spec.split())
    if spec.count("->") > 1:
        raise ValueError("contraction has more than one output separator")
    lhs, *rhs = spec.split("->")
    inputs = tuple(lhs.split(","))
    for term in (*inputs, *rhs):
        if term.count("...") > 1 or any(c not in string.ascii_letters for c in term.replace("...", "")):
            raise ValueError("contraction indices must be ASCII letters or one ellipsis")
    labels = "".join(inputs).replace("...", "")
    output = (
        rhs[0]
        if rhs
        else ("..." if "..." in lhs else "") + "".join(sorted(c for c in set(labels) if labels.count(c) == 1))
    )
    out_labels = output.replace("...", "")
    if len(set(out_labels)) != len(out_labels) or any(c not in labels for c in out_labels):
        raise ValueError("output indices must be unique and present in an operand")
    if "..." in output and "..." not in lhs:
        raise ValueError("output ellipsis has no input ellipsis")
    mapping: dict[str, str] = {}

    def rename(term):
        result = ""
        for c in term:
            if c == ".":
                result += c
            else:
                if c not in mapping:
                    mapping[c] = string.ascii_letters[len(mapping)]
                result += mapping[c]
        return result

    normalized = tuple(rename(term) for term in inputs)
    return Contraction(normalized, rename(output))


@dataclass(frozen=True)
class Delta:
    dimension: int

    def __post_init__(self):
        if not isinstance(self.dimension, int) or isinstance(self.dimension, bool) or self.dimension <= 0:
            raise ValueError("Delta dimension must be a positive integer")


@dataclass(frozen=True)
class Epsilon:
    """Positive three-dimensional Levi-Civita symbol; no rank-three storage."""


def _parity(order):
    return -1 if sum(order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order))) % 2 else 1


def contract(spec: str, *operands):
    """Evaluate a contraction, eliminating declared delta/epsilon identities.

    Explicit-output, no-ellipsis identity fragment; dense-only contractions
    retain the complete ordinary einsum surface. Epsilon pairs expand to six
    products of deltas (the determinant identity); a lone epsilon is evaluated
    from its six nonzero entries without allocating a rank-three tensor.
    """
    form = normal_form(spec)
    if len(form.inputs) != len(operands):
        raise ValueError("contraction operand count mismatch")
    if not any(isinstance(x, (Delta, Epsilon)) for x in operands):
        return np.einsum(form.key, *operands)
    if "..." in form.key or "->" not in spec:
        raise ValueError("identity rewrites require explicit output without ellipsis")
    terms = list(form.inputs)
    values = list(operands)
    dimensions: dict[str, int] = {}
    for term, value in zip(terms, values):
        shape = (
            (value.dimension,) * 2
            if isinstance(value, Delta)
            else (3,) * 3
            if isinstance(value, Epsilon)
            else np.shape(value)
        )
        if len(term) != len(shape):
            raise ValueError("identity contraction operand rank mismatch")
        for c, n in zip(term, shape):
            if c in dimensions and dimensions[c] != n:
                raise ValueError("identity contraction requires equal index extents")
            dimensions[c] = n
    return _evaluate(terms, values, form.output, dimensions)


def _evaluate(terms, values, output, dimensions):
    eps = [i for i, v in enumerate(values) if isinstance(v, Epsilon)]
    if len(eps) >= 2:
        i, j = eps[:2]
        a, b = terms[i], terms[j]
        rest_t = [t for k, t in enumerate(terms) if k not in (i, j)]
        rest_v = [v for k, v in enumerate(values) if k not in (i, j)]
        total = None
        for p in permutations(range(3)):
            pair_terms = [a[k] + b[p[k]] for k in range(3)]
            term = _evaluate(rest_t + pair_terms, rest_v + [Delta(3)] * 3, output, dimensions) * _parity(p)
            total = term if total is None else total + term
        return total
    if eps:
        i = eps[0]
        indices = terms[i]
        result = np.zeros(
            tuple(dimensions[c] for c in output),
            dtype=np.result_type(
                *[np.asarray(v).dtype for v in values if not isinstance(v, (Delta, Epsilon))], np.float64
            ),
        )
        # Antisymmetry canonically sorts the index order, carrying its sign.
        order = sorted(range(3), key=lambda k: indices[k])
        indices = "".join(indices[k] for k in order)
        sign = _parity(order)
        if len(set(indices)) < 3:
            return result
        for p in permutations(range(3)):
            assignment = dict(zip(indices, p))
            vt, vv = [], []
            for k, (t, v) in enumerate(zip(terms, values)):
                if k == i:
                    continue
                arr = np.eye(v.dimension) if isinstance(v, Delta) else np.asarray(v)
                selector = tuple(assignment.get(c, slice(None)) for c in t)
                vt.append("".join(c for c in t if c not in assignment))
                vv.append(arr[selector])
            remaining = "".join(c for c in output if c not in assignment)
            # Output-only epsilon indices are fixed by assignment; all other
            # indices are accounted for by the remaining operands.
            value = np.einsum(",".join(vt) + "->" + remaining, *vv) if vv else np.array(1.0)
            result[tuple(assignment.get(c, slice(None)) for c in output)] += sign * _parity(p) * value
        return result
    # Eliminate delta indices only if this does not identify two distinct
    # output axes. Remaining deltas then represent genuine output identities.
    terms, values = list(terms), list(values)
    factor = 1
    while True:
        changed = False
        for k, (t, v) in enumerate(zip(terms, values)):
            if not isinstance(v, Delta):
                continue
            a, b = t
            rest = "".join(terms[:k] + terms[k + 1 :])
            if a == b:
                if a not in rest and a not in output:
                    factor *= v.dimension
                elif a not in rest:  # diagonal identity leaves a vector of ones
                    terms[k], values[k] = a, np.ones(v.dimension)
                    changed = True
                    break
            elif a in output and b in output:
                continue
            else:
                keep, drop = (a, b) if a in output else (b, a)
                terms = [t.replace(drop, keep) for t in terms]
                output = output.replace(drop, keep)
                if keep not in "".join(terms[:k] + terms[k + 1 :]) and keep not in output:
                    factor *= v.dimension
            del terms[k]
            del values[k]
            changed = True
            break
        if not changed:
            break
    for c in output:
        if c not in "".join(terms):
            terms.append(c)
            values.append(np.ones(dimensions[c]))
    if not values:
        return np.array(factor)
    dense = [np.eye(v.dimension) if isinstance(v, Delta) else v for v in values]
    result = np.einsum(",".join(terms) + "->" + output, *dense)
    return result if factor == 1 else factor * result


def canonical_call(args, kwargs):
    """One call binding for forward execution, tape, JVP and graph tracing."""
    options = dict(kwargs)
    positional = bool(args and isinstance(args[0], str))
    supplied = int(positional) + int("spec" in options) + int("equation" in options)
    if supplied != 1:
        raise TypeError("einsum requires exactly one spec/equation")
    if positional:
        spec, args = args[0], args[1:]
    elif "spec" in options:
        spec = options.pop("spec")
    else:
        spec = options.pop("equation")
    options["equation"] = normal_form(spec).key
    return args, options
