"""PK8a — Graph IR → ``.mtlpackage`` authoring hook.

This is the compiler-facing entry to PK8: instead of naming an op
("softmax") and passing shapes by hand, *recognize* an MPSGraph-lane region
in a real :class:`~tessera.compiler.graph_ir.GraphIRModule` and author the
matching packaged kernel. It is the mechanism ``@jit(target="apple_gpu")``
would call to emit an AOT packaged kernel for a recognized subgraph.

What's recognized today (all fp32, static 2-D shapes):

* **single op** — any op in :data:`tessera.apple_mlpkg.AUTHOR_OPS`
  (unary / rowop / norm / binary) → :func:`author_op_package`; plain
  ``matmul`` → :func:`author_matmul_package`.
* **fused chains** — ``matmul → softmax`` (→ ``matmul_softmax``),
  ``matmul → softmax → matmul`` (→ ``matmul_softmax_matmul``),
  ``rmsnorm → matmul`` (→ ``rmsnorm_matmul``) → :func:`author_chain_package`.

Recognition is intentionally conservative: anything it can't map to a known
authoring primitive returns ``None`` (the caller keeps the normal lowering
path). The recognizer is pure (no device needed) so it can be unit-tested
without a GPU; only the *author* step touches the runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tessera import apple_mlpkg as _mlpkg
from tessera.compiler.graph_ir import GraphIRModule

# Ops we can author as a single packaged kernel, keyed by the bare op name
# (``tessera.`` prefix stripped). matmul is handled separately (3-D shape).
_SINGLE_OPS = frozenset(_mlpkg.AUTHOR_OPS)
_NORM_OPS = frozenset({"rmsnorm", "layer_norm"})

# Fused chains recognized by op-name sequence (bare names, in body order).
# Each maps to (chain_name, dims_extractor_key).
_CHAIN_BY_SEQUENCE: dict[tuple[str, ...], str] = {
    ("matmul", "softmax"): "matmul_softmax",
    ("matmul", "softmax", "matmul"): "matmul_softmax_matmul",
    ("rmsnorm", "matmul"): "rmsnorm_matmul",
}

_TENSOR_RE = re.compile(r"tensor<([^>]*)>")


@dataclass(frozen=True)
class RecognizedOp:
    """A *shape-free* recognition of an authorable region — the op/chain
    identity without concrete dims. This is what fires on a live ``@jit``
    module, whose Graph IR carries op names but not static shapes
    (``tensor<?x?x?>``). Pair it with concrete input shapes (from example
    args) via :func:`plan_from_shapes` to get a dims-bearing
    :class:`AuthorPlan`.

    ``kind`` is ``"matmul"`` / ``"op"`` / ``"chain"``; ``name`` the op or
    chain name; ``arity`` the number of tensor inputs the region consumes;
    ``weighted`` flags norm gamma/beta; ``eps`` the norm epsilon.
    """

    kind: str
    name: str
    arity: int
    weighted: bool = False
    eps: float = 1e-5

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "name": self.name, "arity": self.arity,
            "weighted": self.weighted, "eps": self.eps,
        }


@dataclass(frozen=True)
class AuthorPlan:
    """A recognized authoring action. ``kind`` is ``"matmul"`` / ``"op"`` /
    ``"chain"``; ``name`` is the op or chain name; ``dims`` is the shape
    vector the authoring primitive expects; ``weighted`` flags norm gamma/
    beta inputs; ``eps`` is the norm epsilon."""

    kind: str
    name: str
    dims: tuple[int, ...]
    weighted: bool = False
    eps: float = 1e-5

    def author(self, out_path: str | Path) -> bool:
        """Execute this plan — author the ``.mtlpackage`` at ``out_path``.
        Returns the underlying ``author_*`` boolean (False if the runtime
        is unavailable)."""
        if self.kind == "matmul":
            m, k, n = self.dims
            return _mlpkg.author_matmul_package(out_path, m, k, n)
        if self.kind == "op":
            rows, cols = self.dims
            return _mlpkg.author_op_package(
                out_path, self.name, rows, cols, eps=self.eps,
                weighted=self.weighted,
            )
        if self.kind == "chain":
            return _mlpkg.author_chain_package(
                out_path, self.name, list(self.dims), eps=self.eps,
            )
        return False

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "name": self.name, "dims": list(self.dims),
            "weighted": self.weighted, "eps": self.eps,
        }

    @property
    def output_shape(self) -> Optional[tuple[int, ...]]:
        """The 2-D output shape the dispatched kernel produces, derived from
        ``dims``. Drives ``read_output_at`` byte-count + reshape when a jitted
        call dispatches through the authored package. ``None`` if unknown."""
        d = self.dims
        if self.kind == "matmul" and len(d) == 3:
            return (d[0], d[2])            # (M, N) = A[M,K] @ B[K,N]
        if self.kind == "op" and len(d) == 2:
            return (d[0], d[1])            # elementwise / rowop: same shape
        if self.kind == "chain":
            if self.name == "matmul_softmax" and len(d) == 3:
                return (d[0], d[2])        # (M, N)
            if self.name == "matmul_softmax_matmul" and len(d) == 4:
                return (d[0], d[3])        # (M, P)
            if self.name == "rmsnorm_matmul" and len(d) == 3:
                return (d[0], d[2])        # (M, N)
        return None


def _bare(op_name: str) -> str:
    """Strip the ``tessera.`` dialect prefix from an op name."""
    return op_name.split(".", 1)[1] if "." in op_name else op_name


def _parse_dims(type_str: Optional[str]) -> Optional[tuple[int, ...]]:
    """Parse static dims from an MLIR tensor type like ``tensor<8x16xf32>``.
    Returns ``None`` for dynamic / non-tensor / non-fp32 types."""
    if not type_str:
        return None
    m = _TENSOR_RE.search(type_str)
    if not m:
        return None
    parts = m.group(1).split("x")
    if len(parts) < 2:
        return None
    dtype = parts[-1]
    if dtype not in ("f32", "fp32", "float32"):
        return None
    dims: list[int] = []
    for p in parts[:-1]:
        if not p.isdigit():
            return None  # dynamic / symbolic — authoring needs static shapes
        dims.append(int(p))
    return tuple(dims)


def _compute_ops(fn) -> list:
    """Body ops that carry a result (skip void/metadata-only ops)."""
    return [op for op in fn.body if op.result is not None and op.op_name]


def recognize(module: GraphIRModule) -> Optional[AuthorPlan]:
    """Recognize an authorable MPSGraph-lane region in ``module``.

    Returns an :class:`AuthorPlan` when the single function's compute body
    maps to a known authoring primitive, else ``None``. Pure / device-free.
    """
    if not module.functions or len(module.functions) != 1:
        return None
    fn = module.functions[0]
    ops = _compute_ops(fn)
    if not ops:
        return None

    bare_seq = tuple(_bare(op.op_name) for op in ops)

    # ---- fused chains (try longest match first) -------------------------
    if bare_seq in _CHAIN_BY_SEQUENCE:
        chain = _CHAIN_BY_SEQUENCE[bare_seq]
        dims = _chain_dims(chain, ops)
        if dims is not None:
            eps = _norm_eps(ops)
            return AuthorPlan(kind="chain", name=chain, dims=dims, eps=eps)
        return None

    # ---- single op ------------------------------------------------------
    if len(ops) == 1:
        op = ops[0]
        bare = bare_seq[0]
        if bare == "matmul":
            a = _parse_dims(op.operand_types[0]) if op.operand_types else None
            b = (_parse_dims(op.operand_types[1])
                 if len(op.operand_types) > 1 else None)
            if a and b and len(a) == 2 and len(b) == 2 and a[1] == b[0]:
                return AuthorPlan(kind="matmul", name="matmul",
                                  dims=(a[0], a[1], b[1]))
            return None
        if bare in _SINGLE_OPS:
            x = _parse_dims(op.operand_types[0]) if op.operand_types else None
            if not x or len(x) != 2:
                return None
            # norms: a gamma (and beta) operand beyond the input => weighted.
            weighted = bare in _NORM_OPS and len(op.operand_types) >= 2
            # Review glass-jaw R5 (2026-06-03): a weighted norm binds gamma
            # (and, for layer_norm, beta) over the feature axis K = x[:, K].
            # Validate each extra operand is a rank-1 vector of length K so a
            # mismatched gamma/beta shape is rejected, not silently packaged.
            if weighted:
                feat = x[1]
                for ot in op.operand_types[1:]:
                    g = _parse_dims(ot)
                    if g is None or len(g) != 1 or g[0] != feat:
                        return None
            return AuthorPlan(kind="op", name=bare, dims=x,
                              weighted=weighted, eps=_norm_eps(ops))
    return None


def recognize_op(module: GraphIRModule) -> Optional[RecognizedOp]:
    """Shape-free recognition — identify the authorable op/chain from the
    module's op-name sequence WITHOUT requiring static dims.

    This is the recognizer that fires on a live ``@jit`` module (whose IR
    carries op names but ``tensor<?x?x?>`` shapes). Combine the result with
    concrete input shapes via :func:`plan_from_shapes` to author. Returns
    ``None`` for anything not in the authoring vocabulary. Pure / device-free.
    """
    if not module.functions or len(module.functions) != 1:
        return None
    fn = module.functions[0]
    ops = _compute_ops(fn)
    if not ops:
        return None
    seq = tuple(_bare(op.op_name) for op in ops)
    n_inputs = len(fn.args)

    if seq in _CHAIN_BY_SEQUENCE:
        return RecognizedOp(kind="chain", name=_CHAIN_BY_SEQUENCE[seq],
                            arity=n_inputs, eps=_norm_eps(ops))
    if len(ops) == 1:
        bare = seq[0]
        if bare == "matmul":
            return RecognizedOp(kind="matmul", name="matmul", arity=2)
        if bare in _SINGLE_OPS:
            weighted = bare in _NORM_OPS and n_inputs >= 2
            return RecognizedOp(kind="op", name=bare, arity=n_inputs,
                                weighted=weighted, eps=_norm_eps(ops))
    return None


def plan_from_shapes(
    rec: RecognizedOp, input_shapes: list[tuple[int, ...]]
) -> Optional[AuthorPlan]:
    """Build a dims-bearing :class:`AuthorPlan` from a shape-free
    :class:`RecognizedOp` + the concrete input shapes (e.g. from example
    args at ``emit_package`` time). Returns ``None`` if the shapes don't fit
    the op's contract (wrong rank / mismatched contraction dim)."""
    s = [tuple(int(d) for d in sh) for sh in input_shapes]
    if rec.kind == "matmul":
        if len(s) < 2 or len(s[0]) != 2 or len(s[1]) != 2:
            return None
        a, b = s[0], s[1]
        if a[1] != b[0]:
            return None
        return AuthorPlan(kind="matmul", name="matmul", dims=(a[0], a[1], b[1]))
    if rec.kind == "op":
        if not s or len(s[0]) != 2:
            return None
        # Review glass-jaw R5 (2026-06-03): a weighted norm binds learnable
        # gamma (and, for layer_norm, beta) over the feature axis K = x[:, K].
        # Validate each extra input is a rank-1 vector of length K so a
        # mismatched gamma/beta shape is rejected at plan time, not silently
        # packaged into a kernel whose argument table won't bind.
        if rec.weighted:
            if len(s) < 2:
                return None
            feat = s[0][1]
            for extra in s[1:]:
                if len(extra) != 1 or extra[0] != feat:
                    return None
        return AuthorPlan(kind="op", name=rec.name, dims=s[0],
                          weighted=rec.weighted, eps=rec.eps)
    if rec.kind == "chain":
        if rec.name == "matmul_softmax":
            if len(s) < 2 or len(s[0]) != 2 or len(s[1]) != 2 \
                    or s[0][1] != s[1][0]:
                return None
            return AuthorPlan(kind="chain", name=rec.name,
                              dims=(s[0][0], s[0][1], s[1][1]), eps=rec.eps)
        if rec.name == "matmul_softmax_matmul":
            if len(s) < 3 or any(len(x) != 2 for x in s[:3]) \
                    or s[0][1] != s[1][0] or s[1][1] != s[2][0]:
                return None
            return AuthorPlan(kind="chain", name=rec.name,
                              dims=(s[0][0], s[0][1], s[1][1], s[2][1]),
                              eps=rec.eps)
        if rec.name == "rmsnorm_matmul":
            # inputs: x[M,K], gamma[K], W[K,N]
            if len(s) < 3 or len(s[0]) != 2 or len(s[2]) != 2 \
                    or s[0][1] != s[2][0]:
                return None
            # Review glass-jaw R5 (2026-06-03): the fused rmsnorm→matmul binds a
            # learnable gamma over the contraction axis K = x[:, K] = W[K, :].
            # The earlier check validated x @ W but not gamma[K]; a wrong gamma
            # shape would package a kernel whose norm argument won't bind.
            feat = s[0][1]
            if len(s[1]) != 1 or s[1][0] != feat:  # gamma[K]
                return None
            return AuthorPlan(kind="chain", name=rec.name,
                              dims=(s[0][0], s[0][1], s[2][1]), eps=rec.eps)
    return None


def _norm_eps(ops) -> float:
    """Pull an ``eps`` kwarg/attr off a norm op if present, else default."""
    for op in ops:
        if _bare(op.op_name) in _NORM_OPS:
            val = (op.kwargs or {}).get("eps") if op.kwargs else None
            if isinstance(val, (int, float)):
                return float(val)
    return 1e-5


def _chain_dims(chain: str, ops) -> Optional[tuple[int, ...]]:
    """Derive the chain's dims vector from its ops' operand types."""
    if chain == "matmul_softmax":
        a = _parse_dims(ops[0].operand_types[0])
        b = _parse_dims(ops[0].operand_types[1])
        if a and b and len(a) == 2 and len(b) == 2 and a[1] == b[0]:
            return (a[0], a[1], b[1])
        return None
    if chain == "matmul_softmax_matmul":
        a = _parse_dims(ops[0].operand_types[0])
        b = _parse_dims(ops[0].operand_types[1])
        c = _parse_dims(ops[2].operand_types[1])
        if (a and b and c and len(a) == 2 and len(b) == 2 and len(c) == 2
                and a[1] == b[0] and b[1] == c[0]):
            return (a[0], a[1], b[1], c[1])
        return None
    if chain == "rmsnorm_matmul":
        x = _parse_dims(ops[0].operand_types[0])
        w = _parse_dims(ops[1].operand_types[1])
        if not (x and w and len(x) == 2 and len(w) == 2 and x[1] == w[0]):
            return None
        # Review glass-jaw R5 (2026-06-03): the fused norm binds a learnable
        # gamma over the contraction axis K = x[:, K]. Validate gamma[K] (the
        # norm op's 2nd operand) — previously only x @ W was checked, so a wrong
        # gamma shape would package a kernel whose norm argument won't bind.
        if len(ops[0].operand_types) >= 2:
            g = _parse_dims(ops[0].operand_types[1])
            if g is None or len(g) != 1 or g[0] != x[1]:
                return None
        return (x[0], x[1], w[1])
    return None


def author_package_from_graph_ir(
    module: GraphIRModule, out_path: str | Path
) -> Optional[AuthorPlan]:
    """Recognize ``module`` and, if recognized, author the ``.mtlpackage``
    at ``out_path``. Returns the :class:`AuthorPlan` on success, ``None`` if
    the region wasn't recognized or authoring failed.

    This is the single entry point ``@jit(target="apple_gpu")`` calls to
    turn a recognized Graph IR subgraph into an AOT packaged kernel.
    """
    plan = recognize(module)
    if plan is None:
        return None
    return plan if plan.author(out_path) else None


__all__ = [
    "AuthorPlan",
    "RecognizedOp",
    "recognize",
    "recognize_op",
    "plan_from_shapes",
    "author_package_from_graph_ir",
]
