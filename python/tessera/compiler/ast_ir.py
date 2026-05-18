"""Shared AST → constrained-IR lowering core — M6 Step 1 deliverable.

The ``@clifford_jit`` work (2026-05-17/18) proved a small template
that fits multiple constrained frontends:

  1. Parse a single-return Python function via :mod:`ast`.
  2. Reject anything outside a whitelist of namespace-qualified
     attribute calls (``ga.<op>``, ``energy.<op>``, ...).
  3. Encode int / float / bool constants as inline operand refs
     (``#int:N`` / ``#float:V`` / ``#bool:0|1``).
  4. Emit an SSA-form IR (`%t0`, `%t1`, ...).
  5. Validate every op against a manifest before any call runs.

M6 lifts that template out of ``clifford_jit`` so it can be reused
by a future ``energy_jit`` (M6 Steps 2–4) and a Visual-Complex /
Cauchy-Riemann verifier (M7).  The module is intentionally
*frontend-agnostic*: the receiver namespace, the op-name whitelist,
and the canonical-op resolver are passed in by the caller.

``clifford_jit`` continues to expose its own ``CliffordIROpCall`` /
``CliffordIRProgram`` types; those types are now thin aliases of
the generic :class:`IROpCall` / :class:`IRProgram` defined here so
existing tests + downstream consumers don't have to migrate.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


# ─────────────────────────────────────────────────────────────────────────────
# IR types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IROpCall:
    """One op call in the constrained IR.

    ``op_name`` is the canonical manifest op (e.g.
    ``clifford_rotor_sandwich`` or ``energy_quadratic``).
    ``operand_refs`` are SSA-style names — function-argument names,
    auto-generated intermediates (``%tN``), or inline literal refs
    (``#int:N`` / ``#float:V`` / ``#bool:0|1``).  ``result_name`` is
    the SSA name for this op's output.  ``python_attr`` is the
    receiver-namespace attribute name for debuggability.
    """
    op_name: str
    operand_refs: tuple[str, ...]
    result_name: str
    python_attr: str = ""


@dataclass(frozen=True)
class IRProgram:
    """A lowered constrained-Python function.

    Structured (function signature + ordered op-call body + return
    ref).  JSON-friendly via :meth:`as_metadata` and debug-friendly
    via :meth:`text`.
    """
    arg_names: tuple[str, ...]
    ops: tuple[IROpCall, ...]
    return_ref: str
    namespace: str = "ir"

    def as_metadata(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "arg_names": list(self.arg_names),
            "ops": [
                {"op": c.op_name, "operands": list(c.operand_refs),
                 "result": c.result_name, "python_attr": c.python_attr}
                for c in self.ops
            ],
            "return_ref": self.return_ref,
        }

    def text(self) -> str:
        """Pretty-printed IR — debug-friendly form."""
        lines = [f"{self.namespace}_ir({', '.join(self.arg_names)}):"]
        for c in self.ops:
            lines.append(
                f"  {c.result_name} = {c.op_name}({', '.join(c.operand_refs)})"
                + (f"  # {self.namespace}.{c.python_attr}" if c.python_attr else "")
            )
        lines.append(f"  return {self.return_ref}")
        return "\n".join(lines)


class ASTLoweringError(Exception):
    """Raised when AST lowering fails — source-span tagged."""


# ─────────────────────────────────────────────────────────────────────────────
# Lowerer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LoweringConfig:
    """Frontend-specific configuration for :func:`lower_function`.

    Attributes
    ----------
    namespace
        The receiver-namespace name the lowerer requires (e.g.
        ``"ga"`` for ``clifford_jit``, ``"energy"`` for a future
        ``energy_jit``).  Calls must read as ``<ns>.<op>(...)`` or
        any attribute chain whose immediate parent segment is
        ``<ns>``.
    attr_to_op_name
        Map from receiver-attribute name (e.g. ``"rotor_sandwich"``)
        to the canonical manifest op name (e.g.
        ``"clifford_rotor_sandwich"``).  Calls whose attribute name
        isn't in this map are rejected.
    error_prefix
        Prefix for raised diagnostics — typically ``"clifford_jit"``
        or ``"energy_jit"`` so the error message names the frontend.
    error_class
        Optional :class:`Exception` subclass to raise instead of the
        default :class:`ASTLoweringError`.  ``clifford_jit`` passes
        its own ``CliffordJitError`` so its tests still match the
        old class.
    """
    namespace: str
    attr_to_op_name: Mapping[str, str]
    error_prefix: str = "ast_ir"
    error_class: type[Exception] = ASTLoweringError


class _ASTLowerer(ast.NodeVisitor):
    """Walks a constrained Python function body and emits
    :class:`IROpCall` entries.

    Accepted forms:
      - ``ast.FunctionDef`` whose body is a sequence of single-target
        ``name = EXPR`` assignments followed by exactly one
        ``ast.Return``.  Leading docstrings are skipped.
      - Each expression is composed of nested ``ast.Call`` invocations
        of ``<ns>.<op>(...)`` (where ``<op>`` is in
        :attr:`LoweringConfig.attr_to_op_name`), ``ast.Name``
        references, or numeric literal constants
        (``ast.Constant`` / ``ast.UnaryOp(USub, Constant)``).
      - Both ``ns.foo(...)`` and any attribute chain ending in
        ``.ns.foo(...)`` are accepted; the chain root must be a Name.

    Anything else raises :attr:`LoweringConfig.error_class` with a
    precise location-tagged message.  This is the **compile-time
    guarantee**: if lowering succeeds, every op is namespace-resolvable.
    """

    def __init__(self, fn_name: str, config: LoweringConfig) -> None:
        self._fn_name = fn_name
        self._config = config
        self._tmp_counter = 0
        self.ops: list[IROpCall] = []

    def _next_tmp(self) -> str:
        name = f"%t{self._tmp_counter}"
        self._tmp_counter += 1
        return name

    def _err(self, node: ast.AST, msg: str) -> Exception:
        line = getattr(node, "lineno", "?")
        col = getattr(node, "col_offset", "?")
        return self._config.error_class(
            f"{self._config.error_prefix}({self._fn_name}) lowering: "
            f"{msg} (at line {line}, col {col})"
        )

    def _is_namespaced_call(self, node: ast.AST) -> Optional[str]:
        """If ``node`` is a call whose callee resolves through the
        configured namespace (i.e. ``<ns>.<op>`` or any
        ``<chain>.<ns>.<op>``), return the op attribute name.
        Otherwise return ``None``.

        The receiver must literally end in the configured namespace
        — ``foo.norm(x)``, ``np.linalg.norm(x)``, and
        ``self.<other>(x)`` are rejected.
        """
        if not isinstance(node, ast.Attribute):
            return None
        attr = node.attr
        if attr not in self._config.attr_to_op_name:
            return None
        recv = node.value
        ns = self._config.namespace
        if isinstance(recv, ast.Name):
            if recv.id != ns:
                return None
            return attr
        if isinstance(recv, ast.Attribute):
            if recv.attr != ns:
                return None
            root = recv.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name):
                return None
            return attr
        return None

    def lower_expr(
        self, expr: ast.AST, *, env: Optional[dict[str, str]] = None,
    ) -> str:
        """Lower an expression to an operand reference."""
        if isinstance(expr, ast.Name):
            if env is not None:
                if expr.id not in env:
                    raise self._err(
                        expr,
                        f"name {expr.id!r} is not a function argument or "
                        "earlier-bound assignment",
                    )
                return env[expr.id]
            return expr.id
        if isinstance(expr, ast.Constant):
            v = expr.value
            if isinstance(v, bool):
                return f"#bool:{int(v)}"
            if isinstance(v, int):
                return f"#int:{v}"
            if isinstance(v, float):
                return f"#float:{v!r}"
            raise self._err(
                expr,
                "literal constants are limited to int / float / bool "
                f"(got {type(v).__name__}: {v!r})",
            )
        if (isinstance(expr, ast.UnaryOp)
                and isinstance(expr.op, ast.USub)
                and isinstance(expr.operand, ast.Constant)):
            v = expr.operand.value
            if isinstance(v, bool):
                raise self._err(expr, "cannot negate a boolean literal")
            if isinstance(v, int):
                return f"#int:{-v}"
            if isinstance(v, float):
                return f"#float:{(-v)!r}"
            raise self._err(
                expr,
                "literal constants are limited to int / float "
                f"(got {type(v).__name__}: {v!r})",
            )
        if isinstance(expr, ast.Call):
            attr = self._is_namespaced_call(expr.func)
            if attr is None:
                raise self._err(
                    expr,
                    f"only ``tessera.{self._config.namespace}.<op>(...)`` calls "
                    f"are allowed (got {ast.dump(expr.func)})",
                )
            if expr.keywords:
                raise self._err(
                    expr,
                    f"{self._config.namespace}.{attr}: keyword arguments are "
                    "not supported by the v1 AST lowering",
                )
            operand_refs = tuple(
                self.lower_expr(a, env=env) for a in expr.args
            )
            result = self._next_tmp()
            self.ops.append(IROpCall(
                op_name=self._config.attr_to_op_name[attr],
                operand_refs=operand_refs,
                result_name=result,
                python_attr=attr,
            ))
            return result
        raise self._err(
            expr,
            f"unsupported expression type {type(expr).__name__} — "
            f"v1 AST lowering accepts only Name references, numeric "
            f"literals, and tessera.{self._config.namespace}.<op>(...) calls",
        )


def lower_function(
    fn: Callable[..., Any],
    config: LoweringConfig,
    *,
    source_read_error_message: str = "cannot read source",
) -> IRProgram:
    """Lower a constrained Python function to an :class:`IRProgram`.

    See :class:`LoweringConfig` for the per-frontend knobs.  The
    function must have the form::

        def name(arg1, arg2, ...):
            \"\"\"optional docstring\"\"\"
            x = ns.op(...)
            ...
            return ns.op(...)

    Any deviation raises :attr:`LoweringConfig.error_class`.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise config.error_class(
            f"{config.error_prefix}({fn.__qualname__}): "
            f"{source_read_error_message} "
            f"({exc!s}); try defining the function in a regular .py "
            "file or use a trace-capture fallback"
        )
    source = textwrap.dedent(source)
    tree = ast.parse(source, mode="exec")
    if not tree.body:
        raise config.error_class(
            f"{config.error_prefix}({fn.__qualname__}): empty source after parse"
        )
    func_def: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node  # type: ignore[assignment]
            break
    if func_def is None:
        raise config.error_class(
            f"{config.error_prefix}({fn.__qualname__}): no FunctionDef found"
        )
    if isinstance(func_def, ast.AsyncFunctionDef):
        raise config.error_class(
            f"{config.error_prefix}({fn.__qualname__}): async functions are "
            "not supported by the v1 AST lowering"
        )
    if (func_def.args.kwonlyargs or func_def.args.vararg
            or func_def.args.kwarg):
        raise config.error_class(
            f"{config.error_prefix}({fn.__qualname__}): v1 only supports "
            "positional arguments (no *args / **kwargs / keyword-only)"
        )
    arg_names = tuple(a.arg for a in func_def.args.args)
    body_stmts = list(func_def.body)
    if (len(body_stmts) >= 2
            and isinstance(body_stmts[0], ast.Expr)
            and isinstance(body_stmts[0].value, ast.Constant)
            and isinstance(body_stmts[0].value.value, str)):
        body_stmts = body_stmts[1:]
    if not body_stmts:
        raise config.error_class(
            f"{config.error_prefix}({fn.__qualname__}): empty function body"
        )
    if not isinstance(body_stmts[-1], ast.Return):
        raise config.error_class(
            f"{config.error_prefix}({fn.__qualname__}): v1 requires the "
            "function to end with a single-return statement"
        )
    lowerer = _ASTLowerer(fn.__qualname__, config)
    binding_map: dict[str, str] = {name: name for name in arg_names}
    for stmt in body_stmts[:-1]:
        if not isinstance(stmt, ast.Assign):
            raise config.error_class(
                f"{config.error_prefix}({fn.__qualname__}): v1 only supports "
                "simple ``name = expr`` assignments above the return "
                f"(got {type(stmt).__name__})"
            )
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise config.error_class(
                f"{config.error_prefix}({fn.__qualname__}): v1 assignments "
                "must have a single Name target (no tuples / attributes / "
                "subscripts)"
            )
        rhs_ref = lowerer.lower_expr(stmt.value, env=binding_map)
        binding_map[stmt.targets[0].id] = rhs_ref
    ret_stmt = body_stmts[-1]
    ret_expr = ret_stmt.value
    if ret_expr is None:
        raise config.error_class(
            f"{config.error_prefix}({fn.__qualname__}): function must "
            "return a value"
        )
    return_ref = lowerer.lower_expr(ret_expr, env=binding_map)
    return IRProgram(
        arg_names=arg_names,
        ops=tuple(lowerer.ops),
        return_ref=return_ref,
        namespace=config.namespace,
    )


def resolve_operand(ref: str, env: Mapping[str, Any]) -> Any:
    """Decode an operand ref back to a Python value at execution time.

    Inline literals (``#int:N`` / ``#float:V`` / ``#bool:0|1``) parse
    directly.  Named refs look up in ``env``.

    This is the executor-side counterpart to the constant encoding
    in :class:`_ASTLowerer`; the two ``clifford_jit`` had to
    open-code separately before M6 Step 1 now share it.
    """
    if ref.startswith("#int:"):
        return int(ref[5:])
    if ref.startswith("#float:"):
        return float(ref[7:])
    if ref.startswith("#bool:"):
        return ref[6:] == "1"
    if ref not in env:
        raise KeyError(
            f"operand ref {ref!r} is not in the env (known: {sorted(env)})"
        )
    return env[ref]


__all__ = [
    "IROpCall",
    "IRProgram",
    "LoweringConfig",
    "ASTLoweringError",
    "lower_function",
    "resolve_operand",
]
