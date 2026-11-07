"""Source call scan for effect inference (E2E-REAL-6F).

Effect inference used to lower the whole function through the AST Graph IR
front end (``_OpExtractor``) a second time, only to read the effect of the
records it emitted. This module replaces that with a scan of the function's
call sites, resolved by object identity:

* a call that resolves to a ``tessera.ops`` callable contributes the effect
  the op catalog registers for it;
* a call to a pure builtin (``range``, ``len``, ...) or to a helper defined
  inside the function body (whose own calls are scanned) contributes nothing;
* every other call -- raw numpy, a user helper from another scope, a method on
  a value, ``getattr`` or dict dispatch -- is *unresolved* and joins to
  ``top``.

The scan is the half of effect inference that tracing cannot supply: a trace
records only the operations that touch traced values, so a host call such as
``np.random.rand(4)`` whose result never reaches a tensor op is invisible to
it. Unresolved calls therefore stay ``top`` even when a concrete trace of the
same function is ``pure``; the trace may refine resolved operations, never
relax an unresolved call.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import sys
import textwrap
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

# Builtins whose call cannot perform a side effect on the program's values.
# Resolved by identity, so a module global that shadows one of these names is
# not admitted.
_PURE_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "divmod", "enumerate", "float", "int",
    "isinstance", "len", "list", "max", "min", "range", "reversed", "round",
    "slice", "sorted", "sum", "tuple", "zip",
})


# How deep the scan follows plain Python helpers before treating a call as
# unresolved.
_MAX_HELPER_DEPTH = 8


@dataclass(frozen=True)
class CallScan:
    """Result of scanning one function's call sites."""

    # Join of the registered effects of every resolved call, and ``top`` when
    # any call is unresolved or the source is unavailable.
    effect: Any
    # Canonical op names (``tessera.ops`` public names) of resolved calls
    # whose registered effect is above ``pure``.
    effectful_ops: Tuple[str, ...]
    # Dotted spellings of unresolved calls, in source order.
    unresolved: Tuple[str, ...]
    source_available: bool


def resolve_tessera_ops_callee(fn: Callable, dotted: str) -> Optional[str]:
    """Return the ``tessera.ops`` public name a dotted call resolves to.

    Resolution is by object identity against ``fn``'s globals and closure, so
    every spelling counts (``ts.ops.x``, ``import tessera as T``,
    ``from tessera.ops import x as y``). Only module attributes are traversed
    (and ``SimpleNamespace``, which is what ``tessera.ops`` is) between the
    root and the leaf: a dotted name through a user object is
    never evaluated, since an attribute read there may run arbitrary code.
    Returns ``None`` when the name does not resolve to a ``tessera.ops``
    callable (a local, a numpy call, a user helper).
    """
    return _ops_name(_resolve_dotted(_function_scope(fn), dotted), dotted)


def _ops_name(obj: Any, dotted: str) -> Optional[str]:
    from .. import ops as _ops

    if obj is None or not callable(obj) or isinstance(obj, (types.ModuleType, type)):
        return None
    leaf = dotted.split(".")[-1]
    if getattr(_ops, leaf, None) is obj:
        return leaf
    for name in dir(_ops):
        if not name.startswith("_") and getattr(_ops, name, None) is obj:
            return name
    return None


def _function_scope(fn: Callable) -> Dict[str, Any]:
    scope: Dict[str, Any] = dict(getattr(fn, "__globals__", {}) or {})
    try:
        scope.update(inspect.getclosurevars(fn).nonlocals)
    except (TypeError, ValueError):
        pass
    return scope


def _resolve_dotted(scope: Dict[str, Any], dotted: str) -> Any:
    parts = dotted.split(".")
    if parts[0] not in scope:
        return None
    obj: Any = scope[parts[0]]
    for part in parts[1:]:
        if not isinstance(obj, (types.ModuleType, types.SimpleNamespace)):
            return None
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _is_constraint_declaration(obj: Any) -> bool:
    """``ts.require`` and the ``ts.constraint`` classes record facts for the
    ConstraintSolver; calling one touches no program value."""
    import importlib

    from . import constraints as _constraints

    # ``tessera.compiler.jit`` the module, not the ``jit`` decorator the
    # package re-exports under the same name.
    _jit_module = importlib.import_module("tessera.compiler.jit")

    if obj is getattr(_jit_module, "require", None):
        return True
    return isinstance(obj, type) and obj.__module__ == _constraints.__name__


def _dotted(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return f"{parent}.{node.attr}" if parent is not None else None
    return None


def _function_node(fn: Callable, source_text: Optional[str]) -> Optional[ast.AST]:
    """The ``def`` (or the single ``lambda``) that defines ``fn``."""
    try:
        source = source_text if source_text is not None else inspect.getsource(fn)
        tree = ast.parse(textwrap.dedent(source))
    except (OSError, TypeError, SyntaxError):
        return None
    name = getattr(fn, "__name__", None)
    if name == "<lambda>":
        # ``getsource`` returns the whole statement; only a statement holding
        # exactly one lambda identifies it.
        lambdas = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]
        return lambdas[0] if len(lambdas) == 1 else None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _body_nodes(node: ast.AST) -> list:
    """Nodes that execute when the function is called.

    Decorators, default values and annotations ran once, at definition.
    """
    if isinstance(node, ast.Lambda):
        return list(ast.walk(node.body))
    return [child for stmt in node.body for child in ast.walk(stmt)]  # type: ignore[attr-defined]


def _local_names(node: ast.AST) -> Tuple[frozenset, frozenset]:
    """(value names bound in the function, helpers defined in it)."""
    values: set[str] = set()
    helpers: set[str] = set()
    args = node.args  # type: ignore[attr-defined]
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        values.add(arg.arg)
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            values.add(extra.arg)
    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            helpers.add(child.name)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            values.add(child.id)
        elif isinstance(child, ast.arg):
            values.add(child.arg)
    # A name bound to a lambda is a helper, not a value.
    for child in ast.walk(node):
        if isinstance(child, ast.Assign) and isinstance(child.value, ast.Lambda):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    helpers.add(target.id)
    return frozenset(values - helpers), frozenset(helpers)


def _local_imports(body: list) -> Dict[str, Any]:
    """Names a body binds with ``import``, for modules already imported.

    A module the process has not imported yet is left unbound (so its calls
    stay unresolved): resolving it here would run its import side effects at
    decoration time.
    """
    bound: Dict[str, Any] = {}
    for child in body:
        if isinstance(child, ast.Import):
            for alias in child.names:
                if alias.asname is not None:
                    module = sys.modules.get(alias.name)
                    if module is not None:
                        bound[alias.asname] = module
                else:
                    root = alias.name.split(".")[0]
                    if root in sys.modules:
                        bound[root] = sys.modules[root]
        elif isinstance(child, ast.ImportFrom) and child.module and not child.level:
            # ``tessera.ops`` is a namespace attribute, not a ``sys.modules``
            # entry, so walk the dotted path from the imported root package.
            module = sys.modules.get(child.module)
            if module is None:
                root, _, rest = child.module.partition(".")
                module = (_resolve_dotted({root: sys.modules[root]}, child.module)
                          if root in sys.modules and rest else None)
            if module is None:
                continue
            for alias in child.names:
                if hasattr(module, alias.name):
                    bound[alias.asname or alias.name] = getattr(module, alias.name)
    return bound


def _is_user_function(obj: Any) -> bool:
    """A plain Python function outside the ``tessera`` package.

    Its body is scanned in place of the call. Tessera's own functions are not
    entered: the ones a program may call are registered ``tessera.ops``
    callables or constraint declarations, and anything else is unresolved.
    """
    module = getattr(obj, "__module__", None) or ""
    return isinstance(obj, types.FunctionType) and not (
        module == "tessera" or module.startswith("tessera."))


def scan_call_effects(
    fn: Callable,
    source_text: Optional[str] = None,
    *,
    _active: Tuple[Any, ...] = (),
) -> CallScan:
    """Scan ``fn``'s call sites and join their registered effects.

    A call to a plain Python helper is scanned through (transitively, up to
    ``_MAX_HELPER_DEPTH`` and cycle-safe), so a program composed of helpers
    that only call ``tessera.ops`` is as pure as the ops it calls.
    """
    from .effects import Effect, registered_op_effect

    node = _function_node(fn, source_text)
    if node is None:
        return CallScan(Effect.top, (), ("<source-unavailable>",), False)
    values, helpers = _local_names(node)
    body = _body_nodes(node)
    scope = _function_scope(fn)
    imported = _local_imports(body)
    scope.update(imported)
    values = values - frozenset(imported)
    effect = Effect.pure
    effectful: list[str] = []
    unresolved: list[str] = []
    for child in body:
        if isinstance(child, (ast.Global, ast.Nonlocal)):
            unresolved.append(f"<{type(child).__name__.lower()}>")
        elif isinstance(child, ast.Delete):
            unresolved.append("<del>")
        elif isinstance(child, (ast.Subscript, ast.Attribute)) and isinstance(
                child.ctx, ast.Store):
            # An in-place write (``a[:] = ...``, ``obj.x = ...``) mutates a
            # value the caller may hold; the Graph front end records it as
            # ``tessera.copy``, whose registered effect is the join here.
            copy_effect = registered_op_effect("tessera.copy")
            if copy_effect > Effect.pure:
                effectful.append("copy")
            effect = effect.join(copy_effect)
    for child in body:
        if not isinstance(child, ast.Call):
            continue
        dotted = _dotted(child.func)
        if dotted is None:
            # Calling the result of a call, a subscript or a lambda literal:
            # nothing names the callee, so nothing can be resolved.
            if isinstance(child.func, ast.Attribute):
                unresolved.append(f"<expression>.{child.func.attr}")
            else:
                unresolved.append("<dynamic>")
            continue
        root = dotted.split(".")[0]
        if "." not in dotted and root in helpers:
            continue  # its body is part of this walk
        if root in values:
            unresolved.append(dotted)  # a method on a value, or a bound callable
            continue
        if "." not in dotted and root not in scope:
            if root in _PURE_BUILTINS and getattr(builtins, root, None) is not None:
                continue
            unresolved.append(dotted)
            continue
        obj = _resolve_dotted(scope, dotted)
        if _is_constraint_declaration(obj):
            continue
        op = _ops_name(obj, dotted)
        if op is not None:
            op_effect = registered_op_effect(op)
            if op_effect > Effect.pure:
                effectful.append(op)
            effect = effect.join(op_effect)
            continue
        if (_is_user_function(obj) and obj not in _active
                and len(_active) < _MAX_HELPER_DEPTH):
            inner = scan_call_effects(obj, _active=(*_active, fn))
            effect = effect.join(inner.effect)
            effectful.extend(inner.effectful_ops)
            unresolved.extend(f"{dotted}->{name}" for name in inner.unresolved)
            continue
        unresolved.append(dotted)
    if unresolved:
        effect = Effect.top
    return CallScan(effect, tuple(effectful), tuple(unresolved), True)
