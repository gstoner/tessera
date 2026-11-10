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


# Owner id of the scanned function's own scope.
_ROOT = 0

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
    # (canonical op name, dotted spelling) of every resolved ``tessera.ops``
    # call the body can execute, in source order: helpers it calls are
    # included, while a nested ``def`` or named lambda the body never invokes
    # contributes nothing (``effect`` still joins it, conservatively).
    ops: Tuple[Tuple[str, str], ...] = ()


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
    """Resolve a dotted name through module (and ``SimpleNamespace``)
    attributes only: an attribute read on a user object may run arbitrary
    code, so such a name is never evaluated and resolves to ``None``."""
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


_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _scope_roots(node: ast.AST) -> list:
    """Nodes a call of ``node`` evaluates directly.

    Decorators, default values and annotations ran once, at definition.
    """
    if isinstance(node, ast.Lambda):
        return [node.body]
    return list(node.body)  # type: ignore[attr-defined]


def _scope_nodes(node: ast.AST) -> Tuple[list, list]:
    """(nodes in ``node``'s own lexical scope, nested function scopes).

    A nested ``def`` or ``lambda`` is its own scope: its parameters and
    assignments do not bind names here. Its decorators and defaults, which
    the enclosing scope evaluates when it defines the helper, stay here.
    """
    own: list = []
    nested: list = []
    stack = list(reversed(_scope_roots(node)))
    while stack:
        child = stack.pop()
        own.append(child)
        if isinstance(child, _FUNCTION_NODES):
            nested.append(child)
            evaluated_here = list(child.args.defaults) + [
                d for d in child.args.kw_defaults if d is not None]
            if not isinstance(child, ast.Lambda):
                evaluated_here += child.decorator_list
            stack.extend(reversed(evaluated_here))
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(child))))
    return own, nested


def _import_bindings(own: list) -> Dict[str, Any]:
    """Names a scope binds with ``import``, for modules already imported.

    A module the process has not imported yet is left unbound (so its calls
    stay unresolved): resolving it here would run its import side effects at
    decoration time.
    """
    bound: Dict[str, Any] = {}
    for child in own:
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


# A local binding is one of: a value (parameter, assignment, loop target,
# ``with`` target, an import also assigned elsewhere), a helper defined in the
# function, or an import of an already-loaded module (bound to its object).
_VALUE = "value"
_HELPER = "helper"


def _scope_bindings(node: ast.AST, own: list) -> Dict[str, Any]:
    """This scope's own bindings: name -> ``_VALUE``, ``_HELPER`` or the
    imported object. A name bound more than one way is a value, so a rebound
    import or helper is never resolved to what it was first bound to."""
    kinds: Dict[str, set] = {}

    def bind(name: str, kind: Any) -> None:
        kinds.setdefault(name, set()).add(kind if isinstance(kind, str) else id(kind))

    args = node.args  # type: ignore[attr-defined]
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
        if arg is not None:
            bind(arg.arg, _VALUE)
    lambda_targets: set[int] = set()
    for child in own:
        if isinstance(child, ast.Assign) and isinstance(child.value, ast.Lambda):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    lambda_targets.add(id(target))
                    bind(target.id, _HELPER)
    imports = _import_bindings(own)
    for child in own:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(child.name, _HELPER if not isinstance(child, ast.ClassDef) else _VALUE)
        elif (isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
              and id(child) not in lambda_targets):
            bind(child.id, _VALUE)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                name = alias.asname or alias.name.split(".")[0]
                bind(name, imports.get(name, _VALUE))
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bind(child.name, _VALUE)
    resolved: Dict[str, Any] = {}
    for name, seen in kinds.items():
        if len(seen) != 1:
            resolved[name] = _VALUE
        else:
            (only,) = seen
            resolved[name] = only if isinstance(only, str) else imports.get(name, _VALUE)
    return resolved


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

    Each nested ``def``/``lambda`` is scanned as its own lexical scope, inside
    the enclosing one. A call to a plain Python helper from another scope is
    scanned through (transitively, up to ``_MAX_HELPER_DEPTH`` and
    cycle-safe), so a program composed of helpers that only call
    ``tessera.ops`` is as pure as the ops it calls.
    """
    from .effects import Effect

    node = _function_node(fn, source_text)
    if node is None:
        return CallScan(Effect.top, (), ("<source-unavailable>",), False, ())
    state = _ScanState(fn, _active)
    state.scan_scope(node, {}, _ROOT)
    unresolved = tuple(state.unresolved)
    effect = Effect.top if unresolved else state.effect
    return CallScan(effect, tuple(state.effectful), unresolved, True, state.reachable_ops())


class _ScanState:
    def __init__(self, fn: Callable, active: Tuple[Any, ...]) -> None:
        from .effects import Effect

        self.fn = fn
        self.active = active
        self.globals = _function_scope(fn)
        self.effect = Effect.pure
        self.effectful: list[str] = []
        self.unresolved: list[str] = []
        # (owning scope, op name, dotted spelling); ``reachable_ops`` keeps
        # the ones whose scope the body invokes.
        self.ops: list[Tuple[int, str, str]] = []
        # owning scope -> nested scopes it calls by name.
        self.invokes: Dict[int, set] = {}
        self.scope_names: Dict[int, Dict[str, list]] = {}
        self.current = _ROOT

    def reachable_ops(self) -> Tuple[Tuple[str, str], ...]:
        reached = {_ROOT}
        frontier = [_ROOT]
        while frontier:
            for callee in self.invokes.get(frontier.pop(), ()):
                if callee not in reached:
                    reached.add(callee)
                    frontier.append(callee)
        return tuple((op, dotted) for owner, op, dotted in self.ops if owner in reached)

    def join(self, op: str, op_effect: Any) -> None:
        from .effects import Effect

        if op_effect > Effect.pure:
            self.effectful.append(op)
        self.effect = self.effect.join(op_effect)

    def scan_scope(self, node: ast.AST, enclosing: Dict[str, Any], owner: int,
                   visible: Optional[Dict[str, list]] = None) -> None:
        from .effects import registered_op_effect

        own, nested = _scope_nodes(node)
        bindings = {**enclosing, **_scope_bindings(node, own)}
        # Helpers this scope defines, by the name a call uses. An anonymous
        # lambda (an argument, an immediately-called literal) has no name a
        # later call could use, so it is treated as invoked here.
        lambda_names = {
            id(child.value): target.id
            for child in own
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Lambda)
            for target in child.targets if isinstance(target, ast.Name)}
        local: Dict[str, list] = {}
        for inner in nested:
            name = (inner.name if not isinstance(inner, ast.Lambda)
                    else lambda_names.get(id(inner)))
            if name is None:
                self.invokes.setdefault(owner, set()).add(id(inner))
            else:
                # A name bound twice in one scope may call either body.
                local.setdefault(name, []).append(id(inner))
        # A local definition shadows an enclosing one of the same name.
        names = {**(visible or {}), **local}
        self.scope_names[owner] = names
        self.current = owner
        for child in own:
            if isinstance(child, (ast.Global, ast.Nonlocal)):
                self.unresolved.append(f"<{type(child).__name__.lower()}>")
            elif isinstance(child, ast.Delete):
                self.unresolved.append("<del>")
            elif isinstance(child, ast.ClassDef):
                self.unresolved.append(f"<class {child.name}>")
            elif isinstance(child, (ast.Subscript, ast.Attribute)) and isinstance(
                    child.ctx, ast.Store):
                # An in-place write (``a[:] = ...``, ``obj.x = ...``) mutates a
                # value the caller may hold; the Graph front end records it as
                # ``tessera.copy``, whose registered effect is the join here.
                self.join("copy", registered_op_effect("tessera.copy"))
            elif isinstance(child, ast.Call):
                self.scan_call(child, bindings)
        for inner in nested:
            self.scan_scope(inner, bindings, id(inner), names)

    def scan_call(self, call: ast.Call, bindings: Dict[str, Any]) -> None:
        from .effects import registered_op_effect

        dotted = _dotted(call.func)
        if dotted is None:
            # Calling the result of a call, a subscript or a lambda literal:
            # nothing names the callee, so nothing can be resolved.
            if isinstance(call.func, ast.Attribute):
                self.unresolved.append(f"<expression>.{call.func.attr}")
            else:
                self.unresolved.append("<dynamic>")
            return
        root = dotted.split(".")[0]
        local = bindings.get(root)
        if local == _HELPER and "." not in dotted:
            # Its body is scanned as a nested scope; record the invocation.
            for helper in self.scope_names.get(self.current, {}).get(root, ()):
                self.invokes.setdefault(self.current, set()).add(helper)
            return
        if isinstance(local, str):
            self.unresolved.append(dotted)  # a method on a value, or a rebound name
            return
        if local is not None:
            obj = _resolve_dotted({root: local}, dotted)
        elif root in self.globals:
            obj = _resolve_dotted(self.globals, dotted)
        else:
            if ("." not in dotted and root in _PURE_BUILTINS
                    and getattr(builtins, root, None) is not None):
                return
            self.unresolved.append(dotted)
            return
        if _is_constraint_declaration(obj):
            return
        op = _ops_name(obj, dotted)
        if op is not None:
            self.ops.append((self.current, op, dotted))
            self.join(op, registered_op_effect(op))
            return
        if (_is_user_function(obj) and obj not in self.active
                and len(self.active) < _MAX_HELPER_DEPTH):
            inner = scan_call_effects(obj, _active=(*self.active, self.fn))
            self.effect = self.effect.join(inner.effect)
            self.effectful.extend(inner.effectful_ops)
            self.ops.extend((self.current, op, spelling) for op, spelling in inner.ops)
            self.unresolved.extend(f"{dotted}->{name}" for name in inner.unresolved)
            return
        self.unresolved.append(dotted)
