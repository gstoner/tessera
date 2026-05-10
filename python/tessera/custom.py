"""S13 custom primitive registration surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .autodiff.jvp import register_jvp
from .autodiff.tape import install_op_wrappers
from .autodiff.vjp import register_vjp


Rule = Callable[..., Any]


@dataclass
class CustomPrimitive:
    name: str
    impl: Callable[..., Any]
    effect: str = "pure"
    opaque: bool = False
    shape_rule: Rule | None = None
    dtype_rule: Rule | None = None
    batching_rule: Rule | None = None
    transpose_rule: Rule | None = None
    sharding_rule: Rule | None = None
    masking_rule: Rule | None = None
    lowerings: dict[str, Rule] = field(default_factory=dict)

    def __call__(self, *args, **kwargs):
        from . import ops

        return getattr(ops, self.name)(*args, **kwargs)

    def def_shape_rule(self, fn: Rule) -> Rule:
        self.shape_rule = fn
        _sync_metadata(self)
        return fn

    def def_dtype_rule(self, fn: Rule) -> Rule:
        self.dtype_rule = fn
        _sync_metadata(self)
        return fn

    def def_vjp(self, fn: Rule) -> Rule:
        register_vjp(self.name, fn)
        return fn

    def def_jvp(self, fn: Rule) -> Rule:
        register_jvp(self.name, fn)
        return fn

    def def_batching(self, fn: Rule) -> Rule:
        self.batching_rule = fn
        _sync_metadata(self)
        return fn

    def def_transpose(self, fn: Rule) -> Rule:
        self.transpose_rule = fn
        _sync_metadata(self)
        return fn

    def def_sharding(self, fn: Rule) -> Rule:
        self.sharding_rule = fn
        _sync_metadata(self)
        return fn

    def def_masking(self, fn: Rule) -> Rule:
        self.masking_rule = fn
        _sync_metadata(self)
        return fn

    def def_lowering(self, target: str) -> Callable[[Rule], Rule]:
        def decorate(fn: Rule) -> Rule:
            self.lowerings[str(target)] = fn
            from . import ops

            ops.registry.register_lowering(self.name, fn, custom_primitive=True, target=str(target))
            _sync_metadata(self)
            return fn

        return decorate

    def lower(self, target: str, *args, **kwargs):
        if target not in self.lowerings:
            raise NotImplementedError(f"custom primitive {self.name!r} has no lowering for target {target!r}")
        return self.lowerings[target](*args, **kwargs)


CUSTOM_PRIMITIVES: dict[str, CustomPrimitive] = {}


def custom_primitive(
    name: str,
    *,
    effect: str = "pure",
    shape_rule: Rule | None = None,
    dtype_rule: Rule | None = None,
    vjp: Rule | None = None,
    jvp: Rule | None = None,
    batching_rule: Rule | None = None,
    transpose_rule: Rule | None = None,
    sharding_rule: Rule | None = None,
    masking_rule: Rule | None = None,
) -> Callable[[Callable[..., Any]], CustomPrimitive]:
    """Decorate a Python reference implementation as a Tessera primitive."""

    def decorate(fn: Callable[..., Any]) -> CustomPrimitive:
        prim = CustomPrimitive(
            name=name,
            impl=fn,
            effect=effect,
            shape_rule=shape_rule,
            dtype_rule=dtype_rule,
            batching_rule=batching_rule,
            transpose_rule=transpose_rule,
            sharding_rule=sharding_rule,
            masking_rule=masking_rule,
        )
        _register_primitive(prim)
        if vjp is not None:
            prim.def_vjp(vjp)
        if jvp is not None:
            prim.def_jvp(jvp)
        return prim

    return decorate


def custom_call(name: str, *, effect: str = "pure", **metadata: Any) -> Callable[[Callable[..., Any]], CustomPrimitive]:
    """Register an opaque call that reports compiler metadata."""

    def decorate(fn: Callable[..., Any]) -> CustomPrimitive:
        prim = CustomPrimitive(name=name, impl=fn, effect=effect, opaque=True)
        _register_primitive(prim, **metadata)
        return prim

    return decorate


def custom_vjp(name: str) -> Callable[[Rule], Rule]:
    def decorate(fn: Rule) -> Rule:
        register_vjp(name, fn)
        prim = CUSTOM_PRIMITIVES.get(name)
        if prim is not None:
            _sync_metadata(prim)
        return fn

    return decorate


def custom_jvp(name: str) -> Callable[[Rule], Rule]:
    def decorate(fn: Rule) -> Rule:
        register_jvp(name, fn)
        prim = CUSTOM_PRIMITIVES.get(name)
        if prim is not None:
            _sync_metadata(prim)
        return fn

    return decorate


def custom_batching(name: str) -> Callable[[Rule], Rule]:
    def decorate(fn: Rule) -> Rule:
        prim = CUSTOM_PRIMITIVES.get(name)
        if prim is None:
            raise KeyError(f"unknown custom primitive {name!r}")
        prim.batching_rule = fn
        _sync_metadata(prim)
        return fn

    return decorate


def get_custom_primitive(name: str) -> CustomPrimitive | None:
    return CUSTOM_PRIMITIVES.get(name)


def _register_primitive(prim: CustomPrimitive, **metadata: Any) -> None:
    from . import ops

    def reference(*args, **kwargs):
        coerced = tuple(_to_numpy(a) for a in args)
        return prim.impl(*coerced, **kwargs)

    reference.__name__ = prim.name
    CUSTOM_PRIMITIVES[prim.name] = prim
    setattr(ops, prim.name, reference)
    ops.registry.register_reference(
        prim.name,
        reference,
        custom_primitive=True,
        effect=prim.effect,
        opaque=prim.opaque,
        **metadata,
    )
    _sync_metadata(prim)
    install_op_wrappers()


def _sync_metadata(prim: CustomPrimitive) -> None:
    from . import ops

    entry = ops.registry.get(prim.name)
    if entry is None:
        return
    entry.metadata.update(
        {
            "custom_primitive": True,
            "effect": prim.effect,
            "opaque": prim.opaque,
            "has_shape_rule": prim.shape_rule is not None,
            "has_dtype_rule": prim.dtype_rule is not None,
            "has_batching_rule": prim.batching_rule is not None,
            "has_transpose_rule": prim.transpose_rule is not None,
            "has_sharding_rule": prim.sharding_rule is not None,
            "has_masking_rule": prim.masking_rule is not None,
            "lowering_targets": tuple(sorted(prim.lowerings)),
        }
    )


def _to_numpy(x: Any) -> Any:
    if hasattr(x, "_data"):
        x = x._data
    if hasattr(x, "_data"):
        x = x._data
    if isinstance(x, np.ndarray):
        return x
    return x


__all__ = [
    "CUSTOM_PRIMITIVES",
    "CustomPrimitive",
    "custom_batching",
    "custom_call",
    "custom_jvp",
    "custom_primitive",
    "custom_vjp",
    "get_custom_primitive",
]
