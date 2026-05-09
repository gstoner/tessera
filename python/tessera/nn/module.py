"""Stateful Module / Parameter surface — Tier 1 of the capability-gap audit.

Design notes:
  * `Parameter` wraps a `DistributedArray` and holds a (currently always-`None`)
    `.grad` slot. Autodiff (Tier 2) will populate `.grad`.
  * `Module` uses the torch.nn-style `__setattr__` routing pattern: assigning
    a `Parameter` or `Module` attribute registers it in `_parameters` /
    `_modules` instead of `__dict__`. Lookups fall back through `__getattr__`.
  * `Sequential`, `ModuleList`, `ModuleDict` are thin containers over `Module`.
  * `requires_grad` is a flag only — there is no autograd plumbing in this
    module. Tier 2 introduces `tessera.autodiff.reverse(fn)` which reads the
    flag and writes `.grad`.

Out of scope (deferred to follow-ups, see audit Tier 1 for tracking):
  * Buffers (BatchNorm running stats)
  * `module.to(device)` / `module.to(dtype)` migration
  * DDP / FSDP wrappers
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Any, Iterator

import numpy as np

from ..distributed.array import DistributedArray
from ..distributed.domain import Rect, Replicated


# Maps id(numpy_buffer) → Parameter (weak ref). Used by tessera.autodiff to
# trace gradients back to Parameters even when their `.data._data` buffer is
# what flows through the op graph (e.g., when `_as_array(param)` extracts the
# underlying numpy view before calling a functional op). The registry is
# populated by `Parameter.__init__` and consulted by `autodiff.tape._describe`.
_PARAM_REGISTRY: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _register_parameter(param: "Parameter") -> None:
    _PARAM_REGISTRY[id(param._data._data)] = param


def parameter_for_buffer_id(buffer_id: int):
    """Return the Parameter that owns `id(numpy_buffer) == buffer_id`, or None."""
    return _PARAM_REGISTRY.get(buffer_id)


# ─────────────────────────────────────────────────────────────────────────────
# Buffer
#
# Non-trainable named tensors that ride alongside Parameters in a Module.
# Used for BatchNorm running stats, RoPE precomputed tables, attention
# masks — anything that is part of the module's persistent state but does
# not receive gradients.
#
# Differs from `Parameter` in three ways:
#   * No `.grad` slot
#   * No `requires_grad` flag
#   * Has a `persistent: bool` flag that controls `state_dict` participation
#
# Not subclassed from `Parameter` deliberately — autodiff's
# `_PARAM_REGISTRY` lookup yields `None` for buffers, so gradients flow
# through them harmlessly.
# ─────────────────────────────────────────────────────────────────────────────



_NUMPY_DTYPE_TO_TESSERA = {
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "int8": "int8",
    "uint8": "uint8",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
}


def _normalize_dtype(dtype: str) -> str:
    return _NUMPY_DTYPE_TO_TESSERA.get(dtype, dtype)


def _migrate_distributed_array(arr: DistributedArray, np_dtype: str, tessera_dtype: str) -> None:
    """Cast a DistributedArray's underlying numpy buffer to ``np_dtype`` in place.

    The DistributedArray handle is preserved, but the underlying numpy buffer
    is replaced (numpy doesn't support in-place dtype changes). Caller is
    responsible for re-registering the new buffer id in any external registry
    (e.g., autodiff's `_PARAM_REGISTRY`).
    """
    new_buf = arr._data.astype(np_dtype, copy=True)
    arr._data = new_buf
    arr.dtype = tessera_dtype


# ─────────────────────────────────────────────────────────────────────────────
# Parameter
# ─────────────────────────────────────────────────────────────────────────────


class Parameter:
    """Trainable tensor handle.

    Wraps a `DistributedArray` and provides a `.grad` slot for autodiff. Until
    Tier 2 lands, `.grad` is always `None`; `requires_grad` is a flag only.

    Construction accepts a `DistributedArray`, a numpy array (wrapped in a
    Replicated `DistributedArray` of the inferred shape/dtype), or a shape +
    dtype pair (zero-initialized).
    """

    __slots__ = ("_data", "requires_grad", "_grad", "__weakref__")

    def __init__(
        self,
        data: DistributedArray | np.ndarray | tuple[int, ...] | list[int] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: str = "fp32",
        requires_grad: bool = True,
    ) -> None:
        if isinstance(data, DistributedArray):
            arr = data
        elif isinstance(data, np.ndarray):
            inferred = _normalize_dtype(str(data.dtype))
            arr = DistributedArray.from_domain(
                Rect(tuple(data.shape)),
                dtype=inferred if dtype == "fp32" else dtype,
                distribution=Replicated(),
                fill="empty",
            )
            arr._data[...] = data
        elif isinstance(data, (tuple, list)):
            arr = DistributedArray.from_domain(
                Rect(tuple(data)), dtype=dtype, distribution=Replicated(), fill="zeros"
            )
        elif data is None and shape is not None:
            arr = DistributedArray.from_domain(
                Rect(tuple(shape)), dtype=dtype, distribution=Replicated(), fill="zeros"
            )
        else:
            raise TypeError(
                "Parameter expects a DistributedArray, numpy array, shape tuple, "
                "or shape= keyword"
            )
        object.__setattr__(self, "_data", arr)
        object.__setattr__(self, "requires_grad", bool(requires_grad))
        object.__setattr__(self, "_grad", None)
        # Register the underlying numpy buffer so autodiff can trace back to us
        # even when only `param.data._data` flows through ops.
        _register_parameter(self)

    # Storage / data access -----------------------------------------------------

    @property
    def data(self) -> DistributedArray:
        return self._data

    @property
    def grad(self) -> DistributedArray | None:
        return self._grad

    @grad.setter
    def grad(self, value: DistributedArray | np.ndarray | None) -> None:
        # Used by autodiff (Tier 2) to write back gradients.
        if value is None:
            object.__setattr__(self, "_grad", None)
            return
        if isinstance(value, np.ndarray):
            arr = DistributedArray.from_domain(
                Rect(tuple(value.shape)),
                dtype=self._data.dtype,
                distribution=Replicated(),
                fill="empty",
            )
            arr._data[...] = value
            object.__setattr__(self, "_grad", arr)
            return
        if isinstance(value, DistributedArray):
            object.__setattr__(self, "_grad", value)
            return
        raise TypeError(f"grad must be DistributedArray, ndarray, or None; got {type(value).__name__}")

    def zero_grad(self) -> None:
        """Drop the gradient slot. Cheaper than allocating zeros until autodiff lands."""
        object.__setattr__(self, "_grad", None)

    # Convenience pass-throughs -------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self) -> str:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def numel(self) -> int:
        return self._data.numel

    def numpy(self) -> np.ndarray:
        return self._data.numpy()

    # Make Parameters numpy-friendly so `np.asarray(p)` works in op calls.
    def __array__(self, dtype=None) -> np.ndarray:
        arr = self._data._data
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    def __repr__(self) -> str:
        return (
            f"Parameter(shape={self.shape}, dtype={self.dtype!r}, "
            f"requires_grad={self.requires_grad})"
        )


class Buffer:
    """Non-trainable named tensor that participates in `state_dict`.

    Wraps a `DistributedArray` (or a numpy array / shape spec). Distinct from
    `Parameter` in that it has no gradient slot and is not yielded by
    `Module.parameters()`. Useful for running statistics, precomputed tables,
    masks, and any module state that should be persisted but never
    differentiated.
    """

    __slots__ = ("_data", "persistent", "__weakref__")

    def __init__(
        self,
        data: DistributedArray | np.ndarray | tuple[int, ...] | list[int] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: str = "fp32",
        persistent: bool = True,
    ) -> None:
        if isinstance(data, DistributedArray):
            arr = data
        elif isinstance(data, np.ndarray):
            inferred = _normalize_dtype(str(data.dtype))
            arr = DistributedArray.from_domain(
                Rect(tuple(data.shape)),
                dtype=inferred if dtype == "fp32" else dtype,
                distribution=Replicated(),
                fill="empty",
            )
            arr._data[...] = data
        elif isinstance(data, (tuple, list)):
            arr = DistributedArray.from_domain(
                Rect(tuple(data)), dtype=dtype, distribution=Replicated(), fill="zeros"
            )
        elif data is None and shape is not None:
            arr = DistributedArray.from_domain(
                Rect(tuple(shape)), dtype=dtype, distribution=Replicated(), fill="zeros"
            )
        else:
            raise TypeError(
                "Buffer expects a DistributedArray, numpy array, shape tuple, "
                "or shape= keyword"
            )
        object.__setattr__(self, "_data", arr)
        object.__setattr__(self, "persistent", bool(persistent))

    @property
    def data(self) -> DistributedArray:
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self) -> str:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def numel(self) -> int:
        return self._data.numel

    def numpy(self) -> np.ndarray:
        return self._data.numpy()

    def __array__(self, dtype=None) -> np.ndarray:
        arr = self._data._data
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    def __repr__(self) -> str:
        return (
            f"Buffer(shape={self.shape}, dtype={self.dtype!r}, "
            f"persistent={self.persistent})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────


class Module:
    """Stateful container for `Parameter`s and child `Module`s.

    Subclass and override `forward()`. Use `__call__` to invoke the forward
    pass; this lets future versions hook pre/post-call instrumentation without
    every subclass having to opt in.

    Example:
        class MyBlock(Module):
            def __init__(self, dim):
                super().__init__()
                self.norm = RMSNorm(dim)
                self.proj = Linear(dim, dim, bias=False)

            def forward(self, x):
                return self.proj(self.norm(x))
    """

    def __init__(self) -> None:
        # Use object.__setattr__ to bypass our routing logic during init.
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "training", True)

    # Attribute routing ---------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        params = self.__dict__.get("_parameters")
        buffers = self.__dict__.get("_buffers")
        modules = self.__dict__.get("_modules")
        if params is None or buffers is None or modules is None:
            raise RuntimeError(
                f"Cannot assign attributes before {type(self).__name__}.__init__() "
                "has been called. Did you forget super().__init__()?"
            )

        if isinstance(value, Parameter):
            self.__dict__.pop(name, None)
            buffers.pop(name, None)
            modules.pop(name, None)
            params[name] = value
        elif isinstance(value, Buffer):
            self.__dict__.pop(name, None)
            params.pop(name, None)
            modules.pop(name, None)
            buffers[name] = value
        elif isinstance(value, Module):
            self.__dict__.pop(name, None)
            params.pop(name, None)
            buffers.pop(name, None)
            modules[name] = value
        else:
            params.pop(name, None)
            buffers.pop(name, None)
            modules.pop(name, None)
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        # Only invoked when normal __getattribute__ raises AttributeError.
        params = self.__dict__.get("_parameters")
        if params is not None and name in params:
            return params[name]
        buffers = self.__dict__.get("_buffers")
        if buffers is not None and name in buffers:
            return buffers[name]
        modules = self.__dict__.get("_modules")
        if modules is not None and name in modules:
            return modules[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __delattr__(self, name: str) -> None:
        if name in self.__dict__.get("_parameters", {}):
            del self._parameters[name]
        elif name in self.__dict__.get("_buffers", {}):
            del self._buffers[name]
        elif name in self.__dict__.get("_modules", {}):
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    # Buffer registration (torch.nn parity)
    def register_buffer(
        self,
        name: str,
        value: "Buffer | DistributedArray | np.ndarray | tuple[int, ...] | None",
        *,
        persistent: bool = True,
    ) -> None:
        """Register `value` as a non-trainable named buffer.

        Accepts a `Buffer` directly, or a tensor-like that will be wrapped.
        `persistent=False` excludes the buffer from `state_dict()`.
        """
        if value is None:
            raise TypeError("register_buffer requires a non-None value")
        if isinstance(value, Buffer):
            buf = value
            # Honor explicit `persistent=` arg if it differs from the wrapper default
            if buf.persistent != persistent:
                object.__setattr__(buf, "persistent", bool(persistent))
        else:
            buf = Buffer(value, persistent=persistent)
        # Route through __setattr__ so it lands in _buffers and is reachable
        # via attribute access.
        setattr(self, name, buf)

    # Parameter / module iteration ----------------------------------------------

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for p in self._parameters.values():
            yield p
        if recurse:
            for m in self._modules.values():
                yield from m.parameters(recurse=True)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            yield (f"{prefix}.{name}" if prefix else name, p)
        if recurse:
            for mod_name, m in self._modules.items():
                child_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from m.named_parameters(prefix=child_prefix, recurse=True)

    def buffers(self, recurse: bool = True) -> Iterator["Buffer"]:
        for b in self._buffers.values():
            yield b
        if recurse:
            for m in self._modules.values():
                yield from m.buffers(recurse=True)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, "Buffer"]]:
        for name, b in self._buffers.items():
            yield (f"{prefix}.{name}" if prefix else name, b)
        if recurse:
            for mod_name, m in self._modules.items():
                child_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from m.named_buffers(prefix=child_prefix, recurse=True)

    def children(self) -> Iterator["Module"]:
        yield from self._modules.values()

    def named_children(self) -> Iterator[tuple[str, "Module"]]:
        yield from self._modules.items()

    def modules(self) -> Iterator["Module"]:
        yield self
        for m in self._modules.values():
            yield from m.modules()

    # State dict ----------------------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return `{name: numpy-array}` for every parameter and persistent buffer.

        Non-persistent buffers (registered with `persistent=False`) are
        excluded — they are runtime-only state.
        """
        d: dict[str, np.ndarray] = {
            name: p.numpy().copy() for name, p in self.named_parameters()
        }
        for name, b in self.named_buffers():
            if b.persistent:
                d[name] = b.numpy().copy()
        return d

    def load_state_dict(
        self, sd: dict[str, np.ndarray | DistributedArray], strict: bool = True
    ) -> None:
        """Copy weights from a state dict into parameters and persistent buffers in-place.

        Parameter / Buffer handles are preserved — the underlying numpy buffer
        is overwritten, so any references to `param.data` / `buffer.data`
        remain valid.
        """
        param_dict = dict(self.named_parameters())
        # Only persistent buffers are loadable from state_dict — non-persistent
        # ones are excluded from `state_dict()` so they shouldn't appear in
        # incoming `sd`.
        buffer_dict = {
            name: b for name, b in self.named_buffers() if b.persistent
        }
        loadable = {**param_dict, **buffer_dict}

        if strict:
            extra_keys = set(sd) - set(loadable)
            missing_keys = set(loadable) - set(sd)
            if extra_keys or missing_keys:
                raise KeyError(
                    f"state_dict mismatch — missing: {sorted(missing_keys)}; "
                    f"unexpected: {sorted(extra_keys)}"
                )

        for name, holder in loadable.items():
            if name not in sd:
                continue
            value = sd[name]
            if isinstance(value, DistributedArray):
                value = value.numpy()
            value = np.asarray(value)
            if value.shape != holder.shape:
                raise ValueError(
                    f"shape mismatch loading {name}: expected {holder.shape}, "
                    f"got {value.shape}"
                )
            holder._data._data[...] = value

    # Mode switching ------------------------------------------------------------

    def train(self, mode: bool = True) -> "Module":
        """Set `self.training = mode` and recurse into children. Returns `self`."""
        object.__setattr__(self, "training", bool(mode))
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # Dtype migration -----------------------------------------------------------

    def to(self, dtype: str) -> "Module":
        """Migrate every Parameter and persistent Buffer to ``dtype`` in place.

        Returns ``self`` for chaining (``model.to("fp16").eval()``).

        Non-persistent buffers are left as-is (they're runtime scratch state
        that callers manage explicitly).

        Device migration (``to("cuda")``) is not yet supported — see Phase H of
        ``docs/audit/execution_roadmap.md``. ``dtype`` must be a Tessera dtype
        string (``"fp16"`` / ``"bf16"`` / ``"fp32"`` / ``"fp64"``); numpy
        dtype strings like ``"float32"`` are accepted and normalized.
        """
        normalized = _normalize_dtype(dtype)
        if normalized not in {"fp16", "bf16", "fp32", "fp64"}:
            raise ValueError(
                f"Unknown dtype {dtype!r}. Valid: fp16, bf16, fp32, fp64. "
                f"Device migration (to('cuda')/to('mps')) is not implemented yet."
            )

        np_dtype = _NUMPY_DTYPE_TO_TESSERA  # use the inverse mapping
        # Map Tessera dtype → numpy dtype string for in-place cast
        tessera_to_np = {
            "fp16": "float16",
            "bf16": "float32",  # bf16 stored as fp32 in the numpy reference path
            "fp32": "float32",
            "fp64": "float64",
        }
        target_np = tessera_to_np[normalized]

        for p in self.parameters():
            _migrate_distributed_array(p._data, target_np, normalized)
            # The numpy buffer identity changed — re-register so autodiff can
            # still trace gradients back to this Parameter through the new buffer.
            _register_parameter(p)
            # Drop any stale grad — its dtype no longer matches the parameter's
            if p.grad is not None:
                p.zero_grad()

        for b in self.buffers():
            if b.persistent:
                _migrate_distributed_array(b._data, target_np, normalized)

        return self

    # Gradient lifecycle --------------------------------------------------------

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    # Forward pass --------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__}.forward() is not implemented"
        )

    # Pretty-print --------------------------------------------------------------

    def __repr__(self) -> str:
        cls = type(self).__name__
        if not self._parameters and not self._buffers and not self._modules:
            return f"{cls}()"
        body = []
        for name, p in self._parameters.items():
            body.append(f"  {name}: Parameter(shape={p.shape}, dtype={p.dtype!r})")
        for name, b in self._buffers.items():
            body.append(
                f"  {name}: Buffer(shape={b.shape}, dtype={b.dtype!r}, "
                f"persistent={b.persistent})"
            )
        for name, m in self._modules.items():
            body.append(f"  {name}: {m!r}".replace("\n", "\n  "))
        return cls + "(\n" + "\n".join(body) + "\n)"


# ─────────────────────────────────────────────────────────────────────────────
# Containers
# ─────────────────────────────────────────────────────────────────────────────


class Sequential(Module):
    """Run a list of `Module`s in order. Output of each is the input to the next."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for i, m in enumerate(modules):
            if not isinstance(m, Module):
                raise TypeError(
                    f"Sequential expects Module instances; got {type(m).__name__} at index {i}"
                )
            setattr(self, str(i), m)

    def forward(self, x: Any) -> Any:
        for m in self._modules.values():
            x = m(x)
        return x

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, idx: int) -> Module:
        if idx < 0:
            idx += len(self)
        return self._modules[str(idx)]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())


class ModuleList(Module):
    """An indexable container of `Module`s that registers them as children."""

    def __init__(self, modules: list[Module] | tuple[Module, ...] = ()) -> None:
        super().__init__()
        for i, m in enumerate(modules):
            if not isinstance(m, Module):
                raise TypeError(
                    f"ModuleList expects Module instances; got {type(m).__name__} at index {i}"
                )
            setattr(self, str(i), m)

    def append(self, module: Module) -> "ModuleList":
        if not isinstance(module, Module):
            raise TypeError(f"ModuleList.append expects a Module, got {type(module).__name__}")
        setattr(self, str(len(self._modules)), module)
        return self

    def extend(self, modules) -> "ModuleList":
        for m in modules:
            self.append(m)
        return self

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, idx: int) -> Module:
        if idx < 0:
            idx += len(self)
        return self._modules[str(idx)]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())


class ModuleDict(Module):
    """A keyed container of `Module`s that registers them as children."""

    def __init__(self, modules: dict[str, Module] | None = None) -> None:
        super().__init__()
        if modules:
            for k, m in dict(modules).items():
                if not isinstance(m, Module):
                    raise TypeError(
                        f"ModuleDict expects Module values; got {type(m).__name__} for key {k!r}"
                    )
                setattr(self, k, m)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(f"ModuleDict expects Module values; got {type(module).__name__}")
        setattr(self, key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __contains__(self, key: object) -> bool:
        return key in self._modules

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()


__all__ = [
    "Parameter",
    "Buffer",
    "Module",
    "Sequential",
    "ModuleList",
    "ModuleDict",
]
