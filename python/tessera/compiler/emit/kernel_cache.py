"""Generic synth→compile→cache loop (COMPILER_REFACTOR_PLAN Workstream B4).

B2 gave every backend an emitter (:class:`KernelEmitter`, source) and a runner
(:class:`KernelRunner`, execute) behind one protocol; B3 made the F4 oracle gate
any backend's kernel. B4 closes the loop with the middle steps — **compile** and
**cache** — as an arch-agnostic driver:

    region ──emit──▶ KernelSource ──cache_key──▶ [cache hit? ] ──▶ CompiledKernel
                                                     │ miss
                                                     ▼
                                              compile_fn(source)  (per-arch plugin)

The **compile step is per-arch** — a plugin supplies it (Apple `metallib` /
NVIDIA `ptxas` / ROCm `hipcc` / x86 `clang`). This module owns only the
arch-neutral parts: the content-addressed :func:`cache_key`, the
:class:`KernelCache`, and the :func:`build` loop. It reuses the exact key
discipline the Apple runtime already uses — ``apple_gpu_runtime.mm`` caches
compiled pipelines by ``source + '\\x1f' + entry_point`` — extended with the
specialization metadata (``spec`` / ``shape_key`` / ``dtype`` / ``target``) and
hashed, so bucketed and dtype variants get distinct entries (the D1 arbiter keys
on this).

Apple's registered ``compile_fn`` is deliberately **deferred**: Metal compiles a
kernel on first launch (``newLibraryWithSource`` inside ``run_*``) and caches the
pipeline in the runtime, so the Python-level compile records a compile-on-launch
:class:`CompiledKernel` (``artifact is None``) rather than duplicating the work.
Workstream C backends register a real ahead-of-time ``compile_fn``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

from tessera.compiler.emit.kernel_emitter import (
    KernelSource,
    SpecPolicy,
    emit_kernel,
)

_SEP = "\x1f"  # unit separator — matches apple_gpu_runtime.mm's cache-key joiner


def cache_key(source: KernelSource, *, dtype: str, target: str) -> str:
    """Content-addressed key for a compiled kernel: sha256 over the source text +
    entry point + the specialization metadata that makes two *sources-equal*
    kernels genuinely distinct (dtype, shape bucket, target, policy).

    Mirrors ``apple_gpu_runtime.mm``'s ``source + '\\x1f' + entry`` map key, so
    the Python cache and the runtime's internal pipeline cache agree on identity,
    then extends it so a bucket/dtype/target variant is not aliased to another."""
    h = hashlib.sha256()
    for part in (
        source.source,
        source.entry,
        source.lang,
        source.spec.value,
        repr(source.shape_key),
        dtype,
        target,
    ):
        h.update(part.encode("utf-8"))
        h.update(_SEP.encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class CompiledKernel:
    """A kernel that has been through the compile step (or deferred to launch).

    ``artifact`` is the backend's opaque compiled handle (a metallib blob, a
    CUBIN, a ``.so`` path, …) — ``None`` for a *compile-on-launch* backend (Apple)
    whose kernel is compiled by the runtime at first ``run_*``. ``key`` is the
    :func:`cache_key`; ``deferred`` records the compile-on-launch case explicitly
    so a caller/arbiter never mistakes ``artifact is None`` for a failure."""

    key: str
    source: KernelSource
    target: str
    entry: str
    artifact: Any = None
    deferred: bool = False


class CompileError(RuntimeError):
    """A backend's compile step failed, or no compiler is registered for a target
    (Decision #21: name the gap, never silently no-op)."""


#: A compile function turns a KernelSource into an opaque compiled artifact for
#: its target. Return ``None`` to signal compile-on-launch (deferred).
CompileFn = Callable[[KernelSource], Any]

_COMPILERS: dict[str, CompileFn] = {}


def register_compiler(target: str, compile_fn: CompileFn) -> None:
    """Register the per-arch compile step for ``target`` (the plugin seam
    Workstream C fills with ``ptxas``/``hipcc``/``clang``). Re-registering
    replaces it."""
    if not target:
        raise ValueError("compiler target must be a non-empty backend id")
    _COMPILERS[target] = compile_fn


def get_compiler(target: str) -> CompileFn:
    """The compile fn for ``target`` or a clear diagnostic."""
    try:
        return _COMPILERS[target]
    except KeyError:
        known = ", ".join(sorted(_COMPILERS)) or "<none registered>"
        raise CompileError(
            f"no compiler registered for target {target!r}; known: {known}"
        ) from None


@dataclass
class KernelCache:
    """A content-addressed cache of :class:`CompiledKernel` keyed by
    :func:`cache_key`. Records hit/miss counts so a test (or the arbiter) can
    assert a repeated build reuses a compiled kernel instead of recompiling."""

    _store: dict[str, CompiledKernel] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> CompiledKernel | None:
        hit = self._store.get(key)
        if hit is not None:
            self.hits += 1
        else:
            self.misses += 1
        return hit

    def put(self, kernel: CompiledKernel) -> None:
        self._store[kernel.key] = kernel

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._store)


#: Process-wide default cache (the arbiter/runtime share one).
_DEFAULT_CACHE = KernelCache()


def default_cache() -> KernelCache:
    return _DEFAULT_CACHE


def build(
    region: Any,
    target: str,
    spec: SpecPolicy = SpecPolicy.BUCKET,
    *,
    dtype: str = "f32",
    dims: tuple[int, ...] | None = None,
    cache: KernelCache | None = None,
) -> CompiledKernel:
    """The synth→compile→cache loop: emit ``region`` for ``target``, key it, and
    return a compiled kernel from the cache (compiling on a miss via the
    registered per-arch ``compile_fn``).

    ``compile_fn`` returning ``None`` records a *deferred* (compile-on-launch)
    kernel. Raises :class:`CompileError` if no compiler is registered for
    ``target`` or the compile step fails; the emit step raises via
    ``emit_kernel`` (unknown target / unsupported region or policy)."""
    cache = cache if cache is not None else _DEFAULT_CACHE
    source = emit_kernel(region, target, spec, dtype=dtype, dims=dims)
    key = cache_key(source, dtype=dtype, target=target)

    cached = cache.get(key)
    if cached is not None:
        return cached

    compile_fn = get_compiler(target)
    try:
        artifact = compile_fn(source)
    except CompileError:
        raise
    except Exception as exc:  # a backend toolchain failure — name it, don't swallow
        raise CompileError(
            f"compile step for target {target!r} failed: {exc}") from exc

    compiled = CompiledKernel(
        key=key, source=source, target=target, entry=source.entry,
        artifact=artifact, deferred=artifact is None)
    cache.put(compiled)
    return compiled
