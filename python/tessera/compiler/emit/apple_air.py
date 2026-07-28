"""Ahead-of-time Apple GPU target: synthesized MSL → AIR → ``.metallib``.

The default ``apple_gpu`` target is *compile-on-launch* — its ``compile_fn``
returns ``None`` and Metal builds the kernel inside ``run_*`` via
``newLibraryWithSource:``. That is the right default for a JIT, and the wrong
one for measuring: the first launch of every distinct kernel pays a front-end
compile, and nothing is reusable across processes.

This module registers a **second** target, ``apple_gpu_air``, emitting the same
MSL and compiling it ahead of time through Apple's Metal toolchain:

    MSL ──`xcrun metal -c`──▶ .air ──`xcrun metallib`──▶ .metallib

``.air`` is LLVM bitcode (magic ``dec0 170b``, triple ``air64_v28-apple-*``), so
the artifact is inspectable with the same LLVM the rest of the compiler is
pinned to. Both tools ship in the downloadable Metal Toolchain
(``xcodebuild -downloadComponent MetalToolchain``); without it, this target
declines rather than falling back to the JIT lane, so a measurement never
silently comes from the other path.

Why a separate target rather than a flag on ``apple_gpu``: it makes AOT-vs-JIT
a *measured* arbiter candidate per ``(op, shape-bucket, dtype, target)``
(Decision #28) instead of a build-time switch. Both can be built and timed.

This is the supported, documented half of the Apple AOT question. Emitting AIR
*directly* from LLVM IR — bypassing the MSL front end — is the separate, harder
question tracked in Decision #26a; nothing here depends on it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from tessera.compiler.emit.kernel_cache import CompileError, register_compiler
from tessera.compiler.emit.kernel_emitter import (
    KernelEmitter,
    KernelSource,
    SpecPolicy,
    get_emitter,
    register_emitter,
)

#: Target id. Deliberately distinct from ``apple_gpu`` so the arbiter can hold
#: both and the cache never aliases a JIT kernel to an AOT one.
AIR_TARGET = "apple_gpu_air"

#: Where compiled ``.metallib`` blobs live between runs. Content-addressed, so a
#: rebuild of identical source is a cache hit rather than a recompile.
_CACHE_DIR = Path(
    os.environ.get("TESSERA_APPLE_AIR_CACHE")
    or Path(tempfile.gettempdir()) / "tessera-apple-air"
)


class MetalToolchainError(CompileError):
    """Apple's offline Metal toolchain is missing or failed — names which."""


@lru_cache(maxsize=1)
def metal_toolchain_available() -> bool:
    """Whether ``xcrun metal`` can actually run on this host.

    Presence of the binary is not enough: Xcode ships a ``metal`` driver that
    exits with "cannot execute tool 'metal' due to missing Metal Toolchain"
    until the component is downloaded. Probe by asking for the version.
    """
    if shutil.which("xcrun") is None:
        return False
    try:
        result = subprocess.run(["xcrun", "metal", "--version"],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def toolchain_hint() -> str:
    """Actionable text for why the AOT lane is unavailable."""
    if shutil.which("xcrun") is None:
        return "xcrun not found — this lane needs macOS with Xcode installed"
    return (
        "Apple Metal toolchain unavailable. Point xcode-select at Xcode "
        "(`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`) "
        "and install the component (`xcodebuild -downloadComponent "
        "MetalToolchain`); verify with `xcrun metal --version`."
    )


class AppleAIREmitter(KernelEmitter):
    """Emits the same MSL as ``apple_gpu``; only the compile step differs.

    Delegating rather than duplicating is deliberate — a second copy of the
    synthesis dispatch would drift from the JIT lane, and then AOT and JIT would
    be timing different kernels while claiming to compare compile strategies.
    """

    target = AIR_TARGET
    lang = "msl"

    def _msl(self) -> KernelEmitter:
        return get_emitter("apple_gpu")

    def can_emit(self, region: Any) -> bool:
        return self._msl().can_emit(region)

    def emit(
        self,
        region: Any,
        *,
        spec: SpecPolicy = SpecPolicy.BUCKET,
        dtype: str = "f32",
        dims: tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> KernelSource:
        return self._msl().emit(region, spec=spec, dtype=dtype, dims=dims, **kwargs)


register_emitter(AppleAIREmitter())


def compile_msl_to_metallib(source: str, *, entry: str,
                            cache_dir: Path | None = None) -> Path:
    """Compile MSL to a ``.metallib``, returning its path.

    Content-addressed on ``(source, entry)`` so an identical kernel is compiled
    once per machine. Raises :class:`MetalToolchainError` naming the failing
    stage — never returns a path that was not produced by both tools.
    """
    if not metal_toolchain_available():
        raise MetalToolchainError(toolchain_hint())

    directory = cache_dir if cache_dir is not None else _CACHE_DIR
    digest = hashlib.sha256(
        source.encode("utf-8") + b"\x1f" + entry.encode("utf-8")).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    metallib = directory / f"{digest}.metallib"
    if metallib.is_file():
        return metallib

    work = directory / digest
    work.mkdir(parents=True, exist_ok=True)
    metal, air = work / "k.metal", work / "k.air"
    metal.write_text(source, encoding="utf-8")

    for stage, command in (
        ("metal -c (MSL → AIR)", ["xcrun", "metal", "-c", str(metal), "-o", str(air)]),
        ("metallib (AIR → metallib)",
         ["xcrun", "metallib", str(air), "-o", str(metallib)]),
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            raise MetalToolchainError(f"{stage} could not run: {exc}") from exc
        if result.returncode != 0:
            raise MetalToolchainError(
                f"{stage} failed for entry {entry!r}:\n{result.stderr.strip()}")
    if not metallib.is_file():
        raise MetalToolchainError(
            f"{stage} reported success but produced no metallib for {entry!r}")
    return metallib


def _metallib_coopmat_symbol() -> Any:
    """ctypes binding for the AOT dispatch (`newLibraryWithURL:` + same body).

    The JIT twin is `apple_msl._synth_coopmat_symbol`; both end in the same
    `dispatch_matmul_epilogue_coopmat` in the runtime, so a timing difference
    between them is the compile strategy and not a different dispatch.
    """
    import ctypes

    from tessera.runtime import _load_apple_gpu_runtime

    runtime = _load_apple_gpu_runtime()
    symbol = getattr(
        runtime, "tessera_apple_gpu_metallib_matmul_epilogue_coopmat", None)
    if symbol is None:
        return None
    void_p = ctypes.c_void_p
    symbol.argtypes = [ctypes.c_char_p, ctypes.c_char_p, void_p, void_p, void_p,
                       void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                       ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    symbol.restype = ctypes.c_int32
    return symbol


def run_fused_region_aot(region: Any, A: Any, B: Any, bias: Any = None, *,
                         tile: int = 32) -> tuple[Any, str]:
    """Run a coopmat region from a prebuilt `.metallib`.

    Returns ``(output, execution)`` with ``execution`` one of
    ``"metal_metallib"`` (the AOT kernel ran) or ``"reference"`` (numpy — no
    device, no toolchain, or the dispatch declined). It never reports the AOT
    tag for work the JIT lane did: the only device path here is the metallib
    symbol.
    """
    import ctypes

    import numpy as np

    from tessera.compiler.emit import apple_msl

    dtype = np.float16 if np.asarray(A).dtype == np.float16 else np.float32
    a = np.ascontiguousarray(A, dtype)
    b = np.ascontiguousarray(B, dtype)
    m, k = a.shape
    _, n = b.shape
    out = np.empty((m, n), dtype)

    symbol = _metallib_coopmat_symbol()
    if symbol is not None and metal_toolchain_available():
        source = get_emitter(AIR_TARGET).emit(
            region, dtype="f16" if dtype == np.float16 else "f32",
            dims=(m, n, k), variant=apple_msl.COOPMAT)
        try:
            metallib = compile_msl_to_metallib(source.source, entry=source.entry)
        except MetalToolchainError:
            metallib = None
        if metallib is not None:
            bias_arr = (np.ascontiguousarray(bias, dtype)
                        if bias is not None else None)
            pointer = (lambda arr: arr.ctypes.data_as(ctypes.c_void_p))
            ok = symbol(
                str(metallib).encode("utf-8"), source.entry.encode("utf-8"),
                pointer(a), pointer(b),
                pointer(bias_arr) if bias_arr is not None else None,
                pointer(out), m, n, k, 1 if bias_arr is not None else 0,
                dtype().itemsize, tile)
            if ok:
                return out, "metal_metallib"

    # The region's own reference — the same one the F4 oracle compares against,
    # so a fallback here is numerically identical to the JIT lane's fallback.
    reference = region.reference(np.asarray(a, np.float32),
                                 np.asarray(b, np.float32),
                                 None if bias is None else np.asarray(bias, np.float32),
                                 None)
    return np.asarray(reference, dtype), "reference"


def _apple_air_compile_fn(source: KernelSource) -> str:
    """The ``apple_gpu_air`` compile step — a real artifact, not a deferral.

    Returning a path (rather than ``None``) is what distinguishes this target
    from ``apple_gpu`` in a :class:`~...kernel_cache.CompiledKernel`: ``deferred``
    is False and ``artifact`` is the ``.metallib`` on disk.
    """
    return str(compile_msl_to_metallib(source.source, entry=source.entry))


register_compiler(AIR_TARGET, _apple_air_compile_fn)
