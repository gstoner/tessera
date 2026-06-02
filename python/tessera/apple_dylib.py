"""Lane (c) — MSL-source → serialized ``.metallib`` dynamic-library AOT.

The parallel AOT lane to the MPSGraph ``.mtlpackage`` path (PK8). Where
packaged kernels serialize an MPSGraph ML model, this serializes Tessera's
*MSL-source* custom kernels (rope, flash_attn, gelu, ...) into a reloadable
Metal **dynamic library** — author once, reload with zero recompilation.

Grounded in the on-machine SDK headers (CLAUDE.md Decision #27): compile the
MSL with ``MTLLibraryType.dynamic`` + an ``installName``, wrap in an
``MTLDynamicLibrary``, ``serializeToURL:`` to disk, and reload via
``newDynamicLibraryWithURL:``. Dynamic libraries hold ``[[visible]]``
functions that other Metal libraries link against.

Off-Darwin / pre-macOS-11 the runtime returns failure and these helpers
return ``False`` (no exception), so callers skip cleanly.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol


def dylib_available() -> bool:
    """True when the Apple GPU runtime is buildable/loadable on this host
    (the dynamic-library APIs need a real Metal device)."""
    return apple_gpu_runtime() is not None


def serialize_msl_dylib(
    msl_source: str,
    out_path: str | Path,
    *,
    install_name: str = "@loader_path/tessera_kernels.metallib",
) -> bool:
    """Compile ``msl_source`` as a Metal **dynamic library** and serialize it
    to ``out_path`` (a ``.metallib``). The source must expose ``[[visible]]``
    functions. ``install_name`` is embedded into any library that later links
    against this one (an ``@loader_path``-relative path is the portable
    default). Returns ``True`` on success; ``False`` if the runtime is
    unavailable or any step fails.
    """
    if apple_gpu_runtime() is None:
        return False
    fn = bind_symbol(
        "tessera_apple_gpu_dylib_serialize",
        (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return False
    rc = int(fn(msl_source.encode("utf-8"),
                install_name.encode("utf-8"),
                str(out_path).encode("utf-8")))
    return rc == 1


def load_dylib(path: str | Path) -> bool:
    """Reload a serialized ``.metallib`` dynamic library from ``path``.
    Returns ``True`` if it loads with a non-empty install name (the
    round-trip succeeded), else ``False``."""
    if apple_gpu_runtime() is None:
        return False
    fn = bind_symbol(
        "tessera_apple_gpu_dylib_load",
        (ctypes.c_char_p,),
        restype=ctypes.c_int32,
    )
    if fn is None:
        return False
    return bool(fn(str(path).encode("utf-8")))


# A minimal valid dynamic-library source — one [[visible]] function. Handy as
# a smoke fixture and as a template for the kernel-source AOT path.
SAMPLE_VISIBLE_MSL = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "[[visible]] float tessera_dylib_add(float a, float b) { return a + b; }\n"
)


__all__ = [
    "dylib_available",
    "serialize_msl_dylib",
    "load_dylib",
    "SAMPLE_VISIBLE_MSL",
]
