"""In-process compile cache for x86 native packaging (E2E-REAL-6, 2026-09-28).

Packaging one x86 Graph op through the compiled route runs ``tessera-opt``
several times -- Graph -> Schedule, Schedule -> Tile, the replay of both, and
the ``tessera-x86-executable`` Tile -> Target pipeline -- plus a ``--version``
probe and a read of the prebuilt AVX-512 shared object. Measured on
Princess-Luna that was ~100 ms per package call against ~33 ms for the
retired Python constructors, all of it subprocess and I/O. Every one of those
steps is a pure function of its inputs, so each is memoized on exactly those
inputs:

* a ``tessera-opt`` run is keyed on the compiler's ELF dependency identity
  (``rocm_native._tool_digest``, memoized on compiler and loaded-library stat signatures, so a
  rebuilt compiler or dependency misses -- Decision #11), the pass option, and the complete
  source text. The source is the MLIR the compiler receives: it names the
  Graph op, every attribute, every shape and dtype, the target/arch module
  attributes and the launch bindings, so any change to the input misses;
* the ``--version`` fingerprint is keyed on the same binary digest;
* a shared-object payload is keyed on its path and stat signature
  (device, inode, mtime, size), the rule ``toolchain_identity`` uses.

Nothing above the compiler boundary is cached: descriptors, projections and
replays are rebuilt on every call from the (cached) compiler outputs, so a
forged artifact still fails its replay comparison. A failing run is never
cached. The cache is per process and bounded (LRU).
"""

from __future__ import annotations

import hashlib
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path

_SCHEMA = "tessera.x86_compile_cache.v1"
_MAX_ENTRIES = 4096

_lock = threading.Lock()
_runs: OrderedDict[str, str] = OrderedDict()
_versions: dict[str, str] = {}
_payloads: dict[tuple[str, int, int, int, int], tuple[bytes, str]] = {}
_stats = {"hits": 0, "misses": 0}


def _tool_digest(tool: Path) -> str:
    from .rocm_native import _tool_digest as digest

    return digest(Path(tool))


def run_key(tool: Path, source: str, option: str) -> str:
    """The cache identity of one ``tessera-opt`` run."""
    return hashlib.sha256(
        "\x1f".join((_SCHEMA, _tool_digest(tool), option, source)).encode()
    ).hexdigest()


def run(tool: Path, source: str, option: str) -> str:
    """``scheduled_matmul.run_tessera_opt`` with an exact-input memo."""
    key = run_key(tool, source, option)
    with _lock:
        cached = _runs.get(key)
        if cached is not None:
            _runs.move_to_end(key)
            _stats["hits"] += 1
            return cached
    result = subprocess.run(
        [str(tool), "-", option], input=source, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"scheduled compiler boundary {option} failed: "
            + (result.stderr.strip() or str(result.returncode))
        )
    with _lock:
        _stats["misses"] += 1
        _runs[key] = result.stdout
        _runs.move_to_end(key)
        while len(_runs) > _MAX_ENTRIES:
            _runs.popitem(last=False)
    return result.stdout


def version_fingerprint(tool: Path) -> str:
    """SHA-256 of ``tessera-opt --version``, memoized on the binary digest."""
    digest = _tool_digest(tool)
    with _lock:
        cached = _versions.get(digest)
    if cached is not None:
        return cached
    result = subprocess.run([str(tool), "--version"], capture_output=True, text=True, check=False)
    text = "\n".join(value.strip() for value in (result.stdout, result.stderr) if value.strip())
    fingerprint = hashlib.sha256((text or str(tool)).encode()).hexdigest()
    with _lock:
        _versions[digest] = fingerprint
    return fingerprint


def payload(path: Path) -> tuple[bytes, str]:
    """A shared object's bytes and SHA-256, memoized on its stat signature."""
    resolved = Path(path).resolve()
    st = resolved.stat()
    key = (str(resolved), st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)
    with _lock:
        cached = _payloads.get(key)
    if cached is None:
        data = resolved.read_bytes()
        cached = (data, hashlib.sha256(data).hexdigest())
        with _lock:
            _payloads[key] = cached
    return cached


def stats() -> dict[str, int]:
    with _lock:
        return {**_stats, "entries": len(_runs)}


def clear() -> None:
    with _lock:
        _runs.clear()
        _versions.clear()
        _payloads.clear()
        _stats.update(hits=0, misses=0)


__all__ = ["clear", "payload", "run", "run_key", "stats", "version_fingerprint"]
