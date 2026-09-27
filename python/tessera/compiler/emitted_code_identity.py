"""Identity of the code a synthesized or emitted candidate runs (Decision #11).

The arbiter reuses a measured verdict only while every candidate racing now
runs the code that was timed (``autotune._record_matches_live_delegates``).
Until 2026-09-27 only ``Tier.HAND_TUNED`` candidates had to prove that: a
SYNTHESIZED or EMITTED lane "came from this checkout's emitters under the
pinned toolchain" and was served on the pin-based family identity alone. That
is not an identity. The emitters change without any pin moving, and a verdict
measured for yesterday's generated kernel kept selecting today's (Codex review
P2 on PR #859). Every candidate, of every tier, now establishes an identity for
the code it would run for one workload, or the verdict misses.

:mod:`kernel_code_identity` covers the images ``tessera-opt`` builds (a
normalized instruction stream). This module covers everything else, by how the
candidate produces its code. Each builder returns a flat ``dict[str, str]`` --
what the arbiter stamps in ``evidence.delegate_identities`` -- so
``kernel_code_identity.identity_mismatch`` names the field that moved.

Normalization ``tessera.emitted_source.v1``
-------------------------------------------

* **Python-emitted source** (:func:`kernel_source_identity`,
  :func:`source_identity`): the exact text the lane hands its compiler for this
  ``(region, inputs)`` -- sha256 over each unit's label and text -- plus, when
  the source is a :class:`~tessera.compiler.emit.kernel_emitter.KernelSource`,
  its content-addressed ``kernel_cache.cache_key`` (text + entry + lang + spec
  + shape key + binding layouts + dtype + target), and the ``build`` line: the
  compile flags, offload arch and defines the toolchain pin does not fix. The
  compiler appears by *name* (``nvcc``, ``hipcc``), never by path: its version
  is the family pin, and a path differs between boxes that run the same code.
  Computable host-side, with no device and no compiler.
* **Host-compiled source** (:func:`source_file_identity` plus
  :func:`compiler_version`): a checked-in C/C++ file compiled at run time by
  the host C compiler. No pin fixes that compiler (the ``cpu`` family pins only
  LLVM/MLIR), so its ``--version`` line is part of the identity; a host with no
  compiler has no identity -- and cannot run the lane either. Every local
  header the file reaches through a quoted ``#include`` is a unit too (added
  2026-09-27 without a normalization bump: it changes only the digest of a file
  that has such headers, and that digest was missing code the file compiles).
  A lane that compiles such a file once per process is identified by the bytes
  it compiled (:func:`identify_compilation`), not by the file as it is later.
* **PTX handed to the driver JIT** (:func:`ptx_identity`): the PTX text with
  full-line ``//`` comments and blank lines dropped (llc and the Python
  emitters put only banners there), every other line kept verbatim. The
  ``.version`` re-stamp at registration is a function of the pinned driver JIT
  ISA, so it is covered by the family identity.
* **Python/numpy lanes** (:func:`python_code_identity`): the source text of the
  named functions/classes that implement the lane. This is an approximation
  and is stated as one: it changes when that code changes, but not when code
  it calls without naming it changes (numpy itself, a helper not listed).
* **Several units** (:func:`composite_identity`): a lane that runs more than
  one artifact -- a shipped library plus an emitted epilogue, or two emitted
  kernels selected by a data-dependent branch -- carries every part, each
  field prefixed with the part's label.

Limits, stated so they are not read as covered: host-side Python around a
kernel (operand casts such as ``_composed_operand``, layout materialization,
launch sequencing in a device session) is not digested; the toolchain *pins*
stand for nvcc/hipcc/driver versions, so a box drifting off its pin without the
pin moving is not caught here (the family identity's own limit); and a lane
whose kernel is chosen by a data-dependent test carries every kernel it may
choose (a false miss when only the untaken one changes, never a false hit).

What is deliberately *not* here: loaded-library identities
(``toolchain_identity.delegate_library_identity``), which a composite may
include as a part, and ``tessera-opt`` images (:mod:`kernel_code_identity`).

**Fail closed.** Empty source, an unreadable file, a compiler that will not
report a version, or source ``inspect`` cannot find raise
:class:`EmittedIdentityUnavailable`. :func:`identify` turns any failure into
``None`` -- a miss -- and records why (:func:`miss_reason`); it never returns a
partial identity.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

#: Normalization version. Bump it when a rule above changes: every stamped row
#: then misses, which is the correct outcome for a changed definition.
NORMALIZATION = "tessera.emitted_source.v1"


class EmittedIdentityUnavailable(RuntimeError):
    """The identity of a lane's code could not be established (fail closed)."""


def _units_digest(units: Sequence[tuple[str, str | bytes]]) -> str:
    if not units:
        raise EmittedIdentityUnavailable("no source units to identify")
    h = hashlib.sha256()
    for label, text in units:
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        if not data.strip():
            raise EmittedIdentityUnavailable(f"source unit {label!r} is empty")
        h.update(label.encode("utf-8") + b"\0")
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def source_identity(*, lang: str, entry: str,
                    units: Sequence[tuple[str, str | bytes]],
                    build: Sequence[str],
                    generator: str = "tessera.emit",
                    extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The identity of source text compiled by ``build``.

    ``units`` is ordered ``(label, text)``; ``build`` is the compiler name and
    every flag that changes the binary (never a path). Raises
    :class:`EmittedIdentityUnavailable` on an empty unit or build line."""
    if not build or not all(str(part).strip() for part in build):
        raise EmittedIdentityUnavailable("build line is empty")
    identity = {
        "identity": "emitted_source",
        "normalization": NORMALIZATION,
        "generator": generator,
        "lang": lang,
        "entry": entry,
        "units": ",".join(label for label, _ in units),
        "source_sha256": _units_digest(units),
        "build": " ".join(str(part) for part in build),
    }
    for key, value in (extra or {}).items():
        # An `extra` field may add to the identity, never replace a core one:
        # an overwritten `source_sha256` or `build` would stamp a digest of
        # something other than the code (fail closed).
        if str(key) in identity:
            raise EmittedIdentityUnavailable(
                f"extra identity field {key!r} collides with a core field")
        identity[str(key)] = str(value)
    return identity


def kernel_source_identity(source: Any, *, dtype: str, target: str,
                           build: Sequence[str],
                           generator: str = "tessera.emit") -> dict[str, str]:
    """:func:`source_identity` of one emitted ``KernelSource``, with its
    content-addressed ``kernel_cache.cache_key`` -- the key the compile cache
    already uses to decide whether two emits are the same kernel."""
    from tessera.compiler.emit.kernel_cache import cache_key

    return source_identity(
        lang=str(source.lang), entry=str(source.entry),
        units=[(str(source.entry), source.source)], build=build,
        generator=generator,
        extra={"cache_key": cache_key(source, dtype=dtype, target=target)})


def normalized_ptx(ptx: str) -> str:
    """PTX with full-line ``//`` comments and blank lines dropped; every other
    line verbatim (trailing whitespace stripped)."""
    kept = [line.rstrip() for line in ptx.splitlines()
            if line.strip() and not line.lstrip().startswith("//")]
    if not kept:
        raise EmittedIdentityUnavailable("PTX is empty after normalization")
    return "\n".join(kept) + "\n"


def ptx_identity(ptx: str, *, entry: str, generator: str,
                 build: Sequence[str] = ("driver-jit",)) -> dict[str, str]:
    """:func:`source_identity` of PTX text (see the module rules)."""
    if f" {entry}" not in ptx and f"{entry}(" not in ptx:
        raise EmittedIdentityUnavailable(f"PTX does not define entry {entry}")
    return source_identity(lang="ptx", entry=entry,
                           units=[(entry, normalized_ptx(ptx))],
                           build=build, generator=generator)


_COMPILER_VERSIONS: dict[str, str] = {}


def compiler_version(command: str) -> str:
    """The first line of ``<command> --version`` (cached per command). Raises
    :class:`EmittedIdentityUnavailable` when the compiler cannot say."""
    cached = _COMPILER_VERSIONS.get(command)
    if cached is not None:
        return cached
    try:
        done = subprocess.run([command, "--version"], capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EmittedIdentityUnavailable(
            f"{command} --version failed: {exc}") from exc
    lines = [ln.strip() for ln in (done.stdout or done.stderr).splitlines()
             if ln.strip()]
    if done.returncode != 0 or not lines:
        raise EmittedIdentityUnavailable(
            f"{command} --version did not name a version (rc={done.returncode})")
    _COMPILER_VERSIONS[command] = lines[0]
    return lines[0]


_QUOTED_INCLUDE = re.compile(rb'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.MULTILINE)


def _local_include_units(path: Path, data: bytes,
                         seen: set[Path]) -> list[tuple[str, bytes]]:
    """Every ``#include "..."`` reachable from ``path`` that resolves next to
    the including file, depth-first, each once, labelled by its path relative
    to the root file's directory. A quoted include that does not resolve there
    is left to the compiler's search path and is not covered (stated, not
    guessed)."""
    units: list[tuple[str, bytes]] = []
    for match in _QUOTED_INCLUDE.finditer(data):
        target = (path.parent / match.group(1).decode("utf-8", "replace")).resolve()
        if target in seen or not target.is_file():
            continue
        seen.add(target)
        try:
            text = target.read_bytes()
        except OSError as exc:
            raise EmittedIdentityUnavailable(f"cannot read {target}: {exc}") from exc
        units.append((match.group(1).decode("utf-8", "replace"), text))
        units.extend(_local_include_units(target, text, seen))
    return units


def source_file_identity(path: str | Path, *, lang: str, entry: str,
                         build: Sequence[str], compiler: str) -> dict[str, str]:
    """A checked-in source file compiled by the host ``compiler``: the file's
    bytes and those of every local header it includes by a quoted
    ``#include`` (2026-09-27: `StockhamRadix4.cpp` includes
    ``../Common/FFTPlan.h``, whose code the digest used to miss), the flags,
    and the compiler's version line."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise EmittedIdentityUnavailable(f"cannot read {p}: {exc}") from exc
    units: list[tuple[str, str | bytes]] = [(p.name, data)]
    units.extend(_local_include_units(p, data, {p.resolve()}))
    return source_identity(
        lang=lang, entry=entry, units=units, build=build,
        generator="checked_in_source",
        extra={"compiler": compiler_version(compiler)})


def identify_compilation(compile_step: Callable[[], Any],
                         build_identity: Callable[[], Mapping[str, str]]
                         ) -> tuple[Any, dict[str, str] | None]:
    """Run ``compile_step`` between two computations of the identity of what it
    compiles, and return ``(its result, that identity)``.

    For a lane that compiles once per process and reuses the loaded artifact:
    the artifact is then identified by the code it was compiled from, not by
    the file on disk later (which may have been edited since). The identity is
    ``None`` -- a miss -- when it cannot be computed or when it differs before
    and after (the source changed during the compile, so which bytes the
    artifact holds is unknown)."""
    try:
        before: dict[str, str] | None = dict(build_identity())
    except Exception:  # noqa: BLE001 - an unidentifiable compile is a miss
        before = None
    result = compile_step()
    try:
        after: dict[str, str] | None = dict(build_identity())
    except Exception:  # noqa: BLE001
        after = None
    return result, (before if before is not None and before == after else None)


def loaded_identity(recorded: Mapping[str, str] | None, what: str) -> dict[str, str]:
    """The identity :func:`identify_compilation` recorded for a loaded
    artifact, or :class:`EmittedIdentityUnavailable` when it recorded none."""
    if not recorded:
        raise EmittedIdentityUnavailable(
            f"{what} is loaded but the code it was compiled from is unknown "
            "(its source changed during the compile, or could not be identified)")
    return dict(recorded)


def python_code_identity(*objects: Any, lane: str) -> dict[str, str]:
    """The source text of the Python ``objects`` that implement ``lane``.

    Approximate by construction (module docstring): code they call without
    naming it here is not covered."""
    units: list[tuple[str, str]] = []
    for obj in objects:
        name = f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', repr(obj))}"
        try:
            text = inspect.getsource(obj)
        except (OSError, TypeError) as exc:
            raise EmittedIdentityUnavailable(
                f"no source for {name}: {exc}") from exc
        units.append((name, text))
    return {
        "identity": "python_code",
        "normalization": NORMALIZATION,
        "lane": lane,
        "units": ",".join(label for label, _ in units),
        "source_sha256": _units_digest(units),
    }


def composite_identity(parts: Mapping[str, Mapping[str, str] | None]
                       ) -> dict[str, str]:
    """One identity from several: every field of each part, prefixed with the
    part's label. A missing part (``None``) is a miss for the whole."""
    if not parts:
        raise EmittedIdentityUnavailable("a composite identity needs parts")
    out = {"identity": "composite", "normalization": NORMALIZATION,
           "parts": ",".join(parts)}
    for label, part in parts.items():
        if part is None:
            raise EmittedIdentityUnavailable(f"part {label!r} has no identity")
        for key, value in part.items():
            field = f"{label}.{key}"
            # Labels and keys may contain dots, so ("a.b", "c") and ("a", "b.c")
            # flatten to the same field; never let one part overwrite another.
            if field in out:
                raise EmittedIdentityUnavailable(
                    f"composite identity field {field!r} is produced twice")
            out[field] = str(value)
    return out


_MISS_REASONS: dict[str, str] = {}


def identify(owner: str, build_identity: Callable[[], Mapping[str, str] | None]
             ) -> dict[str, str] | None:
    """Run ``build_identity`` for candidate ``owner``; any failure, or a
    ``None``, is a miss (``None``) with the reason kept in
    :func:`miss_reason`. Never cached: the point is to notice an emitter that
    changed since the verdict was recorded, including within one process."""
    try:
        identity = build_identity()
    except Exception as exc:  # noqa: BLE001 - any failure to identify is a miss
        _MISS_REASONS[owner] = f"{type(exc).__name__}: {exc}"
        return None
    if not identity:
        _MISS_REASONS[owner] = ("returned no identity (no workload operands, or a "
                                "region this lane does not run)")
        return None
    _MISS_REASONS.pop(owner, None)
    return {str(k): str(v) for k, v in identity.items()}


def miss_reason(owner: str) -> str | None:
    """Why :func:`identify` last returned ``None`` for ``owner``."""
    return _MISS_REASONS.get(owner)


def clear_caches() -> None:
    _COMPILER_VERSIONS.clear()
    _MISS_REASONS.clear()


__all__ = [
    "NORMALIZATION",
    "EmittedIdentityUnavailable",
    "clear_caches",
    "compiler_version",
    "composite_identity",
    "identify",
    "identify_compilation",
    "kernel_source_identity",
    "loaded_identity",
    "miss_reason",
    "normalized_ptx",
    "ptx_identity",
    "python_code_identity",
    "source_file_identity",
    "source_identity",
]
