"""Toolchain and delegate identity for measurement caches (Decision #11).

Decision #11 (amended 2026-08-30): an autotune cache entry is a *measurement*,
and a measurement is only valid for the code that produced it. A key built from
``{op, shape, dtype, arch, layout, numeric_policy, movement}`` alone survives a
CUDA/ROCm toolkit upgrade, an LLVM bump, or a rebuilt delegate library, and
then returns a number that no longer describes anything that can run. This
module is the one place that says *which toolchain a measurement was made
under*, so every cache can put it in its key and a stale entry **misses rather
than lies**.

Nothing here probes a toolchain afresh. Each component is read from an existing
source of truth:

* LLVM/MLIR — the exact pin in ``cmake/TesseraToolchainPins.cmake``, read
  through :func:`runtime_abi_audit.cmake_toolchain_pins` (the file's one
  reader, and the one ``runtime_abi_audit`` drift-gates against the Python
  pins).
* NVIDIA — the ``gpu_target.py`` pins: toolkit, PTX ISA, driver floor, the
  driver API release and the PTX ISA the pinned driver's JIT accepts.
* ROCm — the ``rocm_target.py`` pins: ROCm release and HIP version.
* Apple — there is no declared Apple pin, so the family identity is read from
  the machine, from facts that do not depend on the caller's shell: the macOS
  version, and via the system ``/usr/bin/xcrun`` (which resolves through
  xcode-select, never ``PATH``) the SDK version, the Metal compiler version and
  the Xcode version, plus the Apple runtime source fingerprint (the
  hand-written MSL that is what actually gets timed on Apple) and the device
  family tag. The first ``clang`` on ``PATH`` is deliberately *not* an input.
* A per-candidate artifact — :func:`delegate_library_identity`: the content
  digest of a delegate's loaded shared library (its ABI hash) plus, when it sits
  in a configured build tree, its ``runtime_library_build`` optimization record;
  and, for kernels ``tessera-opt`` generates, the digest of the normalized
  instruction stream of the image the candidate would run for the workload
  (:mod:`kernel_code_identity`; 2026-09-26, replacing :func:`tessera_opt_identity`,
  whose binary digest differed on every build). These are stamped per candidate
  by the arbiter (``Candidate.artifact_identity``) and checked against the live
  candidate before a verdict is reused.

**The family identity is pin-based.** The NVIDIA/ROCm/LLVM components are the
pins the fleet is held to (``runtime_abi_audit`` drift-gates them, and the LLVM
pin is exact), so a toolkit upgrade that is *adopted* — the pin moves —
invalidates every entry measured under the old pin. A box that drifts off its
pin without the pin moving is **not** caught by the family identity. What does
catch a changed artifact is the per-candidate identity above: a rebuilt
delegate library changes its content digest, and a generated kernel whose
instructions change changes its instruction-stream digest, and the verdict for
that candidate misses. A rebuilt ``tessera-opt`` that generates the *same*
kernel keeps the verdict -- the point of keying on the code that was timed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "tessera.toolchain_identity.v1"

#: Value recorded for a component whose source of truth is absent (e.g. the
#: CMake pin file in an installed package). Recorded, never guessed.
UNDECLARED = "undeclared"
#: Value recorded for a live Apple fact that could not be read on this host.
UNAVAILABLE = "unavailable"

FAMILIES = ("nvidia", "rocm", "apple", "cpu", "generic")


def target_family(arch_or_target: str) -> str:
    """The toolchain family an arch or target string belongs to.

    Mirrors :meth:`autotune_v2.BayesianAutotuner.target_features` and accepts
    the arbiter's target/device spellings (``nvidia``, ``nvidia:sm_120``,
    ``rocm:gfx1151``, ``apple_gpu`` …)."""
    a = (arch_or_target or "").lower().split(":", 1)[0]
    if a.startswith(("sm", "nvidia", "cuda", "ptx")):
        return "nvidia"
    if a.startswith(("gfx", "rocm", "hip", "amd")):
        return "rocm"
    if a.startswith(("apple", "metal", "mps")):
        return "apple"
    if a in {"cpu", "x86", "x86_64", "avx512", "reference_cpu"} or a.startswith("x86"):
        return "cpu"
    return "generic"


@dataclass(frozen=True)
class ToolchainIdentity:
    """Which toolchain (and, for a delegate, which library build) produced a
    measurement.

    ``toolchain`` holds the family components; ``delegate`` the delegate's
    library identity when the measured candidate was a Tier-3 delegate, else
    empty. :attr:`digest` covers both, so either changing is a cache miss."""

    family: str
    toolchain: Mapping[str, str] = field(default_factory=dict)
    delegate: Mapping[str, str] = field(default_factory=dict)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "family": self.family,
            "toolchain": {str(k): str(v) for k, v in sorted(self.toolchain.items())},
            "delegate": {str(k): str(v) for k, v in sorted(self.delegate.items())},
        }

    @property
    def digest(self) -> str:
        raw = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable form: the components plus their digest."""
        return {**self._payload(), "digest": self.digest}

    def with_delegate(self, delegate: Mapping[str, str] | None) -> "ToolchainIdentity":
        return ToolchainIdentity(self.family, dict(self.toolchain), dict(delegate or {}))


def _llvm_pin() -> str:
    from .runtime_abi_audit import cmake_toolchain_pins

    return cmake_toolchain_pins().get("llvm_version", UNDECLARED)


def _family_components(family: str) -> dict[str, str]:
    return dict(_family_components_cached(family))


@cache
def _family_components_cached(family: str) -> tuple[tuple[str, str], ...]:
    """Pins do not move under a running process; cache per family so a corpus
    load that checks every row does not re-read the pin file per row."""
    components = {"llvm_mlir": _llvm_pin()}
    if family == "nvidia":
        from . import gpu_target as g

        components.update({
            "cuda_toolkit": g.TESSERA_TARGET_CUDA_TOOLKIT,
            "ptx_isa": g.TESSERA_TARGET_PTX_ISA,
            "cuda_driver_min": g.TESSERA_TARGET_CUDA_DRIVER_MIN,
            "cuda_driver_api": g.TESSERA_TARGET_CUDA_DRIVER_API,
            "driver_jit_ptx_isa": g.TESSERA_TARGET_DRIVER_JIT_PTX_ISA,
        })
    elif family == "rocm":
        from . import rocm_target as r

        components.update({"rocm": r.TESSERA_TARGET_ROCM, "hip": r.TESSERA_TARGET_HIP})
    elif family == "apple":
        components.update(_apple_components())
    return tuple(sorted(components.items()))


def _apple_components() -> dict[str, str]:
    """Shell-independent Apple toolchain facts (see the module docstring)."""
    import platform

    from .apple_route_selector import (
        _XCRUN,
        _command_text,
        _runtime_source_fingerprint,
        live_apple_device_tag,
    )

    def first_line(text: str) -> str:
        return text.splitlines()[0].strip() if text else UNAVAILABLE

    return {
        "macos": platform.mac_ver()[0] or UNAVAILABLE,
        "macos_sdk": _command_text(_XCRUN, "--sdk", "macosx", "--show-sdk-version") or UNAVAILABLE,
        "metal_compiler": first_line(_command_text(_XCRUN, "metal", "--version")),
        "xcode": " ".join(_command_text(_XCRUN, "xcodebuild", "-version").split()) or UNAVAILABLE,
        "apple_runtime_source": _runtime_source_fingerprint(),
        "device": live_apple_device_tag(),
    }


def clear_identity_cache() -> None:
    """Forget cached family components (tests that patch a pin call this)."""
    _family_components_cached.cache_clear()


def toolchain_identity(
    arch_or_target: str,
    *,
    delegate: Mapping[str, str] | None = None,
) -> ToolchainIdentity:
    """The (pin-based) toolchain identity a measurement on ``arch_or_target``
    carries. ``delegate`` — :func:`delegate_library_identity` of the delegate
    library measured, or the ``kernel_code_identity`` of a compiler-generated
    kernel, when the cache keys one measurement per artifact (``autotune_v2``).
    Not :func:`tessera_opt_identity`: a compiler binary's digest differs on
    every build and is only a cache key now. The arbiter corpus keeps
    per-candidate identities separately (``evidence.delegate_identities``)."""
    family = target_family(arch_or_target)
    return ToolchainIdentity(family, _family_components(family), dict(delegate or {}))


_FILE_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    cached = _FILE_DIGESTS.get(key)
    if cached is None:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        cached = "sha256:" + h.hexdigest()
        _FILE_DIGESTS[key] = cached
    return cached


_IDENTITIES: dict[tuple[str, int, int, str | None], dict[str, str]] = {}


def delegate_library_identity(
    library: str | os.PathLike[str], *, cmake_target: str | None = None,
) -> dict[str, str]:
    """ABI identity of a loaded delegate library.

    The content digest is the ABI hash: any rebuild, version bump or different
    vendor drop changes it, so a measurement of the old library misses. When
    ``cmake_target`` is given and the library lives in a configured build tree,
    the ``runtime_library_build`` record (RUNTIME-LIB-OPT-1) adds the
    optimization level it was built at; a library outside a build tree records
    ``build_record=unrecorded`` rather than failing, because its digest still
    identifies it — only its optimization level is unknown.
    """
    path = Path(library).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"delegate library {path} does not exist")
    # Per-process cache keyed on the file's (mtime, size): the arbiter asks on
    # every cache hit, and walking the build tree for the record each time is
    # the cost a hit exists to avoid. A rebuilt library changes mtime/size and
    # is re-identified.
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size, cmake_target)
    cached = _IDENTITIES.get(key)
    if cached is not None:
        return dict(cached)
    identity = {"library": path.name, "abi_digest": _file_digest(path)}
    if cmake_target is not None:
        from .runtime_library_build import RuntimeLibraryBuildError, record_for_library

        identity["cmake_target"] = cmake_target
        try:
            record = record_for_library(path, cmake_target)
        except RuntimeLibraryBuildError:
            identity["build_record"] = "unrecorded"
        else:
            identity["optimization"] = str(record["level"])
            identity["cmake_build_type"] = str(record.get("cmake_build_type", ""))
    _IDENTITIES[key] = dict(identity)
    return identity


def loaded_library_identity(library: Any, *, cmake_target: str | None = None,
                            entry: str | None = None) -> dict[str, str] | None:
    """:func:`delegate_library_identity` of a loaded ``ctypes.CDLL`` (its
    ``_name`` is the path it was loaded from), plus the bound ``entry`` symbol.
    ``None`` when there is no library -- a candidate that cannot load its
    library is not available, so it is never timed."""
    path = getattr(library, "_name", None) if library is not None else None
    if not path:
        return None
    identity = delegate_library_identity(path, cmake_target=cmake_target)
    if entry:
        identity["entry"] = entry
    return identity


def tessera_opt_identity() -> dict[str, str] | None:
    """Identity of the ``tessera-opt`` binary (its content digest). ``None``
    when no ``tessera-opt`` is found.

    No longer a verdict key (2026-09-26): the binary's bytes differ between any
    two builds, so keying on it served a committed row only in the tree that
    recorded it. It is now a *cache* key: ``kernel_code_identity`` folds it into
    its per-process identity cache so a rebuilt compiler is re-identified
    rather than served a cached digest."""
    from tessera import runtime as rt

    path = rt._tessera_opt_path()
    if path is None:
        return None
    return {"generator": "tessera-opt", **delegate_library_identity(path)}


__all__ = [
    "FAMILIES",
    "SCHEMA",
    "UNDECLARED",
    "ToolchainIdentity",
    "UNAVAILABLE",
    "clear_identity_cache",
    "delegate_library_identity",
    "loaded_library_identity",
    "target_family",
    "tessera_opt_identity",
    "toolchain_identity",
]
