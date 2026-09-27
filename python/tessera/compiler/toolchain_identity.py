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
* Apple — there is no declared Apple pin, so the family identity is the live
  :func:`apple_route_selector.live_apple_route_context` (macOS version, SDK
  version, host compiler fingerprint): the same identity the strict Apple route
  ledger already requires before retained evidence may select.
* A compiled artifact — a native image's ``compiler_fingerprint`` /
  ``toolchain_fingerprint`` (the tessera-opt build and toolkit it was lowered
  with), passed in by the caller that holds the image.
* A delegate — :func:`delegate_library_identity`: the content digest of the
  loaded shared library (its ABI hash) plus, when the library sits in a
  configured build tree, its ``runtime_library_build`` optimization record.

**Declared pins are declared.** The NVIDIA/ROCm/LLVM components are the pins
the fleet is held to (``runtime_abi_audit`` drift-gates them, and the LLVM pin
is exact), so a toolkit upgrade that is *adopted* — the pin moves — invalidates
every entry measured under the old pin. A box that drifts off its pin without
the pin moving is not caught by the family identity; the per-artifact
fingerprints are what catch that, which is why callers holding a native image
should pass it.
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
        from .apple_route_selector import live_apple_route_context

        ctx = live_apple_route_context()
        components.update({
            "macos": ctx.os_version,
            "macos_sdk": ctx.sdk_version,
            "host_compiler": ctx.compiler_fingerprint,
        })
    return tuple(sorted(components.items()))


def clear_identity_cache() -> None:
    """Forget cached family components (tests that patch a pin call this)."""
    _family_components_cached.cache_clear()


def toolchain_identity(
    arch_or_target: str,
    *,
    native_image: Any = None,
    delegate: Mapping[str, str] | None = None,
) -> ToolchainIdentity:
    """The toolchain identity a measurement on ``arch_or_target`` carries.

    ``native_image`` — a :class:`native_artifact.NativeImageArtifact` (or any
    mapping/object with ``compiler_fingerprint`` / ``toolchain_fingerprint``)
    whose code was measured; its fingerprints join the identity so a rebuilt
    tessera-opt or a drifted toolkit misses even when the pins did not move.
    ``delegate`` — :func:`delegate_library_identity` of a Tier-3 delegate.
    """
    family = target_family(arch_or_target)
    components = _family_components(family)
    if native_image is not None:
        for name in ("compiler_fingerprint", "toolchain_fingerprint"):
            value = (native_image.get(name) if isinstance(native_image, Mapping)
                     else getattr(native_image, name, None))
            if not value:
                raise ValueError(
                    f"native image carries no {name}; a measurement of it cannot "
                    "be keyed to the toolchain that built it (Decision #11)")
            components[f"image_{name}"] = str(value)
    return ToolchainIdentity(family, components, dict(delegate or {}))


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
    return identity


__all__ = [
    "FAMILIES",
    "SCHEMA",
    "UNDECLARED",
    "ToolchainIdentity",
    "clear_identity_cache",
    "delegate_library_identity",
    "target_family",
    "toolchain_identity",
]
