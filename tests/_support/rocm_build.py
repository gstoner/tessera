"""Canonical paths for artifacts produced by a ROCm-enabled Tessera build."""

from __future__ import annotations

from pathlib import Path

from tests._support.build_artifacts import build_roots, built_artifact


_DEFAULTS = ("build-rocm", "build-rocm-7.14-llvm23-clean", "build")

ROCM_OPT_REL = Path(
    "src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt"
)
ROCM_LIT_SITE_REL = Path(
    "src/compiler/codegen/Tessera_ROCM_Backend/test/lit.site.cfg.py"
)
ROCM_GEMM_LIB_REL = Path(
    "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/"
    "libtessera_rocm_gemm.so"
)
ROCM_FLASH_ATTN_LIB_REL = Path(
    "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/"
    "libtessera_rocm_flash_attn.so"
)
TESSERA_RUNTIME_LIB_REL = Path("src/runtime/libtessera_runtime.a")


def rocm_build_roots() -> tuple[Path, ...]:
    return build_roots(defaults=_DEFAULTS)


def rocm_opt_path() -> Path | None:
    return built_artifact(ROCM_OPT_REL, env=("TESSERA_ROCM_OPT",), defaults=_DEFAULTS)


def rocm_lit_site_path() -> Path | None:
    return built_artifact(ROCM_LIT_SITE_REL, defaults=_DEFAULTS)


def rocm_gemm_lib_path() -> Path | None:
    return built_artifact(
        ROCM_GEMM_LIB_REL, env=("TESSERA_ROCM_GEMM_LIB",), defaults=_DEFAULTS
    )


def rocm_flash_attn_lib_path() -> Path | None:
    return built_artifact(
        ROCM_FLASH_ATTN_LIB_REL,
        env=("TESSERA_ROCM_FLASH_ATTN_LIB",),
        defaults=_DEFAULTS,
    )


def tessera_runtime_lib_path() -> Path | None:
    return built_artifact(TESSERA_RUNTIME_LIB_REL, defaults=_DEFAULTS)


def rocm_hsaco_toolkit_root():
    """ROCm toolkit root that can actually serialize HSACO, or ``None``.

    Starts from the runtime's own detector so a test and the lane it exercises
    agree about the host, then applies the check the detector does not: the
    ROCDL target links the **device bitcode** (``<root>/amdgcn/bitcode``, ocml
    and friends) as well as calling ``ld.lld``. The runtime detector accepts any
    root holding an ``ld.lld``, which a plain LLVM install has -- so on a CUDA
    box with ``/usr/lib/llvm-23/bin`` on ``PATH`` it returns an LLVM prefix and
    every gate built on it admits ROCm packaging tests that cannot pass. That is
    how a host with no ROCm at all produced seven red rows instead of seven
    skips (measured on The-Super-Bear 2026-09-18, identical on clean ``main``).
    Requiring the bitcode makes the gate answer the question it is asked.
    """
    from tessera.runtime import _rocm_toolkit_root
    root = _rocm_toolkit_root()
    if root is None:
        return None
    if not (Path(root) / "amdgcn" / "bitcode" / "ocml.bc").is_file():
        return None
    return root


def require_rocm_hsaco_toolkit():
    """Skip when no ROCm toolkit can serialize HSACO on this host.

    ``rocdl-attach-target`` + ``gpu-module-to-binary`` shell out to ``ld.lld``
    under ``ROCM_PATH``; without a toolkit the pipeline fails with ``lld
    invocation failed`` regardless of the IR, so a package test asking for
    ``backend='rocm'`` is a ROCm-host test. On a ROCm box this returns the root
    and the test runs; on the Mac it skips with the reason, never a false red.
    """
    import pytest
    root = rocm_hsaco_toolkit_root()
    if root is None:
        pytest.skip("ROCDL hsaco serialization needs a ROCm toolkit with ld.lld "
                    "and amdgcn device bitcode (ROCM_PATH); none on this host")
    return root


def rocm_host_arch() -> "str | None":
    """The gfx arch this host will launch on: the ``TESSERA_ROCM_CHIP`` pin if
    set, else the live device, else None. The same resolution the runtime uses
    for its corpus keys (`runtime._rocm_device_name`), so a test asks the
    question the launch path will answer."""
    from tessera import runtime as rt

    try:
        return rt._rocm_device_name()
    except Exception:  # pragma: no cover - a probe failure is "no device"
        return None


def require_rocm_host_arch(arch: str, why: str) -> str:
    """Skip unless this host launches on exactly ``arch``.

    For a test that asserts an arch-specific *record* (a certificate naming
    `rocm_gfx1151`, a consumer named for one chip) rather than a device
    result. The family may be promoted on another arch while the record it
    stamps is still chip-named; that gap is owed by name, not hidden by a
    wider assertion. Unlike the report hook, this is an explicit pin: the
    skip reason names both archs and the owning item.
    """
    import pytest

    host = rocm_host_arch()
    if host is None:
        pytest.skip("no ROCm device on this host")
    if host != arch:
        pytest.skip(f"pinned to {arch}, host is {host}: {why}")
    return host


def require_rocm_compiled_family(*families: str) -> str:
    """Skip unless every named family plugin is promoted on this host's arch.

    The executable pipeline refuses, fail-closed, any family without exact-device
    proof on the launch arch (`rocm_pipeline.promoted_families`). A test that
    launches such a family on such a host is not measuring a defect; it is
    measuring the absence of a proof, and the honest result is a skip that says
    so. Before this existed ~1500 `test_rocm_*_compiled.py` tests failed on the
    gfx1201 box with the pipeline's own refusal text.

    Returns the arch so a caller can key on it.
    """
    import pytest
    from tessera.compiler.rocm_pipeline import promoted_families

    arch = rocm_host_arch()
    if arch is None:
        pytest.skip("no ROCm device arch resolved on this host (no pin, no live device)")
    missing = sorted(f for f in families if f not in promoted_families(arch))
    if missing:
        pytest.skip(f"ROCm executable pipeline has no promoted {', '.join(missing)} "
                    f"family plugin for {arch}; this lane needs a host with that proof")
    return arch


def require_rocm_compiled_lane_host() -> str:
    """Skip unless this host's arch has *every* family plugin promoted.

    For a compiled-family test that does not name its family. The generic
    compiled lane exists on an arch only when the pipeline's rule promotes
    every family there (gfx1151 since commissioning; gfx1201 since 2026-09-18,
    the last family being `paged_kv`). A test that names its family should
    prefer `require_rocm_compiled_family`. Anything else on a partially
    promoted arch is asserting a proof the host cannot have.
    """
    import pytest
    from tessera.compiler.rocm_pipeline import generic_lane_families, promoted_families

    arch = rocm_host_arch()
    if arch is None:
        pytest.skip("no ROCm device arch resolved on this host (no pin, no live device)")
    if not generic_lane_families() <= promoted_families(arch):
        pytest.skip(f"the generic ROCm compiled lane needs every family promoted; "
                    f"{arch} has no promoted family profile for some of them")
    return arch


_UNPROMOTED_REFUSAL = "ROCm executable pipeline has no promoted family plugins for "


def refused_for_host_arch(text: str, arch: "str | None" = None) -> bool:
    """Whether `text` is one of the fail-closed refusals about *this host's* arch.

    1. The family rule: "ROCm executable pipeline has no promoted family plugins
       for <arch>; ...".
    2. An ISA contract gated on another arch's silicon that names the host it is
       refusing: "... hardware-verified on gfx1151; target '<arch>' ... arch-gated".
       The 16x16x16 WMMA fragment layout is the first instance (RDNA4 is
       16x16x32). The spectral composite image used to be the second; since
       2026-09-24 each RDNA chip owns a stamped image, so a missing one is a
       failure, not an arch gate.
    3. That same gfx11 WMMA contract reaching the LLVM backend for a non-gfx11
       host: "Cannot select: intrinsic %llvm.amdgcn.wmma.f32.16x16x16.f16". The
       serializer was handed a gfx11 kernel with a gfx12 target; the honest
       refusal happens one layer up in the runtime, and a test that lowers the
       kernel itself never sees it.

    Every pattern must name, or be conditioned on, the host's own arch, so a
    refusal about an arch a test pinned on purpose stays a failure.
    """
    arch = arch or rocm_host_arch()
    if not arch:
        return False
    if text.startswith(_UNPROMOTED_REFUSAL + arch + ";") or (_UNPROMOTED_REFUSAL + arch + ";") in text:
        return True
    # 1b. The same rule spoken by the C++ pass when a test drives
    #     `tessera-rocm-executable{... arch=<host>}` itself: "tessera-rocm-executable:
    #     architecture '<arch>' has no promoted family-plugin profile".
    if f"architecture '{arch}' has no promoted family-plugin profile" in text:
        return True
    if f"target '{arch}'" in text and ("arch-gated" in text or "hardware-verified on" in text):
        return True
    if ("Cannot select: intrinsic %llvm.amdgcn.wmma.f32.16x16x16" in text
            and not arch.startswith("gfx11")):
        return True
    # 4. A gfx1151-owned artifact or lane refusing on a host that is not gfx1151.
    #    These refusals name the owner but not the host: "exact gfx1151 spectral
    #    reverse package is unavailable", "gfx1151 native HSACO module load
    #    failed", "gfx1151 streaming STFT physical package is unavailable",
    #    "solver IFT package is verified for gfx1151, not gfx1201", "attention
    #    backward requires its exact owning ROCm device". On gfx1151 every one of
    #    them is a real failure and stays one; on any other arch it is the proof
    #    not existing here.
    if arch != "gfx1151":
        if "gfx1151" in text and any(k in text for k in (
                "unavailable", "not loadable", "load failed", "verified for gfx1151")):
            return True
        if "requires its exact owning ROCm device" in text:
            return True
    return False


def refused_by_type_for_host_arch(exc: BaseException, arch: "str | None" = None) -> bool:
    """`_RocmCompiledUnavailable` is the runtime's fail-closed class for its
    compiled lanes, which are gfx11-verified; raised on a non-gfx11 host it is
    "no proof here" whatever its message says ("rocm f32 GEMM lane unavailable —
    no chunked SSD"). On gfx11 it is a failure."""
    arch = arch or rocm_host_arch()
    return bool(arch) and not arch.startswith("gfx11") and type(exc).__name__ == "_RocmCompiledUnavailable"


class _RuntimeForHost:
    """`tessera.runtime` with one change: a launch the executable pipeline
    refuses *for lack of proof on this host's arch* becomes a skip.

    Everything else passes through untouched — a numerical mismatch, a launch
    error, a refusal about some other arch a test pinned on purpose, all still
    fail. Only the exact fail-closed text from `ROCMExecutablePipeline`, naming
    the arch this host resolves to, is a skip: that result says "this proof does
    not exist here", and a test that reads it as "this proof is broken" is the
    defect that put ~1500 red tests on the gfx1201 box. Returned by the
    compiled-family guards (`_rocm_or_skip` and its siblings) so no launch site
    changes.
    """

    def __init__(self, runtime_module):
        self._rt = runtime_module

    def __getattr__(self, name):
        return getattr(self._rt, name)

    def _refused_for_this_host(self, text: str) -> bool:
        return refused_for_host_arch(text)

    def launch(self, *args, **kwargs):
        import pytest

        unavailable = getattr(self._rt, "_RocmCompiledUnavailable", None)
        caught: tuple[type[BaseException], ...] = (ValueError,)
        if isinstance(unavailable, type) and issubclass(unavailable, BaseException):
            caught = caught + (unavailable,)
        try:
            result = self._rt.launch(*args, **kwargs)
        except caught as exc:
            if self._refused_for_this_host(str(exc)):
                pytest.skip(str(exc))
            raise
        if (isinstance(result, dict) and result.get("ok") is False
                and self._refused_for_this_host(str(result.get("reason", "")))):
            pytest.skip(str(result["reason"]))
        return result


def runtime_for_host(runtime_module) -> _RuntimeForHost:
    """Wrap the runtime so unpromoted-family refusals for this host skip."""
    return _RuntimeForHost(runtime_module)
