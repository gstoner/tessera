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
    """ROCm toolkit root whose ``ld.lld`` MLIR's ROCDL ``gpu-module-to-binary``
    serializer links with, or ``None``. Mirrors the runtime's own detector so a
    test and the lane it exercises agree about the host."""
    from tessera.runtime import _rocm_toolkit_root
    return _rocm_toolkit_root()


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
                    "(ROCM_PATH); none on this host")
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
    compiled lane is gfx1151-only by the pipeline's rule; only the five families
    with gfx1201 proof may run there, and they must say which one they are
    through `require_rocm_compiled_family`. Anything else on a non-gfx1151 arch
    is asserting a gfx1151 proof on a host that cannot have one.
    """
    import pytest
    from tessera.compiler.rocm_pipeline import FAMILY_PLUGINS, promoted_families

    arch = rocm_host_arch()
    if arch is None:
        pytest.skip("no ROCm device arch resolved on this host (no pin, no live device)")
    if promoted_families(arch) != frozenset(FAMILY_PLUGINS):
        pytest.skip(f"the generic ROCm compiled lane has proof on gfx1151 only; "
                    f"this host launches on {arch}")
    return arch


_UNPROMOTED_REFUSAL = "ROCm executable pipeline has no promoted family plugins for "


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

    def launch(self, *args, **kwargs):
        import pytest

        arch = rocm_host_arch()
        marker = f"{_UNPROMOTED_REFUSAL}{arch};" if arch else None
        try:
            result = self._rt.launch(*args, **kwargs)
        except ValueError as exc:
            if marker and str(exc).startswith(marker):
                pytest.skip(str(exc))
            raise
        if (marker and isinstance(result, dict) and result.get("ok") is False
                and str(result.get("reason", "")).startswith(marker)):
            pytest.skip(str(result["reason"]))
        return result


def runtime_for_host(runtime_module) -> _RuntimeForHost:
    """Wrap the runtime so unpromoted-family refusals for this host skip."""
    return _RuntimeForHost(runtime_module)
