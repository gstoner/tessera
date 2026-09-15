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
