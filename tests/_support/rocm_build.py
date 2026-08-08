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
