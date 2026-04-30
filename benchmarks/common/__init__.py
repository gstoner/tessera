"""Shared benchmark/compiler contract helpers."""

from .artifact_schema import (
    ArtifactLevels,
    BenchmarkOperator,
    BenchmarkRow,
    CompilerPath,
    Correctness,
    Profile,
    RuntimeStatus,
)
from .compiler_contract import (
    CompilerRun,
    compiler_conv2d_ir,
    compiler_flash_attention_ir,
    compiler_matmul_relu,
    compiler_spectral_ir,
)
from .correctness import correctness_report, max_abs_error, relative_error, within_tolerance

__all__ = [
    "ArtifactLevels",
    "BenchmarkOperator",
    "BenchmarkRow",
    "CompilerPath",
    "CompilerRun",
    "Correctness",
    "Profile",
    "RuntimeStatus",
    "compiler_conv2d_ir",
    "compiler_flash_attention_ir",
    "compiler_matmul_relu",
    "compiler_spectral_ir",
    "correctness_report",
    "max_abs_error",
    "relative_error",
    "within_tolerance",
]
