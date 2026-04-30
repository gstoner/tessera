"""Compatibility shim for the shared benchmark compiler contract."""

from benchmarks.common.compiler_contract import (  # noqa: F401
    CompilerRun,
    compiler_conv2d_ir,
    compiler_flash_attention_ir,
    compiler_matmul_relu,
    compiler_spectral_ir,
)
