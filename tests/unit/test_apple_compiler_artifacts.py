"""Host-free compiler proof owned by the Apple backend (APPLE-CI-2)."""
from __future__ import annotations

import subprocess

import pytest

from tests._support.environment import CompilerToolchain


pytestmark = [pytest.mark.compiler_tool, pytest.mark.compiler_apple]


def test_apple_gpu_pipeline_is_registered_in_declared_compiler_build(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """Apple lowering is a compiler capability, independent of Metal hardware."""

    tessera_opt = compiler_toolchain.require_tessera_opt()
    source = """module {
      func.func @f(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
        %0 = tessera.softmax %x : (tensor<4x4xf32>) -> tensor<4x4xf32>
        return %0 : tensor<4x4xf32>
      }
    }
    """
    result = subprocess.run(
        [str(tessera_opt), "-tessera-lower-to-apple_gpu", "-"],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "tessera_apple.gpu.metal_kernel" in result.stdout
    assert 'framework = "MPSGraph"' in result.stdout
