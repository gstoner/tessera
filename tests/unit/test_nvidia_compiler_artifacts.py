"""Host-only compiler-artifact proofs for the NVIDIA Tile lowering path."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests._support.environment import CompilerToolchain


pytestmark = [pytest.mark.compiler_tool, pytest.mark.compiler_nvidia]


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia"
           / "sm120_pointer_fragment_store.mlir")


def test_sm120_tile_fragment_lowers_to_real_nvvm_mma(
    compiler_toolchain: CompilerToolchain,
) -> None:
    """Prove the compiler artifact without assembling or launching on CUDA."""

    nvidia_opt = compiler_toolchain.require_nvidia_opt()
    result = subprocess.run(
        [
            str(nvidia_opt),
            "--lower-tile-to-nvidia=sm=120",
            "--lower-tessera-nvidia-to-nvvm",
            str(FIXTURE),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "llvm.func @pointer_fragment_store" in result.stdout
    assert "nvvm.mma.sync" in result.stdout
    assert "tile.fragment" not in result.stdout
