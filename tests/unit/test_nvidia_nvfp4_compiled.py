"""On-silicon proof for the sm_120a NVFP4 block-scale operand ABI."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "docs/audit/backend/nvidia/spikes/sm120_mma_sync"
          / "nvfp4_gemm.cu")


def _require_sm120a() -> tuple[str, str]:
    nvcc = shutil.which("nvcc")
    cuobjdump = shutil.which("cuobjdump")
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvcc or not cuobjdump or not nvidia_smi:
        pytest.skip("NVFP4 proof requires nvcc, cuobjdump, and nvidia-smi")
    capability = subprocess.run(
        [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
        check=True, capture_output=True, text=True).stdout.splitlines()[0].strip()
    if capability != "12.0":
        pytest.skip(f"NVFP4 proof requires sm_120, found compute capability {capability}")
    return nvcc, cuobjdump


def test_sm120a_nvfp4_nonuniform_scales_match_and_emit_omma(tmp_path: Path) -> None:
    nvcc, cuobjdump = _require_sm120a()
    binary = tmp_path / "nvfp4_gemm"
    cubin = tmp_path / "nvfp4_gemm.cubin"
    flags = ["-gencode", "arch=compute_120a,code=sm_120a", "-O2"]
    subprocess.run([nvcc, *flags, str(SOURCE), "-o", str(binary)], check=True)
    for mode in ("30", "38", "40", "mapped"):
        result = subprocess.run(
            [str(binary), mode], check=True, capture_output=True, text=True,
            timeout=15)
        assert "max abs error vs fp4 ref = 0" in result.stdout
        assert "RESULT: PASS" in result.stdout

    subprocess.run(
        [nvcc, *flags, "-cubin", str(SOURCE), "-o", str(cubin)], check=True)
    sass = subprocess.run(
        [cuobjdump, "--dump-sass", str(cubin)], check=True,
        capture_output=True, text=True).stdout
    assert "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X" in sass
