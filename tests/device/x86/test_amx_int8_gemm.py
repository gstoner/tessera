"""Exact-device correctness for the Intel AMX INT8 kernel."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = [pytest.mark.hardware_amx, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[3]
X86_BACKEND = ROOT / "src/compiler/codegen/tessera_x86_backend"


def _build_amx_int8_library(output: Path) -> None:
    sources = (
        X86_BACKEND / "src/kernels/amx_gemm_int8.cpp",
        X86_BACKEND / "src/runtime/amx_runtime.cpp",
    )
    command = [
        os.environ.get("CXX", "c++"),
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-mamx-int8",
        "-mamx-tile",
        "-mamx-bf16",
        f"-I{X86_BACKEND / 'include'}",
        *(str(source) for source in sources),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, (
        "AMX exact-device proof requires a working AMX C++ toolchain:\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_amx_int8_gemm_does_not_truncate_large_k(tmp_path: Path) -> None:
    """An unaligned K>64 GEMM must equal the complete int32 contraction.

    The native call runs in a child process. An illegal instruction or runtime
    fault therefore fails this node without terminating the proof runner.
    """
    library = tmp_path / "libamx_int8.so"
    _build_amx_int8_library(library)
    program = r"""
import ctypes
import sys

import numpy as np

lib = ctypes.CDLL(sys.argv[1])
supported = lib.tessera_x86_amx_int8_supported
supported.argtypes = []
supported.restype = ctypes.c_bool
if not supported():
    raise SystemExit("exact-device lane requires CPUID AMX-TILE and AMX-INT8")

m, n, k = 17, 17, 128
rng = np.random.default_rng(7)
a = rng.integers(-8, 8, size=(m, k), dtype=np.int8)
b = rng.integers(-8, 8, size=(k, n), dtype=np.int8)
c = np.zeros((m, n), dtype=np.int32)

gemm = lib.tessera_x86_amx_gemm_s8s8_s32
gemm.restype = None
gemm.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
gemm(
    a.ctypes.data_as(ctypes.c_void_p),
    b.ctypes.data_as(ctypes.c_void_p),
    c.ctypes.data_as(ctypes.c_void_p),
    m,
    n,
    k,
    0,
)
np.testing.assert_array_equal(c, a.astype(np.int32) @ b.astype(np.int32))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(library)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode < 0:
        failure = f"native AMX child terminated by signal {-completed.returncode}"
    else:
        failure = f"native AMX child exited with {completed.returncode}"
    assert completed.returncode == 0, f"{failure}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
