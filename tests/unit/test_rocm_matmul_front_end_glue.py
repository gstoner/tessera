"""The Graph -> Tile -> Target-IR lowering of a rocm matmul PRODUCES the
executable `tessera_rocm.wmma_gemm` directive (Decision #19, Stage L glue).

Before this, the compiled lane was reached only because the runtime *synthesized*
the directive from op metadata at launch. Now the IR stack itself emits it: a
`@jit(target="rocm")` matmul lowers (Graph -> Schedule -> Tile -> Target IR) to a
Target IR that contains `tessera_rocm.wmma_gemm{m=n=k=16, dtype}`, the same
directive the `generate-wmma-gemm-kernel` pass expands into a real WMMA kernel.

This test is GPU-free (pure IR): it proves the directive is produced with the
right attributes AND is consumable by the generate pass (emits the gpu.func +
the WMMA op). Execution of that directive is covered by the compiled-lane tests
(`test_rocm_compiled_launch_execute.py` / `test_rocm_wmma_gemm_general.py`),
which exercise the identical op.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import tessera

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"


@tessera.jit(target="rocm")
def _rocm_mm(a, b):
    return tessera.ops.matmul(a, b)


def test_rocm_matmul_target_ir_emits_wmma_gemm_directive():
    """The IR stack lowers tessera.matmul -> ... -> the executable WMMA directive
    (not just the abstract tessera_rocm.mfma marker)."""
    tir = _rocm_mm.target_ir
    assert "tessera_rocm.wmma_gemm" in tir, "matmul Target IR lacks the directive"
    # The abstract matrix-core marker is still emitted (hardware-free contract).
    assert "tessera_rocm.mfma" in tir
    line = next(l for l in tir.splitlines() if '"tessera_rocm.wmma_gemm"' in l)
    # WMMA instruction tile is 16x16x16; the problem size is a runtime kernel arg.
    assert "m = 16 : i64" in line and "n = 16 : i64" in line and \
        "k = 16 : i64" in line
    assert 'name = "gemm"' in line and 'dtype = "f16"' in line


def test_ir_stack_directive_is_consumable_by_generate_pass():
    """The directive the IR stack produced expands through the SAME
    generate-wmma-gemm-kernel pass into a fragment-materialized WMMA kernel —
    closing Graph matmul -> Target-IR directive -> generated kernel."""
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    tir = _rocm_mm.target_ir
    line = next(l.strip() for l in tir.splitlines()
                if '"tessera_rocm.wmma_gemm"' in l)
    module = "module {\n  " + line + "\n}\n"
    r = subprocess.run([str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel"],
                       input=module, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "gpu.func @gemm" in r.stdout       # the kernel the pass generated
    assert "tessera_rocm.wmma" in r.stdout     # the matrix op inside it
    assert '"tessera_rocm.wmma_gemm"' not in r.stdout  # directive consumed
