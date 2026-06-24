"""Fork-A: the compiler-generated flash_attn FORWARD routes through the wave/LDS
pipeline and lowers identically to the direct generator (GPU-free IR parity).

The flash_attn analog of test_rocm_gemm_pipeline_routing.py — but FA exercises
the pipeline on a realistic kernel: an scf.for KV-loop, gpu.barrier
synchronization, and LDS-staged Q/scores. This proves the FunctionOpInterface
walk + the real-accumulator tile.mma lowering generalize from GEMM to FA: the
matmul ops (QK^T, P@V) route through tile.mma and lower back to an identical
ROCDL op multiset.

NOTE (scope): only the MATMUL ops route through here (the K/V staging is still
direct global loads — tessera_rocm.async_copy lowering is an artifact-only
contract marker today, so a pipeline-routed double-buffer is not yet runnable).
This test guards the matmul-routing parity; on-device correctness of the routed
kernel is the same kernel as the direct lane (identical op multiset).
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"

_DIRECTIVE = (
    'module {\n  "tessera_rocm.flash_attn"() {name = "fa", head_dim = 64 : i64, '
    'dtype = "f16"} : () -> ()\n}\n'
)

_DIRECT = ["--generate-wmma-flash-attn-kernel", "--lower-tessera-target-to-rocdl"]
_FORK_A = ["--generate-wmma-flash-attn-kernel=via-tile=true",
           "--rocm-wave-lds-pipeline", "--rocm-wave-lds-legality",
           "--lower-tile-to-rocm=arch=gfx1151",
           "--lower-tessera-target-to-rocdl"]


def _opt(*passes: str) -> str:
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    r = subprocess.run([str(TESSERA_OPT), "-", *passes],
                       input=_DIRECTIVE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_via_tile_emits_tile_mma_for_fa_matmuls():
    ir = _opt("--generate-wmma-flash-attn-kernel=via-tile=true")
    assert "tile.mma" in ir
    assert "tessera_rocm.wmma" not in ir
    assert "gpu.func @fa" in ir
    assert "scf.for" in ir            # the KV loop is still there
    direct = _opt("--generate-wmma-flash-attn-kernel")
    assert "tessera_rocm.wmma" in direct and "tile.mma" not in direct


def _op_multiset(ir: str) -> Counter:
    return Counter(re.findall(r"\b[a-z_]+\.[a-z_.]+\b", ir))


def test_fa_fork_a_final_rocdl_is_identical_to_direct():
    """The pipeline-routed FA forward lowers to the SAME ROCDL op multiset as
    the direct generator — same kernel, perf parity by construction."""
    direct = _op_multiset(_opt(*_DIRECT))
    fork_a = _op_multiset(_opt(*_FORK_A))
    assert direct == fork_a, (
        "FA Fork-A lowered to a different kernel than direct; "
        f"delta={ (direct - fork_a) | (fork_a - direct) }")
