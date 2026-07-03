"""Phase-0 golden-IR snapshot (COMPILER_REFACTOR_PLAN §9.1 / E1).

Committed Target-IR snapshots for a fixed source × target matrix, rendered by the
**Python** emitter.  The test regenerates and byte-compares; any change to the
emitted IR fails the gate.  This is the host-free lead-backend regression
tripwire for the Python emitter — a refactor that unintentionally perturbs the
Apple / ROCm / NVIDIA / x86 Target IR trips here, on the Mac, with no GPU or
backend-specific ``tessera-opt`` build required (the C++ ``tessera-opt`` side is
covered separately by the FileCheck lit fixtures under ``tests/tessera-ir/``).

Determinism is guaranteed by ``test_target_ir_determinism.py`` (byte-stable
across independent lowerings + canonical attr key order), so a byte-exact golden
compare is safe.

Regenerate after an *intentional* IR change:

    TESSERA_UPDATE_GOLDEN=1 python3 -m pytest tests/unit/test_target_ir_golden.py -q

then review the ``tests/unit/golden/target_ir/*.mlir`` diff before committing.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tessera.compiler.frontend import lower_text_to_graph_ir
from tessera.compiler.schedule_ir import lower_graph_to_schedule_ir
from tessera.compiler.target_ir import lower_tile_to_target_ir
from tessera.compiler.tile_ir import lower_schedule_to_tile_ir

_GOLDEN_DIR = Path(__file__).parent / "golden" / "target_ir"

_MATMUL_SOFTMAX = """
module demo {
  func main(A: tensor<2x3xfp32>, B: tensor<3x2xfp32>) -> tensor<2x2xfp32> {
    C = op.matmul(A, B);
    P = op.softmax(C);
    return P;
  }
}
"""

_FLASH_ATTN = """
module demo {
  func flash(Q: tensor<2x4xfp32>, K: tensor<2x4xfp32>, V: tensor<2x4xfp32>) -> tensor<2x4xfp32> {
    O = op.flash_attn(Q, K, V);
    return O;
  }
}
"""

# (golden name, target_kind, source) — matmul→softmax across the backends +
# x86/CPU, plus flash-attn on two GPU lead lanes.
#
# NOTE — nvidia_sm120 is intentionally EXCLUDED: the Python Target-IR lowering
# (target_ir.py::_lower_nvidia_op) currently lumps sm_120 with sm_100 and emits
# `tcgen05_mma` / `tmem_alloc`, but consumer Blackwell sm_120 has NO tcgen05/TMEM
# — its path is warp-level `mma.sync` (the same sm_120≠sm_100-superset bug fixed
# in the capability queries). Snapshotting that output would bless the bug and
# make the eventual mma.sync fix look like a regression. Re-add sm_120 to this
# matrix once the emitter grows a correct mma.sync path (tracked follow-up).
_FIXTURES = [
    ("matmul_softmax", "cpu", _MATMUL_SOFTMAX),
    ("matmul_softmax", "apple_cpu", _MATMUL_SOFTMAX),
    ("matmul_softmax", "apple_gpu", _MATMUL_SOFTMAX),
    ("matmul_softmax", "rocm", _MATMUL_SOFTMAX),
    ("matmul_softmax", "nvidia_sm90", _MATMUL_SOFTMAX),
    ("flash_attn", "apple_gpu", _FLASH_ATTN),
    ("flash_attn", "nvidia_sm90", _FLASH_ATTN),
]

_IDS = [f"{name}.{target}" for name, target, _ in _FIXTURES]


def _emit(target_kind: str, source: str) -> str:
    graph = lower_text_to_graph_ir(source)
    schedule = lower_graph_to_schedule_ir(graph, target_kind=target_kind)
    tile = lower_schedule_to_tile_ir(schedule, target_kind=target_kind)
    return lower_tile_to_target_ir(tile, target_kind=target_kind).to_mlir()


def _golden_path(name: str, target_kind: str) -> Path:
    return _GOLDEN_DIR / f"{name}.{target_kind}.mlir"


_REGEN = (
    "TESSERA_UPDATE_GOLDEN=1 python3 -m pytest tests/unit/test_target_ir_golden.py"
)


@pytest.mark.parametrize("name,target_kind,source", _FIXTURES, ids=_IDS)
def test_target_ir_matches_golden(name: str, target_kind: str, source: str) -> None:
    emitted = _emit(target_kind, source)
    path = _golden_path(name, target_kind)

    if os.environ.get("TESSERA_UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(emitted)
        pytest.skip(f"updated golden {path.name}")

    assert path.exists(), (
        f"missing golden {path}; regenerate with `{_REGEN}`"
    )
    assert emitted == path.read_text(), (
        f"Target IR for {name}/{target_kind} drifted from committed golden "
        f"{path.name}. If intentional, regenerate with `{_REGEN}` and review "
        f"the diff before committing."
    )
