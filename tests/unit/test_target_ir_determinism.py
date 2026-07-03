"""Phase-0 golden-IR determinism guard (COMPILER_REFACTOR_PLAN §9.1(3) / E1).

The golden-IR regression tripwire requires the Python Target-IR emitter to be
byte-deterministic: two independent lowerings of the same source must render
identical IR text, and attribute keys must emit in a canonical (sorted) order so
a refactor that merely reorders attr construction cannot spuriously trip the
gate.  (The C++ ``tessera-opt`` side is already deterministic via the MLIR
printer + the FileCheck lit fixtures; this covers the Python emitter, which was
the flagged gap.)
"""
from __future__ import annotations

import pytest

from tessera.compiler.frontend import lower_text_to_graph_ir
from tessera.compiler.schedule_ir import lower_graph_to_schedule_ir
from tessera.compiler.target_ir import _format_attr_dict, lower_tile_to_target_ir
from tessera.compiler.tile_ir import lower_schedule_to_tile_ir

_MATMUL_SOFTMAX = """
module demo {
  func main(A: tensor<2x3xfp32>, B: tensor<3x2xfp32>) -> tensor<2x2xfp32> {
    C = op.matmul(A, B);
    P = op.softmax(C);
    return P;
  }
}
"""

# One source lowered across all four backends + CPU — the golden-IR tripwire's
# coverage surface (Apple CPU/GPU, ROCm, NVIDIA Hopper + consumer Blackwell, x86).
_TARGETS = ("cpu", "apple_cpu", "apple_gpu", "rocm", "nvidia_sm90", "nvidia_sm120")


def _emit(target_kind: str, source: str = _MATMUL_SOFTMAX) -> str:
    """Lower ``source`` through the full spine to Target IR text (a fresh run)."""
    graph = lower_text_to_graph_ir(source)
    schedule = lower_graph_to_schedule_ir(graph, target_kind=target_kind)
    tile = lower_schedule_to_tile_ir(schedule, target_kind=target_kind)
    target = lower_tile_to_target_ir(tile, target_kind=target_kind)
    return target.to_mlir()


@pytest.mark.parametrize("target_kind", _TARGETS)
def test_target_ir_emission_is_byte_deterministic(target_kind: str) -> None:
    """Two *independent* lowerings of the same source render identical IR text.

    Independent runs (not ``to_mlir()`` twice on one object) catch nondeterminism
    in the lowering passes themselves — set/dict iteration, hash-ordered naming.
    """
    first = _emit(target_kind)
    second = _emit(target_kind)
    assert first == second, (
        f"non-deterministic Target IR for {target_kind!r} — the golden-IR gate "
        f"would false-positive.  First diverging char at index "
        f"{next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), len(first))}."
    )


def test_format_attr_dict_emits_keys_in_sorted_order() -> None:
    """Attribute keys render in canonical (sorted) order regardless of insertion
    order — construction-path-independent output (the sort prerequisite)."""
    rendered = _format_attr_dict({"z_last": 1, "a_first": 2, "m_middle": 3})
    assert rendered.index("a_first") < rendered.index("m_middle") < rendered.index("z_last")
    # A single-key dict and the empty dict are trivially canonical.
    assert _format_attr_dict({}) == "{}"
    assert _format_attr_dict({"only": 1}).startswith("{only = ")
