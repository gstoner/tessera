"""Content-addressed Schedule→Tile gates for the PR #568 physical lanes."""

from __future__ import annotations

import pytest

from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.scheduled_reference_tier import lower_scheduled_reference_tier


def _tridiagonal_module(n: int = 8, batch: int = 17) -> GraphIRModule:
    diagonal = IRType(f"tensor<{n}xf32>", (str(n),), "fp32")
    rhs = IRType(f"tensor<{batch}x{n}xf32>", (str(batch), str(n)), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="scheduled_tridiagonal",
                args=[
                    IRArg("dl", diagonal),
                    IRArg("diagonal", diagonal),
                    IRArg("du", diagonal),
                    IRArg("rhs", rhs),
                ],
                result_types=[rhs],
                body=[
                    IROp(
                        result="result",
                        op_name="tessera.tridiagonal_solve",
                        operands=["%dl", "%diagonal", "%du", "%rhs"],
                        operand_types=[str(diagonal), str(diagonal), str(diagonal), str(rhs)],
                        result_type=str(rhs),
                    )
                ],
                return_values=["%result"],
            )
        ]
    )


def _coalition_module(op_name: str = "tessera.game_subset_zeta", *, size: int = 32) -> GraphIRModule:
    values = IRType(f"tensor<3x{size}xf32>", ("3", str(size)), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="scheduled_coalition",
                args=[IRArg("values", values)],
                result_types=[values],
                body=[
                    IROp(
                        result="result",
                        op_name=op_name,
                        operands=["%values"],
                        operand_types=[str(values)],
                        result_type=str(values),
                    )
                ],
                return_values=["%result"],
            )
        ]
    )


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
@pytest.mark.parametrize(
    ("target", "algorithm", "workgroup"),
    [
        ("x86", "batch_vector_thomas_v1", 1),
        ("rocm_gfx1151", "cooperative_lds_pcr_v1", 8),
    ],
)
def test_tridiagonal_lowers_to_one_exact_artifact(target: str, algorithm: str, workgroup: int) -> None:
    artifact = lower_scheduled_reference_tier(_tridiagonal_module(), target=target)
    assert artifact.algorithm == algorithm
    assert artifact.schedule_ir.count("schedule.tridiagonal_solve") == 1
    assert artifact.tile_ir.count("tile.tridiagonal_solve_kernel") == 1
    assert 'accum = "f64"' in artifact.tile_ir
    assert f"workgroup_size = {workgroup}" in artifact.tile_ir
    assert "tessera.tridiagonal_solve" not in artifact.tile_ir


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
@pytest.mark.parametrize(
    ("op_name", "half", "sign"),
    [
        ("tessera.game_subset_zeta", 1, 1),
        ("tessera.game_subset_mobius", 1, -1),
        ("tessera.game_superset_zeta", 0, 1),
        ("tessera.game_superset_mobius", 0, -1),
    ],
)
def test_four_transforms_share_one_butterfly_contract(op_name: str, half: int, sign: int) -> None:
    artifact = lower_scheduled_reference_tier(_coalition_module(op_name), target="x86")
    assert artifact.family == "coalition_butterfly"
    assert artifact.tile_ir.count("tile.coalition_butterfly_kernel") == 1
    assert 'stage_order = "ascending_bit_yates_v1"' in artifact.tile_ir
    assert 'accum = "f64"' in artifact.tile_ir
    assert f"half = {half}" in artifact.tile_ir
    assert f"sign = {sign}" in artifact.tile_ir
    assert op_name not in artifact.tile_ir


def test_unowned_rocm_architectures_fail_closed() -> None:
    for target in ("rocm_gfx1200", "rocm_gfx1250"):
        with pytest.raises(ValueError, match="support x86 and gfx1151"):
            lower_scheduled_reference_tier(_tridiagonal_module(), target=target)


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_graph_verifier_rejects_non_power_of_two_coalition_axis() -> None:
    with pytest.raises(RuntimeError, match="power-of-two coalition trailing extent"):
        lower_scheduled_reference_tier(_coalition_module(size=24), target="x86")


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_gfx1151_coalition_rejects_an_lds_impossible_lattice() -> None:
    with pytest.raises(RuntimeError, match="REF-TIER-PHYS-1"):
        lower_scheduled_reference_tier(
            _coalition_module(size=16_384), target="rocm_gfx1151"
        )
