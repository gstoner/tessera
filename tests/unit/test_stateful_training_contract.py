from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tessera import runtime
from tessera.compiler.stateful_training import (
    build_adafactor_vjp_state_contract,
    build_lion_vjp_state_contract,
    build_sequence_mixer_backward_state_contract,
    lower_scheduled_adafactor_vjp,
    lower_scheduled_lion_vjp,
    lower_scheduled_sequence_mixer_backward,
    validate_sequence_mixer_backward_state_contract,
    validate_lion_vjp_state_contract,
    validate_scheduled_lion_vjp_metadata,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt

# These lower through the production `tessera-opt`, which the CI unit lane does
# not build. The library correctly RAISES rather than silently degrading, so
# the tests must skip there rather than fail.
_needs_opt = pytest.mark.skipif(
    find_tessera_opt() is None, reason="tessera-opt not built"
)


_KWARGS = {"lr": 1.0e-4, "beta2": 0.99, "weight_decay": 0.01}


def _native_lion_metadata(target: str, shape: tuple[int, ...]) -> dict:
    from tessera.compiler.frontend_authority import (
        certify_frontends_non_reexecuting,
    )
    from tessera.compiler.graph_ir import (
        GraphIRFunction,
        GraphIRModule,
        IRArg,
        IROp,
        tensor_ir_type,
    )
    from tessera.compiler.native_lion_vjp import build_native_lion_vjp_package

    tensor_type = tensor_ir_type(shape, "fp32")
    operation = IROp(
        result="new_param,new_moment",
        op_name="tessera.lion",
        operands=["%p", "%g", "%m"],
        operand_types=[str(tensor_type)] * 3,
        result_type=f"({tensor_type}, {tensor_type})",
        inferred_type=tensor_type,
        inferred_types=(tensor_type, tensor_type),
        kwargs=dict(_KWARGS),
    )
    function = GraphIRFunction(
        name="lion",
        args=[IRArg(name, tensor_type) for name in ("p", "g", "m")],
        result_types=[tensor_type, tensor_type],
        body=[operation],
        return_values=["%new_param", "%new_moment"],
    )
    legacy = GraphIRModule(functions=[function])
    tracer = copy.deepcopy(legacy)
    tracer.module_attrs["tessera.frontend.authority"] = '"tracer"'
    certificate = certify_frontends_non_reexecuting(
        legacy_module=legacy,
        tracer_module=tracer,
        graph_consumers=("tessera.lion",),
    )
    values = tuple(np.zeros(shape, np.float32) for _ in range(3))
    cotangents = tuple(np.zeros(shape, np.float32) for _ in range(2))
    package = build_native_lion_vjp_package(
        source_graph_ir=tracer.to_mlir(canonical=True),
        source=SimpleNamespace(op_name="tessera.lion", kwargs=dict(_KWARGS)),
        target=target,
        ordered_inputs=values,
        arg_names=("p", "g", "m"),
        out_cotangents=cotangents,
        frontend_certificate=certificate,
    )
    return package.runtime_metadata()


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_lion_vjp_contract_content_addresses_buffer_lineage(target: str) -> None:
    contract = build_lion_vjp_state_contract(
        target=target, shape=(5, 19), kwargs=_KWARGS
    )

    assert len(contract["artifact_hash"]) == 64
    assert contract["mutation"] == {
        "mode": "functional",
        "alias_policy": "no_input_output_alias",
        "ordered_writes": ["d_p", "d_g", "d_m"],
        "state_transition": "m@0-read-only;d_m@1-fresh",
    }
    assert {item["access"] for item in contract["inputs"]} == {"read_only"}
    assert {item["access"] for item in contract["outputs"]} == {"fresh_write"}
    input_ids = {item["lineage_id"] for item in contract["inputs"]}
    output_ids = {item["lineage_id"] for item in contract["outputs"]}
    assert input_ids.isdisjoint(output_ids)
    assert all(set(item["parents"]) <= input_ids for item in contract["outputs"])


def test_lion_vjp_contract_rejects_lineage_and_numeric_tampering() -> None:
    contract = build_lion_vjp_state_contract(
        target="x86", shape=(7,), kwargs=_KWARGS
    )
    for path, value in (
        (("inputs", 2, "access"), "read_write"),
        (("outputs", 2, "parents"), []),
        (("mutation", "alias_policy"), "may_alias"),
        (("numeric", "beta2"), 0.9),
    ):
        altered = copy.deepcopy(contract)
        cursor = altered
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(ValueError, match="stale or malformed"):
            validate_lion_vjp_state_contract(
                altered, target="x86", shape=(7,), kwargs=_KWARGS
            )


@_needs_opt
@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_lion_vjp_lowers_to_one_exact_launch_tile(target: str) -> None:
    artifact = lower_scheduled_lion_vjp(
        target=target, shape=(5, 19), kwargs=_KWARGS
    )

    artifact.validate()
    assert artifact.schedule_ir.count("schedule.lion_vjp") == 1
    assert artifact.tile_ir.count("tile.training_kernel") == 1
    assert "schedule." not in artifact.tile_ir
    assert "arith.constant 95 : i64" in artifact.tile_ir


@_needs_opt
def test_scheduled_lion_vjp_rejects_tile_policy_tampering() -> None:
    artifact = lower_scheduled_lion_vjp(
        target="rocm_gfx1151", shape=(7,), kwargs=_KWARGS
    )
    metadata = artifact.metadata()
    metadata["tile_ir"] = metadata["tile_ir"].replace(
        'alias_policy = "no_input_output_alias"', 'alias_policy = "may_alias"'
    )

    with pytest.raises(ValueError, match="policy is stale"):
        validate_scheduled_lion_vjp_metadata(
            metadata,
            target="rocm_gfx1151",
            shape=(7,),
            state_contract=artifact.state_contract,
        )


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
@pytest.mark.parametrize(
    ("topology", "shape", "state_count"),
    [("factored", (5, 19), 2), ("full", (19,), 1)],
)
@_needs_opt
def test_adafactor_vjp_contract_and_tile_artifact(
    target: str, topology: str, shape: tuple[int, ...], state_count: int
) -> None:
    kwargs = {"lr": 1e-2, "beta2": 0.9, "eps": 1e-6}
    contract = build_adafactor_vjp_state_contract(
        target=target,
        parameter_shape=shape,
        topology=topology,
        kwargs=kwargs,
    )
    artifact = lower_scheduled_adafactor_vjp(
        target=target,
        parameter_shape=shape,
        topology=topology,
        kwargs=kwargs,
    )

    assert contract == artifact.state_contract
    assert len(contract["inputs"]) == 3 + state_count
    assert len(contract["outputs"]) == 2 + state_count
    assert all(item["access"] == "read_only" for item in contract["inputs"])
    assert all(item["access"] == "fresh_write" for item in contract["outputs"])
    artifact.validate()
    assert artifact.schedule_ir.count("schedule.adafactor_vjp") == 1
    assert artifact.tile_ir.count("tile.training_kernel") == 1
    assert "schedule." not in artifact.tile_ir


@_needs_opt
def test_adafactor_vjp_contract_rejects_alias_tampering() -> None:
    kwargs = {"lr": 1e-2, "beta2": 0.9, "eps": 1e-6}
    artifact = lower_scheduled_adafactor_vjp(
        target="rocm_gfx1151",
        parameter_shape=(5, 19),
        topology="factored",
        kwargs=kwargs,
    )
    altered = copy.deepcopy(artifact.state_contract)
    altered["mutation"]["alias_policy"] = "may_alias"

    from tessera.compiler.stateful_training import (
        validate_adafactor_vjp_state_contract,
    )

    with pytest.raises(ValueError, match="lineage contract is stale"):
        validate_adafactor_vjp_state_contract(
            altered,
            target="rocm_gfx1151",
            parameter_shape=(5, 19),
            topology="factored",
            kwargs=kwargs,
        )


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
@pytest.mark.parametrize(
    "family", [
        "gated_deltanet",
        "kimi_delta_attention",
        "modified_delta_attention",
    ]
)
@_needs_opt
def test_sequence_mixer_backward_contract_lowers_to_one_tile(
    target: str, family: str
) -> None:
    artifact = lower_scheduled_sequence_mixer_backward(
        target=target,
        family=family,
        q_shape=(2, 3, 5, 7),
        v_shape=(2, 3, 5, 11),
        erase=family != "kimi_delta_attention",
        chunk_size=4,
        parallel_chunks=True,
    )

    assert artifact.schedule_ir.count("schedule.sequence_mixer_backward") == 1
    assert artifact.tile_ir.count("tile.training_kernel") == 1
    assert "schedule." not in artifact.tile_ir
    assert artifact.state_contract["mutation"]["workspace_owner"] == "program_launch"
    assert artifact.state_contract["mutation"]["phase_order"] == [
        "checkpoint", "chunk_summary", "chunk_prefix", "chunk_fill", "reverse"
    ]
    assert [item["access"] for item in artifact.state_contract["outputs"]] == [
        "fresh_write"
    ] * 6


def test_sequence_mixer_backward_rejects_workspace_tampering() -> None:
    contract = build_sequence_mixer_backward_state_contract(
        target="rocm_gfx1151",
        family="gated_deltanet",
        q_shape=(1, 1, 5, 3),
        v_shape=(1, 1, 5, 3),
        erase=False,
        chunk_size=2,
        parallel_chunks=True,
    )
    altered = copy.deepcopy(contract)
    altered["mutation"]["workspace_owner"] = "process_global"

    with pytest.raises(ValueError, match="lineage is stale"):
        validate_sequence_mixer_backward_state_contract(
            altered,
            target="rocm_gfx1151",
            family="gated_deltanet",
            q_shape=(1, 1, 5, 3),
            v_shape=(1, 1, 5, 3),
            erase=False,
            chunk_size=2,
            parallel_chunks=True,
        )


@pytest.mark.parametrize(
    ("compiler_path", "target"),
    [
        ("x86_lion_bwd_compiled", "x86"),
        ("rocm_lion_bwd_compiled", "rocm_gfx1151"),
    ],
)
@_needs_opt
def test_lion_runtime_rejects_stale_contract_before_backend_launch(
    compiler_path: str, target: str
) -> None:
    shape = (5, 19)
    runtime_target = "x86" if target == "x86" else "rocm"
    metadata = _native_lion_metadata(runtime_target, shape)
    assert metadata["compiler_path"] == compiler_path
    metadata["state_contract"]["outputs"][0]["version"] = 0
    artifact = runtime.RuntimeArtifact(metadata=metadata)
    values = tuple(np.zeros(shape, dtype=np.float32) for _ in range(5))
    result = runtime.launch(artifact, values)

    assert result["ok"] is False
    assert "state/buffer lineage contract is stale" in result["reason"]


@pytest.mark.parametrize(
    ("compiler_path", "target"),
    [
        ("x86_lion_bwd_compiled", "x86"),
        ("rocm_lion_bwd_compiled", "rocm"),
    ],
)
@_needs_opt
def test_lion_runtime_rejects_missing_lineage(
    compiler_path: str, target: str
) -> None:
    shape = (7,)
    metadata = _native_lion_metadata(target, shape)
    assert metadata["compiler_path"] == compiler_path
    metadata.pop("state_contract")
    artifact = runtime.RuntimeArtifact(metadata=metadata)
    result = runtime.launch(
        artifact, tuple(np.zeros(shape, dtype=np.float32) for _ in range(5))
    )

    assert result["ok"] is False
    assert "state lineage is stale" in result["reason"]
