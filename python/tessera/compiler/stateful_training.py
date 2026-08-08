"""Content-addressed buffer lineage for stateful training packages.

The contract deliberately describes logical buffers, not Python object ids or
device addresses.  A backend may allocate fresh storage or reuse an input only
when the declared mutation policy permits it.  This keeps package identity
stable across compilation and makes state transitions reviewable before a
stateful family crosses the Schedule -> Tile boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .scheduled_matmul import find_tessera_opt, run_tessera_opt


_SCHEMA = "tessera.state_buffer_lineage.v1"
_LION_INPUTS = ("parameter", "gradient", "moment", "dparameter", "dmoment")
_LION_OUTPUTS = ("d_parameter", "d_gradient", "d_moment")


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _buffer(
    *,
    name: str,
    role: str,
    shape: Sequence[int],
    version: int,
    access: str,
    parents: Sequence[str] = (),
) -> dict[str, Any]:
    identity = {
        "name": name,
        "role": role,
        "shape": [int(dim) for dim in shape],
        "dtype": "f32",
        "version": int(version),
        "access": access,
        "parents": list(parents),
    }
    return {**identity, "lineage_id": _digest(identity)}


def build_lion_vjp_state_contract(
    *, target: str, shape: Sequence[int], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Build Lion's functional, no-alias VJP buffer contract.

    Lion's canonical stop-gradient-through-sign VJP reads the forward subjects
    and two output cotangents, then produces three fresh cotangents.  It does
    not mutate the parameter or carried moment.  That distinction matters:
    Lion forward is stateful, while this first E2E-REAL-5C slice is a pure state
    adjoint and therefore has no hidden in-place update.
    """

    if target not in {"x86", "rocm_gfx1151"}:
        raise ValueError("Lion VJP state lineage supports x86 and rocm_gfx1151")
    normalized_shape = tuple(int(dim) for dim in shape)
    if not normalized_shape or any(dim <= 0 for dim in normalized_shape):
        raise ValueError("Lion VJP state lineage requires a positive static shape")
    # Validate the coefficients as floats BEFORE mixing them into a dict that
    # also carries a policy string. The previous form checked
    # `list(numeric.values())[:3]`, which silently depended on dict insertion
    # order to skip the string -- reorder the literal and it would call
    # `isfinite` on a policy name.
    lr = float(kwargs.get("lr", 1.0e-4))
    beta2 = float(kwargs.get("beta2", 0.99))
    weight_decay = float(kwargs.get("weight_decay", 0.0))
    if not all(math.isfinite(value) for value in (lr, beta2, weight_decay)):
        raise ValueError("Lion VJP state lineage requires finite coefficients")
    numeric = {
        "lr": lr,
        "beta2": beta2,
        "weight_decay": weight_decay,
        "derivative_policy": "stop_gradient_through_sign",
    }

    inputs = [
        _buffer(
            name=name,
            role=role,
            shape=normalized_shape,
            version=0,
            access="read_only",
        )
        for name, role in zip(
            ("p", "g", "m", "dparam_out", "dmoment_out"),
            _LION_INPUTS,
            strict=True,
        )
    ]
    by_name = {buffer["name"]: buffer["lineage_id"] for buffer in inputs}
    parent_names = (
        ("p", "dparam_out"),
        ("g", "dmoment_out"),
        ("m", "dmoment_out"),
    )
    outputs = [
        _buffer(
            name=name,
            role=role,
            shape=normalized_shape,
            version=1,
            access="fresh_write",
            parents=tuple(by_name[parent] for parent in parents),
        )
        for name, role, parents in zip(
            ("d_p", "d_g", "d_m"), _LION_OUTPUTS, parent_names, strict=True
        )
    ]
    body: dict[str, Any] = {
        "schema": _SCHEMA,
        "family": "lion_vjp",
        "phase": "backward",
        "target": target,
        "architecture": "zen5-avx512" if target == "x86" else "gfx1151",
        "numeric": numeric,
        "inputs": inputs,
        "outputs": outputs,
        "mutation": {
            "mode": "functional",
            "alias_policy": "no_input_output_alias",
            "ordered_writes": ["d_p", "d_g", "d_m"],
            "state_transition": "m@0-read-only;d_m@1-fresh",
        },
    }
    return {**body, "artifact_hash": _digest(body)}


def build_adafactor_vjp_state_contract(
    *,
    target: str,
    parameter_shape: Sequence[int],
    topology: str,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if target not in {"x86", "rocm_gfx1151"}:
        raise ValueError("Adafactor VJP lineage supports x86 and rocm_gfx1151")
    shape = tuple(int(dim) for dim in parameter_shape)
    if any(dim <= 0 for dim in shape) or topology not in {"factored", "full"}:
        raise ValueError("Adafactor VJP requires positive static shape and topology")
    if topology == "factored" and len(shape) < 2:
        raise ValueError("factored Adafactor requires rank-2+ parameter state")
    if topology == "full" and len(shape) >= 2:
        raise ValueError("full Adafactor requires rank-0/1 parameter state")
    numeric = {
        "lr": float(kwargs.get("lr", 1.0e-3)),
        "beta2": float(kwargs.get("beta2", 0.999)),
        "eps": float(kwargs.get("eps", 1.0e-30)),
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("Adafactor VJP lineage requires finite coefficients")
    state_shapes = (
        (shape[:-1], (shape[-1],)) if topology == "factored" else (shape,)
    )
    input_specs = [("p", "parameter", shape), ("g", "gradient", shape)]
    input_specs.extend(
        (f"state{index}", f"optimizer_state_{index}", state_shape)
        for index, state_shape in enumerate(state_shapes)
    )
    input_specs.append(("dy", "dparameter", shape))
    inputs = [
        _buffer(
            name=name,
            role=role,
            shape=buffer_shape,
            version=0,
            access="read_only",
        )
        for name, role, buffer_shape in input_specs
    ]
    by_name = {buffer["name"]: buffer["lineage_id"] for buffer in inputs}
    output_specs = [
        ("d_p", "d_parameter", shape, ("p", "dy")),
        (
            "d_g",
            "d_gradient",
            shape,
            tuple(["g", "dy", *[f"state{i}" for i in range(len(state_shapes))]]),
        ),
    ]
    output_specs.extend(
        (
            f"d_state{index}",
            f"d_optimizer_state_{index}",
            state_shape,
            (f"state{index}", "g", "dy"),
        )
        for index, state_shape in enumerate(state_shapes)
    )
    outputs = [
        _buffer(
            name=name,
            role=role,
            shape=buffer_shape,
            version=1,
            access="fresh_write",
            parents=tuple(by_name[parent] for parent in parents),
        )
        for name, role, buffer_shape, parents in output_specs
    ]
    body: dict[str, Any] = {
        "schema": _SCHEMA,
        "family": "adafactor_vjp",
        "phase": "backward",
        "topology": topology,
        "target": target,
        "architecture": "zen5-avx512" if target == "x86" else "gfx1151",
        "numeric": numeric,
        "inputs": inputs,
        "outputs": outputs,
        "mutation": {
            "mode": "functional",
            "alias_policy": "no_input_output_alias",
            "ordered_writes": [item["name"] for item in outputs],
            "state_transition": "state@0-read-only;d_state@1-fresh",
        },
    }
    return {**body, "artifact_hash": _digest(body)}


def validate_lion_vjp_state_contract(
    contract: Mapping[str, Any],
    *,
    target: str,
    shape: Sequence[int],
    kwargs: Mapping[str, Any],
) -> None:
    """Fail closed if a Lion package's lineage or mutation identity is stale."""

    expected = build_lion_vjp_state_contract(
        target=target, shape=shape, kwargs=kwargs
    )
    if dict(contract) != expected:
        raise ValueError("Lion VJP state/buffer lineage contract is stale or malformed")


_TILE_HASH_RE = re.compile(
    r'tile\.training_kernel[\s\S]*?tessera\.schedule_hash = "([0-9a-f]{64})"'
)


@dataclass(frozen=True)
class ScheduledLionVJPArtifact:
    target: str
    architecture: str
    shape: tuple[int, ...]
    state_contract: Mapping[str, Any]
    schedule_ir: str
    tile_ir: str

    @property
    def schedule_digest(self) -> str:
        return str(self.state_contract["artifact_hash"])

    def validate(self) -> None:
        validate_lion_vjp_state_contract(
            self.state_contract,
            target=self.target,
            shape=self.shape,
            kwargs=self.state_contract["numeric"],
        )
        if self.schedule_ir.count("schedule.lion_vjp") != 1:
            raise ValueError("Lion VJP requires one typed Schedule operation")
        if self.tile_ir.count("tile.training_kernel") != 1:
            raise ValueError("Lion VJP requires one launch-level Tile artifact")
        if "schedule." in self.tile_ir:
            raise ValueError("Lion VJP Tile artifact retains Schedule IR")
        if _TILE_HASH_RE.findall(self.tile_ir) != [self.schedule_digest]:
            raise ValueError("Lion VJP Tile artifact has stale lineage identity")
        for marker in (
            'family = "lion_vjp"',
            'storage = "f32"',
            'derivative_policy = "stop_gradient_through_sign"',
            'mutation_mode = "functional"',
            'alias_policy = "no_input_output_alias"',
            'state_transition = "m@0-read-only;d_m@1-fresh"',
        ):
            if marker not in self.tile_ir:
                raise ValueError(f"Lion VJP Tile artifact lost {marker}")

    def metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "family": "lion_vjp",
            "target": self.target,
            "architecture": self.architecture,
            "shape": list(self.shape),
            "schedule_digest": self.schedule_digest,
            "tile_ir": self.tile_ir,
        }


def lower_scheduled_lion_vjp(
    *, target: str, shape: Sequence[int], kwargs: Mapping[str, Any]
) -> ScheduledLionVJPArtifact:
    contract = build_lion_vjp_state_contract(
        target=target, shape=shape, kwargs=kwargs
    )
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled Lion VJP requires production tessera-opt")
    normalized_shape = tuple(int(dim) for dim in shape)
    tensor_type = "tensor<" + "x".join(map(str, normalized_shape)) + "xf32>"
    digest = str(contract["artifact_hash"])
    body = {key: value for key, value in contract.items() if key != "artifact_hash"}
    payload = _payload(body)
    numeric = contract["numeric"]
    mutation = contract["mutation"]
    architecture = str(contract["architecture"])
    compiler_target = "x86" if target == "x86" else "rocm"
    args = ", ".join(
        f"%{name}: {tensor_type}"
        for name in ("p", "g", "m", "dparam_out", "dmoment_out")
    )
    inputs = ", ".join(
        f"%{name}" for name in ("p", "g", "m", "dparam_out", "dmoment_out")
    )
    input_types = ", ".join([tensor_type] * 5)
    result_types = ", ".join([tensor_type] * 3)
    schedule_ir = f'''module attributes {{tessera.target = "{compiler_target}", tessera.arch = "{architecture}"}} {{
  func.func @tessera_lion_vjp({args}) -> ({result_types}) {{
    %grads:3 = schedule.lion_vjp {inputs} {{artifact_hash = "{digest}", lineage_payload = {json.dumps(payload)}, arch = "{architecture}", learning_rate = {float(numeric["lr"]):.9g} : f32, beta2 = {float(numeric["beta2"]):.9g} : f32, weight_decay = {float(numeric["weight_decay"]):.9g} : f32, derivative_policy = "stop_gradient_through_sign", mutation_mode = "{mutation["mode"]}", alias_policy = "{mutation["alias_policy"]}", state_transition = "{mutation["state_transition"]}", ordered_writes = array<i64: 0, 1, 2>, workgroup_size = {256 if target == "rocm_gfx1151" else 1} : i64}} : {input_types} -> {result_types}
    schedule.artifact {{hash = "{digest}", arch = "{architecture}", shape_key = "family=lion_vjp;shape={'x'.join(map(str, normalized_shape))};storage=f32", numeric_policy = "f32;stop_gradient_through_sign;functional_no_alias"}}
    return %grads#0, %grads#1, %grads#2 : {result_types}
  }}
}}
'''
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    artifact = ScheduledLionVJPArtifact(
        target, architecture, normalized_shape, contract, schedule_ir, tile_ir
    )
    artifact.validate()
    return artifact


def validate_scheduled_lion_vjp_metadata(
    metadata: Mapping[str, Any],
    *,
    target: str,
    shape: Sequence[int],
    state_contract: Mapping[str, Any],
) -> None:
    expected_target = target
    expected_shape = tuple(int(dim) for dim in shape)
    expected_hash = str(state_contract["artifact_hash"])
    if (
        metadata.get("family") != "lion_vjp"
        or metadata.get("target") != expected_target
        or tuple(metadata.get("shape", ())) != expected_shape
        or metadata.get("schedule_digest") != expected_hash
    ):
        raise ValueError("Lion VJP scheduled artifact metadata is stale")
    tile_ir = str(metadata.get("tile_ir", ""))
    if tile_ir.count("tile.training_kernel") != 1 or "schedule." in tile_ir:
        raise ValueError("Lion VJP scheduled artifact is not exact launch Tile IR")
    if _TILE_HASH_RE.findall(tile_ir) != [expected_hash]:
        raise ValueError("Lion VJP scheduled artifact lineage is stale")
    expected_arch = str(state_contract["architecture"])
    required_markers = (
        f'arch = "{expected_arch}"',
        'family = "lion_vjp"',
        'storage = "f32"',
        'derivative_policy = "stop_gradient_through_sign"',
        'mutation_mode = "functional"',
        'alias_policy = "no_input_output_alias"',
        'state_transition = "m@0-read-only;d_m@1-fresh"',
        "ordered_writes = array<i64: 0, 1, 2>",
        f"arith.constant {math.prod(expected_shape)} : i64",
    )
    if any(marker not in tile_ir for marker in required_markers):
        raise ValueError("Lion VJP scheduled artifact policy is stale")
    for name, key in (
        ("learning_rate", "lr"),
        ("beta2", "beta2"),
        ("weight_decay", "weight_decay"),
    ):
        match = re.search(rf"{name} = ([+\-0-9.eE]+) : f32", tile_ir)
        if match is None or not math.isclose(
            float(match.group(1)),
            float(state_contract["numeric"][key]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-9,
        ):
            raise ValueError("Lion VJP scheduled artifact coefficients are stale")


def validate_adafactor_vjp_state_contract(
    contract: Mapping[str, Any],
    *,
    target: str,
    parameter_shape: Sequence[int],
    topology: str,
    kwargs: Mapping[str, Any],
) -> None:
    expected = build_adafactor_vjp_state_contract(
        target=target,
        parameter_shape=parameter_shape,
        topology=topology,
        kwargs=kwargs,
    )
    if dict(contract) != expected:
        raise ValueError("Adafactor VJP state/buffer lineage contract is stale")


@dataclass(frozen=True)
class ScheduledAdafactorVJPArtifact:
    target: str
    architecture: str
    parameter_shape: tuple[int, ...]
    topology: str
    state_contract: Mapping[str, Any]
    schedule_ir: str
    tile_ir: str

    @property
    def schedule_digest(self) -> str:
        return str(self.state_contract["artifact_hash"])

    def validate(self) -> None:
        validate_adafactor_vjp_state_contract(
            self.state_contract,
            target=self.target,
            parameter_shape=self.parameter_shape,
            topology=self.topology,
            kwargs=self.state_contract["numeric"],
        )
        metadata = self.metadata(validate=False)
        validate_scheduled_adafactor_vjp_metadata(
            metadata,
            target=self.target,
            parameter_shape=self.parameter_shape,
            topology=self.topology,
            state_contract=self.state_contract,
        )

    def metadata(self, *, validate: bool = True) -> dict[str, Any]:
        value = {
            "family": "adafactor_vjp",
            "target": self.target,
            "architecture": self.architecture,
            "parameter_shape": list(self.parameter_shape),
            "topology": self.topology,
            "schedule_digest": self.schedule_digest,
            "tile_ir": self.tile_ir,
        }
        if validate:
            self.validate()
        return value


def _tensor_type(shape: Sequence[int]) -> str:
    return "tensor<" + ("x".join(map(str, shape)) + "x" if shape else "") + "f32>"


def lower_scheduled_adafactor_vjp(
    *,
    target: str,
    parameter_shape: Sequence[int],
    topology: str,
    kwargs: Mapping[str, Any],
) -> ScheduledAdafactorVJPArtifact:
    contract = build_adafactor_vjp_state_contract(
        target=target,
        parameter_shape=parameter_shape,
        topology=topology,
        kwargs=kwargs,
    )
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled Adafactor VJP requires production tessera-opt")
    shape = tuple(int(dim) for dim in parameter_shape)
    state_shapes = (shape[:-1], (shape[-1],)) if topology == "factored" else (shape,)
    names = ["p", "g", *[f"state{i}" for i in range(len(state_shapes))], "dy"]
    input_shapes = [shape, shape, *state_shapes, shape]
    output_shapes = [shape, shape, *state_shapes]
    input_types = [_tensor_type(value) for value in input_shapes]
    output_types = [_tensor_type(value) for value in output_shapes]
    args = ", ".join(
        f"%{name}: {tensor_type}" for name, tensor_type in zip(names, input_types)
    )
    inputs = ", ".join(f"%{name}" for name in names)
    digest = str(contract["artifact_hash"])
    body = {key: value for key, value in contract.items() if key != "artifact_hash"}
    payload = _payload(body)
    numeric = contract["numeric"]
    mutation = contract["mutation"]
    architecture = str(contract["architecture"])
    compiler_target = "x86" if target == "x86" else "rocm"
    write_order = ", ".join(map(str, range(len(output_shapes))))
    schedule_ir = f'''module attributes {{tessera.target = "{compiler_target}", tessera.arch = "{architecture}"}} {{
  func.func @tessera_adafactor_vjp({args}) -> ({", ".join(output_types)}) {{
    %grads:{len(output_types)} = schedule.adafactor_vjp {inputs} {{artifact_hash = "{digest}", lineage_payload = {json.dumps(payload)}, arch = "{architecture}", topology = "{topology}", learning_rate = {float(numeric["lr"]):.9e} : f32, beta2 = {float(numeric["beta2"]):.9e} : f32, epsilon = {float(numeric["eps"]):.9e} : f32, mutation_mode = "{mutation["mode"]}", alias_policy = "{mutation["alias_policy"]}", state_transition = "{mutation["state_transition"]}", ordered_writes = array<i64: {write_order}>, workgroup_size = {256 if target == "rocm_gfx1151" else 1} : i64}} : {", ".join(input_types)} -> {", ".join(output_types)}
    schedule.artifact {{hash = "{digest}", arch = "{architecture}", shape_key = "family=adafactor_vjp;topology={topology};shape={'x'.join(map(str, shape))};storage=f32", numeric_policy = "f32;functional_no_alias"}}
    return {", ".join(f"%grads#{index}" for index in range(len(output_types)))} : {", ".join(output_types)}
  }}
}}
'''
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    artifact = ScheduledAdafactorVJPArtifact(
        target, architecture, shape, topology, contract, schedule_ir, tile_ir
    )
    artifact.validate()
    return artifact


def validate_scheduled_adafactor_vjp_metadata(
    metadata: Mapping[str, Any],
    *,
    target: str,
    parameter_shape: Sequence[int],
    topology: str,
    state_contract: Mapping[str, Any],
) -> None:
    shape = tuple(int(dim) for dim in parameter_shape)
    digest = str(state_contract["artifact_hash"])
    if (
        metadata.get("family") != "adafactor_vjp"
        or metadata.get("target") != target
        or tuple(metadata.get("parameter_shape", ())) != shape
        or metadata.get("topology") != topology
        or metadata.get("schedule_digest") != digest
    ):
        raise ValueError("Adafactor VJP scheduled artifact metadata is stale")
    tile_ir = str(metadata.get("tile_ir", ""))
    if tile_ir.count("tile.training_kernel") != 1 or "schedule." in tile_ir:
        raise ValueError("Adafactor VJP artifact is not exact launch Tile IR")
    if _TILE_HASH_RE.findall(tile_ir) != [digest]:
        raise ValueError("Adafactor VJP scheduled artifact lineage is stale")
    required = (
        'family = "adafactor_vjp"',
        f'topology = "{topology}"',
        f'arch = "{state_contract["architecture"]}"',
        'mutation_mode = "functional"',
        'alias_policy = "no_input_output_alias"',
        'state_transition = "state@0-read-only;d_state@1-fresh"',
    )
    if any(marker not in tile_ir for marker in required):
        raise ValueError("Adafactor VJP scheduled artifact policy is stale")
    for name, key in (
        ("learning_rate", "lr"),
        ("beta2", "beta2"),
        ("epsilon", "eps"),
    ):
        match = re.search(rf"{name} = ([+\-0-9.eE]+) : f32", tile_ir)
        if match is None or not math.isclose(
            float(match.group(1)),
            float(state_contract["numeric"][key]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-35,
        ):
            raise ValueError("Adafactor VJP scheduled coefficients are stale")


_SEQUENCE_PHASES = (
    "checkpoint",
    "chunk_summary",
    "chunk_prefix",
    "chunk_fill",
    "reverse",
)


def build_sequence_mixer_backward_state_contract(
    *,
    target: str,
    family: str,
    q_shape: Sequence[int],
    v_shape: Sequence[int],
    erase: bool,
    chunk_size: int,
    parallel_chunks: bool,
) -> dict[str, Any]:
    """Describe the bounded physical DeltaNet backward buffer program."""

    if target not in {"x86", "rocm_gfx1151"}:
        raise ValueError("sequence-mixer lineage supports x86 and rocm_gfx1151")
    if family not in {
        "gated_deltanet",
        "kimi_delta_attention",
        "modified_delta_attention",
    }:
        raise ValueError("sequence-mixer lineage requires a proven physical family")
    q_dims = tuple(int(dim) for dim in q_shape)
    v_dims = tuple(int(dim) for dim in v_shape)
    if (
        len(q_dims) != 4
        or len(v_dims) != 4
        or q_dims[:3] != v_dims[:3]
        or any(dim <= 0 for dim in (*q_dims, *v_dims))
        or int(chunk_size) <= 0
    ):
        raise ValueError("sequence-mixer lineage requires compatible static rank-4 shapes")
    scalar_shape = q_dims[:3]
    input_specs = (
        ("q", "query", q_dims),
        ("k", "key", q_dims),
        ("v", "value", v_dims),
        ("gate", "gate", v_dims),
        ("beta", "update_scale", scalar_shape),
        ("decay", "state_decay", scalar_shape),
        ("dy", "output_cotangent", v_dims),
    )
    inputs = [
        _buffer(
            name=name,
            role=role,
            shape=shape,
            version=0,
            access="read_only",
        )
        for name, role, shape in input_specs
    ]
    by_name = {item["name"]: item["lineage_id"] for item in inputs}
    output_specs = (
        ("dq", "query_cotangent", q_dims, ("q", "dy")),
        ("dk", "key_cotangent", q_dims, ("k", "dy")),
        ("dv", "value_cotangent", v_dims, ("v", "dy")),
        ("dgate", "gate_cotangent", v_dims, ("gate", "dy")),
        ("dbeta", "update_scale_cotangent", scalar_shape, ("beta", "dy")),
        ("ddecay", "state_decay_cotangent", scalar_shape, ("decay", "dy")),
    )
    outputs = [
        _buffer(
            name=name,
            role=role,
            shape=shape,
            version=1,
            access="fresh_write",
            parents=tuple(by_name[parent] for parent in parents),
        )
        for name, role, shape, parents in output_specs
    ]
    body: dict[str, Any] = {
        "schema": _SCHEMA,
        "family": "sequence_mixer_backward",
        "mixer_family": family,
        "phase": "backward",
        "target": target,
        "architecture": "zen5-avx512" if target == "x86" else "gfx1151",
        "numeric": {
            "erase": bool(erase),
            "chunk_size": int(chunk_size),
            "parallel_chunks": bool(parallel_chunks),
        },
        "inputs": inputs,
        "outputs": outputs,
        "mutation": {
            "mode": "functional",
            "alias_policy": "no_input_output_alias",
            "ordered_writes": [item["name"] for item in outputs],
            "workspace_owner": "program_launch",
            "phase_order": list(_SEQUENCE_PHASES),
        },
    }
    return {**body, "artifact_hash": _digest(body)}


def validate_sequence_mixer_backward_state_contract(
    contract: Mapping[str, Any],
    *,
    target: str,
    family: str,
    q_shape: Sequence[int],
    v_shape: Sequence[int],
    erase: bool,
    chunk_size: int,
    parallel_chunks: bool,
) -> None:
    expected = build_sequence_mixer_backward_state_contract(
        target=target,
        family=family,
        q_shape=q_shape,
        v_shape=v_shape,
        erase=erase,
        chunk_size=chunk_size,
        parallel_chunks=parallel_chunks,
    )
    if dict(contract) != expected:
        raise ValueError("sequence-mixer state/buffer lineage is stale or malformed")


@dataclass(frozen=True)
class ScheduledSequenceMixerBackwardArtifact:
    target: str
    architecture: str
    family: str
    q_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    state_contract: Mapping[str, Any]
    schedule_ir: str
    tile_ir: str

    @property
    def schedule_digest(self) -> str:
        return str(self.state_contract["artifact_hash"])

    def metadata(self, *, validate: bool = True) -> dict[str, Any]:
        value = {
            "family": "sequence_mixer_backward",
            "mixer_family": self.family,
            "target": self.target,
            "architecture": self.architecture,
            "q_shape": list(self.q_shape),
            "v_shape": list(self.v_shape),
            "schedule_digest": self.schedule_digest,
            "tile_ir": self.tile_ir,
        }
        if validate:
            validate_scheduled_sequence_mixer_backward_metadata(
                value,
                target=self.target,
                family=self.family,
                q_shape=self.q_shape,
                v_shape=self.v_shape,
                state_contract=self.state_contract,
            )
        return value

    def validate(self) -> None:
        numeric = self.state_contract["numeric"]
        validate_sequence_mixer_backward_state_contract(
            self.state_contract,
            target=self.target,
            family=self.family,
            q_shape=self.q_shape,
            v_shape=self.v_shape,
            erase=bool(numeric["erase"]),
            chunk_size=int(numeric["chunk_size"]),
            parallel_chunks=bool(numeric["parallel_chunks"]),
        )
        self.metadata()


def lower_scheduled_sequence_mixer_backward(
    *,
    target: str,
    family: str,
    q_shape: Sequence[int],
    v_shape: Sequence[int],
    erase: bool,
    chunk_size: int = 64,
    parallel_chunks: bool = True,
) -> ScheduledSequenceMixerBackwardArtifact:
    contract = build_sequence_mixer_backward_state_contract(
        target=target,
        family=family,
        q_shape=q_shape,
        v_shape=v_shape,
        erase=erase,
        chunk_size=chunk_size,
        parallel_chunks=parallel_chunks,
    )
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled sequence-mixer backward requires tessera-opt")
    q_dims = tuple(int(dim) for dim in q_shape)
    v_dims = tuple(int(dim) for dim in v_shape)
    scalar_dims = q_dims[:3]
    q_type, v_type, scalar_type = map(
        _tensor_type, (q_dims, v_dims, scalar_dims)
    )
    names = ("q", "k", "v", "gate", "beta", "decay", "dy")
    types = (q_type, q_type, v_type, v_type, scalar_type, scalar_type, v_type)
    result_types = (q_type, q_type, v_type, v_type, scalar_type, scalar_type)
    digest = str(contract["artifact_hash"])
    body = {key: value for key, value in contract.items() if key != "artifact_hash"}
    payload = _payload(body)
    mutation = contract["mutation"]
    architecture = str(contract["architecture"])
    compiler_target = "x86" if target == "x86" else "rocm"
    phase_attrs = ", ".join(json.dumps(phase) for phase in _SEQUENCE_PHASES)
    args = ", ".join(f"%{name}: {typ}" for name, typ in zip(names, types))
    schedule_ir = f'''module attributes {{tessera.target = "{compiler_target}", tessera.arch = "{architecture}"}} {{
  func.func @tessera_sequence_mixer_backward({args}) -> ({", ".join(result_types)}) {{
    %grads:6 = schedule.sequence_mixer_backward {", ".join(f"%{name}" for name in names)} {{artifact_hash = "{digest}", lineage_payload = {json.dumps(payload)}, arch = "{architecture}", family = "{family}", erase = {str(bool(erase)).lower()}, chunk_size = {int(chunk_size)} : i64, parallel_chunks = {str(bool(parallel_chunks)).lower()}, mutation_mode = "{mutation["mode"]}", alias_policy = "{mutation["alias_policy"]}", workspace_owner = "{mutation["workspace_owner"]}", phase_order = [{phase_attrs}], workgroup_size = {256 if target == "rocm_gfx1151" else 1} : i64}} : {", ".join(types)} -> {", ".join(result_types)}
    schedule.artifact {{hash = "{digest}", arch = "{architecture}", shape_key = "family=sequence_mixer_backward;mixer={family};q={'x'.join(map(str, q_dims))};v={'x'.join(map(str, v_dims))};storage=f32", numeric_policy = "f32;functional_no_alias;ordered_reverse"}}
    return {", ".join(f"%grads#{index}" for index in range(6))} : {", ".join(result_types)}
  }}
}}
'''
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    artifact = ScheduledSequenceMixerBackwardArtifact(
        target, architecture, family, q_dims, v_dims, contract, schedule_ir, tile_ir
    )
    artifact.validate()
    return artifact


def validate_scheduled_sequence_mixer_backward_metadata(
    metadata: Mapping[str, Any],
    *,
    target: str,
    family: str,
    q_shape: Sequence[int],
    v_shape: Sequence[int],
    state_contract: Mapping[str, Any],
) -> None:
    q_dims = tuple(int(dim) for dim in q_shape)
    v_dims = tuple(int(dim) for dim in v_shape)
    digest = str(state_contract["artifact_hash"])
    if (
        metadata.get("family") != "sequence_mixer_backward"
        or metadata.get("mixer_family") != family
        or metadata.get("target") != target
        or tuple(metadata.get("q_shape", ())) != q_dims
        or tuple(metadata.get("v_shape", ())) != v_dims
        or metadata.get("schedule_digest") != digest
    ):
        raise ValueError("sequence-mixer scheduled metadata is stale")
    tile_ir = str(metadata.get("tile_ir", ""))
    if tile_ir.count("tile.training_kernel") != 1 or "schedule." in tile_ir:
        raise ValueError("sequence-mixer artifact is not exact launch Tile IR")
    required = (
        f'tessera.schedule_hash = "{digest}"',
        'family = "sequence_mixer_backward"',
        f'mixer_family = "{family}"',
        f'arch = "{state_contract["architecture"]}"',
        'mutation_mode = "functional"',
        'alias_policy = "no_input_output_alias"',
        'workspace_owner = "program_launch"',
        'phase_order = ["checkpoint", "chunk_summary", "chunk_prefix", "chunk_fill", "reverse"]',
    )
    if any(marker not in tile_ir for marker in required):
        raise ValueError("sequence-mixer scheduled policy is stale")


__all__ = [
    "ScheduledAdafactorVJPArtifact",
    "ScheduledLionVJPArtifact",
    "ScheduledSequenceMixerBackwardArtifact",
    "build_adafactor_vjp_state_contract",
    "build_lion_vjp_state_contract",
    "build_sequence_mixer_backward_state_contract",
    "lower_scheduled_adafactor_vjp",
    "lower_scheduled_lion_vjp",
    "lower_scheduled_sequence_mixer_backward",
    "validate_adafactor_vjp_state_contract",
    "validate_scheduled_adafactor_vjp_metadata",
    "validate_scheduled_lion_vjp_metadata",
    "validate_scheduled_sequence_mixer_backward_metadata",
    "validate_lion_vjp_state_contract",
    "validate_sequence_mixer_backward_state_contract",
]
