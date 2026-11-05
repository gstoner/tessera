"""Native Apple GPU domain value contracts (E2E-REAL-6, 2026-11-04).

One isolated static Apple GPU Graph operation -- ``tessera.rl.ppo_policy_loss``,
``tessera.ebm.{energy_quadratic,langevin_step,refinement,partition_exact}`` or
the cl(3,0) ``tessera.clifford.geometric_product`` -- lowers through the native
compiler: ``--tessera-graph-to-schedule`` records a content-addressed contract
(``src/compiler/programming_model/lib/NativeAppleGPUDomain.h``),
``--tessera-schedule-to-tile`` emits the registered Tile value op, and
``--tile-to-apple_gpu=value=true`` turns it into a verified
``tessera_apple.gpu.kernel_call``. The package
(``apple_native.package_scheduled_apple_gpu_domain``) is projected from the
*replayed* contract, never from the Python Graph object.

The delegate behind every family is a hand-written MSL / MPSGraph C ABI entry
in ``apple_gpu_runtime.mm``: a Decision #28 Tier-3 implementation reached
through the declared, verified Target IR op, which is one lowering path
(Decision #31), not a second authority. The retired Graph-owned constructor is
the declared differential oracle in
``tests/_support/apple_gpu_domain_baseline.py`` (Decision #31(a)).

These families also have compiled-kernel routes on other targets (the Clifford
arena route in ``native_clifford_gpu`` through ``ts-clifford-opt``, the EBM
Langevin row program through ``tessera-row-program-to-gpu``). Those end in
ROCDL / NVVM, which Apple has no counterpart for -- the Apple GPU pipeline
cannot yet emit an MSL body from MLIR -- so on Apple the compiled route ends at
the declared runtime delegate, exactly as the Apple GELU and softmax slices do.

:func:`supports_scheduled_apple_gpu_domain` is the host-free admission. It
mirrors what the native contract accepts: every semantic value the delegate
reads is materialized, and every semantic it does not implement is refused
(Decision #21a).
"""

from __future__ import annotations

import copy
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any

from .graph_ir import GraphIRModule

#: Graph op -> (kind, minimum inputs, maximum inputs). Mirrors
#: ``kAppleGPUDomainSpecs`` in ``NativeAppleGPUDomain.h``.
APPLE_GPU_DOMAIN_OPS: dict[str, tuple[str, int, int]] = {
    "tessera.rl.ppo_policy_loss": ("ppo_policy_loss", 3, 6),
    "tessera.ebm.energy_quadratic": ("ebm_energy_quadratic", 2, 2),
    "tessera.ebm.langevin_step": ("ebm_langevin_step", 3, 3),
    "tessera.ebm.refinement": ("ebm_refinement", 2, 2),
    "tessera.ebm.partition_exact": ("ebm_partition_exact", 1, 1),
    "tessera.clifford.geometric_product": ("clifford_geometric_product", 2, 2),
}

#: kind -> (C ABI symbol, delegate routine, framework). The PPO entry is the
#: strict three-tensor ABI; any side tensor selects ``_PPO_EXTENDED``. These
#: are the symbols ``tile-to-apple_gpu{value=true}`` selects.
APPLE_GPU_DOMAIN_SYMBOLS: dict[str, tuple[str, str, str]] = {
    "ppo_policy_loss": ("tessera_apple_gpu_ppo_policy_loss_f32", "mpsgraph_ppo_policy_loss_f32", "MPSGraph"),
    "ebm_energy_quadratic": ("tessera_apple_gpu_ebm_energy_quadratic_value_f32",
                             "msl_ebm_energy_quadratic_value_f32", "Metal"),
    "ebm_langevin_step": ("tessera_apple_gpu_ebm_langevin_step_value_f32",
                          "msl_ebm_langevin_step_value_f32", "Metal"),
    "ebm_refinement": ("tessera_apple_gpu_ebm_refinement_value_f32", "msl_ebm_refinement_value_f32", "Metal"),
    "ebm_partition_exact": ("tessera_apple_gpu_ebm_partition_exact_value_f32",
                            "msl_ebm_partition_exact_value_f32", "Metal"),
    "clifford_geometric_product": ("tessera_apple_gpu_clifford_geo_product_cl30_value_f32",
                                   "msl_clifford_geo_product_cl30_value_f32", "Metal"),
}
_PPO_EXTENDED = ("tessera_apple_gpu_ppo_policy_loss_ex_f32", "mpsgraph_ppo_policy_loss_ex_f32", "MPSGraph")

#: The semantic keys each family may carry on its Target op, i.e. the keys a
#: replayed call is allowed to hold. Anything else on the call is refused.
_SEMANTIC_KEYS: dict[str, frozenset[str]] = {
    "ppo_policy_loss": frozenset({"clip_epsilon", "kl_coef", "entropy_coef", "reduction",
                                  "has_mask", "has_ref_kl", "has_entropy"}),
    "ebm_energy_quadratic": frozenset(),
    "ebm_langevin_step": frozenset({"eta", "noise_scale", "has_noise"}),
    "ebm_refinement": frozenset({"eta", "steps"}),
    "ebm_partition_exact": frozenset({"temperature", "reduction"}),
    "clifford_geometric_product": frozenset({"p", "q", "coefficient_layout"}),
}
_INT32_MAX = 2**31 - 1
TARGET, ARCHITECTURE = "apple_gpu", "apple7"


@dataclass(frozen=True)
class AppleGPUDomainRequest:
    """The normalized Graph request one Apple GPU domain kernel lowers from."""

    graph_op: str
    kind: str
    symbol: str
    abi: str
    framework: str
    input_names: tuple[str, ...]
    output_name: str
    shapes: tuple[tuple[int, ...], ...]
    semantics: dict[str, Any]
    scalars: dict[str, int]


def _static(shape: Any) -> tuple[int, ...]:
    try:
        dims = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("Apple GPU native domain kernel requires static shapes") from exc
    if any(value <= 0 for value in dims) or math.prod(dims) > _INT32_MAX:
        raise ValueError("Apple GPU native domain kernel requires non-empty int32-extent tensors")
    return dims


def _real(raw: dict[str, Any], name: str, default: float | None, kind: str) -> float:
    value = raw.pop(name, default)
    # The Graph op's ODS type is F64Attr: an int or bool is not that spelling.
    if value is None or isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"Apple GPU native {kind} {name} must be a finite float")
    return float(value)


_SEGMENTS_RE = re.compile(r"\s*operandSegmentSizes = array<i32: ([0-9, ]*)>\s*")


def _ppo_segments(raw_attrs: str, operands: int) -> tuple[bool, bool, bool]:
    """``[mask, ref_logp, entropy]`` presence, read the way Graph IR emits it:
    positional from the operand count, unless the op's raw attribute text
    states the ODS segments explicitly."""
    if raw_attrs:
        match = _SEGMENTS_RE.fullmatch(raw_attrs)
        if match is None:
            raise ValueError("Apple GPU native ppo_policy_loss cannot mirror raw attribute text")
        segments = [int(v) for v in match[1].split(",") if v.strip()]
    else:
        side = max(0, min(3, operands - 3))
        segments = [1, 1, 1] + [1 if i < side else 0 for i in range(3)]
    if (len(segments) != 6 or segments[:3] != [1, 1, 1] or any(v not in (0, 1) for v in segments[3:])
            or sum(segments) != operands):
        raise ValueError("Apple GPU native ppo_policy_loss operand segments are malformed")
    return segments[3] == 1, segments[4] == 1, segments[5] == 1


def admit(module: GraphIRModule) -> AppleGPUDomainRequest:
    """Admit one Apple GPU native domain request or raise ``ValueError``."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        raise ValueError("Apple GPU native domain kernel requires one Graph function and operation")
    function = module.functions[0]
    op = function.body[0]
    spec = APPLE_GPU_DOMAIN_OPS.get(op.op_name)
    if spec is None:
        raise ValueError(f"{op.op_name} has no Apple GPU native domain contract")
    kind, low, high = spec
    args = {arg.name: arg for arg in function.args}
    inputs = tuple(value.removeprefix("%") for value in op.operands)
    outputs = tuple(value.removeprefix("%") for value in function.return_values)
    if (not low <= len(inputs) <= high or len(set(inputs)) != len(inputs) or set(inputs) != set(args)
            or len(function.args) != len(inputs)):
        raise ValueError("Apple GPU native domain operands must be its distinct entry arguments")
    if (len(outputs) != 1 or len(function.result_types) != 1 or not outputs[0]
            or outputs[0] in inputs):
        raise ValueError("Apple GPU native domain kernel binds one result to a distinct name")
    dtypes = [str(args[name].ir_type.dtype) for name in inputs] + [str(function.result_types[0].dtype)]
    if any(dtype != "fp32" for dtype in dtypes):
        raise ValueError("Apple GPU native domain kernel storage is f32 only")
    shapes = [_static(args[name].ir_type.shape) for name in inputs]
    shapes.append(_static(function.result_types[0].shape))
    first, result = shapes[0], shapes[-1]
    # None-valued keywords mean "not given" (the frontend's spelling).
    raw = {key: value for key, value in op.kwargs.items() if value is not None}
    raw_attrs = (op.attrs or "").strip()
    # Raw attribute text is mirrored only for the one spelling Graph IR uses
    # (PPO's ODS segments); anything else is the native contract's to read,
    # and this admission does not guess at it.
    if raw_attrs and kind != "ppo_policy_loss":
        raise ValueError(f"Apple GPU native {kind} cannot mirror raw attribute text")
    semantics: dict[str, Any] = {}
    scalars: dict[str, int] = {}
    symbol, abi, framework = APPLE_GPU_DOMAIN_SYMBOLS[kind]
    if kind == "ppo_policy_loss":
        mask, ref, entropy = _ppo_segments(raw_attrs, len(inputs))
        if any(shape != first for shape in shapes[:-1]) or result != ():
            raise ValueError("Apple GPU ppo_policy_loss requires one operand shape and a rank-0 result")
        clip = _real(raw, "clip_epsilon", 0.2, kind)
        kl = _real(raw, "kl_coef", 0.0, kind)
        ent = _real(raw, "entropy_coef", 0.0, kind)
        if clip <= 0.0 or kl < 0.0 or ent < 0.0 or (kl and not ref) or (ent and not entropy):
            raise ValueError("Apple GPU ppo_policy_loss coefficients are outside the delegate's contract")
        if raw.pop("reduction", "mean") != "mean":
            raise ValueError("Apple GPU ppo_policy_loss implements reduction = 'mean' only")
        semantics = {"clip_epsilon": clip, "entropy_coef": ent, "has_entropy": entropy,
                     "has_mask": mask, "has_ref_kl": ref, "kl_coef": kl, "reduction": "mean"}
        if mask or ref or entropy:
            symbol, abi, framework = _PPO_EXTENDED
        scalars = {"N": math.prod(first)}
    elif kind == "ebm_energy_quadratic":
        if len(first) != 2 or shapes[1] != first or result != (first[0],):
            raise ValueError("Apple GPU ebm_energy_quadratic requires x/y [B,D] and energies [B]")
        scalars = {"B": first[0], "D": first[1]}
    elif kind == "ebm_langevin_step":
        if any(shape != first for shape in shapes):
            raise ValueError("Apple GPU ebm_langevin_step requires y/grad/noise/result of one shape")
        eta = _real(raw, "eta", None, kind)
        scale = _real(raw, "noise_scale", 0.0, kind)
        if eta <= 0.0 or scale < 0.0 or "temperature" in raw:
            raise ValueError("Apple GPU ebm_langevin_step requires eta > 0, noise_scale >= 0 and no temperature")
        semantics = {"eta": eta, "has_noise": True, "noise_scale": scale}
        scalars = {"N": math.prod(first)}
    elif kind == "ebm_refinement":
        if any(shape != first for shape in shapes):
            raise ValueError("Apple GPU ebm_refinement requires y/grad/result of one shape")
        eta = _real(raw, "eta", None, kind)
        steps = raw.pop("steps", None)
        if (eta <= 0.0 or type(steps) is not int or not 0 < steps <= _INT32_MAX
                or "temperature" in raw or "noise_scale" in raw):
            raise ValueError("Apple GPU ebm_refinement implements the deterministic eta > 0, steps > 0 loop only")
        semantics = {"eta": eta, "steps": steps}
        scalars = {"N": math.prod(first)}
    elif kind == "ebm_partition_exact":
        if result != ():
            raise ValueError("Apple GPU ebm_partition_exact requires a rank-0 result")
        temperature = _real(raw, "temperature", 1.0, kind)
        if temperature <= 0.0 or raw.pop("reduction", "logsumexp") != "logsumexp":
            raise ValueError("Apple GPU ebm_partition_exact requires temperature > 0 and reduction 'logsumexp'")
        semantics = {"reduction": "logsumexp", "temperature": temperature}
        scalars = {"N": math.prod(first)}
    else:  # clifford_geometric_product
        p, q = raw.pop("p", None), raw.pop("q", None)
        signature = raw.pop("signature", None)
        layout = raw.pop("coefficient_layout", "blade_last")
        if type(p) is not int or type(q) is not int or (p, q) != (3, 0):
            raise ValueError("Apple GPU clifford_geometric_product implements cl(3,0) only")
        if signature is not None and list(signature) != [1, 1, 1]:
            raise ValueError("Apple GPU clifford_geometric_product implements signature [1,1,1] only")
        if layout != "blade_last":
            raise ValueError("Apple GPU clifford_geometric_product implements coefficient_layout 'blade_last' only")
        if not first or first[-1] != 8 or any(shape != first for shape in shapes):
            raise ValueError("Apple GPU clifford_geometric_product requires blade-last [...,8] tensors of one shape")
        semantics = {"coefficient_layout": "blade_last", "p": 3, "q": 0}
        scalars = {"Batch": math.prod(first) // 8}
    # Keywords the delegate does not compute are refused, not ignored. The
    # effect marker is the only non-semantic keyword the Graph op may carry.
    if raw.pop("tessera.effect_kind", "pure") != "pure" or raw:
        raise ValueError(f"Apple GPU native {kind} has unsupported keyword(s) {sorted(raw)}")
    return AppleGPUDomainRequest(
        graph_op=op.op_name, kind=kind, symbol=symbol, abi=abi, framework=framework,
        input_names=inputs, output_name=outputs[0], shapes=tuple(shapes), semantics=semantics,
        scalars=scalars,
    )


def supports_scheduled_apple_gpu_domain(module: GraphIRModule, *, target: str = TARGET) -> bool:
    """Whether the native Apple GPU domain contract admits ``module``."""
    if target != TARGET:
        return False
    try:
        admit(module)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ScheduledAppleGPUDomainArtifact:
    """One replayable Graph -> Schedule -> Tile -> Target Apple GPU domain kernel."""

    target: str
    architecture: str
    kind: str
    graph_op: str
    symbol: str
    input_names: tuple[str, ...]
    output_name: str
    shapes: tuple[tuple[int, ...], ...]
    semantics: dict[str, Any]
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str
    schedule_digest: str

    def validate(self) -> None:
        if (self.target, self.architecture) != (TARGET, ARCHITECTURE):
            raise ValueError("Apple GPU native domain artifact names another target")
        spec = APPLE_GPU_DOMAIN_OPS.get(self.graph_op)
        if spec is None or spec[0] != self.kind:
            raise ValueError("Apple GPU native domain artifact kind disagrees with its op")
        if self.symbol not in {APPLE_GPU_DOMAIN_SYMBOLS[self.kind][0]} | (
                {_PPO_EXTENDED[0]} if self.kind == "ppo_policy_loss" else set()):
            raise ValueError("Apple GPU native domain artifact symbol disagrees with its op")
        if not spec[1] <= len(self.input_names) <= spec[2] or len(self.shapes) != len(self.input_names) + 1:
            raise ValueError("Apple GPU native domain artifact arity disagrees with its op")
        if re.fullmatch(r"[0-9a-f]{64}", self.schedule_digest) is None:
            raise ValueError("Apple GPU native domain artifact requires a schedule digest")


def _value_lowering_flag() -> str:
    return "--tile-to-apple_gpu=value=true"


def _boundaries(tool: Any, graph_ir: str) -> tuple[str, str, str]:
    from .scheduled_matmul import run_tessera_opt

    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    target_ir = run_tessera_opt(tool, tile_ir, _value_lowering_flag())
    return schedule_ir, tile_ir, target_ir


def _production_tool() -> Any:
    """Production ``tessera-opt`` that registers the Apple GPU value lowering."""
    from .scheduled_matmul import find_tessera_opt

    tool = find_tessera_opt()
    if tool is None or not apple_gpu_value_lowering_available(str(tool)):
        raise RuntimeError(
            "Apple GPU native domain kernel requires production tessera-opt with the Apple GPU "
            "value lowering (--tile-to-apple_gpu=value=true); "
            + ("no tessera-opt was found" if tool is None
               else f"{tool} does not register it (build with TESSERA_BUILD_APPLE_BACKEND=ON)"))
    return tool


_LOWERING_PROBES: dict[str, bool] = {}


def apple_gpu_value_lowering_available(tool: str) -> bool:
    """Whether ``tool`` registers ``--tile-to-apple_gpu=value=true``.

    A ``tessera-opt`` built without the Apple backend parses Graph IR but
    cannot produce ``tessera_apple.gpu.kernel_call``, so these families are not
    packageable with it. Probed once per binary rather than assumed.
    """
    from pathlib import Path

    from .scheduled_matmul import run_tessera_opt

    if tool not in _LOWERING_PROBES:
        probe = ('module {\n  func.func @f(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2xf32> {\n'
                 '    %0 = tile.ebm_energy_quadratic %a, %b {source = "tessera.ebm.energy_quadratic"} : '
                 '(tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2xf32>\n'
                 '    return %0 : tensor<2xf32>\n  }\n}\n')
        try:
            text = run_tessera_opt(Path(tool), probe, _value_lowering_flag())
            _LOWERING_PROBES[tool] = "tessera_apple.gpu.kernel_call" in text
        except (OSError, RuntimeError):
            _LOWERING_PROBES[tool] = False
    return _LOWERING_PROBES[tool]


def lowering_available() -> bool:
    """Whether this host can produce the compiled route at all."""
    from .scheduled_matmul import find_tessera_opt

    tool = find_tessera_opt()
    return tool is not None and apple_gpu_value_lowering_available(str(tool))


def lower_scheduled_apple_gpu_domain(
    module: GraphIRModule, *, target: str = TARGET,
) -> ScheduledAppleGPUDomainArtifact:
    """Graph -> Schedule -> Tile -> Target for one admitted request."""
    if target != TARGET:
        raise ValueError("Apple GPU native domain kernel lowers for apple_gpu only")
    request = admit(module)
    tool = _production_tool()
    targeted = copy.deepcopy(module)
    targeted.module_attrs["tessera.target"] = f'"{TARGET}"'
    targeted.module_attrs["tessera.arch"] = f'"{ARCHITECTURE}"'
    targeted.module_attrs["tessera.launch_bindings"] = json.dumps(
        list(request.input_names + (request.output_name,)))
    graph_ir = targeted.to_mlir(target=TARGET, canonical=True)
    schedule_ir, tile_ir, target_ir = _boundaries(tool, graph_ir)
    digests = re.findall(r'schedule\.artifact .*?hash = "([0-9a-f]{64})"', schedule_ir)
    if len(digests) != 1:
        raise ValueError("Apple GPU native domain kernel Schedule requires exactly one artifact record")
    return ScheduledAppleGPUDomainArtifact(
        target=TARGET, architecture=ARCHITECTURE, kind=request.kind, graph_op=request.graph_op,
        symbol=request.symbol, input_names=request.input_names, output_name=request.output_name,
        shapes=request.shapes, semantics=dict(request.semantics), graph_ir=graph_ir,
        schedule_ir=schedule_ir, tile_ir=tile_ir, target_ir=target_ir, schedule_digest=digests[0],
    )


def _float_token(text: str, ty: str) -> float:
    """A printed MLIR float, including the hex form MLIR uses when a decimal
    spelling would not round-trip."""
    if text.startswith("0x"):
        bits = int(text, 16)
        if ty == "f64":
            return struct.unpack("<d", bits.to_bytes(8, "little"))[0]
        if ty == "f32":
            return struct.unpack("<f", bits.to_bytes(4, "little"))[0]
        raise ValueError(f"Apple GPU native domain contract has an unsupported hex float type {ty!r}")
    return float(text)


def _attr_mapping(text: str, start: int) -> dict[str, Any]:
    """Parse the printed attribute dictionary that opens just before ``start``."""
    from .native_x86_kernel import _AttrParser

    class _Parser(_AttrParser):
        def value(self) -> Any:
            token = self._peek()
            if token["num"] is not None and token["num"].startswith("0x") and (token["ty"] or "").startswith("f"):
                self._next()
                return _float_token(token["num"], token["ty"])
            return super().value()

    parser = _Parser(text)
    parser.pos = start
    return parser.mapping()


def _parse_contract(tile_ir: str) -> dict[str, Any]:
    marker = "tessera.apple_gpu_domain_contract = {"
    if tile_ir.count(marker) != 1:
        raise ValueError("Apple GPU native domain Tile artifact requires one tessera.apple_gpu_domain_contract")
    return _attr_mapping(tile_ir, tile_ir.index(marker) + len(marker))


@dataclass(frozen=True)
class AppleGPUDomainProjection:
    kind: str
    symbol: str
    abi: str
    framework: str
    bindings: tuple[str, ...]
    inputs: int
    shapes: tuple[tuple[int, ...], ...]
    scalars: dict[str, int]
    semantics: dict[str, Any]
    schedule_digest: str


def verify_apple_gpu_domain_projection(artifact: ScheduledAppleGPUDomainArtifact) -> AppleGPUDomainProjection:
    """Replay every boundary and project the serialized contract.

    Each stored stage must be byte-equal to the production compiler's replay of
    the stage before it, and every descriptor field the artifact carries must
    equal the contract the native Schedule serialized. The Target IR must hold
    exactly one verified ``tessera_apple.gpu.kernel_call`` whose symbol, ABI,
    framework, semantic values and schedule digest are the contract's, and
    which carries no semantic key the contract does not.
    """
    artifact.validate()
    tool = _production_tool()
    schedule_ir, tile_ir, target_ir = _boundaries(tool, artifact.graph_ir)
    if schedule_ir != artifact.schedule_ir:
        raise ValueError("Apple GPU native domain Schedule disagrees with Graph replay")
    if tile_ir != artifact.tile_ir:
        raise ValueError("Apple GPU native domain Tile disagrees with Schedule replay")
    if target_ir != artifact.target_ir:
        raise ValueError("Apple GPU native domain Target IR disagrees with Tile replay")
    header = re.match(r'\s*module attributes \{([^{}]*)\}', artifact.tile_ir)
    if header is None or any(
        re.search(r'(?:^|,)\s*' + re.escape(key) + r' = "' + value + r'"(?:,|$)', header[1]) is None
        for key, value in (("tessera.target", TARGET), ("tessera.arch", ARCHITECTURE))
    ):
        raise ValueError("Apple GPU native domain parent target disagrees")
    contract = _parse_contract(artifact.tile_ir)
    if re.findall(r'tessera\.schedule_hash = "([0-9a-f]{64})"', artifact.tile_ir) != [artifact.schedule_digest]:
        raise ValueError("Apple GPU native domain Tile artifact has a stale schedule digest")
    tile_ops = re.findall(r"(?m)^\s*(?:%\S+ = )?(tile\.[A-Za-z_0-9]+)", artifact.tile_ir)
    if tile_ops != [f"tile.{artifact.kind}"] or artifact.graph_op + " " in artifact.tile_ir:
        raise ValueError("Apple GPU native domain Tile artifact requires exactly its one value op")
    bindings = contract.get("bindings")
    shapes = tuple(tuple(shape) for shape in contract.get("shapes", ()))
    semantics = contract.get("semantics")
    scalars = contract.get("scalars")
    if (not isinstance(bindings, list) or not isinstance(semantics, dict)
            or not isinstance(scalars, dict) or contract.get("layout") != "row_major"
            or contract.get("storage") != ["f32"] * len(shapes)):
        raise ValueError("Apple GPU native domain contract is malformed")
    if (contract.get("kind") != artifact.kind or contract.get("graph_op") != artifact.graph_op
            or contract.get("symbol") != artifact.symbol
            or tuple(bindings) != artifact.input_names + (artifact.output_name,)
            or contract.get("inputs") != len(artifact.input_names)
            or shapes != artifact.shapes or semantics != artifact.semantics
            or set(semantics) != _SEMANTIC_KEYS[artifact.kind]):
        raise ValueError("Apple GPU native domain descriptor fields disagree with native IR")
    abi, framework = str(contract.get("abi", "")), str(contract.get("framework", ""))
    expected_symbol = (_PPO_EXTENDED if artifact.kind == "ppo_policy_loss" and any(
        semantics[k] for k in ("has_mask", "has_ref_kl", "has_entropy")) else APPLE_GPU_DOMAIN_SYMBOLS[artifact.kind])
    if (artifact.symbol, abi, framework) != expected_symbol:
        raise ValueError("Apple GPU native domain contract names another delegate")
    calls = re.findall(r"(?m)^\s*(?:%\S+ = )?(tessera_apple\.[A-Za-z_.0-9]+)(.*)$", artifact.target_ir)
    if len(calls) != 1 or calls[0][0] != "tessera_apple.gpu.kernel_call" or "tile." in artifact.target_ir:
        raise ValueError("Apple GPU native domain Target IR requires exactly one tessera_apple.gpu.kernel_call")
    open_at = calls[0][1].find("{")
    if open_at < 0:
        raise ValueError("Apple GPU native domain Target IR call carries no attributes")
    attrs = _attr_mapping(calls[0][1], open_at + 1)
    expected = {"op_kind": artifact.kind, "symbol": artifact.symbol, "abi": abi, "status": "executable",
                "framework": framework, "tessera.schedule_hash": artifact.schedule_digest, **semantics}
    if attrs != expected:
        raise ValueError("Apple GPU native domain Target IR call disagrees with its contract")
    if any(not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= _INT32_MAX
           for value in scalars.values()):
        raise ValueError("Apple GPU native domain ABI scalars must be positive int32")
    return AppleGPUDomainProjection(
        kind=artifact.kind, symbol=artifact.symbol, abi=abi, framework=framework, bindings=tuple(bindings),
        inputs=len(artifact.input_names), shapes=shapes, scalars=dict(scalars), semantics=dict(semantics),
        schedule_digest=artifact.schedule_digest,
    )


__all__ = [
    "APPLE_GPU_DOMAIN_OPS", "APPLE_GPU_DOMAIN_SYMBOLS", "AppleGPUDomainProjection", "AppleGPUDomainRequest",
    "ScheduledAppleGPUDomainArtifact", "admit", "apple_gpu_value_lowering_available",
    "lower_scheduled_apple_gpu_domain", "lowering_available", "supports_scheduled_apple_gpu_domain",
    "verify_apple_gpu_domain_projection",
]
