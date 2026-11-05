"""E2E-REAL-6, Apple GPU PPO / EBM / Clifford value families: retired
Graph-owned packaging vs the compiled route.

The generic value branch of ``apple_native.package_native`` used to read the
Python Graph object and build the MSL / MPSGraph launch descriptor for
``rl.ppo_policy_loss``, ``ebm.{energy_quadratic, langevin_step, refinement,
partition_exact}`` and ``clifford.geometric_product``. They now lower through
the native Graph -> Schedule -> Tile boundaries (``NativeAppleGPUDomain.h``)
and the Apple GPU value lowering, and package the replayed artifact, reaching
the C ABI through a verified ``tessera_apple.gpu.kernel_call``. The frozen
constructor in ``tests/_support/apple_gpu_domain_baseline.py`` is the declared
oracle (Decision #31(a)); these tests are its differential.

Host-free half: every descriptor/ABI field the old route produced is
reproduced over the envelope both admit; what the old route admitted and
computed wrongly (or not at all) is now refused by name; a tampered artifact
fails closed before the runtime is touched; and a host whose ``tessera-opt``
lacks the Apple GPU value lowering fails closed naming it. Device half (the
M1 Max): both images run on the same inputs and must agree bit-for-bit, and
with a float64 reference.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tessera.compiler import apple_native, scheduled_apple_gpu_domain as native
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests._support import apple_gpu_domain_baseline as baseline

PIPELINE = "tessera-lower-to-apple_gpu"
ROOT = Path(__file__).resolve().parents[2]
_ELEMENT = {"fp32": "f32", "fp16": "f16"}

#: The values the delegate reads when the retired call carried no keyword
#: (``runtime._dispatch_gpu_*``).
_DELEGATE_DEFAULTS = {
    "clip_epsilon": 0.2, "kl_coef": 0.0, "entropy_coef": 0.0, "reduction": None,
    "has_mask": False, "has_ref_kl": False, "has_entropy": False, "noise_scale": 0.0,
    "temperature": 1.0, "p": 3, "q": 0,
}

_needs_compiler = pytest.mark.skipif(
    not native.lowering_available(),
    reason="requires production tessera-opt with the Apple GPU value lowering",
)


def _type(shape, dtype: str = "fp32") -> IRType:
    text = f"tensor<{'x'.join(map(str, shape))}x{_ELEMENT[dtype]}>" if shape else f"tensor<{_ELEMENT[dtype]}>"
    return IRType(text, tuple(map(str, shape)), dtype)


def _module(op_name, inputs, result, *, dtype="fp32", kwargs=None) -> GraphIRModule:
    args = [IRArg(f"a{i}", _type(shape, dtype)) for i, shape in enumerate(inputs)]
    out = _type(result, dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name="apple_gpu_domain", args=args, result_types=[out],
        body=[IROp(result="out", op_name=op_name, operands=[f"%{a.name}" for a in args],
                   operand_types=[str(a.ir_type) for a in args], result_type=str(out),
                   kwargs=kwargs or {})],
        return_values=["%out"],
    )])


def _ppo(n_inputs, shape=(4, 5), **kwargs):
    return _module("tessera.rl.ppo_policy_loss", [shape] * n_inputs, (), kwargs=kwargs)


def _energy(b, d):
    return _module("tessera.ebm.energy_quadratic", [(b, d), (b, d)], (b,))


def _langevin(shape, **kwargs):
    return _module("tessera.ebm.langevin_step", [shape] * 3, shape, kwargs=kwargs)


def _refine(shape, **kwargs):
    return _module("tessera.ebm.refinement", [shape] * 2, shape, kwargs=kwargs)


def _partition(shape, **kwargs):
    return _module("tessera.ebm.partition_exact", [shape], (), kwargs=kwargs)


def _clifford(shape, **kwargs):
    return _module("tessera.clifford.geometric_product", [shape] * 2, shape,
                   kwargs={"p": 3, "q": 0, **kwargs})


_ADMITTED = {
    "ppo-strict": _ppo(3, clip_epsilon=0.2, reduction="mean"),
    "ppo-defaults": _ppo(3, shape=(7,)),
    "ppo-mask": _ppo(4, clip_epsilon=0.1),
    "ppo-mask-ref": _ppo(5, clip_epsilon=0.25, kl_coef=0.05),
    "ppo-all": _ppo(6, clip_epsilon=0.2, reduction="mean", kl_coef=0.03, entropy_coef=0.02),
    "energy-2x3": _energy(2, 3), "energy-9x17": _energy(9, 17),
    "langevin": _langevin((2, 3), eta=0.125, noise_scale=0.3),
    "langevin-no-scale": _langevin((5, 4, 3), eta=0.01),
    "langevin-hex-eta": _langevin((6,), eta=0.123456789, noise_scale=0.7),
    "refine": _refine((2, 3), eta=0.125, steps=4),
    "refine-1step": _refine((11,), eta=0.5, steps=1),
    "partition": _partition((2, 3), temperature=0.75, reduction="logsumexp"),
    "partition-defaults": _partition((13,)),
    "clifford-2": _clifford((2, 8)), "clifford-3x5": _clifford((3, 5, 8)),
    "clifford-layout": _clifford((4, 8), coefficient_layout="blade_last", signature=[1, 1, 1]),
}


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    library = tmp_path / "libTesseraAppleRuntime.dylib"
    library.write_bytes(b"apple-gpu-domain-migration-fixture")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: library)
    return library


# ---------------------------------------------------------------------------
# Differential: the compiled route reproduces the retired descriptor
# ---------------------------------------------------------------------------


@_needs_compiler
@pytest.mark.parametrize("module", list(_ADMITTED.values()), ids=list(_ADMITTED))
def test_compiled_route_reproduces_retired_descriptor(module, fake_runtime):
    assert baseline.retired_contract(module)
    old = baseline.package(module, pipeline_name=PIPELINE)
    new = apple_native.package_native(module, pipeline_name=PIPELINE)
    for field in ("entry_symbol", "abi_id", "buffers", "shape_guards", "scalars", "geometry", "ordering"):
        assert getattr(new.descriptor, field) == getattr(old.descriptor, field), field
    for key in ("route", "op_kind"):
        assert new.descriptor.provenance[key] == old.descriptor.provenance[key], key
    old_call, new_call = old.descriptor.provenance["value_call"], new.descriptor.provenance["value_call"]
    for key in ("op", "op_kind", "symbol", "status"):
        assert new_call[key] == old_call[key], key
    # The retired call spread whatever keywords the Graph op carried; the new
    # one materializes every value the delegate reads, so a default is written
    # rather than assumed. They agree on the value the runtime reads.
    for key in (set(old_call) | set(new_call)) - {"op", "op_kind", "symbol", "status"}:
        if key in {"has_noise", "signature"} or (key == "coefficient_layout" and key not in old_call):
            continue  # read by no delegate (Tile-verifier carriage / refused unless Euclidean)
        fallback = _DELEGATE_DEFAULTS.get(key)
        retired = old_call.get(key, fallback)
        if key == "reduction" and retired is None:
            retired = new_call[key]  # the delegate's own default spelling
        assert new_call[key] == retired, key
    assert new.image.entry_points == old.image.entry_points
    assert new.image.payload == old.image.payload
    assert (new.image.target, new.image.architecture, new.image.binary_format) == (
        old.image.target, old.image.architecture, old.image.binary_format)
    # The new route is the scheduled one, and says so.
    provenance = new.descriptor.provenance
    kind = provenance["op_kind"]
    assert provenance["work_item"] == "E2E-REAL-6"
    assert provenance["target_ir_op"] == "tessera_apple.gpu.kernel_call"
    digest = provenance["schedule_digest"]
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    graph_op = module.functions[0].body[0].op_name
    assert f"{graph_op} " not in new.tile_ir and f"tile.{kind} " in new.tile_ir
    assert new.target_ir.count("tessera_apple.gpu.kernel_call") == 1
    assert f'tessera.schedule_hash = "{digest}"' in new.target_ir
    assert f'symbol = "{new.descriptor.entry_symbol}"' in new.target_ir


@_needs_compiler
@pytest.mark.parametrize("name", ["ppo-all", "energy-2x3", "langevin", "refine", "partition", "clifford-2"])
def test_driver_routes_the_families_through_the_scheduled_package(name, fake_runtime, monkeypatch):
    from tessera.compiler.driver import compile_graph_module

    from tessera.compiler.driver import canonical_compile_options

    monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(fake_runtime))
    # The canonical wrapper claims the family where the compiled route exists.
    assert canonical_compile_options(_ADMITTED[name], target="apple_gpu")["package_native"] is True
    bundle = compile_graph_module(
        _ADMITTED[name], source_origin="apple-gpu-domain-migration-driver", target="apple_gpu",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["work_item"] == "E2E-REAL-6"
    assert bundle.launch_descriptor.provenance["schedule_digest"]
    assert "tessera_apple.gpu.kernel_call" in bundle.target_ir.text


def test_ppo_side_tensors_follow_their_graph_segments(fake_runtime):
    """The retired constructor inferred the side tensors from the operand count;
    a Graph op whose one side tensor is ``ref_logp`` was bound as ``mask``.
    The native contract reads the ODS segments."""
    module = _ppo(4, kl_coef=0.05)
    module.functions[0].body[0].attrs = "operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>"
    request = native.admit(module)
    assert (request.semantics["has_mask"], request.semantics["has_ref_kl"]) == (False, True)
    assert baseline.package(module, pipeline_name=PIPELINE).descriptor.provenance["value_call"]["has_mask"]
    if native.lowering_available():
        package = apple_native.package_native(module, pipeline_name=PIPELINE)
        call = package.descriptor.provenance["value_call"]
        assert (call["has_mask"], call["has_ref_kl"]) == (False, True)


# ---------------------------------------------------------------------------
# What the retired route admitted and the native contract refuses, by name
# ---------------------------------------------------------------------------

_NARROWED = {
    # The delegates compute one formula each; these select another, which the
    # retired route passed straight through and the delegate ignored.
    "langevin-temperature": _langevin((2, 3), eta=0.1, temperature=2.0),
    "refine-temperature": _refine((2, 3), eta=0.1, steps=2, temperature=0.0),
    "partition-sum": _partition((5,), reduction="sum"),
    "clifford-blade-major": _clifford((2, 8), coefficient_layout="blade_major"),
    "clifford-signature": _clifford((2, 8), signature=[1, -1, 1]),
    "ppo-entropy-without-tensor": _ppo(3, entropy_coef=0.1),
    "ppo-kl-without-ref": _ppo(4, kl_coef=0.1),
    "ppo-sum": _ppo(3, reduction="sum"),
    # A Langevin step without its noise tensor has no delegate.
    "langevin-two-operands": _module("tessera.ebm.langevin_step", [(2, 3)] * 2, (2, 3), kwargs={"eta": 0.1}),
    # The runtime refused these at launch; now the compiler does.
    "energy-rank3": _module("tessera.ebm.energy_quadratic", [(2, 3, 4)] * 2, (2,)),
    "energy-wrong-result": _module("tessera.ebm.energy_quadratic", [(2, 3)] * 2, (3,)),
    "clifford-not-cl30": _clifford((2, 4), p=2),
    "langevin-zero-eta": _langevin((2, 3), eta=0.0),
    "partition-vector-result": _module("tessera.ebm.partition_exact", [(2, 3)], (2,)),
    # Keywords the delegate does not compute are refused, not ignored, and a
    # float semantic is an F64Attr, not an integer.
    "policy-keyword": _module("tessera.ebm.energy_quadratic", [(2, 3)] * 2, (2,),
                              kwargs={"numeric_policy": "fp32"}),
    "integer-eta": _refine((2, 3), eta=1, steps=2),
}


@pytest.mark.parametrize("module", list(_NARROWED.values()), ids=list(_NARROWED))
def test_native_contract_refuses_what_the_retired_route_admitted(module):
    assert baseline.retired_contract(module)
    assert not native.supports_scheduled_apple_gpu_domain(module)
    assert apple_native.native_package_kind(module) is None
    with pytest.raises(ValueError):
        native.admit(module)


@pytest.mark.parametrize("module", [
    _module("tessera.ebm.energy_quadratic", [(2, 3)] * 2, (2,), dtype="fp16"),
    _module("tessera.clifford.geometric_product", [(2, 8)] * 2, (2, 8), dtype="fp16", kwargs={"p": 3, "q": 0}),
], ids=["f16-energy", "f16-clifford"])
def test_neither_route_admits_non_f32_storage(module, fake_runtime):
    assert not native.supports_scheduled_apple_gpu_domain(module)
    with pytest.raises(ValueError):
        baseline.package(module, pipeline_name=PIPELINE)


def test_admission_names_apple_gpu_only():
    module = _energy(2, 3)
    assert native.supports_scheduled_apple_gpu_domain(module, target="apple_gpu")
    for target in ("apple_cpu", "x86", "rocm_gfx1151", "nvidia_sm120"):
        assert not native.supports_scheduled_apple_gpu_domain(module, target=target)
    with pytest.raises(ValueError, match="apple_gpu only"):
        native.lower_scheduled_apple_gpu_domain(module, target="apple_cpu")


def test_classifier_keeps_the_descriptor_kinds():
    expected = {
        "ppo-all": "value_rl_ppo_policy_loss", "energy-2x3": "value_ebm_energy_quadratic",
        "langevin": "value_ebm_langevin_step", "refine": "value_ebm_refinement",
        "partition": "value_ebm_partition_exact", "clifford-2": "value_clifford_geometric_product",
    }
    for name, kind in expected.items():
        assert apple_native.native_package_kind(_ADMITTED[name]) == kind, name


def test_missing_value_lowering_fails_closed_naming_tessera_opt(monkeypatch, fake_runtime):
    """A host whose tessera-opt lacks the Apple GPU value lowering cannot
    produce the compiled route: the driver does not claim the family, and an
    explicit request fails closed, naming the tool -- it never falls back to
    the retired Graph-owned constructor."""
    from tessera.compiler.driver import canonical_compile_options

    monkeypatch.setattr(native, "apple_gpu_value_lowering_available", lambda tool: False)
    module = _ADMITTED["langevin"]
    assert apple_native.native_package_kind(module) == "value_ebm_langevin_step"
    assert not apple_native.native_package_available(module)
    assert canonical_compile_options(module, target="apple_gpu")["package_native"] is False
    with pytest.raises(RuntimeError, match=r"tessera-opt.*tile-to-apple_gpu=value=true"):
        apple_native.package_native(module, pipeline_name=PIPELINE)
    # Families outside this contract do not consult the probe. (Transpose was
    # the example until it moved onto the descriptor contract, which needs the
    # same lowering; a static softmax packages from the semantic kernel.)
    softmax = _module("tessera.softmax", [(4, 8)], (4, 8))
    assert apple_native.native_package_kind(softmax) == "softmax"
    assert apple_native.native_package_available(softmax)


# ---------------------------------------------------------------------------
# Tampering fails closed before the runtime is touched
# ---------------------------------------------------------------------------


def _no_runtime(monkeypatch):
    monkeypatch.setattr(apple_native, "_runtime_library_path",
                        lambda: pytest.fail("runtime touched before rejection"))


def _package(artifact):
    return apple_native.package_scheduled_apple_gpu_domain(artifact, pipeline_name=PIPELINE)


@_needs_compiler
@pytest.mark.parametrize("field,value", [
    ("kind", "ebm_refinement"), ("graph_op", "tessera.ebm.refinement"),
    ("symbol", "tessera_apple_gpu_ebm_refinement_value_f32"),
    ("input_names", ("a1", "a0", "a2")), ("output_name", "other"),
    ("shapes", ((2, 4), (2, 4), (2, 4), (2, 4))),
    ("semantics", {"eta": 0.25, "has_noise": True, "noise_scale": 0.3}),
    ("semantics", {"eta": 0.125, "has_noise": True, "noise_scale": 0.3, "temperature": 1.0}),
    ("schedule_digest", "0" * 64), ("architecture", "apple8"), ("target", "apple_cpu"),
])
def test_descriptor_field_tampering_fails_closed(field, value, monkeypatch):
    artifact = native.lower_scheduled_apple_gpu_domain(_ADMITTED["langevin"])
    _no_runtime(monkeypatch)
    with pytest.raises(ValueError):
        _package(replace(artifact, **{field: value}))


@_needs_compiler
@pytest.mark.parametrize("stage,old,new", [
    ("graph_ir", "tensor<2x3xf32>", "tensor<2x4xf32>"),
    ("graph_ir", "eta = 0.125", "eta = 0.25"),
    ("schedule_ir", "eta = 1.250000e-01 : f64, has_noise", "eta = 2.500000e-01 : f64, has_noise"),
    ("schedule_ir", "N = 6 : i64", "N = 7 : i64"),
    ("schedule_ir", 'symbol = "tessera_apple_gpu_ebm_langevin_step_value_f32"',
     'symbol = "tessera_apple_gpu_ebm_refinement_value_f32"'),
    ("tile_ir", "noise_scale = 3.000000e-01 : f64, source", "noise_scale = 9.000000e-01 : f64, source"),
    ("tile_ir", 'abi = "msl_ebm_langevin_step_value_f32"', 'abi = "msl_ebm_refinement_value_f32"'),
    ("target_ir", 'symbol = "tessera_apple_gpu_ebm_langevin_step_value_f32"',
     'symbol = "tessera_apple_gpu_ebm_refinement_value_f32"'),
    ("target_ir", "noise_scale = 3.000000e-01 : f64", "noise_scale = 0.000000e+00 : f64"),
])
def test_stage_tampering_fails_closed(stage, old, new, monkeypatch):
    artifact = native.lower_scheduled_apple_gpu_domain(_ADMITTED["langevin"])
    text = getattr(artifact, stage)
    assert old in text, (stage, old)
    _no_runtime(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        _package(replace(artifact, **{stage: text.replace(old, new)}))


@_needs_compiler
def test_schedule_to_tile_refuses_an_altered_contract():
    """A Schedule record edited without its hash is refused by the native
    Schedule -> Tile boundary itself, before any Target IR exists."""
    artifact = native.lower_scheduled_apple_gpu_domain(_ADMITTED["langevin"])
    tampered = artifact.schedule_ir.replace("N = 6 : i64", "N = 7 : i64")
    assert tampered != artifact.schedule_ir
    with pytest.raises(RuntimeError, match="contract was altered"):
        run_tessera_opt(find_tessera_opt(), tampered, "--tessera-schedule-to-tile")


# ---------------------------------------------------------------------------
# Native compiler owns the admission; the tables agree
# ---------------------------------------------------------------------------


def _graph_text(body: str, args: str, results: str, *, bindings: str, arch: str = "apple7") -> str:
    return (
        f'module attributes {{tessera.arch = "{arch}", tessera.launch_bindings = [{bindings}], '
        f'tessera.target = "apple_gpu"}} {{\n'
        f"  func.func @f({args}) -> {results} {{\n    {body}\n  }}\n}}\n"
    )


_ENERGY = ("%e = tessera.ebm.energy_quadratic %x, %y : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2xf32>\n"
           "    return %e : tensor<2xf32>")
_ENERGY_ARGS = "%x: tensor<2x3xf32>, %y: tensor<2x3xf32>"


@_needs_compiler
@pytest.mark.parametrize("text", [
    _graph_text(_ENERGY, _ENERGY_ARGS, "tensor<2xf32>", bindings='"x", "x", "e"'),
    _graph_text(_ENERGY, _ENERGY_ARGS, "tensor<2xf32>", bindings='"x", "y"'),
    _graph_text(_ENERGY, _ENERGY_ARGS, "tensor<2xf32>", bindings='"x", "y", "e"', arch="apple8"),
    _graph_text("%e = tessera.ebm.energy_quadratic %x, %x : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2xf32>\n"
                "    return %e : tensor<2xf32>", "%x: tensor<2x3xf32>", "tensor<2xf32>", bindings='"x", "e"'),
    _graph_text("%e = tessera.ebm.energy_quadratic %x, %y : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2xf32>\n"
                "    %z = tessera.ebm.partition_exact %e : (tensor<2xf32>) -> tensor<f32>\n"
                "    return %z : tensor<f32>", _ENERGY_ARGS, "tensor<f32>", bindings='"x", "y", "z"'),
], ids=["aliased-bindings", "missing-binding", "wrong-arch", "repeated-operand", "two-ops"])
def test_native_graph_to_schedule_refuses_outside_the_envelope(text):
    with pytest.raises(RuntimeError, match="Apple GPU native domain kernel"):
        run_tessera_opt(find_tessera_opt(), text, "--tessera-graph-to-schedule")


def test_contract_tables_agree_across_python_and_cpp():
    header = (ROOT / "src/compiler/programming_model/lib/NativeAppleGPUDomain.h").read_text()
    cpp = {op: (kind, int(low), int(high)) for op, kind, low, high in re.findall(
        r'\{"(tessera\.[\w.]+)", "(\w+)", (\d), (\d)\}', header)}
    assert cpp == native.APPLE_GPU_DOMAIN_OPS
    for kind, (symbol, abi, framework) in {**native.APPLE_GPU_DOMAIN_SYMBOLS,
                                           "ppo_ex": native._PPO_EXTENDED}.items():
        assert f'"{symbol}"' in header and f'"{abi}"' in header, kind
    # The value lowering picks the Target symbol and routine on its own; it
    # must name the same entries (TileToApple.cpp, GPU value mode).
    lowering = (ROOT / "src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Lowering/"
                "TileToApple.cpp").read_text()
    for symbol, abi, _ in (*native.APPLE_GPU_DOMAIN_SYMBOLS.values(), native._PPO_EXTENDED):
        assert f'"{symbol}"' in lowering and f'"{abi}"' in lowering, symbol


def test_retired_branch_no_longer_owns_the_families():
    """The Graph-owned value branch no longer names any of these families."""
    assert not set(getattr(apple_native, "_VALUE_SYMBOLS", {})) & set(native.APPLE_GPU_DOMAIN_OPS)
    source = (ROOT / "python/tessera/compiler/apple_native.py").read_text()
    assert "ppo_policy_loss_ex_f32" not in source


# ---------------------------------------------------------------------------
# Census: the prune dashboard sees the routes
# ---------------------------------------------------------------------------


def test_prune_census_sees_the_compiled_routes():
    from tessera.compiler import bootstrap_prune_audit

    rows = {(t, f): (route, status) for t, f, route, status in bootstrap_prune_audit.family_rows()}
    for family in ("value_rl_ppo_policy_loss", "value_ebm_energy_quadratic", "value_ebm_langevin_step",
                   "value_ebm_refinement", "value_ebm_partition_exact", "value_clifford_geometric_product"):
        assert rows[("apple_gpu", family)] == (
            "scheduled_apple_gpu_domain.supports_scheduled_apple_gpu_domain", "compiled"), family
    # The per-target declaration must not leak to the Apple CPU rows.
    assert not any(t == "apple_cpu" and r.startswith("scheduled_apple_gpu_domain")
                   for (t, _), (r, _) in rows.items())


# ---------------------------------------------------------------------------
# Device differential (M1 Max, Metal)
# ---------------------------------------------------------------------------


def _inputs(name: str, module: GraphIRModule) -> list[np.ndarray]:
    rng = np.random.default_rng(1104 + len(name))
    shapes = [tuple(int(d) for d in a.ir_type.shape) for a in module.functions[0].args]
    values = [rng.standard_normal(shape).astype(np.float32) for shape in shapes]
    if name.startswith("ppo") and len(values) >= 4:
        values[3] = (rng.random(shapes[3]) > 0.3).astype(np.float32)  # a 0/1 mask
    return values


def _clifford_reference(a, b):
    # Bitmask blade order: index bit k set <=> e_(k+1) present (e12 is 3).
    blades = tuple(tuple(k + 1 for k in range(3) if i >> k & 1) for i in range(8))
    index = {blade: i for i, blade in enumerate(blades)}

    def product(x, y):
        sign, items = 1, list(x) + list(y)
        for i in range(len(items)):  # bubble sort, counting swaps
            for j in range(len(items) - 1 - i):
                if items[j] > items[j + 1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
                    sign = -sign
        out = []
        for item in items:  # e_i e_i = +1 in cl(3,0)
            if out and out[-1] == item:
                out.pop()
            else:
                out.append(item)
        return sign, index[tuple(out)]

    a, b = a.astype(np.float64), b.astype(np.float64)
    out = np.zeros_like(a)
    for i, x in enumerate(blades):
        for j, y in enumerate(blades):
            sign, k = product(x, y)
            out[..., k] += sign * a[..., i] * b[..., j]
    return out


def _reference(name, module, values):
    op = module.functions[0].body[0]
    kw = {k: v for k, v in op.kwargs.items() if v is not None}
    x = [v.astype(np.float64) for v in values]
    if name.startswith("ppo"):
        new, old, adv = x[:3]
        has_mask, has_ref, has_entropy = (len(x) >= 4, len(x) >= 5, len(x) >= 6)
        clip = kw.get("clip_epsilon", 0.2)
        ratio = np.exp(new - old)
        loss = -np.minimum(ratio * adv, np.clip(ratio, 1 - clip, 1 + clip) * adv)
        if has_ref:
            delta = x[4] - new
            loss = loss + kw.get("kl_coef", 0.0) * (np.exp(delta) - delta - 1.0)
        if has_entropy:
            loss = loss - kw.get("entropy_coef", 0.0) * x[5]
        mask = x[3] if has_mask else np.ones_like(new)
        return np.sum(loss * mask) / max(float(np.sum(mask)), 1.0)
    if name.startswith("energy"):
        return 0.5 * np.sum((x[0] - x[1]) ** 2, axis=1)
    if name.startswith("langevin"):
        return x[0] - kw["eta"] * x[1] + kw.get("noise_scale", 0.0) * x[2]
    if name.startswith("refine"):
        return x[0] - kw["steps"] * kw["eta"] * x[1]
    if name.startswith("partition"):
        # The delegate returns the partition function Z (the runtime's
        # ``_dispatch_gpu_ebm_partition_exact`` and APPLE-E2E-1's reference).
        return np.sum(np.exp(-x[0] / kw.get("temperature", 1.0)))
    return _clifford_reference(x[0], x[1])


def _run(package, values):
    from tessera.runtime import RuntimeArtifact, launch

    artifact = RuntimeArtifact(
        metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
        target_ir=package.target_ir, native_image=package.image, launch_descriptor=package.descriptor,
    )
    inputs = [b for b in package.descriptor.buffers if b.direction == "input"]
    output = next(b for b in package.descriptor.buffers if b.direction == "output")
    shape = tuple(g.value for g in package.descriptor.shape_guards if g.binding == output.name)
    args = {binding.name: value for binding, value in zip(inputs, values)}
    args[output.name] = np.full(shape, np.nan, dtype=np.float32)
    launched = launch(artifact, args)
    assert launched["ok"], launched
    return np.array(launched["output"], dtype=np.float32, copy=True)


@pytest.mark.hardware_apple_gpu
@_needs_compiler
@pytest.mark.parametrize("name", list(_ADMITTED))
def test_compiled_and_retired_images_agree_bit_for_bit_on_metal(name):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    # On an Apple chip the runtime dylib must build; its absence is a failure.
    assert apple_native.tools_available()
    module = _ADMITTED[name]
    values = _inputs(name, module)
    new = _run(apple_native.package_native(module, pipeline_name=PIPELINE), values)
    old = _run(baseline.package(module, pipeline_name=PIPELINE), values)
    assert new.shape == old.shape
    assert np.array_equal(new.view(np.uint32), old.view(np.uint32))
    for _ in range(2):  # replay reuses the packaged route
        assert np.array_equal(_run(apple_native.package_native(module, pipeline_name=PIPELINE), values)
                              .view(np.uint32), new.view(np.uint32))
    np.testing.assert_allclose(new, _reference(name, module, values), rtol=2e-4, atol=2e-5)
