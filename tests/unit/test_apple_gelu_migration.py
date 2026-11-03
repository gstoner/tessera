"""E2E-REAL-6, Apple GPU GELU: retired Graph-owned packaging vs the compiled route.

``apple_native.package_gelu`` used to read the Python Graph object, run
``--tessera-gelu-to-apple_gpu`` (Graph straight to the Target ``func.call``)
and build the launch descriptor from that text. Static GELU now lowers through
the native Graph -> Schedule -> Tile activation route (``scheduled_kernel``)
and packages the replayed artifact. The frozen constructor in
``tests/_support/apple_gelu_baseline.py`` is the declared oracle (Decision
#31(a)); these tests are its differential.

Host-free half: every descriptor/ABI field the old route produced is
reproduced by the compiled route over the envelope the old contract admitted,
everything it refused is still refused, and a tampered artifact fails closed
before the runtime is touched. Device half (``hardware_apple_gpu``, the M1 Max):
both images run on the same inputs and must agree bit-for-bit, and with a
float64 tanh-GELU oracle.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from tessera.compiler import apple_native, scheduled_kernel
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests._support import apple_gelu_baseline as baseline

PIPELINE = "tessera-lower-to-apple_gpu"
_needs_compiler = pytest.mark.skipif(find_tessera_opt() is None, reason="requires production tessera-opt")


#: The retired route's Graph->Target pass lives in the Apple backend, which
#: the Linux fleet builds do not configure; the oracle needs it, the
#: production route does not.
_ORACLE_PASS = "tessera-gelu-to-apple_gpu"
_ELEMENT = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}
_SHAPES = [(1, 1), (5, 11), (3, 17), (64, 129)]


def _type(shape, dtype: str) -> IRType:
    dims = "x".join(map(str, shape))
    return IRType(f"tensor<{dims}x{_ELEMENT[dtype]}>", tuple(map(str, shape)), dtype)


def _module(shape, dtype="fp32", *, out_shape=None, out_dtype=None, kwargs=None) -> GraphIRModule:
    in_type = _type(shape, dtype)
    out_type = _type(out_shape or shape, out_dtype or dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name="gelu_migration", args=[IRArg("a0", in_type)], result_types=[out_type],
        body=[IROp(result="out", op_name="tessera.gelu", operands=["%a0"],
                   operand_types=[str(in_type)], result_type=str(out_type), kwargs=kwargs or {})],
        return_values=["%out"],
    )])


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    library = tmp_path / "libTesseraAppleRuntime.dylib"
    library.write_bytes(b"apple-gelu-migration-fixture")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: library)
    return library


@pytest.fixture
def oracle_tool(compiler_toolchain, monkeypatch):
    """A tessera-opt that registers the retired route's pass, else skip.

    Both routes are pinned to that one binary so the differential compares
    the two lowerings, not two builds.
    """
    tool = compiler_toolchain.require_tessera_opt(_ORACLE_PASS)
    monkeypatch.setenv("TESSERA_OPT", str(tool))
    return tool


# ---------------------------------------------------------------------------
# Host-free differential
# ---------------------------------------------------------------------------


@_needs_compiler
@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("shape", _SHAPES)
def test_compiled_route_reproduces_retired_descriptor(dtype, shape, fake_runtime, oracle_tool):
    module = _module(shape, dtype)
    old = baseline.package_gelu(module, pipeline_name=PIPELINE)
    new = apple_native.package_gelu(module, pipeline_name=PIPELINE)
    for field in ("entry_symbol", "abi_id", "buffers", "shape_guards", "scalars",
                  "geometry", "ordering"):
        assert getattr(new.descriptor, field) == getattr(old.descriptor, field), field
    for key in ("route", "op_kind", "shape", "storage", "dynamic_shape",
                "scalar_contract", "accumulation"):
        assert new.descriptor.provenance[key] == old.descriptor.provenance[key], key
    assert new.image.entry_points == old.image.entry_points
    assert new.image.payload == old.image.payload
    # The new route is the scheduled activation, and says so.
    provenance = new.descriptor.provenance
    assert provenance["work_item"] == "E2E-REAL-6"
    assert provenance["family"] == "activation" and provenance["approximation"] == "tanh"
    assert provenance["synthesized"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["schedule_digest"])
    assert "tile.activation_kernel" in new.tile_ir and "tessera.gelu" not in new.tile_ir
    # The runtime MSL kernel is a declared delegate, never a generated body.
    assert new.target_ir.startswith(f"tessera_apple.gpu.kernel_call @{new.descriptor.entry_symbol} ")
    assert 'implementation = "runtime_msl"' in new.target_ir
    assert "msl_kernel" not in new.target_ir


@_needs_compiler
def test_driver_routes_static_gelu_through_scheduled_activation(fake_runtime, monkeypatch):
    from tessera.compiler.driver import compile_graph_module

    monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(fake_runtime))
    bundle = compile_graph_module(
        _module((5, 11)), source_origin="apple-gelu-migration-driver", target="apple_gpu",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.abi_id == apple_native.APPLE_GELU_F32_ABI
    assert bundle.launch_descriptor.provenance["work_item"] == "E2E-REAL-6"
    assert bundle.launch_descriptor.provenance["family"] == "activation"


_REFUSED = [
    _module((5, 11), "fp16", out_dtype="bf16"),
    _module((2, 5, 11)),
    _module((55,)),
    _module((5, 11), out_shape=(11, 5)),
    _module((5, 11), kwargs={"approximate": "none"}),
    _module((5, 11), kwargs={"approximate": "tanh"}),
]
_REFUSED_IDS = ["mixed-storage", "rank3", "rank1", "transposed-result", "erf-request", "named-tanh"]


@_needs_compiler
@pytest.mark.parametrize("module", _REFUSED, ids=_REFUSED_IDS)
def test_compiled_route_refuses_outside_the_retired_envelope(module, fake_runtime):
    # The retired regex admitted no attribute beyond the pure effect, so a
    # named approximation (even "tanh") was refused; the compiled contract
    # keeps that boundary rather than guessing what the keyword means.
    assert not scheduled_kernel.supports_scheduled_kernel(module, target="apple_gpu")
    with pytest.raises(ValueError):
        apple_native.package_gelu(module, pipeline_name=PIPELINE)


@pytest.mark.parametrize("module", _REFUSED, ids=_REFUSED_IDS)
def test_retired_route_refused_the_same_envelope(module, fake_runtime, oracle_tool):
    with pytest.raises(ValueError):
        baseline.package_gelu(module, pipeline_name=PIPELINE)


def test_gelu_is_not_a_scheduled_activation_on_other_targets():
    # Evidence never transfers: the semantic activation contract admits GELU
    # on apple7 only. (x86 serves GELU through its own native elementwise
    # contract, ``native_x86_kernel``, which this slice does not touch.)
    for target in ("nvidia_sm120", "rocm_gfx1151", "rocm_gfx1201", "x86"):
        with pytest.raises(ValueError):
            scheduled_kernel._graph_contract(_module((5, 11)), target)
    for target in ("nvidia_sm120", "rocm_gfx1151", "rocm_gfx1201"):
        assert not scheduled_kernel.supports_scheduled_kernel(_module((5, 11)), target=target)


@_needs_compiler
def test_dynamic_gelu_keeps_its_route_and_static_cannot_reenter_it(fake_runtime, oracle_tool):
    with pytest.raises(ValueError, match="dynamic rank-two contract"):
        apple_native.lower_gelu_artifact(_module((5, 11)))
    # A static artifact built by the oracle is refused by the production
    # dynamic packager: it can no longer serve the retired static route.
    with pytest.raises(ValueError, match="replayed Schedule artifact"):
        apple_native.package_gelu_artifact(baseline.lower_gelu_artifact(_module((5, 11))),
                                           pipeline_name=PIPELINE)


# ---------------------------------------------------------------------------
# IR-derived ABI: tampering fails closed before the runtime is touched
# ---------------------------------------------------------------------------


@_needs_compiler
@pytest.mark.parametrize("field,value", [
    ("rows", 6), ("columns", 12), ("input_shape", (5, 12)), ("output_shape", (5, 12)),
    ("dtype", "fp16"), ("storage", "f16"), ("kind", "relu"), ("workgroup_size", 128),
    ("input_name", "out"), ("function_name", "other"),
])
def test_descriptor_field_tampering_fails_closed(field, value, monkeypatch):
    artifact = scheduled_kernel.lower_scheduled_kernel(_module((5, 11)), target="apple_gpu")
    monkeypatch.setattr(apple_native, "_runtime_library_path",
                        lambda: pytest.fail("runtime touched before rejection"))
    with pytest.raises(ValueError):
        apple_native.package_scheduled_kernel(replace(artifact, **{field: value}),
                                              pipeline_name=PIPELINE)


@_needs_compiler
@pytest.mark.parametrize("old,new", [
    ('approximation = "tanh"', 'approximation = "erf"'),
    ('kind = "gelu"', 'kind = "relu"'),
    ('zero_mode = "preserve"', 'zero_mode = "positive"'),
    ('workgroup_size = 1 : i64', 'workgroup_size = 2 : i64'),
    ('arch = "apple7"', 'arch = "apple8"'),
])
def test_schedule_tampering_fails_closed(old, new, monkeypatch):
    artifact = scheduled_kernel.lower_scheduled_kernel(_module((5, 11)), target="apple_gpu")
    line = next(l for l in artifact.schedule_ir.splitlines() if "schedule.activation" in l)
    assert old in line
    tampered = artifact.schedule_ir.replace(line, line.replace(old, new))
    monkeypatch.setattr(apple_native, "_runtime_library_path",
                        lambda: pytest.fail("runtime touched before rejection"))
    with pytest.raises((ValueError, RuntimeError)):
        apple_native.package_scheduled_kernel(replace(artifact, schedule_ir=tampered),
                                              pipeline_name=PIPELINE)


@_needs_compiler
def test_tile_tampering_fails_closed(monkeypatch):
    artifact = scheduled_kernel.lower_scheduled_kernel(_module((5, 11)), target="apple_gpu")
    tampered = artifact.tile_ir.replace("%c11_i64 = arith.constant 11", "%c11_i64 = arith.constant 12")
    assert tampered != artifact.tile_ir
    monkeypatch.setattr(apple_native, "_runtime_library_path",
                        lambda: pytest.fail("runtime touched before rejection"))
    with pytest.raises(ValueError, match="replay"):
        apple_native.package_scheduled_kernel(replace(artifact, tile_ir=tampered),
                                              pipeline_name=PIPELINE)


# ---------------------------------------------------------------------------
# Native verifiers own the contract, not the Python wrapper
# ---------------------------------------------------------------------------


def _graph_text(element: str = "f32", shape: str = "5x11", attrs: str = "") -> str:
    attr = f" {{{attrs}}}" if attrs else ""
    return (
        'module attributes {tessera.arch = "apple7", tessera.target = "apple_gpu"} {\n'
        f"  func.func @f(%x: tensor<{shape}x{element}>) -> tensor<{shape}x{element}> {{\n"
        f"    %y = tessera.gelu %x{attr} : (tensor<{shape}x{element}>) -> tensor<{shape}x{element}>\n"
        f"    return %y : tensor<{shape}x{element}>\n  }}\n}}\n"
    )


@_needs_compiler
@pytest.mark.parametrize("text", [
    _graph_text(shape="2x5x11"),
    _graph_text(attrs='approximate = "none"'),
    _graph_text(attrs='tessera.effect_kind = "memory"'),
], ids=["rank3", "erf", "impure"])
def test_native_graph_to_schedule_refuses_outside_the_envelope(text):
    with pytest.raises(RuntimeError, match="semantic-kernel contract"):
        run_tessera_opt(find_tessera_opt(), text, "--tessera-graph-to-schedule")


@_needs_compiler
def test_native_graph_to_schedule_claims_only_an_isolated_gelu():
    composite = (
        'module attributes {tessera.arch = "apple7", tessera.target = "apple_gpu"} {\n'
        "  func.func @f(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {\n"
        "    %y = tessera.gelu %x : (tensor<5x11xf32>) -> tensor<5x11xf32>\n"
        "    %z = tessera.gelu %y : (tensor<5x11xf32>) -> tensor<5x11xf32>\n"
        "    return %z : tensor<5x11xf32>\n  }\n}\n"
    )
    scheduled = run_tessera_opt(find_tessera_opt(), composite, "--tessera-graph-to-schedule")
    assert "schedule.activation" not in scheduled


@_needs_compiler
def test_native_schedule_verifier_rejects_a_hand_built_erf_activation():
    artifact = scheduled_kernel.lower_scheduled_kernel(_module((5, 11)), target="apple_gpu")
    tampered = artifact.schedule_ir.replace('approximation = "tanh"', 'approximation = "erf"', 1)
    with pytest.raises(RuntimeError, match="tanh GELU"):
        run_tessera_opt(find_tessera_opt(), tampered, "--canonicalize")


# ---------------------------------------------------------------------------
# Census: the prune dashboard sees a real route
# ---------------------------------------------------------------------------


def test_prune_census_sees_the_generic_compiled_route():
    from tessera.compiler import bootstrap_prune_audit

    rows = {(t, f): status for t, f, _, status in bootstrap_prune_audit.family_rows()}
    assert rows[("apple_gpu", "gelu")] == "generic_compiled"
    assert rows[("apple_gpu", "dynamic_gelu")] == "gap"


# ---------------------------------------------------------------------------
# Device differential (M1 Max)
# ---------------------------------------------------------------------------


def _tanh_gelu(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


@pytest.mark.hardware_apple_gpu
@_needs_compiler
@pytest.mark.parametrize("dtype,tolerance", [("fp32", 2e-5), ("fp16", 4e-3), ("bf16", 2e-2)])
@pytest.mark.parametrize("shape", [(5, 11), (64, 129), (257, 1025)])
def test_compiled_and_retired_images_agree_bit_for_bit_on_metal(dtype, tolerance, shape, oracle_tool):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    from ml_dtypes import bfloat16

    from tessera.runtime import RuntimeArtifact, launch

    storage = {"fp32": np.float32, "fp16": np.float16, "bf16": bfloat16}[dtype]
    rng = np.random.default_rng(1103)
    x = (rng.standard_normal(shape) * 3.0).astype(storage)
    # Exercise the signed zero and the large-|x| saturation on both sides.
    x.flat[0], x.flat[1], x.flat[2] = -0.0, 12.0, -12.0
    module = _module(shape, dtype)
    outputs = []
    for package in (baseline.package_gelu(module, pipeline_name=PIPELINE),
                    apple_native.package_gelu(module, pipeline_name=PIPELINE)):
        artifact = RuntimeArtifact(
            metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
            target_ir=package.target_ir, native_image=package.image,
            launch_descriptor=package.descriptor,
        )
        launched = launch(artifact, {"a0": x, "out": np.empty_like(x)})
        assert launched["ok"], launched
        assert launched["execution_kind"] == "native_gpu"
        outputs.append(np.asarray(launched["output"]))
    old, new = outputs
    assert old.dtype == new.dtype == np.dtype(storage)
    assert old.tobytes() == new.tobytes()
    np.testing.assert_allclose(new.astype(np.float64), _tanh_gelu(x), rtol=tolerance, atol=tolerance)
    assert np.signbit(new.flat[0]) and new.flat[0] == 0
