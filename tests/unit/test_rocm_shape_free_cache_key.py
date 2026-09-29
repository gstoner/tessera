"""Shape-free kernel identity for the ROCm scheduled softmax/reduction images.

FOUNDATION-BATCH-2-2026-09-27 (follow-up from PR #870). The compiled
Graph -> Schedule -> Tile route replays a Schedule record whose digest binds the
static shape, and the Tile function carries the Graph symbol, so keying the
image on Tile text made every new shape a cold compile of an identical binary.
``rocm_native._compile_shape_free_tile_ir`` keys and compiles the image from
the Target IR directive alone (``_shape_free_target_ir``), named by a symbol
derived from that directive.

Host-free half: the projection keeps every directive attribute and drops only
host scaffolding; two shapes / two Graph symbols share one compile and one
image; any binary-affecting change (directive attribute, arch, compiler binary)
misses; unaudited Target IR fails closed. Device half (exact gfx1151 with
``TESSERA_ROCM_E2E_DEVICE_TEST=1``; exact gfx1201 with
``TESSERA_GFX1201_DEVICE_PROOF=1``): the real compiler produces one image for
two shapes and two symbols, both launches are numerically right, and a
storage/kind change produces a different image.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from tessera.compiler import rocm_native, scheduled_kernel
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

PIPELINE = "tessera-lower-to-rocm"
_ELEMENT = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}

_HEADER = (
    'module attributes {tessera.arch = "gfx1151", tessera.ir.version = "1.0", '
    'tessera.pipeline.arch = "gfx1151", tessera.pipeline.backend_codegen = "rocdl_hsaco", '
    'tessera.pipeline.family = "reduction", tessera.pipeline.output = "target", '
    'tessera.pipeline.schema = "tessera.executable_pipeline.v1", '
    'tessera.pipeline.target_ir_consumer = "tessera_rocm", '
    'tessera.pipeline.tile_producer = "content_addressed_tile", tessera.target = "rocm"} {'
)

_REDUCE_ATTRS = {
    "accum": '"f32"', "arch": '"gfx1151"', "axis": "1 : i64", "dtype": '"bf16"',
    "keepdims": "false", "kind": '"mean"', "layout": '"outer_axis_inner"',
    "nan_mode": '"propagate"', "output_dtype": '"f32"', "schedule": '"workgroup_256"',
    "source": '"tile.reduce_kernel"',
}


def _directive(attrs: dict[str, str], name: str) -> str:
    items = dict(attrs, name=f'"{name}"')
    return "tessera_rocm.reduce {" + ", ".join(f"{key} = {items[key]}" for key in sorted(items)) + "}"


def _target(shape: tuple[int, int, int], *, symbol: str = "gfx1151_unary", attrs=None,
            header: str = _HEADER, extra: str = "") -> str:
    """A per-shape Target IR module in the form TileToROCM prints (Princess-Luna, 2026-09-27)."""
    outer, axis, inner = shape
    src, dst = f"memref<{outer}x{axis}x{inner}xbf16>", f"memref<{outer}x{inner}xf32>"
    tensor_src, tensor_dst = f"tensor<{outer}x{axis}x{inner}xbf16>", f"tensor<{outer}x{inner}xf32>"
    return f"""{header}
  func.func @{symbol}(%arg0: {tensor_src}) -> {tensor_dst} {{
    %0 = bufferization.to_buffer %arg0 : {tensor_src} to {src}
    %intptr = memref.extract_aligned_pointer_as_index %0 : {src} -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %alloc = memref.alloc() : {dst}
    %intptr_0 = memref.extract_aligned_pointer_as_index %alloc : {dst} -> index
    %3 = arith.index_cast %intptr_0 : index to i64
    %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
    %c{outer}_i64 = arith.constant {outer} : i64
    %c{axis}_i64 = arith.constant {axis} : i64
    %c{inner}_i64 = arith.constant {inner} : i64
    {_directive(attrs or _REDUCE_ATTRS, symbol)}{extra}
    %5 = bufferization.to_tensor %alloc : {dst} to {tensor_dst}
    return %5 : {tensor_dst}
  }}
}}
"""


def _project(target_ir: str) -> str:
    return rocm_native._shape_free_target_ir(target_ir, family="reduction", directive="tessera_rocm.reduce")


# --- the projection -----------------------------------------------------------------------------


def test_projection_is_independent_of_shape_and_graph_symbol() -> None:
    first = _project(_target((2, 3, 17)))
    assert _project(_target((2, 8, 64))) == first
    assert _project(_target((4, 257, 1), symbol="some_other_graph_function")) == first
    # Only the header and the one directive survive; no extent or Graph symbol does.
    assert first.count("\n") == 3 and "func.func" not in first and "arith.constant" not in first
    assert "gfx1151_unary" not in first and "257" not in first
    symbol = rocm_native._directive_symbol(first, "tessera_rocm.reduce")
    assert symbol.startswith("tessera_rocm_reduction_") and len(symbol) == len("tessera_rocm_reduction_") + 16


@pytest.mark.parametrize("key,value", [
    ("dtype", '"f16"'), ("kind", '"max"'), ("axis", "0 : i64"), ("keepdims", "true"),
    ("arch", '"gfx1201"'), ("nan_mode", '"ignore"'), ("accum", '"f16"'), ("schedule", '"serial"'),
])
def test_every_directive_attribute_is_in_the_identity(key, value) -> None:
    base = _project(_target((2, 3, 17)))
    changed = _project(_target((2, 3, 17), attrs=dict(_REDUCE_ATTRS, **{key: value})))
    assert changed != base
    assert rocm_native._directive_symbol(changed, "tessera_rocm.reduce") != (
        rocm_native._directive_symbol(base, "tessera_rocm.reduce"))


def test_an_added_directive_attribute_is_in_the_identity() -> None:
    """``inner_is_one`` selects a different kernel body and is present only when true."""
    base = _project(_target((2, 3, 1)))
    assert _project(_target((2, 3, 1), attrs=dict(_REDUCE_ATTRS, inner_is_one="true"))) != base


def test_the_module_header_is_in_the_identity() -> None:
    base = _project(_target((2, 3, 17)))
    assert _project(_target((2, 3, 17), header=_HEADER.replace('tessera.arch = "gfx1151"',
                                                               'tessera.arch = "gfx1201"'))) != base


@pytest.mark.parametrize("extra,match", [
    ("\n    %9 = arith.addi %1, %1 : i64", "unaudited Target IR operation 'arith.addi'"),
    ("\n    gpu.module @k {\n    }", "unaudited Target IR operation 'gpu.module'"),
    ("\n    " + _directive(_REDUCE_ATTRS, "second"), "exactly one tessera_rocm.reduce directive"),
    ("\n    tessera_rocm.softmax {name = \"s\"}", "exactly one tessera_rocm.reduce directive"),
])
def test_unaudited_target_ir_fails_closed(extra, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        _project(_target((2, 3, 17), extra=extra))


def test_directive_without_exactly_one_name_fails_closed() -> None:
    nameless = _target((2, 3, 17)).replace('name = "gfx1151_unary", ', "")
    with pytest.raises(RuntimeError, match="exactly one kernel name"):
        _project(nameless)
    with pytest.raises(RuntimeError, match="exactly one named"):
        rocm_native._directive_symbol('module {\n  tessera_rocm.reduce {kind = "sum"}\n}\n',
                                      "tessera_rocm.reduce")


def test_a_family_without_an_audited_identity_is_refused() -> None:
    with pytest.raises(ValueError, match="no audited shape-free kernel identity"):
        rocm_native._compile_shape_free_tile_ir("module {}", family="binary", architecture="gfx1151")


# --- the cache ----------------------------------------------------------------------------------


class _FakeCompiler:
    """Stand-in for ``tessera-opt``: Tile->Target returns the per-shape Target IR
    the test put in the "Tile" text; the binary step records one compile and
    returns an HSACO that is a function of the module it was given."""

    def __init__(self) -> None:
        self.binary_inputs: list[str] = []
        self.target_runs = 0

    def run_opt(self, _tool: Path, source: str, pipeline: str) -> str:
        if "output=target" in pipeline:
            self.target_runs += 1
            return source
        assert "input=directive" in pipeline, pipeline
        self.binary_inputs.append(source)
        return "gpu.binary @" + hashlib.sha256(source.encode()).hexdigest()

    @staticmethod
    def extract(text: str) -> bytes:
        return b"\x7fELF" + text.encode()


@pytest.fixture
def fake_compiler(monkeypatch, tmp_path):
    tool = tmp_path / "tessera-opt"
    tool.write_bytes(b"compiler build 1")
    fake = _FakeCompiler()
    monkeypatch.setattr(rocm_native, "_cache", {})
    monkeypatch.setattr(rocm_native, "_shape_free_targets", {})
    monkeypatch.setattr(rocm_native, "_tessera_opt", lambda: tool)
    monkeypatch.setattr(rocm_native, "_run_opt", fake.run_opt)
    monkeypatch.setattr(rocm_native, "_extract_hsaco", fake.extract)
    monkeypatch.setattr(rocm_native, "_driver_selected_device_libraries", lambda **_kw: ())
    monkeypatch.setattr(rocm_native, "_version_fingerprint", lambda _tool: "fingerprint")
    monkeypatch.setattr(rocm_native, "_rocm_clang", lambda _path: None)
    monkeypatch.setattr(rocm_native, "warn_if_generator_is_stale", lambda _tool=None: None)
    fake.tool = tool
    return fake


def _compile(target_ir: str, architecture: str = "gfx1151"):
    return rocm_native._compile_shape_free_tile_ir(target_ir, family="reduction", architecture=architecture)


def test_two_shapes_share_one_compile_and_one_image(fake_compiler) -> None:
    cold = _compile(_target((2, 3, 17)))
    warm = _compile(_target((2, 8, 64)))
    assert cold[-1] == "cold" and warm[-1] == "warm_cache"
    assert len(fake_compiler.binary_inputs) == 1
    assert warm[:6] == cold[:6]  # target IR, backend IR, payload, fingerprints, libraries
    # Each new Tile text still goes through the compiler's Tile -> Target.
    assert fake_compiler.target_runs == 3  # two per-shape runs + one directive-level run
    # The binary was compiled from exactly the Target IR the image binds.
    assert fake_compiler.binary_inputs[0] == _project(_target((2, 3, 17)))


def test_a_graph_symbol_only_difference_reuses_the_image(fake_compiler) -> None:
    first = _compile(_target((2, 3, 17), symbol="caller_a"))
    second = _compile(_target((2, 3, 17), symbol="caller_b"))
    assert second[-1] == "warm_cache" and second[2] == first[2]
    assert len(fake_compiler.binary_inputs) == 1


def test_an_exact_repeat_skips_the_target_run(fake_compiler) -> None:
    _compile(_target((2, 3, 17)))
    runs = fake_compiler.target_runs
    assert _compile(_target((2, 3, 17)))[-1] == "warm_cache"
    assert fake_compiler.target_runs == runs


@pytest.mark.parametrize("key,value", [("dtype", '"f16"'), ("kind", '"max"'), ("keepdims", "true")])
def test_a_binary_affecting_directive_change_compiles_again(fake_compiler, key, value) -> None:
    base = _compile(_target((2, 3, 17)))
    changed = _compile(_target((2, 3, 17), attrs=dict(_REDUCE_ATTRS, **{key: value})))
    assert changed[-1] == "cold" and changed[2] != base[2]
    assert len(fake_compiler.binary_inputs) == 2


def test_a_different_architecture_compiles_again(fake_compiler) -> None:
    _compile(_target((2, 3, 17)))
    assert _compile(_target((2, 3, 17)), architecture="gfx1201")[-1] == "cold"
    assert len(fake_compiler.binary_inputs) == 2


def test_a_rebuilt_compiler_misses(fake_compiler) -> None:
    """Decision #11: the compiler binary is in the key, and a rebuild is seen."""
    _compile(_target((2, 3, 17)))
    fake_compiler.tool.write_bytes(b"compiler build 2 -- a different size")
    assert _compile(_target((2, 8, 64)))[-1] == "cold"
    assert len(fake_compiler.binary_inputs) == 2


def test_tool_digest_is_the_binary_content(tmp_path) -> None:
    tool = tmp_path / "tessera-opt"
    tool.write_bytes(b"abc")
    assert rocm_native._tool_digest(tool) == hashlib.sha256(b"abc").hexdigest()
    tool.write_bytes(b"abcd")
    assert rocm_native._tool_digest(tool) == hashlib.sha256(b"abcd").hexdigest()


# --- exact device -------------------------------------------------------------------------------


def _module(name: str, op_name: str, shape, dtype, out_shape, out_dtype, kwargs) -> GraphIRModule:
    def ir_type(dims, element):
        text = "x".join(map(str, dims))
        return IRType(f"tensor<{text}x{_ELEMENT[element]}>", tuple(map(str, dims)), element)

    source, result = ir_type(shape, dtype), ir_type(out_shape, out_dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name=name, args=[IRArg("x", source)], result_types=[result],
        body=[IROp(result="o", op_name=op_name, operands=["%x"], operand_types=[str(source)],
                   result_type=str(result), kwargs=dict(kwargs))],
        return_values=["%o"],
    )])


def _live_arch() -> str | None:
    if os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") == "1":
        return "gfx1201"
    if os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") == "1":
        return "gfx1151"
    return None


_device = pytest.mark.skipif(
    _live_arch() is None,
    reason="set TESSERA_ROCM_E2E_DEVICE_TEST=1 (gfx1151) or TESSERA_GFX1201_DEVICE_PROOF=1 (gfx1201) "
           "on the exact owning host",
)


def _package(module: GraphIRModule, arch: str):
    artifact = scheduled_kernel.lower_scheduled_kernel(module, target=f"rocm_{arch}")
    return artifact, rocm_native.package_scheduled_kernel(artifact, pipeline_name=PIPELINE)


def _launch(artifact, package, x: np.ndarray) -> np.ndarray:
    from tessera import runtime as rt

    output = np.zeros(artifact.output_shape, np.float32 if artifact.family == "reduce" else x.dtype)
    runtime = rt.RuntimeArtifact(
        metadata={"target": package.image.target}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    scalars = ({"Rows": artifact.rows, "K": artifact.columns} if artifact.family == "softmax"
               else {"Outer": artifact.outer, "AxisExtent": artifact.axis_extent, "Inner": artifact.inner})
    result = rt.launch(runtime, {"buffers": {"x": x, "o": output}, "scalars": scalars})
    assert result["ok"] and result["execution_kind"] == "native_gpu", result.get("reason")
    return output


@_device
@pytest.mark.hardware_rocm
@pytest.mark.parametrize("family", ["softmax", "reduce"])
def test_exact_device_new_shape_and_symbol_reuse_one_image(family) -> None:
    from tessera import runtime as rt

    arch = _live_arch()
    assert rt._rocm_live_arch() == arch, "requires the exact owning device"
    rocm_native._cache.clear()
    rocm_native._shape_free_targets.clear()

    def build(name, shape):
        if family == "softmax":
            return _module(name, "tessera.softmax", shape, "fp32", shape, "fp32", {"axis": -1})
        out = (shape[0],) + shape[2:]
        return _module(name, "tessera.mean", shape, "fp32", out, "fp32", {"axis": 1, "keepdims": False})

    shapes = [(3, 17), (5, 300)] if family == "softmax" else [(2, 3, 5), (4, 33, 7)]
    first_art, first = _package(build("caller_a", shapes[0]), arch)
    second_art, second = _package(build("caller_b", shapes[1]), arch)
    assert first.image.compile_state == "cold"
    assert second.image.compile_state == "warm_cache"
    assert second.image.payload == first.image.payload
    assert second.image.image_digest == first.image.image_digest
    assert second.descriptor.entry_symbol == first.descriptor.entry_symbol
    assert first.descriptor.provenance["graph_symbol"] == "caller_a"
    assert second.descriptor.provenance["graph_symbol"] == "caller_b"
    assert first.descriptor.shape_guards != second.descriptor.shape_guards

    rng = np.random.default_rng(1201)
    for artifact, package, shape in ((first_art, first, shapes[0]), (second_art, second, shapes[1])):
        x = rng.normal(size=shape).astype(np.float32)
        got = _launch(artifact, package, x)
        if family == "softmax":
            e = np.exp(x.astype(np.float64) - x.max(axis=-1, keepdims=True))
            expected = e / e.sum(axis=-1, keepdims=True)
        else:
            expected = x.astype(np.float64).mean(axis=1)
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-6)

    # A binary-affecting change is a new image, not a reuse.
    changed = (_module("caller_a", "tessera.sum", (2, 3, 5), "fp32", (2, 5), "fp32", {"axis": 1})
               if family == "reduce" else None)
    if changed is None and arch == "gfx1151":
        changed = _module("caller_a", "tessera.softmax", shapes[0], "fp16", shapes[0], "fp16", {"axis": -1})
    if changed is not None:
        _, other = _package(changed, arch)
        assert other.image.compile_state == "cold"
        assert other.image.payload != first.image.payload
        assert other.descriptor.entry_symbol != first.descriptor.entry_symbol


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("architecture", ["gfx1151", "gfx1201"])
def test_scheduled_attention_shape_and_symbol_reuse_one_image(architecture):
    gate = "TESSERA_ROCM_E2E_DEVICE_TEST" if architecture == "gfx1151" else "TESSERA_GFX1201_DEVICE_PROOF"
    if os.environ.get(gate) != "1":
        pytest.skip(f"set {gate}=1 on the exact owning device")
    from tessera import runtime as rt
    from tessera.compiler import scheduled_attention
    from tessera.compiler.attention_contract import reference_streaming_attention
    from tests.unit.test_scheduled_attention_consumers import _module

    assert rt._rocm_live_arch() == architecture
    rocm_native._cache.clear()
    rocm_native._shape_free_targets.clear()
    packages = []
    for index, (b, hq, hkv, sq, sk) in enumerate(((1, 4, 2, 17, 19), (2, 6, 3, 23, 37))):
        module = _module(target="rocm", query_rows=sq, rocm_dims=(b, hq, hkv, sk))
        module.functions[0].name = f"attention_shape_{index}"
        artifact = scheduled_attention.lower_scheduled_attention(module, target=f"rocm_{architecture}")
        package = rocm_native.package_scheduled_attention(artifact, pipeline_name=PIPELINE)
        packages.append(package)
        assert package.image.compile_state == ("cold" if index == 0 else "warm_cache")
        rng = np.random.default_rng(120 + index)
        q, k, v = [(rng.normal(size=shape) * 0.2).astype(np.float16)
                   for shape in ((b, hq, sq, 64), (b, hkv, sk, 64), (b, hkv, sk, 64))]
        out = np.full((b, hq, sq, 64), np.nan, np.float32)
        runtime = rt.RuntimeArtifact(metadata={"target": f"rocm_{architecture}"},
            native_image=package.image, launch_descriptor=package.descriptor,
            tile_ir=package.tile_ir, target_ir=package.target_ir)
        result = rt.launch(runtime, dict(q=q, k=k, v=v, o=out, Sq=sq, Sk=sk,
                         Scale=0.125, Causal=1, Hq=hq, KvRatio=hq // hkv, Window=64))
        assert result["ok"] and result["execution_kind"] == "native_gpu", result
        expected = reference_streaming_attention(q, k, v, block_size=16, scale=0.125,
                                                causal=True, window_left=64, window_right=0)
        np.testing.assert_allclose(out, expected, rtol=3e-2, atol=3e-2)
    assert packages[0].image.image_digest == packages[1].image.image_digest
    assert packages[0].tile_ir != packages[1].tile_ir
    assert packages[0].descriptor.shape_guards != packages[1].descriptor.shape_guards
    assert packages[0].descriptor.provenance["schedule_digest"] != packages[1].descriptor.provenance["schedule_digest"]
    changed = _module(target="rocm", bias=True)
    artifact = scheduled_attention.lower_scheduled_attention(changed, target=f"rocm_{architecture}")
    biased = rocm_native.package_scheduled_attention(artifact, pipeline_name=PIPELINE)
    assert biased.image.compile_state == "cold"
    assert biased.image.image_digest != packages[0].image.image_digest
