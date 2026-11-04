"""E2E-REAL-6, ROCm MoE dispatch: retired Graph-authored packaging vs the Tile projection.

``rocm_native.package_moe_dispatch`` used to take its kernel from the native
Graph -> Schedule -> Tile replay but author every launch field -- buffers,
ABI, scalar list, shape guards, geometry -- from the Python Graph object.
``package_scheduled_moe_dispatch`` now projects them from the replayed Tile
``tessera.native_contract``. The frozen packager in
``tests/_support/rocm_moe_dispatch_baseline.py`` is the declared oracle
(Decision #31(a)); these tests are its differential.

Host-free half: over the admitted envelope the projected descriptor equals the
retired one field for field, refusals are unchanged, and a tampered artifact
fails closed before target compilation. Device half
(``TESSERA_ROCM_MOVEMENT_DEVICE_PROOF=1`` with ``TESSERA_ROCM_CHIP=gfx1151``
on Princess-Luna): both packages build the same image and their outputs agree
bit-for-bit and with a NumPy oracle. The native owner admits gfx1151 only, so
gfx1201 has no device row here; its refusal is checked host-free.
"""

from __future__ import annotations

import copy
import os
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.rocm.benchmark_rocm_e2e_movement import _moe_module
from tessera.compiler import rocm_native, scheduled_moe_dispatch
from tessera.compiler.graph_ir import IRType
from tessera.compiler.native_artifact import DeviceLibraryRecord
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support import rocm_moe_dispatch_baseline as baseline

PIPELINE = "tessera-lower-to-rocm"
ARCH = "gfx1151"
_needs_compiler = pytest.mark.skipif(find_tessera_opt() is None, reason="requires production tessera-opt")

#: (tokens, slots, hidden): single slot, more slots than tokens (repeats),
#: fewer slots than tokens, a non-power-of-two width, and a wide row.
EXTENTS = ((1, 1, 1), (7, 9, 13), (11, 5, 17), (3, 2, 2), (64, 128, 256))


def _recording_compile(calls: list[str]):
    def compile_(tile_ir: str, **_kwargs):
        calls.append(tile_ir)
        return ('module {\n  tessera_rocm.moe_dispatch {name = "tessera_rocm_moe_fake", route = "direct_gather"}\n}',
                "backend", b"\x7fELFrocm-moe", "compiler", "toolchain",
                (DeviceLibraryRecord("rocm.ocml", "1" * 64, "compiler_driver"),), "cold")
    return compile_


def _assert_same_launch_contract(old, new) -> None:
    for field in ("abi_id", "entry_symbol", "buffers", "scalars", "shape_guards", "geometry", "ordering"):
        assert getattr(old.descriptor, field) == getattr(new.descriptor, field), field
    # Every retired provenance key survives with the same value; the only new
    # key names the consumer, so a regression in either direction is visible.
    assert {k: v for k, v in new.descriptor.provenance.items() if k != "consumer"} == old.descriptor.provenance
    assert new.descriptor.provenance["consumer"] == "native_schedule_tile_projection"
    assert "consumer" not in old.descriptor.provenance
    assert new.image.target == old.image.target == f"rocm_{ARCH}"
    assert new.image.entry_points == old.image.entry_points
    assert new.tile_ir == old.tile_ir


def _swapped_operands(extent):
    """Graph arguments in (token, x) order; operands still name x then token."""
    module = _moe_module(*extent)
    module.functions[0].args.reverse()
    return module


@_needs_compiler
@pytest.mark.parametrize("swapped", [False, True], ids=["x_first", "token_first"])
@pytest.mark.parametrize("extent", EXTENTS)
def test_projected_descriptor_reproduces_retired_contract(monkeypatch, extent, swapped) -> None:
    calls: list[str] = []
    monkeypatch.setattr(rocm_native, "_compile_moe_dispatch_tile_ir", _recording_compile(calls))
    module = _swapped_operands(extent) if swapped else _moe_module(*extent)
    assert scheduled_moe_dispatch.supports_scheduled_moe_dispatch(module, target=f"rocm_{ARCH}")
    old = baseline.baseline_moe_dispatch(module, pipeline_name=PIPELINE, architecture=ARCH)
    new = rocm_native.package_moe_dispatch(module, pipeline_name=PIPELINE, architecture=ARCH)
    _assert_same_launch_contract(old, new)
    assert calls[0] == calls[1]
    contract = scheduled_moe_dispatch.lower_scheduled_moe_dispatch(module).native_contract()
    assert contract.architecture == ARCH and contract.layout == "row_major"
    assert contract.dims == baseline._moe_dispatch_contract(module)[3]
    assert contract.names == baseline._moe_dispatch_contract(module)[:3]


@_needs_compiler
def test_consumer_never_reads_the_graph(monkeypatch) -> None:
    """``package_scheduled_moe_dispatch`` sees only the artifact; admission is not re-run."""
    monkeypatch.setattr(rocm_native, "_compile_moe_dispatch_tile_ir", _recording_compile([]))
    artifact = scheduled_moe_dispatch.lower_scheduled_moe_dispatch(_moe_module(*EXTENTS[1]))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the MoE consumer re-entered Graph admission")

    monkeypatch.setattr(scheduled_moe_dispatch, "admit_moe_dispatch", forbidden)
    monkeypatch.setattr(scheduled_moe_dispatch, "supports_scheduled_moe_dispatch", forbidden)
    package = rocm_native.package_scheduled_moe_dispatch(artifact, pipeline_name=PIPELINE)
    assert package.descriptor.abi_id == rocm_native.GFX_MOE_DISPATCH_F32_ABI
    assert package.image.architecture == ARCH
    assert [item.name for item in package.descriptor.buffers] == ["x", "token", "o"]
    assert package.descriptor.provenance["shape"] == [7, 9, 13]


def _refused_modules():
    def module():
        return _moe_module(*EXTENTS[1])

    transport = module()
    transport.functions[0].body[0].kwargs["transport"] = "all_to_all"
    token_dtype = module()
    token_dtype.functions[0].args[1].ir_type = IRType("tensor<9xi64>", ("9",), "int64")
    x_dtype = module()
    x_dtype.functions[0].args[0].ir_type = IRType("tensor<7x13xf16>", ("7", "13"), "fp16")
    result_shape = module()
    result_shape.functions[0].result_types[0] = IRType("tensor<9x14xf32>", ("9", "14"), "fp32")
    result_dtype = module()
    result_dtype.functions[0].result_types[0] = IRType("tensor<9x13xf16>", ("9", "13"), "fp16")
    dynamic = module()
    dynamic.functions[0].args[0].ir_type = IRType("tensor<?x13xf32>", ("?", "13"), "fp32")
    rank3 = module()
    rank3.functions[0].args[0].ir_type = IRType("tensor<7x13x1xf32>", ("7", "13", "1"), "fp32")
    token_rank2 = module()
    token_rank2.functions[0].args[1].ir_type = IRType("tensor<9x1xi32>", ("9", "1"), "int32")
    extra_op = module()
    extra_op.functions[0].body.append(copy.deepcopy(extra_op.functions[0].body[0]))
    return {"transport": transport, "token_i64": token_dtype, "x_f16": x_dtype,
            "result_shape": result_shape, "result_f16": result_dtype, "dynamic_x": dynamic,
            "x_rank3": rank3, "token_rank2": token_rank2, "two_ops": extra_op}


@pytest.mark.parametrize("case", sorted(_refused_modules()))
def test_retired_refusals_still_refuse(case) -> None:
    module = _refused_modules()[case]
    assert baseline._moe_dispatch_contract(module) is None
    assert scheduled_moe_dispatch.admit_moe_dispatch(module) is None
    assert not rocm_native.supports_moe_dispatch(module)
    assert rocm_native.native_package_kind(module) != "moe_dispatch"
    with pytest.raises(ValueError, match="MoE dispatch packaging requires"):
        rocm_native.package_moe_dispatch(module, pipeline_name=PIPELINE, architecture=ARCH)


def test_admission_is_gfx1151_only() -> None:
    """The native owner admits gfx1151; other targets fall through, not raise."""
    module = _moe_module(*EXTENTS[1])
    assert baseline._moe_dispatch_contract(module) is not None
    for target in ("rocm_gfx1201", "rocm", "nvidia_sm120", "apple_gpu"):
        assert not scheduled_moe_dispatch.supports_scheduled_moe_dispatch(module, target=target)
    with pytest.raises(ValueError, match="MoE dispatch packaging requires"):
        rocm_native.package_moe_dispatch(module, pipeline_name=PIPELINE, architecture="gfx1201")
    with pytest.raises(ValueError, match="only admits gfx1151"):
        scheduled_moe_dispatch.project_scheduled_moe_dispatch_graph(module, target="rocm_gfx1201")


def _tile_tamperings(artifact):
    tile = artifact.tile_ir
    return {
        "shape": tile.replace("shape = array<i64: 7, 9, 13>", "shape = array<i64: 7, 9, 12>"),
        "binding": tile.replace('bindings = ["x", "token", "o"]', 'bindings = ["x", "token", "y"]'),
        "duplicate_binding": tile.replace('bindings = ["x", "token", "o"]', 'bindings = ["x", "token", "x"]'),
        "layout": tile.replace('layout = "row_major"', 'layout = "col_major"'),
        "route": tile.replace('route = "direct_gather"', 'route = "indirect"'),
        "missing_field": tile.replace('layout = "row_major", ', ""),
        "extra_field": tile.replace('route = "direct_gather"', 'route = "direct_gather", donation = "owned"'),
        "kernel_storage": tile.replace('index_storage = "i32", storage = "f32"',
                                       'index_storage = "i32", storage = "f16"'),
        "kernel_index_storage": tile.replace('index_storage = "i32", storage = "f32"',
                                             'index_storage = "i64", storage = "f32"'),
        "contract_arch": tile.replace('arch = "gfx1151", bindings', 'arch = "gfx1201", bindings'),
        "contract_target": tile.replace('target = "rocm_gfx1151"}', 'target = "rocm_gfx1201"}'),
        "module_target": tile.replace('tessera.target = "rocm_gfx1151"', 'tessera.target = "rocm_gfx1201"'),
        "entry": tile.replace("@tessera_tile_moe_dispatch_f32_direct", "@tessera_tile_moe_dispatch_f32_other"),
        "duplicate_kernel": tile.replace("    llvm.return", "    tile.moe_dispatch_kernel %arg0 {}\n    llvm.return"),
    }


@_needs_compiler
def test_tampered_tile_contract_fails_closed(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a tampered MoE artifact reached target compilation")

    monkeypatch.setattr(rocm_native, "_compile_moe_dispatch_tile_ir", forbidden)
    artifact = scheduled_moe_dispatch.lower_scheduled_moe_dispatch(_moe_module(*EXTENTS[1]))
    artifact.native_contract()
    for name, tile in _tile_tamperings(artifact).items():
        assert tile != artifact.tile_ir, name
        changed = replace(artifact, tile_ir=tile)
        # The projection alone refuses the text, independent of replay...
        with pytest.raises(ValueError):
            changed.native_contract()
        # ...and the consumer refuses it before any target compilation.
        with pytest.raises(ValueError):
            rocm_native.package_scheduled_moe_dispatch(changed, pipeline_name=PIPELINE)


@_needs_compiler
def test_relabelled_artifact_fields_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(rocm_native, "_compile_moe_dispatch_tile_ir", _recording_compile([]))
    artifact = scheduled_moe_dispatch.lower_scheduled_moe_dispatch(_moe_module(*EXTENTS[1]))
    for stale in (replace(artifact, names=("x", "token", "out")),
                  replace(artifact, names=("token", "x", "o")),
                  replace(artifact, dims=(7, 9, 12))):
        with pytest.raises(ValueError):
            stale.native_contract()
        with pytest.raises(ValueError):
            rocm_native.package_scheduled_moe_dispatch(stale, pipeline_name=PIPELINE)


@_needs_compiler
def test_supplied_artifact_must_be_this_graph(monkeypatch) -> None:
    monkeypatch.setattr(rocm_native, "_compile_moe_dispatch_tile_ir", _recording_compile([]))
    module = _moe_module(*EXTENTS[1])
    other = scheduled_moe_dispatch.lower_scheduled_moe_dispatch(_moe_module(*EXTENTS[2]))
    with pytest.raises(ValueError, match="caller Graph or target"):
        rocm_native.package_moe_dispatch(module, pipeline_name=PIPELINE, architecture=ARCH,
                                         scheduled_artifact=other)


@_needs_compiler
def test_driver_packages_the_scheduled_artifact(monkeypatch) -> None:
    from tessera.compiler.driver import compile_graph_module

    monkeypatch.setattr(rocm_native, "_compile_moe_dispatch_tile_ir", _recording_compile([]))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ROCm MoE packaging must consume the scheduled artifact")

    monkeypatch.setattr(rocm_native, "package_native", forbidden)
    bundle = compile_graph_module(_moe_module(*EXTENTS[1]), source_origin="unit",
                                  target=f"rocm_{ARCH}", options={"package_native": True},
                                  enable_tool_validation=False)
    assert bundle.orchestration_state == "launchable"
    assert bundle.launch_descriptor.abi_id == rocm_native.GFX_MOE_DISPATCH_F32_ABI
    assert bundle.launch_descriptor.provenance["consumer"] == "native_schedule_tile_projection"


def test_prune_dashboard_sees_the_rocm_consumer() -> None:
    from tessera.compiler import bootstrap_prune_audit

    rows = {(target, family): (route, status)
            for target, family, route, status in bootstrap_prune_audit.family_rows()}
    assert rows[("rocm_gfx1151", "moe_dispatch")] == (
        "scheduled_moe_dispatch.supports_scheduled_moe_dispatch", "compiled")


# --- exact-device differential (gfx1151 only: the native owner's one arch) ----------------------

_device = pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_MOVEMENT_DEVICE_PROOF") != "1"
    or os.environ.get("TESSERA_ROCM_CHIP") != ARCH,
    reason="set TESSERA_ROCM_MOVEMENT_DEVICE_PROOF=1 and TESSERA_ROCM_CHIP=gfx1151 on Princess-Luna",
)


def _launch(package, x, token, out_shape):
    from tessera import runtime as rt

    descriptor = package.descriptor
    scalars = dict(zip(("T", "S", "H"), descriptor.provenance["shape"], strict=True))
    out = np.full(out_shape, np.nan, np.float32)
    buffers = {descriptor.buffers[0].name: x, descriptor.buffers[1].name: token,
               descriptor.buffers[2].name: out}
    artifact = rt.RuntimeArtifact(graph_ir="graph", tile_ir=package.tile_ir, target_ir=package.target_ir,
                                  metadata={"target": f"rocm_{ARCH}"}, native_image=package.image,
                                  launch_descriptor=descriptor)
    result = rt.launch(artifact, {"buffers": buffers, "scalars": scalars})
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_gpu"
    return np.array(result["output"], copy=True)


@_device
@pytest.mark.hardware_rocm
@pytest.mark.parametrize("extent", EXTENTS)
def test_exact_device_moe_images_agree_bitwise(extent) -> None:
    from tessera import runtime as rt

    assert rt._rocm_live_arch() == ARCH, "requires the exact owning device"
    module = _moe_module(*extent)
    old = baseline.baseline_moe_dispatch(module, pipeline_name=PIPELINE, architecture=ARCH)
    new = rocm_native.package_moe_dispatch(module, pipeline_name=PIPELINE, architecture=ARCH)
    assert new.image.image_digest == old.image.image_digest
    tokens, slots, hidden = extent
    rng = np.random.default_rng(1151 + tokens * 31 + slots)
    x = rng.standard_normal((tokens, hidden)).astype(np.float32)
    # Signed zero and a subnormal survive a bitwise gather unchanged.
    x.flat[0] = -0.0
    x.flat[-1] = np.float32(1e-40)
    token = rng.integers(0, tokens, size=slots).astype(np.int32)
    token[0] = tokens - 1
    old_out = _launch(old, x, token, (slots, hidden))
    new_out = _launch(new, x, token, (slots, hidden))
    np.testing.assert_array_equal(new_out.view(np.uint32), old_out.view(np.uint32))
    np.testing.assert_array_equal(new_out.view(np.uint32), x[token].view(np.uint32))
    rt._clear_rocm_native_image_cache()
