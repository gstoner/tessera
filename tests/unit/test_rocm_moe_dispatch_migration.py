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
(``TESSERA_ROCM_MOVEMENT_DEVICE_PROOF=1`` with ``TESSERA_ROCM_CHIP`` set to the
live arch: gfx1151 on Princess-Luna, gfx1201 on Tajasarus): both packages build
the same image and their outputs agree bit-for-bit and with a NumPy oracle.
The native owner admits exactly the archs whose ``moe_dispatch`` family is
promoted (gfx1151, and gfx1201 since 2026-11-04); each arch's rows are proven
on its own box, and evidence never transfers between them.
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
ARCHES = ("gfx1151", "gfx1201")
ARCH = ARCHES[0]
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


def _assert_same_launch_contract(old, new, arch=ARCH) -> None:
    for field in ("abi_id", "entry_symbol", "buffers", "scalars", "shape_guards", "geometry", "ordering"):
        assert getattr(old.descriptor, field) == getattr(new.descriptor, field), field
    # Every retired provenance key survives with the same value; the only new
    # key names the consumer, so a regression in either direction is visible.
    assert {k: v for k, v in new.descriptor.provenance.items() if k != "consumer"} == old.descriptor.provenance
    assert new.descriptor.provenance["consumer"] == "native_schedule_tile_projection"
    assert "consumer" not in old.descriptor.provenance
    assert new.image.target == old.image.target == f"rocm_{arch}"
    assert new.descriptor.geometry.policy == f"{arch}_moe_dispatch_direct_256"
    assert new.image.entry_points == old.image.entry_points
    assert new.tile_ir == old.tile_ir


def _swapped_operands(extent):
    """Graph arguments in (token, x) order; operands still name x then token."""
    module = _moe_module(*extent)
    module.functions[0].args.reverse()
    return module


@_needs_compiler
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("swapped", [False, True], ids=["x_first", "token_first"])
@pytest.mark.parametrize("extent", EXTENTS)
def test_projected_descriptor_reproduces_retired_contract(monkeypatch, extent, swapped, arch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(rocm_native, "_compile_moe_dispatch_tile_ir", _recording_compile(calls))
    module = _swapped_operands(extent) if swapped else _moe_module(*extent)
    assert scheduled_moe_dispatch.supports_scheduled_moe_dispatch(module, target=f"rocm_{arch}")
    old = baseline.baseline_moe_dispatch(module, pipeline_name=PIPELINE, architecture=arch)
    new = rocm_native.package_moe_dispatch(module, pipeline_name=PIPELINE, architecture=arch)
    _assert_same_launch_contract(old, new, arch)
    assert calls[0] == calls[1]
    contract = scheduled_moe_dispatch.lower_scheduled_moe_dispatch(
        module, target=f"rocm_{arch}").native_contract()
    assert contract.architecture == arch and contract.target == f"rocm_{arch}"
    assert contract.layout == "row_major"
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


#: Targets that must never be admitted: archs with no ``moe_dispatch`` proof,
#: the generic ``rocm`` name, and other backends.
UNPROVEN_TARGETS = ("rocm_gfx1200", "rocm_gfx1250", "rocm_gfx1100", "rocm_gfx942",
                    "rocm", "nvidia_sm120", "apple_gpu")


def test_admission_is_derived_from_the_promotion_table() -> None:
    """Admission is exactly the archs whose ``moe_dispatch`` family is promoted."""
    from tessera.compiler.rocm_pipeline import promoted_families
    from tessera.compiler.rocm_target import AMDArch, rocm_arch_string

    expected = {f"rocm_{rocm_arch_string(a)}" for a in AMDArch
                if "moe_dispatch" in promoted_families(rocm_arch_string(a))}
    assert scheduled_moe_dispatch.ROCM_MOE_DISPATCH_TARGETS == expected
    assert expected == {f"rocm_{arch}" for arch in ARCHES}


def test_unproven_targets_fall_through_or_refuse() -> None:
    """Every other target falls through admission and refuses packaging."""
    module = _moe_module(*EXTENTS[1])
    assert baseline._moe_dispatch_contract(module) is not None
    for arch in ARCHES:
        assert scheduled_moe_dispatch.supports_scheduled_moe_dispatch(module, target=f"rocm_{arch}")
    for target in UNPROVEN_TARGETS:
        assert not scheduled_moe_dispatch.supports_scheduled_moe_dispatch(module, target=target)
        with pytest.raises(ValueError, match="admits only ROCm architectures"):
            scheduled_moe_dispatch.project_scheduled_moe_dispatch_graph(module, target=target)
    with pytest.raises(ValueError, match="MoE dispatch packaging requires"):
        rocm_native.package_moe_dispatch(module, pipeline_name=PIPELINE, architecture="gfx1200")


def _owner_graph(arch: str) -> str:
    """The projected Graph for a proven arch, relabelled to ``arch``."""
    text = scheduled_moe_dispatch.project_scheduled_moe_dispatch_graph(
        _moe_module(*EXTENTS[1]), target="rocm_gfx1151")
    assert text.count('"rocm_gfx1151"') == 1 and text.count('"gfx1151"') == 1
    return text.replace('"rocm_gfx1151"', f'"rocm_{arch}"').replace('"gfx1151"', f'"{arch}"')


@_needs_compiler
def test_native_owner_admits_exactly_the_python_targets() -> None:
    """``NativeMoeDispatch.h`` and ROCM_MOE_DISPATCH_TARGETS state one set.

    Every modelled AMD arch is replayed through the C++ owner; it must lower
    the proven archs and refuse every other one with its stable diagnostic.
    """
    from tessera.compiler.rocm_target import AMDArch, rocm_arch_string
    from tessera.compiler.scheduled_matmul import run_tessera_opt

    tool = find_tessera_opt()
    for arch in sorted({rocm_arch_string(a) for a in AMDArch}):
        graph = _owner_graph(arch)
        admitted = f"rocm_{arch}" in scheduled_moe_dispatch.ROCM_MOE_DISPATCH_TARGETS
        if admitted:
            schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
            assert "schedule.moe_dispatch" in schedule, arch
        else:
            with pytest.raises(RuntimeError, match="gfx1151 or gfx1201 Graph entry"):
                run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")


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
        "unproven_arch": tile.replace('arch = "gfx1151", bindings', 'arch = "gfx1200", bindings')
                             .replace('target = "rocm_gfx1151"}', 'target = "rocm_gfx1200"}')
                             .replace('tessera.target = "rocm_gfx1151"', 'tessera.target = "rocm_gfx1200"')
                             .replace('tessera.arch = "gfx1151"', 'tessera.arch = "gfx1200"'),
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
def test_supplied_artifact_must_be_this_graph_and_target(monkeypatch) -> None:
    monkeypatch.setattr(rocm_native, "_compile_moe_dispatch_tile_ir", _recording_compile([]))
    module = _moe_module(*EXTENTS[1])
    other = scheduled_moe_dispatch.lower_scheduled_moe_dispatch(_moe_module(*EXTENTS[2]))
    with pytest.raises(ValueError, match="caller Graph or target"):
        rocm_native.package_moe_dispatch(module, pipeline_name=PIPELINE, architecture=ARCH,
                                         scheduled_artifact=other)
    # A gfx1201 artifact never packages as gfx1151, and vice versa.
    for built, asked in (("gfx1201", "gfx1151"), ("gfx1151", "gfx1201")):
        artifact = scheduled_moe_dispatch.lower_scheduled_moe_dispatch(module, target=f"rocm_{built}")
        with pytest.raises(ValueError, match="caller Graph or target"):
            rocm_native.package_moe_dispatch(module, pipeline_name=PIPELINE, architecture=asked,
                                             scheduled_artifact=artifact)
    package = rocm_native.package_scheduled_moe_dispatch(
        scheduled_moe_dispatch.lower_scheduled_moe_dispatch(module, target="rocm_gfx1201"),
        pipeline_name=PIPELINE)
    assert package.image.architecture == "gfx1201" and package.image.target == "rocm_gfx1201"


@_needs_compiler
@pytest.mark.parametrize("arch", ARCHES)
def test_driver_packages_the_scheduled_artifact(monkeypatch, arch) -> None:
    from tessera.compiler.driver import compile_graph_module

    monkeypatch.setattr(rocm_native, "_compile_moe_dispatch_tile_ir", _recording_compile([]))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ROCm MoE packaging must consume the scheduled artifact")

    monkeypatch.setattr(rocm_native, "package_native", forbidden)
    bundle = compile_graph_module(_moe_module(*EXTENTS[1]), source_origin="unit",
                                  target=f"rocm_{arch}", options={"package_native": True},
                                  enable_tool_validation=False)
    assert bundle.orchestration_state == "launchable"
    assert bundle.launch_descriptor.abi_id == rocm_native.GFX_MOE_DISPATCH_F32_ABI
    assert bundle.launch_descriptor.geometry.policy == f"{arch}_moe_dispatch_direct_256"
    assert bundle.launch_descriptor.provenance["consumer"] == "native_schedule_tile_projection"


def test_prune_dashboard_sees_the_rocm_consumer() -> None:
    from tessera.compiler import bootstrap_prune_audit

    rows = {(target, family): (route, status)
            for target, family, route, status in bootstrap_prune_audit.family_rows()}
    assert rows[("rocm_gfx1151", "moe_dispatch")] == (
        "scheduled_moe_dispatch.supports_scheduled_moe_dispatch", "compiled")


def test_gfx1201_canonical_options_default_to_the_native_package(monkeypatch) -> None:
    from tessera.compiler.driver import canonical_compile_options

    module = _moe_module(*EXTENTS[1])
    monkeypatch.setattr(rocm_native, "native_packaging_available", lambda: True)
    assert canonical_compile_options(module, target="rocm_gfx1201")["package_native"]
    assert not canonical_compile_options(
        module, target="rocm_gfx1201", options={"package_native": False})["package_native"]
    monkeypatch.setattr(rocm_native, "native_packaging_available", lambda: False)
    assert not canonical_compile_options(module, target="rocm_gfx1201")["package_native"]


# --- exact-device differential (each owning ROCm host separately) ------------------------------

_device = pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_MOVEMENT_DEVICE_PROOF") != "1",
    reason="set TESSERA_ROCM_MOVEMENT_DEVICE_PROOF=1 and TESSERA_ROCM_CHIP on the owning ROCm host",
)


def _launch(package, x, token, out_shape):
    from tessera import runtime as rt

    descriptor = package.descriptor
    scalars = dict(zip(("T", "S", "H"), descriptor.provenance["shape"], strict=True))
    out = np.full(out_shape, np.nan, np.float32)
    buffers = {descriptor.buffers[0].name: x, descriptor.buffers[1].name: token,
               descriptor.buffers[2].name: out}
    artifact = rt.RuntimeArtifact(graph_ir="graph", tile_ir=package.tile_ir, target_ir=package.target_ir,
                                  metadata={"target": package.image.target}, native_image=package.image,
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

    arch = os.environ.get("TESSERA_ROCM_CHIP")
    assert arch in ARCHES and rt._rocm_live_arch() == arch, "requires the exact owning device"
    module = _moe_module(*extent)
    old = baseline.baseline_moe_dispatch(module, pipeline_name=PIPELINE, architecture=arch)
    new = rocm_native.package_moe_dispatch(module, pipeline_name=PIPELINE, architecture=arch)
    assert new.image.image_digest == old.image.image_digest
    assert new.image.architecture == arch and new.image.target == f"rocm_{arch}"
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
