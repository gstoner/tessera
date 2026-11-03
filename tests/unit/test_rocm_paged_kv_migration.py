"""E2E-REAL-6, ROCm paged-KV: retired Graph-authored packaging vs the Tile projection.

``rocm_native.package_paged_kv_read`` used to take its kernel from the native
Graph -> Schedule -> Tile replay but author every launch field -- buffers,
ABI, scalar list, shape guards, geometry, page layout -- from the Python Graph
object. ``package_scheduled_paged_kv`` now projects them from the replayed
Tile ``tessera.native_contract``. The frozen packager in
``tests/_support/rocm_paged_kv_baseline.py`` is the declared oracle (Decision
#31(a)); these tests are its differential.

Host-free half: over the admitted envelope (both layouts, both ROCm archs) the
projected descriptor equals the retired one field for field, refusals are
unchanged, and a tampered artifact fails closed before target compilation.
Device half (``TESSERA_ROCM_MOVEMENT_DEVICE_PROOF=1`` on the owning host,
``TESSERA_ROCM_CHIP`` naming it): both packages build the same image and
their outputs agree bit-for-bit and with a NumPy oracle. gfx1151 and gfx1201
results are separate evidence; neither transfers.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.rocm.benchmark_rocm_e2e_movement import _paged_module
from tessera.compiler import rocm_native, scheduled_paged_kv
from tessera.compiler.graph_ir import IRType
from tessera.compiler.native_artifact import DeviceLibraryRecord
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support import rocm_paged_kv_baseline as baseline

PIPELINE = "tessera-lower-to-rocm"
ARCHES = ("gfx1151", "gfx1201")
_needs_compiler = pytest.mark.skipif(find_tessera_opt() is None, reason="requires production tessera-opt")

#: (pages, page_size, heads, dim, start, tokens): single token, interior,
#: whole capacity, and a non-power-of-two head/dim pair.
EXTENTS = ((4, 4, 3, 8, 0, 1), (4, 4, 3, 8, 3, 7), (4, 4, 3, 8, 0, 16), (5, 3, 2, 5, 2, 9))
CASES = [(arch, layout, extent) for arch in ARCHES for layout in ("row_major", "strided")
         for extent in EXTENTS]


def _module(extent, layout="row_major"):
    module = _paged_module(*extent)
    if layout != "row_major":
        module.functions[0].args[0].layout = layout
    return module


def _recording_compile(calls: list[str]):
    def compile_(tile_ir: str, **_kwargs):
        calls.append(tile_ir)
        return ('module {\n  tessera_rocm.paged_kv_read {name = "tessera_rocm_paged_kv_fake", route = "direct"}\n}',
                "backend", b"\x7fELFrocm-paged-kv", "compiler", "toolchain",
                (DeviceLibraryRecord("rocm.ocml", "1" * 64, "compiler_driver"),), "cold")
    return compile_


def _assert_same_launch_contract(old, new, arch) -> None:
    for field in ("abi_id", "entry_symbol", "buffers", "scalars", "shape_guards", "geometry", "ordering"):
        assert getattr(old.descriptor, field) == getattr(new.descriptor, field), field
    # Every retired provenance key survives with the same value; the only new
    # key names the consumer, so a regression in either direction is visible.
    assert {k: v for k, v in new.descriptor.provenance.items() if k != "consumer"} == old.descriptor.provenance
    assert new.descriptor.provenance["consumer"] == "native_schedule_tile_projection"
    assert "consumer" not in old.descriptor.provenance
    assert new.image.target == old.image.target == f"rocm_{arch}"
    assert new.image.entry_points == old.image.entry_points
    assert new.tile_ir == old.tile_ir


@_needs_compiler
@pytest.mark.parametrize("arch,layout,extent", CASES)
def test_projected_descriptor_reproduces_retired_contract(monkeypatch, arch, layout, extent) -> None:
    calls: list[str] = []
    monkeypatch.setattr(rocm_native, "_compile_paged_kv_tile_ir", _recording_compile(calls))
    module = _module(extent, layout)
    assert scheduled_paged_kv.supports_scheduled_paged_kv(module, target=f"rocm_{arch}")
    old = baseline.baseline_paged_kv_read(module, pipeline_name=PIPELINE, architecture=arch)
    new = rocm_native.package_paged_kv_read(module, pipeline_name=PIPELINE, architecture=arch)
    _assert_same_launch_contract(old, new, arch)
    assert calls[0] == calls[1]
    contract = scheduled_paged_kv.lower_scheduled_paged_kv_graph(module, target=f"rocm_{arch}").native_contract()
    assert contract.layout == layout and contract.architecture == arch
    assert contract.dims == baseline._paged_kv_contract(module)[3]


@_needs_compiler
def test_consumer_never_reads_the_graph(monkeypatch) -> None:
    """``package_scheduled_paged_kv`` sees only the artifact; admission is not re-run."""
    monkeypatch.setattr(rocm_native, "_compile_paged_kv_tile_ir", _recording_compile([]))
    artifact = scheduled_paged_kv.lower_scheduled_paged_kv_graph(
        _module(EXTENTS[1], "strided"), target="rocm_gfx1201")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the paged consumer re-entered Graph admission")

    monkeypatch.setattr(scheduled_paged_kv, "admit_paged_kv", forbidden)
    monkeypatch.setattr(scheduled_paged_kv, "paged_storage_layout", forbidden)
    package = rocm_native.package_scheduled_paged_kv(artifact, pipeline_name=PIPELINE)
    assert package.descriptor.abi_id == rocm_native.GFX_PAGED_KV_STRIDED_F32_ABI
    assert package.image.architecture == "gfx1201"
    assert package.descriptor.buffers[0].layout == "strided"


def _refused_modules():
    out_of_bounds = _module((4, 4, 3, 8, 0, 16))
    out_of_bounds.functions[0].body[0].kwargs["end"] = 17
    negative = _module(EXTENTS[1])
    negative.functions[0].body[0].kwargs["start"] = -1
    bool_bound = _module(EXTENTS[1])
    bool_bound.functions[0].body[0].kwargs["start"] = True
    table_dtype = _module(EXTENTS[1])
    table_dtype.functions[0].args[1].ir_type = IRType("tensor<4xi64>", ("4",), "int64")
    page_dtype = _module(EXTENTS[1])
    page_dtype.functions[0].args[0].ir_type = IRType("tensor<4x4x3x8xf16>", ("4", "4", "3", "8"), "fp16")
    result_shape = _module(EXTENTS[1])
    result_shape.functions[0].result_types[0] = IRType("tensor<7x3x9xf32>", ("7", "3", "9"), "fp32")
    dynamic = _module(EXTENTS[1])
    dynamic.functions[0].args[0].ir_type = IRType("tensor<?x4x3x8xf32>", ("?", "4", "3", "8"), "fp32")
    return {"end_past_capacity": out_of_bounds, "negative_start": negative, "bool_start": bool_bound,
            "table_i64": table_dtype, "pages_f16": page_dtype, "result_shape": result_shape,
            "dynamic_pages": dynamic}


@pytest.mark.parametrize("case", sorted(_refused_modules()))
def test_retired_refusals_still_refuse(case) -> None:
    module = _refused_modules()[case]
    assert baseline._paged_kv_contract(module) is None
    assert scheduled_paged_kv.admit_paged_kv(module) is None
    assert not rocm_native.supports_paged_kv_read(module)
    assert rocm_native.native_package_kind(module) != "paged_kv"
    for arch in ARCHES:
        with pytest.raises(ValueError, match="paged-KV packaging requires"):
            rocm_native.package_paged_kv_read(module, pipeline_name=PIPELINE, architecture=arch)


def test_unknown_page_layout_is_refused_at_admission() -> None:
    """The retired route admitted it and raised in packaging; admission now refuses."""
    module = _module(EXTENTS[1], "col_major")
    assert baseline._paged_kv_contract(module) is not None
    assert not scheduled_paged_kv.supports_scheduled_paged_kv(module, target="rocm_gfx1151")
    assert not scheduled_paged_kv.supports_scheduled_paged_kv(_module(EXTENTS[1]), target="nvidia_sm120")


def _tile_tamperings(artifact):
    tile = artifact.tile_ir
    return {
        "shape": tile.replace("shape = array<i64: 4, 4, 4, 3, 8, 3, 7>", "shape = array<i64: 4, 4, 4, 3, 8, 3, 6>"),
        "binding": tile.replace('bindings = ["pages", "page_table", "slice"]',
                                'bindings = ["pages", "page_table", "other"]'),
        "layout": tile.replace('layout = "row_major"', 'layout = "strided"'),
        "missing_policy": tile.replace('table_bounds = "runtime_checked_physical_page_indices", ', ""),
        "extra_policy": tile.replace('page_ownership = "read_only_borrow"',
                                     'page_ownership = "read_only_borrow", donation = "owned"'),
        "kernel_storage": tile.replace('storage = "f32", table_storage', 'storage = "f16", table_storage'),
        "kernel_route": tile.replace('route = "direct"', 'route = "indirect"'),
        "contract_arch": tile.replace('arch = "gfx1151", bindings', 'arch = "gfx1201", bindings'),
        "duplicate_kernel": tile.replace("    llvm.return", "    tile.paged_kv_read_kernel %arg0 {}\n    llvm.return"),
    }


@_needs_compiler
def test_tampered_tile_contract_fails_closed(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a tampered paged artifact reached target compilation")

    monkeypatch.setattr(rocm_native, "_compile_paged_kv_tile_ir", forbidden)
    artifact = scheduled_paged_kv.lower_scheduled_paged_kv_graph(_module(EXTENTS[1]), target="rocm_gfx1151")
    artifact.native_contract()
    for name, tile in _tile_tamperings(artifact).items():
        assert tile != artifact.tile_ir, name
        changed = replace(artifact, tile_ir=tile)
        # The projection alone refuses the text, independent of replay...
        with pytest.raises(ValueError):
            changed.native_contract()
        # ...and the consumer refuses it before any target compilation.
        with pytest.raises(ValueError):
            rocm_native.package_scheduled_paged_kv(changed, pipeline_name=PIPELINE)


@_needs_compiler
def test_relabelled_artifact_fields_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(rocm_native, "_compile_paged_kv_tile_ir", _recording_compile([]))
    artifact = scheduled_paged_kv.lower_scheduled_paged_kv_graph(_module(EXTENTS[1]), target="rocm_gfx1151")
    for stale in (replace(artifact, names=("pages", "page_table", "out")),
                  replace(artifact, dims=(4, 4, 4, 3, 8, 2, 7)),
                  replace(artifact, page_layout="strided", entry="tessera_tile_paged_kv_read_f32_strided")):
        with pytest.raises(ValueError):
            rocm_native.package_scheduled_paged_kv(stale, pipeline_name=PIPELINE)


@_needs_compiler
def test_supplied_artifact_must_be_this_graph_and_target(monkeypatch) -> None:
    monkeypatch.setattr(rocm_native, "_compile_paged_kv_tile_ir", _recording_compile([]))
    module = _module(EXTENTS[1])
    gfx1201 = scheduled_paged_kv.lower_scheduled_paged_kv_graph(module, target="rocm_gfx1201")
    with pytest.raises(ValueError, match="caller Graph or target"):
        rocm_native.package_paged_kv_read(module, pipeline_name=PIPELINE, architecture="gfx1151",
                                          scheduled_artifact=gfx1201)
    strided = scheduled_paged_kv.lower_scheduled_paged_kv_graph(
        _module(EXTENTS[1], "strided"), target="rocm_gfx1201")
    with pytest.raises(ValueError, match="caller Graph or target"):
        rocm_native.package_paged_kv_read(module, pipeline_name=PIPELINE, architecture="gfx1201",
                                          scheduled_artifact=strided)


@_needs_compiler
@pytest.mark.parametrize("arch", ARCHES)
def test_driver_packages_the_scheduled_artifact(monkeypatch, arch) -> None:
    from tessera.compiler.driver import compile_graph_module

    monkeypatch.setattr(rocm_native, "_compile_paged_kv_tile_ir", _recording_compile([]))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ROCm paged packaging must consume the scheduled artifact")

    monkeypatch.setattr(rocm_native, "package_native", forbidden)
    for layout in ("row_major", "strided"):
        bundle = compile_graph_module(_module(EXTENTS[1], layout), source_origin="unit",
                                      target=f"rocm_{arch}", options={"package_native": True},
                                      enable_tool_validation=False)
        assert bundle.orchestration_state == "launchable"
        assert bundle.launch_descriptor.provenance["consumer"] == "native_schedule_tile_projection"
        assert bundle.launch_descriptor.buffers[0].layout == layout


def test_prune_dashboard_sees_the_rocm_consumer() -> None:
    from tessera.compiler import bootstrap_prune_audit

    rows = {(target, family): (route, status)
            for target, family, route, status in bootstrap_prune_audit.family_rows()}
    assert rows[("rocm_gfx1151", "paged_kv")] == (
        "scheduled_paged_kv.supports_scheduled_paged_kv", "compiled")
    # MoE dispatch is a separate slice and is still Graph-owned.
    assert rows[("rocm_gfx1151", "moe_dispatch")][1] == "gap"


# --- exact-device differential (each owning ROCm host separately) ------------------------------

_device = pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_MOVEMENT_DEVICE_PROOF") != "1",
    reason="set TESSERA_ROCM_MOVEMENT_DEVICE_PROOF=1 and TESSERA_ROCM_CHIP on the owning ROCm host",
)


def _pages(extent, layout, seed):
    pages, page_size, heads, dim, _, _ = extent
    rng = np.random.default_rng(seed)
    if layout == "row_major":
        return rng.standard_normal((pages, page_size, heads, dim)).astype(np.float32)
    # A padded, permuted parent: positive element strides, not C-contiguous.
    parent = rng.standard_normal((dim * 2 + 1, heads + 1, page_size, pages + 2)).astype(np.float32)
    return parent[1:2 * dim + 1:2, :heads, :, 1:pages + 1].transpose(3, 2, 1, 0)


def _launch(package, arch, pages, table, out_shape):
    from tessera import runtime as rt

    descriptor = package.descriptor
    shape = descriptor.provenance["shape"]
    scalars = dict(zip(("P", "LP", "PageSize", "H", "D", "Start", "Tokens"), shape, strict=True))
    if descriptor.abi_id == rocm_native.GFX_PAGED_KV_STRIDED_F32_ABI:
        scalars.update(zip(("StrideP", "StridePage", "StrideH", "StrideD"),
                           (s // pages.itemsize for s in pages.strides), strict=True))
    out = np.full(out_shape, np.nan, np.float32)
    buffers = {descriptor.buffers[0].name: pages, descriptor.buffers[1].name: table,
               descriptor.buffers[2].name: out}
    artifact = rt.RuntimeArtifact(graph_ir="graph", tile_ir=package.tile_ir, target_ir=package.target_ir,
                                  metadata={"target": f"rocm_{arch}"}, native_image=package.image,
                                  launch_descriptor=descriptor)
    result = rt.launch(artifact, {"buffers": buffers, "scalars": scalars})
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_gpu"
    return np.array(result["output"], copy=True)


@_device
@pytest.mark.hardware_rocm
@pytest.mark.parametrize("layout", ["row_major", "strided"])
@pytest.mark.parametrize("extent", EXTENTS)
def test_exact_device_paged_images_agree_bitwise(layout, extent) -> None:
    from tessera import runtime as rt

    arch = os.environ.get("TESSERA_ROCM_CHIP")
    assert arch in ARCHES and rt._rocm_live_arch() == arch, "requires the exact owning device"
    module = _module(extent, layout)
    old = baseline.baseline_paged_kv_read(module, pipeline_name=PIPELINE, architecture=arch)
    new = rocm_native.package_paged_kv_read(module, pipeline_name=PIPELINE, architecture=arch)
    assert new.image.image_digest == old.image.image_digest
    pages_count, page_size, heads, dim, start, tokens = extent
    pages = _pages(extent, layout, 1151 + 1201 * len(layout) + start)
    assert pages.flags.c_contiguous == (layout == "row_major")
    table = np.random.default_rng(start + tokens).permutation(pages_count).astype(np.int32)
    out_shape = (tokens, heads, dim)
    old_out = _launch(old, arch, pages, table, out_shape)
    new_out = _launch(new, arch, pages, table, out_shape)
    np.testing.assert_array_equal(new_out.view(np.uint32), old_out.view(np.uint32))
    expected = np.ascontiguousarray(pages[table]).reshape(-1, heads, dim)[start:start + tokens]
    np.testing.assert_array_equal(new_out.view(np.uint32), expected.view(np.uint32))
    rt._clear_rocm_native_image_cache()
