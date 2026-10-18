"""Frontend source -> native paired AD -> typed ROCm member/lifetime export.

Host-free: verifies compiler artifacts, ABI lineage and metadata refusals;
numerical execution belongs to separately gated tests/device/rocm fixtures.
"""
import base64
import json
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts

from benchmarks.nvidia.active_source_jvp_reference import product, product_reversed

pytestmark = pytest.mark.compiler_route


def source(arch="gfx1201", storage="f16", permuted=False, shape=(17, 19, 35)):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    m, n, k = shape
    roots = (np.zeros((m, k), dtype), np.zeros((k, n), dtype))
    target = "rocm_" + arch
    fn = ts.jit(target=target, autodiff="reverse", wrt=("a",))(
        product_reversed if permuted else product)
    graph = fn._specialized_autodiff_module(roots[::-1] if permuted else roots, {})
    graph.module_attrs.update({"tessera.arch": '"' + arch + '"',
                               "tessera.target": '"' + target + '"',
                               "tessera.native.source_reverse_request": "unit"})
    return graph.to_mlir(target=target, canonical=True)


def run(tool, text, flag):
    result = subprocess.run([str(tool), flag, "-"], input=text, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("permuted", [False, True])
def test_rocm_source_reverse_retains_original_ad_edges_and_private_lifetimes(
        compiler_toolchain, arch, storage, permuted):
    import re
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    original = source(arch, storage, permuted)
    outlined = run(tool, original, "--tessera-autodiff-paired")
    records = re.findall(r'tessera.native.rocm_reverse_program_json = "([^"]+)"', outlined)
    assert len(records) == 1, "native ROCm source reverse program export is missing"
    plan = json.loads(base64.b64decode(records[0], validate=True))
    assert plan["schema"] == "tessera.native.rocm_source_reverse_outline.v1"
    assert plan["architecture"] == arch and plan["target"] == "rocm_" + arch
    assert plan["execution_state"] == "artifact_only"
    roles = [1, 0, 2] if permuted else [0, 1, 2]
    assert plan["role_indices"] == roles
    assert [step["inputs"] for step in plan["steps"]] == [[2, roles[1]], [roles[0], 3]]
    assert [b["first_write"] for b in plan["buffers"]] == [-1, -1, -1, 0, 1]
    assert plan["buffers"][3]["ownership"] == "private_scratch"
    assert plan["buffers"][3]["last_read"] == 1
    assert plan["outputs"] == [4]
    assert len(plan["member_graphs"]) == 2
    for index, graph in enumerate(plan["member_graphs"]):
        assert 'tessera.target = "rocm_' + arch + '"' in graph
        schedule = run(tool, graph, "--tessera-graph-to-schedule")
        tile = run(tool, schedule, "--tessera-schedule-to-tile")
        assert ("tile.scalar_matmul_kernel" if index == 0 else "tile.row_region_kernel") in tile
        if index == 0:
            frames = re.findall(r'tessera.native.scalar_matmul_frame_json = "([^"]+)"', tile)
            assert len(frames) == 1
            frame = json.loads(base64.b64decode(frames[0], validate=True))
            assert frame["scalars"] == [17, 35, 19]
            assert frame["geometry"] == [5, 2, 1, 32, 1, 1]
            assert [b["storage"] for b in frame["buffers"]] == ["f32", storage, storage]
            assert frame["transposeA"] is False and frame["transposeB"] is True
        if index == 1:
            assert "schema = 7" in tile
            assert "tile.buffer_view" in tile
        target = run(tool, tile, "--lower-tile-to-rocm=arch=" + arch)
        assert ("tessera_rocm.scalar_matmul" if index == 0 else "tessera_rocm.row_region_kernel") in target
        if index == 1:
            frames = re.findall(r'tessera.rocm.source_row_member_json = "([^"]+)"', target)
            assert len(frames) == 1
            frame = json.loads(base64.b64decode(frames[0], validate=True))
            assert frame["architecture"] == arch and frame["row_schema"] == 7
            assert frame["abi"] == "flattened_rank1_memrefs_then_i64"
            assert frame["scalars"] == [17, 35]
            assert frame["geometry"] == [1, 1, 1, 128, 1, 1]
            from tessera.compiler.rocm_pipeline import ROCMExecutablePipeline, ROCMOutputLevel
            pipeline = ROCMExecutablePipeline(family="source_reverse_row", arch=arch).pass_pipeline(output=ROCMOutputLevel.BINARY)
            binary = run(tool, tile, "--pass-pipeline=" + pipeline)
            assert "gpu.binary" in binary
            assert "tessera_rocm.row_region_kernel" not in binary


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("permuted", [False, True])
def test_rocm_source_reverse_package_roundtrips_without_compiler(
        compiler_toolchain, arch, storage, permuted, monkeypatch):
    from tessera.compiler.native_rocm_source_reverse import package_source_reverse, RocmSourceReverseProgram
    compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    package = package_source_reverse(source(arch, storage, permuted))
    encoded = package.to_json()
    with monkeypatch.context() as offline:
        offline.setattr(subprocess, "run", lambda *a, **k: pytest.fail("portable validation invoked compiler"))
        restored = RocmSourceReverseProgram.from_json(encoded)
        restored.validate()
    assert restored.manifest["architecture"] == arch
    assert restored.images == package.images
    assert len(restored.images) == 2


@pytest.fixture(scope="module")
def native_rocm_package(compiler_toolchain):
    from tessera.compiler.native_rocm_source_reverse import package_source_reverse
    compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    return package_source_reverse(source())


@pytest.mark.parametrize("fault", ["row_geometry", "row_bytes", "image", "outline", "native_stage", "row_boolean"])
def test_source_reverse_frame_edits_fail_before_device_probe(native_rocm_package, fault, monkeypatch):
    from dataclasses import replace
    from tessera import runtime
    package = native_rocm_package
    if fault in {"row_geometry", "row_bytes", "row_boolean"}:
        frame = json.loads(package.row_frame_json)
        if fault == "row_geometry":
            frame["geometry"][3] = 64
        elif fault == "row_boolean":
            frame["inputs"] = [False, True]
        else:
            frame["buffers"][2]["bytes"] += 2
        changed = replace(package, row_frame_json=json.dumps(frame))
    elif fault == "image":
        changed = replace(package, row_image=package.row_image + b"edited")
    elif fault == "outline":
        plan = json.loads(package.program_json)
        plan["source_graph_ir"] += "\n// edited original source"
        changed = replace(package, program_json=json.dumps(plan))
    else:
        stages = tuple(tuple(text + "\n// edited" if j == 3 else text for j, text in enumerate(row))
                       if i == 1 else row for i, row in enumerate(package.stages))
        changed = replace(package, stages=stages)
    from benchmarks.math.source_reverse_reference import inputs
    valid_roots = inputs("f16", False)
    monkeypatch.setattr(runtime, "_rocm_live_arch", lambda: pytest.fail("invalid frame probed GPU"))
    with pytest.raises(ValueError):
        changed.prepare(valid_roots)


def test_invalid_source_root_dtype_is_checked_before_hardware(native_rocm_package, monkeypatch):
    from tessera import runtime
    from benchmarks.math.source_reverse_reference import inputs
    roots = inputs("f16", False)
    roots = (*roots[:2], roots[2].astype(np.float64))
    monkeypatch.setattr(runtime, "_rocm_live_arch", lambda: pytest.fail("invalid input probed GPU"))
    with pytest.raises(ValueError, match="root type/shape"):
        native_rocm_package.prepare(roots)
