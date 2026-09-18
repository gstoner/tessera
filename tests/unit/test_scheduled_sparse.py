"""Public admission of the checked 2:4 sparse half matmul on gfx1201.

Host-free half: the logical matmul lowers through the production
Graph->Schedule->Tile boundaries into the compiler-built SWMMAC kernel with
its validity words, the artifact replays, and the family is RDNA4-only.
Device half (Tajasarus): the packaged kernel executes through
`runtime.launch` under its own ABI, a non-2:4 A block refuses the launch,
and the public jit front door keeps its logical native backward.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tessera.compiler import scheduled_sparse
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type
from tessera.compiler.scheduled_matmul import find_tessera_opt


def sparse_module(m=32, k=64, n=32, storage="fp16", output="fp32"):
    a, b = tensor_ir_type((m, k), storage), tensor_ir_type((k, n), storage)
    out = tensor_ir_type((m, n), output)
    return GraphIRModule(functions=[GraphIRFunction(
        name="sparse_product", args=[IRArg("a", a), IRArg("b", b)], result_types=[out],
        return_values=["%o"], body=[IROp(result="o", op_name="tessera.matmul", operands=["%a", "%b"],
                                        operand_types=[str(a), str(b)], result_type=str(out))])])


def two_to_four(rng, m, k, dtype):
    a = (rng.normal(size=(m, k)) * 0.2).astype(dtype)
    a.reshape(m, k // 4, 4)[:, :, 2:] = 0
    return a


@pytest.mark.parametrize("storage,output", [("fp16", "fp32"), ("bf16", "bf16"), ("fp16", "fp16")])
@pytest.mark.parametrize("selection", ["checked_2to4", "auto_2to4"])
def test_sparse_matmul_lowers_through_the_production_boundaries(storage, output, selection):
    if find_tessera_opt() is None:
        pytest.skip("production tessera-opt unavailable")
    artifact = scheduled_sparse.lower_scheduled_sparse_matmul(sparse_module(storage=storage, output=output), selection=selection)
    assert artifact.shape == (32, 32, 64) and artifact.tiles == 4
    assert artifact.storage == {"fp16": "f16", "bf16": "bf16"}[storage]
    assert artifact.output == {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[output]
    assert artifact.selection == selection and artifact.architecture == "gfx1201"
    assert "schedule.sparse_mma" in artifact.schedule_ir and "tile.sparse_mma" in artifact.tile_ir
    assert ("gpu.shuffle" in artifact.tile_ir) == (selection == "auto_2to4")
    assert (artifact.a_name, artifact.b_name, artifact.output_name) == ("a", "b", "o")
    artifact.validate()
    from dataclasses import replace
    with pytest.raises(ValueError, match="identity"):
        replace(artifact, tile_ir=artifact.tile_ir + "\n").validate()
    with pytest.raises(ValueError, match="descriptor"):
        replace(artifact, m=48, schedule_digest=artifact.schedule_digest).validate()


def test_sparse_admission_refuses_what_the_native_lowering_refuses():
    if find_tessera_opt() is None:
        pytest.skip("production tessera-opt unavailable")
    with pytest.raises(ValueError, match="unknown sparse selection"):
        scheduled_sparse.lower_scheduled_sparse_matmul(sparse_module(), selection="dense")
    with pytest.raises(ValueError):
        scheduled_sparse.lower_scheduled_sparse_matmul(sparse_module(storage="fp32", output="fp32"))
    module = sparse_module()
    module.functions[0].body[0].kwargs["activation"] = "relu"
    with pytest.raises(ValueError, match="policy|epilogue"):
        scheduled_sparse.lower_scheduled_sparse_matmul(module)


def test_sparse_family_is_rdna4_only():
    from tessera.compiler.rocm_pipeline import ROCMExecutablePipeline
    ROCMExecutablePipeline(family="sparse_matmul_2to4", arch="gfx1201")
    with pytest.raises(ValueError, match="SWMMAC"):
        ROCMExecutablePipeline(family="sparse_matmul_2to4", arch="gfx1151")
    from tessera.compiler import rocm_native
    with pytest.raises(ValueError, match="ScheduledSparseMatmulArtifact"):
        rocm_native.package_sparse_matmul(object(), pipeline_name="tessera-lower-to-rocm")


def _gfx1201_or_skip():
    if os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1":
        pytest.skip("explicit gfx1201 owning-device gate")
    from tessera import runtime as rt
    assert rt._rocm_live_arch() == "gfx1201"
    return rt


def _launch(rt, package, a, b, output):
    status = np.zeros(package.descriptor.provenance["validity_words"], np.int32)
    artifact = rt.RuntimeArtifact(metadata={"target": package.image.target}, native_image=package.image,
                                  launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir)
    return rt.launch(artifact, {"buffers": {"a": a, "b": b, "o": output, "status": status}, "scalars": {}}), status


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("storage,output,dtype,out_dtype", [
    ("fp16", "fp32", np.float16, np.float32), ("fp16", "fp16", np.float16, np.float16),
    ("bf16", "fp32", "bfloat16", np.float32)])
@pytest.mark.parametrize("shape", [(32, 64, 32), (64, 128, 48), (16, 32, 16)])
def test_gfx1201_sparse_package_executes_and_refuses_dense_tiles(storage, output, dtype, out_dtype, shape):
    rt = _gfx1201_or_skip()
    from tessera.compiler import rocm_native
    if dtype == "bfloat16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    m, k, n = shape
    artifact = scheduled_sparse.lower_scheduled_sparse_matmul(sparse_module(m, k, n, storage, output))
    package = rocm_native.package_sparse_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
    assert package.descriptor.abi_id == rocm_native.GFX_SPARSE_MATMUL_2TO4_ABI
    assert package.image.architecture == "gfx1201" and "swmmac" in package.target_ir
    rng = np.random.default_rng(m * n)
    a = two_to_four(rng, m, k, dtype)
    b = (rng.normal(size=(k, n)) * 0.2).astype(dtype)
    out = np.zeros((m, n), out_dtype)
    result, status = _launch(rt, package, a, b, out)
    assert result["ok"] and result["execution_kind"] == "native_gpu", result
    assert np.all(status == 1)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(out.astype(np.float32), expected, rtol=4e-3, atol=2e-3)
    # One dense group of four in one tile: the launch refuses, the output is untouched.
    dense = a.copy(); dense[0, :4] = 1
    out.fill(7)
    refused, status = _launch(rt, package, dense, b, out)
    assert refused["ok"] is False and "2:4" in str(refused.get("reason")), refused
    assert np.all(out == 7)


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("density", ["sparse", "dense", "mixed"])
@pytest.mark.parametrize("storage,output,dtype,out_dtype", [("fp16", "fp32", np.float16, np.float32), ("bf16", "bf16", "bfloat16", "bfloat16")])
def test_gfx1201_auto_selection_package_takes_dense_and_mixed_tiles(density, storage, output, dtype, out_dtype):
    """`auto_2to4` through the public package: every K tile picks SWMMAC or
    the dense branch by wave-uniform agreement, so dense and mixed A blocks
    compute correctly and no validity word refuses (the isolated worker's
    rows proved the same selection; these are its public-route twins)."""
    rt = _gfx1201_or_skip()
    from tessera.compiler import rocm_native
    if dtype == "bfloat16":
        ml = pytest.importorskip("ml_dtypes"); dtype = out_dtype = ml.bfloat16
    m, k, n = 64, 128, 48
    artifact = scheduled_sparse.lower_scheduled_sparse_matmul(sparse_module(m, k, n, storage, output), selection="auto_2to4")
    assert artifact.selection == "auto_2to4" and "gpu.shuffle" in artifact.tile_ir
    package = rocm_native.package_sparse_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
    assert package.descriptor.provenance["selection"] == "auto_2to4"
    rng = np.random.default_rng(len(density))
    if density == "sparse":
        a = two_to_four(rng, m, k, dtype)
    elif density == "dense":
        a = (rng.normal(size=(m, k)) * 0.2).astype(dtype)
    else:
        a = two_to_four(rng, m, k, dtype)
        a[:16, :32] = (rng.normal(size=(16, 32)) * 0.2).astype(dtype)  # one dense K tile of the first row block
    b = (rng.normal(size=(k, n)) * 0.2).astype(dtype)
    out = np.zeros((m, n), out_dtype)
    result, status = _launch(rt, package, a, b, out)
    assert result["ok"] and result["execution_kind"] == "native_gpu", result
    assert np.all(status == 1)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(out.astype(np.float32), expected, rtol=4e-3, atol=2e-3)


@pytest.mark.hardware_rocm
def test_gfx1201_public_sparse_packaging_keeps_logical_native_backward():
    rt = _gfx1201_or_skip()
    import tessera as ts

    @ts.jit(target="rocm", autodiff="reverse", wrt=("a", "b"))
    def multiply(a, b):
        return ts.ops.matmul(a, b)
    a = np.zeros((16, 32), np.float16); a[:, ::4] = .5
    b = np.full((32, 16), .25, np.float16)
    artifact = multiply.package_sparse_2to4(a, b)
    assert artifact.launch_descriptor.abi_id.startswith("tessera.rocm.sparse_matmul_2to4")
    out = np.zeros((16, 16), np.float16)
    status = np.zeros(artifact.launch_descriptor.provenance["validity_words"], np.int32)
    names = [x.name for x in artifact.launch_descriptor.buffers]
    result = rt.launch(artifact, {"buffers": dict(zip(names, (a, b, out, status))), "scalars": {}})
    assert result["ok"], result
    np.testing.assert_array_equal(out, a @ b)
    da, db = multiply.native_backward(a, b, out_cotangents=np.ones((16, 16), np.float16))
    np.testing.assert_allclose(da, np.ones((16, 16), np.float32) @ b.T.astype(np.float32))
    np.testing.assert_allclose(db, a.T.astype(np.float32) @ np.ones((16, 16), np.float32))
    assert multiply.last_backward_execution["evidence_target"] == "rocm_gfx1201"
