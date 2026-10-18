"""Native projection and sealed package contract for ROCm mixed contractions."""
import pytest

from tessera.compiler import scheduled_matmul
from tessera.compiler.graph_ir import IRType
from tests.unit.test_scheduled_matmul_consumers import _module


def module(arch="gfx1201", half="fp16", side="a", output="fp32", orientation=(False, True), shape_mnk=(17, 19, 35)):
    a, b = ("fp32", half) if side == "a" else (half, "fp32")
    m, n, k = shape_mnk
    graph = _module(target="rocm_" + arch, shape=(m, k, n),
                    dtype=a, b_dtype=b, output_dtype=output)
    for arg, transpose in zip(graph.functions[0].args, orientation, strict=True):
        if transpose:
            old = arg.ir_type
            shape = old.shape[::-1]
            element = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}[old.dtype]
            arg.ir_type = IRType(f"tensor<{shape[0]}x{shape[1]}x{element}>", shape, old.dtype)
    op = graph.functions[0].body[0]
    op.operand_types = [str(arg.ir_type) for arg in graph.functions[0].args]
    op.kwargs.update(transposeA=orientation[0], transposeB=orientation[1])
    return graph


@pytest.mark.compiler_route
@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("half", ["fp16", "bf16"])
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("output", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("orientation", [(False, False), (False, True), (True, False), (True, True)])
def test_native_rocm_scalar_projection(compiler_toolchain, arch, half, side, output, orientation):
    compiler_toolchain.require_tessera_opt("--tessera-graph-to-schedule")
    graph = module(arch, half, side, output, orientation)
    before = graph.to_mlir(canonical=True)
    artifact = scheduled_matmul.lower_scheduled_matmul(graph, target="rocm_" + arch)
    assert graph.to_mlir(canonical=True) == before
    assert artifact.target == "rocm" and artifact.architecture == arch
    assert artifact.b_layout == "row_major"
    assert artifact.function_name.endswith("_scalar_fma_kernel")
    artifact.validate()
    scheduled_matmul.verify_matmul_projection(artifact)


def checked_fixture():
    from tessera.compiler.native_artifact import (
        BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
        NativeImageArtifact, OrderingSemantics, ScalarArgument, ShapeGuard,
    )
    from tessera.compiler.rocm_scalar_matmul import SCALAR_MATMUL_ABI
    import hashlib
    target_ir = ('module { tessera_rocm.scalar_matmul {a_element_type = f32, arch = "gfx1201", '
                 'b_element_type = f16, name = "tessera_rocm_scalar_matmul_0123456789abcdef", '
                 'output_element_type = f32, transpose_a = false, transpose_b = true} }')
    digest = "e" * 64
    schedule_ir = ('shape_key = "M=17;N=19;K=35;dtype=f32" ' + " ".join([digest] * 3))
    image = NativeImageArtifact(
        target="rocm_gfx1201", architecture="gfx1201", pipeline_name="tessera-rocm-executable",
        compiler_fingerprint="unit-only-compiler", toolchain_fingerprint="unit-only-toolchain",
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="hsaco", payload=b"unit-only-not-executed",
        entry_points=(NativeEntryPoint("tessera_rocm_scalar_matmul_0123456789abcdef", SCALAR_MATMUL_ABI),),
        compile_state="cold")
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol="tessera_rocm_scalar_matmul_0123456789abcdef", abi_id=SCALAR_MATMUL_ABI,
        buffers=(BufferBinding(0, "a", "input", "fp32", 2, "row_major", 4),
                 BufferBinding(1, "b", "input", "fp16", 2, "row_major", 2),
                 BufferBinding(2, "out", "output", "fp32", 2, "row_major", 4)),
        scalars=tuple(ScalarArgument(i + 3, name, "int64") for i, name in enumerate(("M", "N", "K"))),
        shape_guards=tuple(ShapeGuard(name, axis, "eq", size)
                          for name, shape in (("a", (17, 35)), ("b", (19, 35)), ("out", (17, 19)))
                          for axis, size in enumerate(shape)),
        geometry=LaunchGeometry(grid=(3, 2, 1), workgroup=(32, 1, 1)),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("completion",)),
        provenance={"route": "canonical_scheduled_scalar_fma", "physical_route": "ordered_fp32_scalar_fma",
                    "shape": [17, 19, 35], "scalar_types": ["fp32", "fp16", "fp32"],
                    "transposeA": False, "transposeB": True,
                    "native_target_ir": target_ir, "native_schedule_ir": schedule_ir,
                    "graph_digest": "b" * 64, "tile_ir_digest": "c" * 64,
                    "schedule_ir_digest": hashlib.sha256(schedule_ir.encode()).hexdigest(),
                    "schedule_digest": digest})
    return image, descriptor


def test_scalar_descriptor_can_be_checked_without_a_device(monkeypatch):
    from tessera import runtime
    from tessera.compiler.rocm_scalar_matmul import checked_plan
    monkeypatch.setattr(runtime, "_rocm_live_arch", lambda: pytest.fail("offline validation probed device"))
    image, descriptor = checked_fixture()
    assert checked_plan(image, descriptor)[:2] == ((17, 19, 35), ("fp32", "fp16", "fp32"))


@pytest.mark.parametrize("field,value", [
    ("route", "wmma"), ("physical_route", "wmma"), ("shape", [True, 19, 35]),
    ("shape", [17, 19, 0]), ("shape", [17, 19, 35.0]), ("shape", [17, 19]),
    ("shape", [65535 * 16 + 1, 19, 35]), ("shape", [17, 2**31, 35]),
    ("scalar_types", ["fp16", "fp16", "fp32"]),
    ("scalar_types", ["fp32", "int16", "fp32"]),
    ("scalar_types", ["fp32", "fp16", "fp64"]),
    ("transposeA", 0), ("transposeB", 1), ("transposeB", None),
])
def test_scalar_descriptor_rejects_changed_numerical_contract(field, value):
    from dataclasses import replace
    from tessera.compiler.rocm_scalar_matmul import checked_plan
    image, descriptor = checked_fixture()
    with pytest.raises(ValueError):
        checked_plan(image, replace(descriptor, provenance={**descriptor.provenance, field: value}))


@pytest.mark.parametrize("mutation", ["geometry", "buffer", "scalars", "guards", "ordering", "local"])
def test_scalar_descriptor_rejects_changed_physical_abi(mutation):
    from dataclasses import replace
    from tessera.compiler.native_artifact import LaunchGeometry, OrderingSemantics
    from tessera.compiler.rocm_scalar_matmul import checked_plan
    image, descriptor = checked_fixture()
    if mutation == "geometry":
        changed = replace(descriptor, geometry=LaunchGeometry(grid=(3, 2, 1), workgroup=(64, 1, 1)))
    elif mutation == "buffer":
        changed = replace(descriptor, buffers=(replace(descriptor.buffers[0], dtype="fp16"),
                                               *descriptor.buffers[1:]))
    elif mutation == "scalars":
        changed = replace(descriptor, scalars=descriptor.scalars[:-1])
    elif mutation == "guards":
        changed = replace(descriptor, shape_guards=descriptor.shape_guards[:-1])
    elif mutation == "ordering":
        changed = replace(descriptor, ordering=OrderingSemantics(
            ordered_submission=True, residency="inputs", synchronization=("completion",)))
    else:
        changed = replace(descriptor, dynamic_local_memory_bytes=64)
    with pytest.raises(ValueError):
        checked_plan(image, changed)


@pytest.mark.compiler_route
@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
def test_native_image_identity_reuses_shapes_and_preserves_physical_recipe(compiler_toolchain, arch):
    import subprocess
    tool = compiler_toolchain.require_tessera_opt("--tessera-rocm-project-kernel-identity")

    def identity(shape, *, half="fp16", orientation=(False, True), rename=False):
        graph = module(arch, half, "a", "fp32", orientation, shape)
        if rename:
            graph.functions[0].name = "different_frontend_function"
        artifact = scheduled_matmul.lower_scheduled_matmul(graph, target="rocm_" + arch)
        result = subprocess.run([str(tool), "--lower-tile-to-rocm=arch=" + arch, "-"],
                                input=artifact.tile_ir, text=True, capture_output=True, check=True)
        projected = subprocess.run([str(tool), "--tessera-rocm-project-kernel-identity=family=scalar_matmul", "-"],
                                  input=result.stdout, text=True, capture_output=True, check=True)
        assert "schedule_hash" not in projected.stdout
        return projected.stdout

    first = identity((17, 19, 35))
    assert identity((7, 11, 13), rename=True) == first
    assert identity((7, 11, 13), half="bf16") != first
    assert identity((7, 11, 13), orientation=(True, True)) != first

@pytest.mark.parametrize("mutation", ["dimensions", "output_type", "transpose"])
def test_coordinated_descriptor_edits_cannot_reinterpret_a_native_image(mutation):
    from dataclasses import replace
    from tessera.compiler.native_artifact import LaunchGeometry
    from tessera.compiler.rocm_scalar_matmul import checked_plan
    image, descriptor = checked_fixture()
    info = dict(descriptor.provenance)
    if mutation == "dimensions":
        info["shape"] = [7, 19, 35]
        guards = tuple(replace(g, value=7) if g.dimension == 0 and g.binding in {"a", "out"} else g
                       for g in descriptor.shape_guards)
        descriptor = replace(descriptor, provenance=info, shape_guards=guards,
                             geometry=LaunchGeometry(grid=(3, 1, 1), workgroup=(32, 1, 1)))
    elif mutation == "output_type":
        info["scalar_types"] = ["fp32", "fp16", "fp16"]
        descriptor = replace(descriptor, provenance=info, buffers=(
            *descriptor.buffers[:2], replace(descriptor.buffers[2], dtype="fp16", alignment=2)))
    else:
        info["transposeA"] = True
        guards = tuple(replace(g, value=35 if g.dimension == 0 else 17) if g.binding == "a" else g
                       for g in descriptor.shape_guards)
        descriptor = replace(descriptor, provenance=info, shape_guards=guards)
    with pytest.raises(ValueError):
        checked_plan(image, descriptor)
