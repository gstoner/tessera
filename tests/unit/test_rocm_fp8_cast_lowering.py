"""Typed FP8 storage legalization, independently of quantization frontends."""
import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.compiler_route

SOURCE = 'module attributes {gpu.container_module} {\n gpu.module @probe {\n  gpu.func @fp8_cast(%input: memref<?xf32>, %codes: memref<?xf8E4M3FN>, %count: index) kernel {\n   %tid = gpu.thread_id x\n   %bid = gpu.block_id x\n   %c256 = arith.constant 256 : index\n   %base = arith.muli %bid, %c256 : index\n   %index = arith.addi %base, %tid : index\n   %active = arith.cmpi ult, %index, %count : index\n   scf.if %active {\n    %value = memref.load %input[%index] : memref<?xf32>\n    %code = arith.truncf %value : f32 to f8E4M3FN\n    memref.store %code, %codes[%index] : memref<?xf8E4M3FN>\n   }\n   gpu.return\n  }\n }\n}\n'


def source(arch="gfx1201"):
    return SOURCE.replace("gpu.container_module", f'gpu.container_module, tessera.arch = "{arch}"')


@pytest.mark.usefixtures("production_compiler")
def test_scalar_fp32_to_e4m3_reaches_real_rocdl_storage_conversion():
    output = run_tessera_opt(find_tessera_opt(), source(),
                            "--lower-tessera-target-to-rocdl")
    assert "rocdl.cvt.pk.fp8.f32" in output
    assert "arith.truncf" not in output
    assert "arith.bitcast" in output


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("arch", ["gfx1151", "gfx1200"])
def test_scalar_e4m3_conversion_does_not_transfer_architecture_admission(arch):
    output = run_tessera_opt(find_tessera_opt(), source(arch),
                            "--lower-tessera-target-to-rocdl")
    assert "rocdl.cvt.pk.fp8.f32" not in output
    assert "arith.truncf" in output


@pytest.mark.usefixtures("production_compiler")
def test_e4m3_legalization_never_injects_device_intrinsics_into_host_functions():
    host = """module attributes {tessera.arch = "gfx1201"} {
      func.func @host(%input: f32) -> f8E4M3FN {
        %code = arith.truncf %input : f32 to f8E4M3FN
        return %code : f8E4M3FN
      }
    }"""
    output = run_tessera_opt(find_tessera_opt(), host,
                            "--lower-tessera-target-to-rocdl")
    assert "arith.truncf" in output
    assert "rocdl.cvt.pk.fp8.f32" not in output


@pytest.mark.usefixtures("production_compiler")
def test_e4m3_cast_preserves_an_explicit_directed_rounding_request():
    directed = source().replace(
        "%code = arith.truncf %value : f32 to f8E4M3FN",
        '%code = arith.truncf %value toward_zero : f32 to f8E4M3FN')
    output = run_tessera_opt(find_tessera_opt(), directed,
                            "--lower-tessera-target-to-rocdl")
    assert "arith.truncf" in output and "toward_zero" in output
    assert "rocdl.cvt.pk.fp8.f32" not in output


@pytest.mark.usefixtures("production_compiler")
def test_e4m3_legalization_uses_pipeline_selected_architecture_without_authored_copy():
    selected = SOURCE.replace("gpu.container_module",
        'gpu.container_module, tessera.pipeline.arch = "gfx1201"')
    output = run_tessera_opt(find_tessera_opt(), selected,
                            "--lower-tessera-target-to-rocdl")
    assert "rocdl.cvt.pk.fp8.f32" in output and "arith.truncf" not in output


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("authored,selected", [("gfx1201","gfx1151"),("gfx1151","gfx1201")])
def test_e4m3_legalization_rejects_conflicting_authored_and_selected_architectures(authored,selected):
    conflicting = source(authored).replace("gpu.container_module",
        f'gpu.container_module, tessera.pipeline.arch = "{selected}"')
    with pytest.raises(RuntimeError,match="pipeline architecture differs"):
        run_tessera_opt(find_tessera_opt(), conflicting,"--lower-tessera-target-to-rocdl")
