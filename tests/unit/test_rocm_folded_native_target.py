"""Native Target admission preserves the declared folded numeric/schedule contract."""
import os
from pathlib import Path
import subprocess
import pytest
from tessera.compiler.rocm_mxfp4_folded_frontend import author_folded_scaled_matmul_graph
from tessera.compiler.scheduled_matmul import find_tessera_opt

@pytest.fixture(scope="module")
def target():
    tool=find_tessera_opt()
    if tool is None:pytest.skip("requires native compiler")
    graph=author_folded_scaled_matmul_graph(256,80,128)
    result=subprocess.run([str(tool),"--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile","--lower-tile-to-rocm=arch=gfx1201"],
        input=graph,text=True,capture_output=True)
    if result.returncode and "Unknown command line argument" in result.stderr:
        pytest.skip("requires ROCm compiler build")
    assert result.returncode==0,result.stderr
    return tool,result.stdout

@pytest.mark.parametrize("old,new",[
    ('accum = "f32"','accum = "f16"'),
    ('execution_mode = "folded_row_reference_explicit_approximate"','execution_mode = "exact_per_block"'),
    ('scale_format = "e8m0_row_reference"','scale_format = "ue4m3"'),
    ('partial_combine = "row_reference_after_full_k"','partial_combine = "per_group"'),
    ('stage_k = 64','stage_k = 128'),
    ('instruction_k = 16','instruction_k = 32'),
    ('block_m = 256','block_m = 128'),
    ('workgroup_mode = "wgp"','workgroup_mode = "invalid"'),
    ('output = "bf16"','output = "f16"'),
])
def test_native_folded_rejects_conflicting_target(target,old,new):
    tool,text=target
    assert old in text, text
    result=subprocess.run([str(tool),"--generate-wmma-gemm-kernel=via-tile=true"],
        input=text.replace(old,new),text=True,capture_output=True)
    assert result.returncode!=0
    assert "ROCM_FOLDED_NATIVE_CONTRACT" in result.stderr,result.stderr

@pytest.mark.parametrize("options",[
    "blockscale-stage-k=128","blockscale-prefetch=0","blockscale-lds-pad-bytes=0",
])
def test_native_folded_rejects_conflicting_pass_options(target,options):
    tool,text=target
    assert 'staging_prefetch = "register_next_slab"' in text
    result=subprocess.run([str(tool),"--generate-wmma-gemm-kernel=via-tile=true "+options],
        input=text,text=True,capture_output=True)
    assert result.returncode!=0
    assert "ROCM_FOLDED_NATIVE_CONTRACT" in result.stderr,result.stderr

@pytest.mark.parametrize("mode,feature",[("wgp","-cumode"),("cu","+cumode")])
def test_native_folded_accepts_matching_explicit_options(target,mode,feature):
    tool,text=target
    assert 'workgroup_mode = "wgp"' in text
    text=text.replace('workgroup_mode = "wgp"', f'workgroup_mode = "{mode}"')
    result=subprocess.run([str(tool),"--generate-wmma-gemm-kernel=via-tile=true blockscale-stage-k=64 blockscale-prefetch=1 blockscale-lds-pad-bytes=16"],
        input=text,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert "tile.fragment_folded_scale" in result.stdout
    assert "target-features" in result.stdout and feature in result.stdout

@pytest.mark.parametrize("m,n,whole_m,whole_n",[
    (256,128,True,True),(257,128,False,True),
    (256,129,True,False),(257,129,False,False),
])
def test_native_folded_derives_complete_panel_proof(target,m,n,whole_m,whole_n):
    tool,_=target
    graph=author_folded_scaled_matmul_graph(m,n,128)
    result=subprocess.run([str(tool),"--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile","--lower-tile-to-rocm=arch=gfx1201",
        "--generate-wmma-gemm-kernel=via-tile=true"],
        input=graph,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    stores=[line for line in result.stdout.splitlines() if "tile.store " in line]
    assert len(stores)==8
    # Masked stores carry five index operands/types; complete stores carry three.
    expected_index_count=3 if whole_m and whole_n else 5
    assert all(line.partition(": !tile.tile")[2].count("index")==expected_index_count for line in stores)

@pytest.mark.parametrize("old,new",[
    ('m = 256 : i64','m = 64 : i64'),
    ('n = 80 : i64','n = 0 : i64'),
    ('accum = "f32"','accum = "f16"'),
    ('scale_format = "e8m0_row_reference"','scale_format = "ue4m3"'),
    ('stage_k = 64','stage_k = 128'),
    ('partial_combine = "row_reference_after_full_k"','partial_combine = "per_group"'),
])
def test_folded_identity_checks_original_contract_before_projection(target,old,new):
    tool,text=target
    assert old in text
    result=subprocess.run([str(tool),"--tessera-rocm-project-kernel-identity=family=folded_matmul"],
        input=text.replace(old,new),text=True,capture_output=True)
    assert result.returncode!=0
    assert "ROCM_FOLDED_NATIVE_CONTRACT" in result.stderr,result.stderr

def test_folded_identity_removes_launch_shape_and_keeps_kernel_contract(target):
    tool,text=target
    result=subprocess.run([str(tool),"--tessera-rocm-project-kernel-identity=family=folded_matmul"],
        input=text,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    for fragment in ('runtime_mn','m = 0 : i64','n = 0 : i64','k = 128 : i64',
                     'whole_m = true','whole_n = false','scale_k = 128 : i64',
                     'macro_k = 128 : i64','name = "tessera_rocm_folded_matmul_'):
        assert fragment in result.stdout,result.stdout
    assert "tessera.schedule_hash" not in result.stdout
    assert "func.func" not in result.stdout
    generated=subprocess.run([str(tool),"--generate-wmma-gemm-kernel=via-tile=true"],
        input=result.stdout,text=True,capture_output=True)
    assert generated.returncode==0,generated.stderr
    assert "tile.fragment_folded_scale" in generated.stdout

@pytest.mark.parametrize("attrs,diagnostic",[
    ("runtime_mn = true","unit attribute"),
    ("runtime_mn, whole_m = true","ROCM_FOLDED_NATIVE_CONTRACT"),
    ("whole_m = true","ROCM_FOLDED_NATIVE_CONTRACT"),
])
def test_folded_generator_rejects_partial_runtime_contract(target,attrs,diagnostic):
    tool,text=target
    text=text.replace('abi = "a_bfold_sa_rowref_d_m_n_k",',
                      'abi = "a_bfold_sa_rowref_d_m_n_k", '+attrs+',',1)
    result=subprocess.run([str(tool),"--generate-wmma-gemm-kernel=via-tile=true"],
        input=text,text=True,capture_output=True)
    assert result.returncode!=0
    assert diagnostic in result.stderr,result.stderr

@pytest.mark.parametrize("old,new",[
    ('raster_group_m = 4','raster_group_m = 8'),
    ('workgroup_mode = "wgp"','workgroup_mode = "cu"'),
    ('row_guard = "cta"','row_guard = "wave"'),
    ('staging_prefetch = "register_next_slab"','staging_prefetch = "none"'),
    ('epilogue_schedule = "complete_tile_vector_scales"','epilogue_schedule = "predicated_scalar_scales"'),
])
@pytest.mark.parametrize("runtime_k",[False,True])
def test_folded_identity_preserves_each_physical_key(target,old,new,runtime_k):
    tool,text=target
    assert old in text
    images=[]
    for source in (text,text.replace(old,new)):
        result=subprocess.run([str(tool),"--tessera-rocm-project-kernel-identity=family=folded_matmul"+(" runtime-k=true" if runtime_k else "")],
            input=source,text=True,capture_output=True)
        assert result.returncode==0,result.stderr
        images.append(result.stdout)
    assert images[0]!=images[1]
    assert new in images[1]

def test_folded_identity_refuses_sibling_architecture(target):
    tool,text=target
    result=subprocess.run([str(tool),"--tessera-rocm-project-kernel-identity=family=folded_matmul"],
        input=text.replace('tessera.arch = "gfx1201"','tessera.arch = "gfx1151"'),
        text=True,capture_output=True)
    assert result.returncode!=0
    assert "ROCM_FOLDED_NATIVE_CONTRACT" in result.stderr,result.stderr

def _project_runtime_k(target):
    tool,text=target
    result=subprocess.run([str(tool),
        "--tessera-rocm-project-kernel-identity=family=folded_matmul runtime-k=true"],
        input=text,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    return tool,result.stdout

def test_folded_runtime_k_projects_full_k_scope_and_positive_loop(target):
    tool,text=_project_runtime_k(target)
    for fragment in ("runtime_k", "runtime_mn", "k = 0 : i64",
                     "scale_k = 0 : i64", "macro_k = 0 : i64",
                     "stage_k = 64 : i64", "row_reference_after_full_k"):
        assert fragment in text,text
    result=subprocess.run([str(tool),"--generate-wmma-gemm-kernel=via-tile=true"],
        input=text,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert "llvm.intr.assume" in result.stdout
    assert "tile.fragment_folded_scale" in result.stdout

@pytest.mark.parametrize("old,new",[
    ("runtime_k,", "runtime_k = true,"),
    ("scale_k = 0 : i64", "scale_k = 128 : i64"),
    ("macro_k = 0 : i64", "macro_k = 128 : i64"),
    ("k = 0 : i64,", "k = 128 : i64,"),
    ("runtime_mn,", ""),
])
def test_folded_runtime_k_rejects_incomplete_scope(target,old,new):
    tool,text=_project_runtime_k(target)
    assert old in text
    result=subprocess.run([str(tool),"--generate-wmma-gemm-kernel=via-tile=true"],
        input=text.replace(old,new,1),text=True,capture_output=True)
    assert result.returncode!=0
    assert ("ROCM_FOLDED_NATIVE_CONTRACT" in result.stderr
            or "unit attribute" in result.stderr),result.stderr

def test_folded_scale_likelihood_reaches_native_llvm(target):
    from tessera.compiler.rocm_pipeline import (
        ROCMExecutablePipeline,ROCMInputLevel,ROCMOutputLevel)
    tool,text=target
    pipeline=ROCMExecutablePipeline(family="matmul",arch="gfx1201",
        input_level=ROCMInputLevel.DIRECTIVE).pass_pipeline(output=ROCMOutputLevel.BINARY)
    result=subprocess.run([str(tool),"--pass-pipeline="+pipeline,
        "--mlir-print-ir-before=gpu-module-to-binary","--mlir-disable-threading"],
        input=text,text=True,capture_output=True)
    assert result.returncode==0,result.stderr[-3000:]
    assert "llvm.intr.expect" in result.stderr
    assert "gpu.binary" in result.stdout

@pytest.mark.parametrize("m,n,k,runtime_mn,vectorized", [
    (256, 1024, 1024, False, True),
    (256, 4096, 5120, False, True),
    (200, 1024, 1024, False, False),
    (256, 1024, 128, False, False),
    (256, 80, 1024, False, False),
    (256, 1024, 1024, True, False),
])
def test_packed_vector_scale_band_preserves_seed_and_runtime_identity(
        target, m, n, k, runtime_mn, vectorized):
    from tessera.compiler.rocm_mxfp4_packed_folded import author_packed_folded_shape_graph
    tool, _ = target
    passes = ["--tessera-graph-to-schedule", "--tessera-schedule-to-tile",
              "--lower-tile-to-rocm=arch=gfx1201"]
    if runtime_mn:
        passes.append("--tessera-rocm-project-kernel-identity=family=folded_matmul")
    passes.append("--generate-wmma-gemm-kernel=via-tile=true")
    result = subprocess.run([str(tool), *passes],
        input=author_packed_folded_shape_graph(m, n, k),
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    scales = [line for line in result.stdout.splitlines()
              if "tile.fragment_folded_scale " in line]
    assert scales
    assert all(("vector_scales = true" in line) == vectorized for line in scales)
