"""Runtime descriptor maps must retain logical-shape proofs through GPU lowering."""
import os
from pathlib import Path
import subprocess
import pytest
from tessera.compiler.native_gpu_storage import _run
from tessera.compiler.native_public_result import _prepare
from tessera.compiler.rocm_pipeline import ROCMExecutablePipeline

ROOT=Path(__file__).resolve().parents[2]


def compiler():
    path=Path(os.environ.get('TESSERA_OPT','/nonexistent'))
    if not path.is_file():pytest.skip('native compiler required')
    return path


def buffered(source,role):
    path=compiler()
    exported=_run(path,'--tessera-autodiff-paired=box-product-scalars=true export-product='+role,source=source)
    native=_run(path,'--tessera-to-linalg',source=exported)
    return _run(Path('/usr/lib/llvm-23/bin/mlir-opt'),'--allow-unregistered-dialect','--convert-elementwise-to-linalg',
        '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map',
        '--convert-linalg-to-loops','--canonicalize',source=native)


@pytest.mark.parametrize('backend',['nvidia','rocm'])
@pytest.mark.parametrize('role',['forward','backward'])
def test_runtime_strided_map_native_shape_contract(backend,role):
    source=(ROOT/'tests/fixtures/source_maps/runtime_even_columns.mlir').read_text()
    gpu,metadata,_=_prepare(buffered(source,role),compiler(),backend,16,16)
    assert metadata['results'][0]['rank']==2
    assert metadata['results'][0]['capacity']==16
    assert 'tessera.native_result_abi' in gpu


def test_zero_divisor_does_not_prove_storage_bounds():
    source=(ROOT/'tests/fixtures/source_maps/runtime_even_columns.mlir').read_text().replace('%two = arith.constant 2 : index','%two = arith.constant 0 : index')
    with pytest.raises(subprocess.CalledProcessError):_prepare(buffered(source,'forward'),compiler(),'nvidia',16,16)


def test_cooperative_pipeline_is_explicit_and_has_distinct_identity():
    default=ROCMExecutablePipeline('depth_attention')
    candidate=ROCMExecutablePipeline('depth_attention',depth_cooperative=True)
    assert 'depth-cooperative=true' not in default.pass_pipeline()
    assert 'depth-cooperative=true' in candidate.pass_pipeline()
    assert default.cache_key()!=candidate.cache_key()
    with pytest.raises(ValueError):ROCMExecutablePipeline('matmul',depth_cooperative=True)


def test_depth_cooperative_generator_has_uniform_shared_tree():
    tool=Path(os.environ.get('TESSERA_ROCM_OPT',str(ROOT/'build/src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt')))
    if not tool.is_file():pytest.skip('standalone ROCm compiler required')
    source=(ROOT/'src/compiler/codegen/Tessera_ROCM_Backend/test/rocm/gfx1151_tile_depth_attention_kernel.mlir').read_text()
    base=_run(tool,'--allow-unregistered-dialect','--lower-tile-to-rocm=arch=gfx1151','--generate-rocm-depth-attention-kernel',source=source)
    candidate=_run(tool,'--allow-unregistered-dialect','--lower-tile-to-rocm=arch=gfx1151','--generate-rocm-depth-attention-kernel=cooperative-width=true',source=source)
    assert 'memref<2x256xf32' not in base
    assert 'memref<2x256xf32' in candidate
    assert candidate.count('gpu.barrier')>=10


def test_gpu_exception_ad_refuses_without_product_aware_completion():
    import json
    from tessera.compiler.native_public_result import materialize_ad_public_results
    contract=json.dumps(json.dumps({'error_specs': [[[1],'f32']]}))
    source='module attributes {tessera.source_state = '+contract+'} {}'
    with pytest.raises(ValueError,match='checked forward binding'):
        materialize_ad_public_results(source,compiler='/missing',llvm_bin='/missing',backend='nvidia',chip='sm_120',capacity=4)


@pytest.mark.parametrize('mutation',['remove','invert'])
def test_slice_adjoint_shape_proof_requires_dominating_equality(mutation):
    import re
    source=(ROOT/'tests/fixtures/source_maps/runtime_even_columns.mlir').read_text()
    native=buffered(source,'backward')
    assert 'slice cotangent shape mismatch' in native
    if mutation=='remove':native=re.sub(r'^\s*cf\.assert[^\n]*\n','',native,flags=re.MULTILINE)
    else:native=native.replace('arith.cmpi eq,','arith.cmpi ne,')
    with pytest.raises(subprocess.CalledProcessError):_prepare(native,compiler(),'nvidia',16,16)
