"""Operand-pair totality against the ISA, and native compiler refusal boundaries."""
import json
from pathlib import Path
import subprocess

import pytest
from tessera.compiler.rocm_target import AMDArch, wmma_dtype_forms
from benchmarks.rocm.record_gfx1201_wmma_types import pair_source
from tests.unit.test_rocm_arch_fragment_compiler import _lower, _serialize

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('arch,directory', [(AMDArch.GFX_1151,'rdna35'), (AMDArch.GFX_1201,'rdna4')])
@pytest.mark.parametrize('sparse', [False, True])
def test_operand_catalog_is_exact_isa_inventory(arch, directory, sparse):
    records = json.loads((ROOT/'docs/reference/isa/rdna'/directory/'instructions.json').read_text())
    prefix = 'V_SWMMAC_' if sparse else 'V_WMMA_'
    expected = {r['name'] for r in records if r['name'].startswith(prefix)}
    forms = wmma_dtype_forms(arch, sparse=sparse)
    assert {f.instruction for f in forms} == expected
    assert all(f.k in (16,32,64) for f in forms)
    assert all(f.a not in ('fp32','fp64','fp4','fp6','tf32') for f in forms)

@pytest.mark.compiler_rocm
@pytest.mark.parametrize('a,b,k,accum,signed_a,signed_b,intrinsic', [
    ('e4m3','e5m2',16,'f32',True,True,'fp8_bf8'),
    ('e5m2','e4m3',16,'f32',True,True,'bf8_fp8'),
    ('int4','int4',16,'i32',False,True,'i32.16x16x16.iu4'),
    ('int4','int4',32,'i32',True,False,'i32.16x16x32.iu4'),
    ('int8','int8',16,'i32',False,False,'i32.16x16x16.iu8'),
    ('f16','f16',16,'f16',True,True,'f16.16x16x16.f16'),
    ('bf16','bf16',16,'bf16',True,True,'bf16.16x16x16.bf16'),
])
def test_dense_operand_form_compiles(compiler_toolchain,a,b,k,accum,signed_a,signed_b,intrinsic):
    source = pair_source(a,b,k,signed_a,signed_b,accum=accum)
    lowered = _lower(compiler_toolchain,'gfx1201',source, generic=True)
    assert intrinsic in lowered
    if a.startswith('int'):
        assert f'signA = {str(signed_a).lower()}' in lowered
        assert f'signB = {str(signed_b).lower()}' in lowered
    assert 'gpu.binary' in _serialize(compiler_toolchain,lowered,'gfx1201')

@pytest.mark.compiler_rocm
@pytest.mark.parametrize('a,b,k,accum', [('e4m3','e5m2',16,'f32'), ('int4','int4',32,'i32')])
def test_rdna4_only_forms_stay_refused_on_gfx1151(compiler_toolchain,a,b,k,accum):
    with pytest.raises(AssertionError, match='ROCM_'):
        _lower(compiler_toolchain,'gfx1151',pair_source(a,b,k,accum=accum))

@pytest.mark.compiler_rocm
def test_target_bf16_cannot_reinterpret_wrong_a_type(compiler_toolchain):
    pipeline = '--pass-pipeline=builtin.module(lower-tessera-target-to-rocdl)'
    tool = compiler_toolchain.require_tessera_opt(pipeline)
    source = '''module { func.func @bad(%a: vector<8xf32>, %b: vector<8xbf16>, %c: vector<8xf32>) -> vector<8xf32> {
    %r = "tessera_rocm.wmma"(%a,%b,%c) {fragment_family = "rdna4_wmma"} : (vector<8xf32>,vector<8xbf16>,vector<8xf32>) -> vector<8xf32>
    return %r : vector<8xf32>
    } }'''
    result = subprocess.run([str(tool),'-',pipeline],input=source,text=True,capture_output=True)
    assert result.returncode != 0
    assert 'wmma' in result.stderr
