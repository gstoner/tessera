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


def test_gfx1201_has_no_fp4_matrix_form_and_tessera_refuses_the_storage():
    """gfx1201 has no `v_wmma_*_fp4` in any form, dense or sparse.

    The failure this pins is not a crash: the common workaround elsewhere is to
    dequantize an FP4-stored weight to fp16 before the MMA, which runs at the
    *fp16* ceiling while the request still says FP4 -- a throughput claim that
    is off by 2x with nothing red anywhere. Storage dtype is a Decision #21a
    semantic key, so the arch row omits `fp4_e2m1` and the instruction table
    enumerates no FP4 form; an FP4 lane here would have to be a declared
    dequantize-and-fp16-MMA route that says so. CDNA 4 (gfx950) is the arch
    that does have native FP4/FP6 -- evidence never transfers.
    """
    from tessera.compiler.rocm_target import _ROCM_DTYPES

    for sparse in (False, True):
        forms = wmma_dtype_forms(AMDArch.GFX_1201, sparse=sparse)
        assert forms, "gfx1201 must enumerate matrix forms"
        assert not [f for f in forms if "fp4" in (f.a, f.b) or "e2m1" in f.instruction.lower()]
    assert "fp4_e2m1" not in _ROCM_DTYPES[AMDArch.GFX_1201]
    assert "fp4_e2m1" not in _ROCM_DTYPES[AMDArch.GFX_1151]
    assert "fp4_e2m1" in _ROCM_DTYPES[AMDArch.GFX_950]


def test_throughput_table_names_only_families_the_isa_has():
    """The ceilings page is the denominator every gfx1201 GEMM number is quoted
    against, so its instruction column may not name a family the arch lacks.
    Each row's family pattern must match at least one enumerated form, and the
    fp4 row must claim none."""
    doc = (ROOT / "docs/backends/rocm/wmma-fragment-layout.md").read_text()
    table = [ln for ln in doc.splitlines() if ln.startswith("| ") and "`V_" in ln]
    assert len(table) >= 5, "throughput table lost its instruction rows"
    dense = {f.instruction for f in wmma_dtype_forms(AMDArch.GFX_1201)}
    sparse = {f.instruction for f in wmma_dtype_forms(AMDArch.GFX_1201, sparse=True)}
    known = dense | sparse
    for line in table:
        family = line.split("|")[2].strip().strip("`")
        stem = family.split("{")[0].rstrip("_")
        assert any(i.startswith(stem) for i in known), f"{family} names no gfx1201 form"
    assert "**no form exists**" in doc
