"""Diagnostic only: execute all bounded shapes; retain strict canary/input checks."""
import os, json, hashlib
from pathlib import Path
import numpy as np
from original_bounded_fixture import test_bounded_pack_zero_fill_and_canaries
from tests.unit.test_rocm_bounded_fragment_pack import SHAPES
root=Path(__file__).resolve().parent
real_assert=np.testing.assert_array_equal
records={}; arrays={}
class Tools:
    def __init__(self,path): self.path=path
    def require_tessera_opt(self,*args): return self.path
    def require_mlir_opt(self,*args): return self.path
orderses=[('row_major','col_major'),('col_major','col_major'),('row_major','row_major'),('col_major','row_major')]
for label,binary in [('control','tessera-opt-control'),('candidate','tessera-opt-vector')]:
    tool=root/'tools'/binary
    os.environ['TESSERA_OPT']=str(tool)
    for dtype in ('f16','bf16'):
        for oi,orders in enumerate(orderses):
            counter=[0]; result=[]
            def capture(actual,expected,*args,**kwargs):
                index=counter[0];counter[0]+=1
                if index%4:
                    return real_assert(actual,expected,*args,**kwargs)
                shape=SHAPES[index//4];m,n,k=shape
                key=f'{dtype}_{oi}_{m}_{n}_{k}'
                aa=np.asarray(actual).copy();ee=np.asarray(expected).copy()
                arrays[f'{label}_{key}']=aa;arrays[f'expected_{key}']=ee
                invalid=np.ones((16,16),dtype=bool);invalid[:m,:n]=False
                result.append(dict(shape=list(shape),max_abs_error=float(np.max(np.abs(aa-ee))),
                    differing_elements=int(np.count_nonzero(aa!=ee)),
                    invalid_max_abs=float(np.max(np.abs(aa[invalid]))) if invalid.any() else 0.,
                    output_sha256=hashlib.sha256(aa.tobytes()).hexdigest()))
            np.testing.assert_array_equal=capture
            try:
                test_bounded_pack_zero_fill_and_canaries(Tools(tool),dtype,dtype,orders)
            finally:
                np.testing.assert_array_equal=real_assert
            assert counter[0]==len(SHAPES)*4,counter
            records[f'{label}_{dtype}_{oi}']=result
comparisons=[]
for key in list(arrays):
    if key.startswith('control_'):
        name=key.removeprefix('control_');a=arrays[key];b=arrays[f'candidate_{name}']
        comparisons.append(dict(case=name,bitwise_equal=bool(np.array_equal(a.view(np.uint32),b.view(np.uint32))),max_abs_delta=float(np.max(np.abs(a-b)))))
np.savez_compressed(root/'gfx1151-floating-control-candidate.npz',**arrays)
receipt=dict(architecture='gfx1151',records=records,comparisons=comparisons,
    compiler_sha256={label:hashlib.sha256((root/'tools'/binary).read_bytes()).hexdigest() for label,binary in [('control','tessera-opt-control'),('candidate','tessera-opt-vector')]},
    canary_and_input_checks='strict assertions executed for every case',all_bitwise_equal=all(c['bitwise_equal'] for c in comparisons))
(root/'gfx1151-floating-control-candidate.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(cases=len(comparisons),all_bitwise_equal=receipt['all_bitwise_equal'],max_abs_error=max(r['max_abs_error'] for group in records.values() for r in group),invalid_max_abs=max(r['invalid_max_abs'] for group in records.values() for r in group))))
