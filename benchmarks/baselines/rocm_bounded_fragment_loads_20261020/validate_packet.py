import json,math
from pathlib import Path
E=Path(__file__).resolve().parent
V='276e09dfe344a910b345a71de4d39eeaf430a3a18a290fc052b1489cc648adb2';C='db3a4e86bb42c937d4da70fb98a45266e8e758241c7c773115906d9a6ab871ec'
result={'format_runs':[],'exact_program_runs':[]}
first=None
for file in ['fp8_formats_first.json','fp8_formats_counterbalanced.json']:
 x=json.loads((E/file).read_text());assert x['architecture']=='gfx1201';assert x['compiler_sha256']==V;assert x['reference_compiler_sha256']==C;assert len(x['rows'])==24
 keyed={};disagreements=[];durations=[]
 for r in x['rows']:
  key=(tuple(r['shape_mnk']),r['format']);assert key not in keyed;keyed[key]=r
  assert r['timing_admission']['status']=='duration_admitted'
  for arm in ['candidate','reference']:
   assert len(r['samples'][arm])==8
   assert r['metadata'][arm]['loaded_resources']['block_threads']==math.prod(r['metadata'][arm]['block'])
   for s in r['samples'][arm]:
    assert math.isfinite(s['device_window_ms']) and s['device_window_ms']>=10
    durations.append(s['device_window_ms']);disagreements.append(s['device_event_disagreement'])
  if r['format']=='mxfp4_folded':assert r['approximate_policy']!='none'
  else:assert r['approximate_policy']=='none'
 if first:
  assert set(first)==set(keyed)
  assert all(r['input_sha256']==first[k]['input_sha256'] for k,r in keyed.items())
 else:first=keyed
 result['format_runs'].append(dict(file=file,rows=24,min_device_window_ms=min(durations),max_device_event_disagreement=max(disagreements),byte_identical_repeat_inputs=True))
for file in ['exact_mxfp4_paired_program.json','exact_mxfp4_paired_program_repeat.json']:
 x=json.loads((E/file).read_text());assert x['architecture']=='gfx1201';assert x['compiler_sha256']=={'candidate':V,'reference':C};assert len(x['rows'])==2
 for r in x['rows']:
  assert r['admission']['status']=='duration_admitted'
  assert r['metadata']['candidate']['program_digest']==r['metadata']['reference']['program_digest']
  for arm,samples in r['samples'].items():
   assert len(samples)==(8 if 'repeat' not in file else 16)
   for s in samples:assert s['device_window_ms']>=10 and math.isfinite(s['device_window_ms'])
 result['exact_program_runs'].append(dict(file=file,rows=len(x['rows']),ratios=[r['median_paired_ratio'] for r in x['rows']]))
x=json.loads((E/'exact_mxfp4_member_domains.json').read_text());assert len(x['rows'])==4
assert all(r['max_abs_error']==0 for r in x['rows'])
x=json.loads((E/'gfx1151/gfx1151-device-identity.json').read_text());assert x['architecture']=='gfx1151';assert x['compiler_hashes']['tessera-opt-vector']==V
assert x['final_vector_gate']['passed']==16 and x['final_vector_gate']['skipped']==16
x=json.loads((E/'gfx1151/gfx1151-floating-control-candidate.json').read_text());assert x['all_bitwise_equal'];assert len(x['comparisons'])==96;assert x['compiler_sha256']=={'control':C,'candidate':V}
assert all(r['invalid_max_abs']==0 for group in x['records'].values() for r in group)
x=json.loads((E/'gfx1151/direct-wmma-receipt.json').read_text());assert len(x['cases'])==24 and x['all_bitwise_equal']
identity=json.loads((E/'frozen_vector_identity.json').read_text())
correction=json.loads((E/'rejected_geometry.json').read_text())['source_provenance_correction']
result['native_source_sha256']={'TileToROCM.cpp':identity['candidate_source_sha256']['src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp'],'GenerateWMMAGemmKernel.cpp':correction['corrected_value']}
result['status']='passed';result['claim']='Measurement/provenance consistency only; not aggregate compiler, model-quality, or performance closure'
(E/'packet_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
