"""Original two-quantizer NVFP4 gate/up-to-down native SSA composition."""
import re
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
from tests.unit.test_native_quantized_gate_up_program import source as gate_up_source
from tests.unit.test_native_quantized_nvfp4_product_program import composed_source
from tests.unit.test_native_quantized_product_program import manifest,OPTION

pytestmark=pytest.mark.compiler_route


def source(m=3,width=32,k=48,n=19):
    gate=gate_up_source(m,width,k)
    down=composed_source(m,n,width)
    start=down.index('(',down.index('func.func'))+1;end=down.index(') ->',start)
    args=down[start:end].split(', ')[1:]
    body=down[down.index('{',end)+1:down.rindex('}')].rsplit('}',1)[0]
    body=re.sub(r'%([A-Za-z_][A-Za-z_0-9]*)',r'%down_\1',body)
    body=body.replace('%down_x','%activation')
    before,arrow,after=body.rpartition(f'-> tensor<{m}x{n}xf32>')
    assert arrow
    body=before+f'-> tensor<{m}x{n}xbf16>'+after
    body=body.replace(f'return %down_result : tensor<{m}x{n}xf32>',f'return %down_result : tensor<{m}x{n}xbf16>')
    args=[re.sub(r'%([A-Za-z_][A-Za-z_0-9]*)',r'%down_\1',arg) for arg in args]
    start=gate.index('(',gate.index('func.func'))+1;end=gate.index(') ->',start)
    gate=gate[:end]+', '+', '.join(args)+gate[end:]
    old=f') -> tensor<{m}x{width}xbf16>'
    assert old in gate
    gate=gate.replace(old,f') -> tensor<{m}x{n}xbf16>',1)
    old=f'return %activation : tensor<{m}x{width}xbf16>'
    assert old in gate
    return gate.replace(old,body.strip())


@pytest.mark.usefixtures('production_compiler')
@pytest.mark.parametrize('m,width,k,n',[(1,16,16,1),(3,32,48,19)])
def test_native_seven_members_derive_full_mlp_private_lifetimes(m,width,k,n):
    text=source(m,width,k,n)
    run_tessera_opt(find_tessera_opt(),text,'--verify-each')
    plan=manifest(text)
    assert plan['schema']=='tessera.native.quantized_product.v4'
    assert len(plan['steps'])==7 and len(plan['buffers'])==18
    assert plan['output']==17
    assert [s['inputs'] for s in plan['steps']]==[[0],[1,2,3],[7,9,8,10],[11],[12],[4,5,6],[13,15,14,16]]
    assert [s['outputs'] for s in plan['steps']]==[[7,8],[9,10],[11],[12],[13,14],[15,16],[17]]
    assert [(b['first_write'],b['last_read']) for b in plan['buffers'][7:]]==[(0,2),(0,2),(1,2),(1,2),(2,3),(3,4),(4,6),(4,6),(5,6),(5,6),(6,7)]
    assert plan['buffers'][12]['storage']=='bf16'
    assert plan['buffers'][12]['shape']==[m,width]
    assert plan['buffers'][17]['storage']=='bf16'
    assert plan['buffers'][17]['shape']==[m,n]
    for i,member in enumerate(plan['member_graphs']):
        assert run_tessera_opt(find_tessera_opt(),text,OPTION+f' select-scaled-member={i}')==member
        schedule=run_tessera_opt(find_tessera_opt(),member,'--tessera-graph-to-schedule')
        tile=run_tessera_opt(find_tessera_opt(),schedule,'--tessera-schedule-to-tile')
        assert 'tile.' in tile


@pytest.mark.usefixtures('production_compiler')
def test_native_mlp_root_permutation_preserves_both_weight_roles():
    text=source()
    start=text.index('(',text.index('func.func'))+1;end=text.index(') ->',start)
    args=text[start:end].split(', ');assert len(args)==7
    text=text[:start]+', '.join(args[i] for i in (6,2,0,5,1,4,3))+text[end:]
    plan=manifest(text)
    assert plan['steps'][0]['inputs']==[2]
    assert plan['steps'][1]['inputs']==[4,1,6]
    assert plan['steps'][5]['inputs']==[5,3,0]


@pytest.mark.usefixtures('production_compiler')
@pytest.mark.parametrize('old,new',[
    ('kind = "max"','kind = "sum"'),
    ('dense<448.0>','dense<449.0>'),
    ('execution_mode = "exact_per_block"','execution_mode = "approximate"'),
    ('f64_product_to_f32_rne','truncate'),
])
def test_native_mlp_preserves_and_rejects_conflicting_down_math(old,new):
    text=source()
    run_tessera_opt(find_tessera_opt(),text,'--verify-each')
    offset=text.index('%down_blocks =')
    assert old in text[offset:]
    text=text[:offset]+text[offset:].replace(old,new)
    with pytest.raises(RuntimeError):manifest(text)
