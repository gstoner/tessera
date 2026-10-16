"""Typed scalar memory views reject malformed native contracts before lowering."""
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt


@pytest.fixture(scope='module')
def tool():
    result=find_tessera_opt()
    if result is None:pytest.skip('requires matching production compiler')
    return result


def source(*,dtype='f16',order='row_major',access='read',address=0,rows=3,columns=5,pitch=8,
           store=False,result_dtype=None):
    pointer='!llvm.ptr' if address==0 else f'!llvm.ptr<{address}>'
    view=f'!tile.buffer_view<{dtype}, "{access}", "{order}">'
    element=dtype if result_dtype is None else result_dtype
    operation=(f'"tile.store_element"(%v,%i,%i,%value) : ({view},i64,i64,{element}) -> ()'
               if store else f'%value = "tile.load_element"(%v,%i,%i) : ({view},i64,i64) -> {element}')
    value=f'%value = arith.constant 0.0 : {element}' if store else ''
    return f'''module {{
      llvm.func @view(%base: {pointer}) {{
        %rows = arith.constant {rows} : i64
        %columns = arith.constant {columns} : i64
        %pitch = arith.constant {pitch} : i64
        %i = arith.constant 0 : i64
        %v = "tile.buffer_view"(%base,%rows,%columns,%pitch) : ({pointer},i64,i64,i64) -> {view}
        {value}
        {operation}
        llvm.return
      }}
    }}'''


@pytest.mark.compiler_route
@pytest.mark.parametrize('dtype',['f16','bf16','f32'])
@pytest.mark.parametrize('order',['row_major','col_major'])
@pytest.mark.parametrize('address',[0,1,3])
@pytest.mark.parametrize('access',['read','write'])
def test_storage_order_and_native_address_spaces(tool,dtype,order,address,access):
    text=run_tessera_opt(tool,source(dtype=dtype,order=order,address=address,
                                   access=access,store=access=='write'),'--verify-each')
    operation='tile.store_element' if access=='write' else 'tile.load_element'
    assert operation in text and '!tile.buffer_view' in text


@pytest.mark.compiler_route
@pytest.mark.parametrize('options,diagnostic',[
    ({'rows':0},'must be positive'),({'columns':-1},'must be positive'),
    ({'pitch':0},'must be positive'),({'pitch':4},'smaller than the minor extent'),
    ({'order':'col_major','pitch':2},'smaller than the minor extent'),
    ({'address':2},'generic, global or shared memory'),
    ({'store':True},'read/write permission'),
    ({'access':'write'},'read/write permission'),
    ({'result_dtype':'f32'},'matching storage element'),
])
def test_malformed_descriptor_or_access_is_rejected(tool,options,diagnostic):
    with pytest.raises(RuntimeError,match=diagnostic):
        run_tessera_opt(tool,source(**options),'--verify-each')
