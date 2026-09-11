import json
from pathlib import Path
import pytest
from tessera.compiler.source_exception_heap import pack_exception_table
from tessera.compiler.gpu_exception_heap import emit_gpu_exception_heap, materialize_gpu_exception_heap
from tessera.compiler.scheduled_matmul import find_tessera_opt


def source():
    table = pack_exception_table([('ValueError',['@tensor','a']),('RuntimeError',['@tensor','b'])])
    contract = dict(exception_heap=table,error_dynamic=True,error_payload_sites=['a','b'])
    return 'module attributes {tessera.source_state = '+json.dumps(json.dumps(contract))+'} {}'


def test_gpu_heap_checked_transaction_and_source_replay():
    from dataclasses import replace
    tool = find_tessera_opt()
    if tool is None or not Path('/usr/lib/llvm-23/bin/mlir-opt').exists():
        pytest.skip('native GPU compiler required')
    program = materialize_gpu_exception_heap(source(),capacity=8,compiler=tool,llvm_bin='/usr/lib/llvm-23/bin',backend='nvidia',chip='sm_120')
    assert program.validate()[3].shape == (2,3)
    altered = replace(program.package,arena_ir=program.package.arena_ir.replace('arith.addi','arith.subi'))
    altered = replace(altered,binding_digest=altered._digest())
    with pytest.raises(ValueError,match='replay'):
        replace(program,package=altered).validate()


@pytest.mark.parametrize('capacity',[0,True,262145])
def test_gpu_heap_capacity_refuses(capacity):
    with pytest.raises(ValueError,match='capacity'):
        emit_gpu_exception_heap(source(),capacity)
