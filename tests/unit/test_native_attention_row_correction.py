"""Typed native residual contract; execution is proved by the device lane separately."""
import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.compiler_route


def correction_graph(storage='f16', output='f16', *, bias=True, seeded=True):
    types = {'do': f'tensor<1x4x7x6x{output}>',
             'q': f'tensor<1x4x7x8x{storage}>',
             'k': f'tensor<1x2x129x8x{storage}>',
             'v': f'tensor<1x2x129x6x{storage}>'}
    if bias:
        types['bias'] = 'tensor<1x4x1x129xf32>'
    types['lse'] = 'tensor<1x4x7xf32>'
    if seeded:
        types['seed'] = 'tensor<1x4x7xf32>'
    result = 'tensor<1x4x7xf32>'
    args = ', '.join(f'%{name}: {type_}' for name, type_ in types.items())
    operands = ', '.join('%' + name for name in types)
    return f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
  func.func @correction({args}) -> {result} {{
    %r = "tessera_attn.checkpoint_row_correction"({operands}) {{
      scale = 0.5 : f32, causal = true, bias = {str(bias).lower()},
      lse_cotangent = {str(seeded).lower()}
    }} : ({', '.join(types.values())}) -> {result}
    return %r : {result}
  }}
}}'''


def verify(text):
    tool = find_tessera_opt()
    assert tool is not None, 'production tessera-opt is required for compiler-route proof'
    return run_tessera_opt(tool, text, '--canonicalize')


@pytest.mark.parametrize('storage', ['f16', 'bf16'])
@pytest.mark.parametrize('wide_output', [False, True])
@pytest.mark.parametrize('bias,seeded', [(False, False), (False, True), (True, False), (True, True)])
def test_row_correction_retains_typed_inputs_and_f32_rows(storage, wide_output, bias, seeded):
    text = correction_graph(storage, 'f32' if wide_output else storage, bias=bias, seeded=seeded)
    result = verify(text)
    assert 'tessera_attn.checkpoint_row_correction' in result
    assert 'tensor<1x4x7xf32>' in result
    assert 'tensor<1x4x7x8x' + storage + '>' in result


@pytest.mark.parametrize('old,new,message', [
    ('tensor<1x2x129x8xf16>', 'tensor<1x2x129x8xbf16>', 'homogeneous half'),
    ('tensor<1x4x7x6xf16>', 'tensor<1x4x7x6xbf16>', 'matching half or f32 dO'),
    ('tensor<1x4x7xf32>', 'tensor<1x4x7xf16>', 'LSE, seed or result shapes'),
    ('tensor<1x4x1x129xf32>', 'tensor<1x4x2x129xf32>', 'bias axes'),
    ('tensor<1x4x1x129xf32>', 'tensor<1x4x1x129xf16>', 'bias requires rank-four f32'),
    ('tensor<1x2x129x6xf16>', 'tensor<1x2x129x5xf16>', 'data, LSE, seed or result shapes'),
    ('1x2x129', '1x3x129', 'data, LSE, seed or result shapes'),
    ('tensor<1x4x7x8xf16>', 'tensor<1x4x?x8xf16>', 'explicit native shape bounds'),
    ('tensor<1x4x7x8xf16>', 'tensor<1x4x0x8xf16>', 'positive or bounded'),
    ('scale = 0.5', 'scale = -0.5', 'finite positive'),
    ('scale = 0.5', 'scale = 0.0', 'finite positive'),
    ('bias = true', 'bias = false', 'input count'),
    ('lse_cotangent = true', 'lse_cotangent = false', 'input count'),
    ('causal = true', 'causal = true, execution_mode = "approximate"', 'unsupported policy attribute'),
])
def test_row_correction_rejects_inconsistent_physical_or_numeric_contract(old, new, message):
    with pytest.raises(RuntimeError, match=message):
        verify(correction_graph().replace(old, new))


def test_row_correction_result_must_be_f32_even_with_valid_lse_and_seed():
    text = correction_graph().replace('-> tensor<1x4x7xf32>', '-> tensor<1x4x7xf16>')
    text = text.replace('return %r : tensor<1x4x7xf32>', 'return %r : tensor<1x4x7xf16>')
    with pytest.raises(RuntimeError, match='LSE, seed or result shapes'):
        verify(text)
