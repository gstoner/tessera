"""MSW-9 native consumer: infer frozen affine compositions from parsed SSA."""
import re
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def source(*, dynamic=False, activation=False, shared=False, policy=False, overflow=False):
    args=', %w1: tensor<2x2xf32>' if dynamic else ''
    value='3.4028234663852886e38' if overflow else '2.0'
    constants='' if dynamic else f'%w1 = arith.constant dense<{value}> : tensor<2x2xf32>'
    intermediate='%r' if activation else '%a'
    relu='%r = "tessera.relu"(%a) : (tensor<3x2xf32>) -> tensor<3x2xf32>' if activation else ''
    result='(tensor<3x2xf32>, tensor<3x2xf32>)' if shared else 'tensor<3x2xf32>'
    ret='%o, %a : tensor<3x2xf32>, tensor<3x2xf32>' if shared else '%o : tensor<3x2xf32>'
    attrs=' attributes {numeric_policy = "strict"}' if policy else ''
    return f'''module {{
func.func @ann(%x: tensor<3x2xf32>{args}) -> {result}{attrs} {{
{constants}
%w2 = arith.constant dense<3.0> : tensor<2x2xf32>
%b1 = arith.constant dense<1.0> : tensor<3x2xf32>
%b2 = arith.constant dense<4.0> : tensor<3x2xf32>
%m = "tessera.matmul"(%x, %w1) : (tensor<3x2xf32>, tensor<2x2xf32>) -> tensor<3x2xf32>
%a = "tessera.add"(%m, %b1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
{relu}
%n = "tessera.matmul"({intermediate}, %w2) : (tensor<3x2xf32>, tensor<2x2xf32>) -> tensor<3x2xf32>
%o = "tessera.add"(%n, %b2) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
return {ret}
}}
}}'''


def compile(text, enabled=True):
    tool=find_tessera_opt()
    if tool is None:
        pytest.skip('requires native compiler')
    return run_tessera_opt(tool,text,'--tessera-canonicalize=ann-reassociate='+str(enabled).lower())


def test_native_composition_discovers_and_folds_frozen_chain():
    output=compile(source())
    assert output.count('tessera.matmul')==1
    assert output.count('tessera.add')==1
    assert re.search(r'dense<1\.2[0-9]*e\+01>',output)
    assert re.search(r'dense<1\.0[0-9]*e\+01>',output)


def test_native_composition_requires_reassociation_opt_in():
    assert compile(source(),False).count('tessera.matmul')==2


@pytest.mark.parametrize('option',['dynamic','activation','shared','policy','overflow'])
def test_native_composition_refuses_unsafe_candidates(option):
    assert compile(source(**{option:True})).count('tessera.matmul')==2


def test_native_composition_preserves_distinct_bias_rows():
    text=source().replace('dense<1.0> : tensor<3x2xf32>',
        'dense<[[1.0,2.0],[3.0,4.0],[5.0,6.0]]> : tensor<3x2xf32>')
    output=compile(text)
    assert output.count('tessera.matmul')==1
    for value in ('1.300000e+01','2.500000e+01','3.700000e+01'):
        assert value in output


def test_native_composition_does_not_ignore_transpose_policy():
    text=source().replace('"tessera.matmul"(%x, %w1)',
                          '"tessera.matmul"(%x, %w1) {transposeB = true}')
    assert compile(text).count('tessera.matmul')==2
