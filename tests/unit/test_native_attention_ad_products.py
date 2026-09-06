"""Native attention AD interfaces preserve the checkpoint policy and LSE SSA."""
from pathlib import Path
import subprocess
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt


def source(mode, wrt='', dropout='0.0'):
    return '''module {
      func.func @attention(%q: tensor<1x2x4x8xf32>, %k: tensor<1x1x6x8xf32>, %v: tensor<1x1x6x8xf32>) -> tensor<1x2x4x8xf32>
      attributes {tessera.autodiff = "MODE" WRT} {
        %o = tessera.flash_attn %q, %k, %v {head_dim = 8 : i64, causal = true, dropout_p = DROP : f64, operandSegmentSizes = array<i32: 1, 1, 1, 0>}
          : (tensor<1x2x4x8xf32>, tensor<1x1x6x8xf32>, tensor<1x1x6x8xf32>) -> tensor<1x2x4x8xf32>
        return %o : tensor<1x2x4x8xf32>
      }
    }'''.replace('MODE',mode).replace('WRT',wrt).replace('DROP',dropout)


def run(text, mode):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler unavailable')
    return subprocess.run([str(tool), '--tessera-autodiff-'+mode], input=text, text=True, capture_output=True, timeout=30)


def test_reverse_attention_uses_explicit_lse_producer_consumer():
    result = run(source('reverse'), 'paired')
    assert result.returncode == 0, result.stderr
    assert 'tessera_attn.checkpoint_forward' in result.stdout
    assert 'tessera_attn.checkpoint_backward' in result.stdout
    assert 'tensor<1x2x4xf32>' in result.stdout
    assert 'causal = true' in result.stdout
    # No quadratic score materialization or handwritten derivative expression.
    assert 'tessera.matmul' not in result.stdout


def test_value_only_jvp_uses_linear_checkpoint_product():
    result = run(source('forward', ', tessera.autodiff.wrt_indices = [2]'), 'forward')
    assert result.returncode == 0, result.stderr
    assert 'tessera_attn.checkpoint_forward' in result.stdout


@pytest.mark.parametrize('mode,flag,wrt,dropout', [
    ('forward', 'forward', ', tessera.autodiff.wrt_indices = [0]', '0.0'),
    ('reverse', 'paired', '', '0.2'),
])
def test_unimplemented_attention_products_fail_closed(mode,flag,wrt,dropout):
    result = run(source(mode,wrt,dropout),flag)
    assert result.returncode != 0


def test_interfaces_are_registered_on_the_public_attention_op():
    text=(Path(__file__).resolve().parents[2]/'src/compiler/ir/TesseraOps.td').read_text()
    body=text.split('def Tessera_FlashAttnOp :',1)[1].split('let summary',1)[0]
    assert 'Tessera_AdjointInterface' in body and 'Tessera_TangentInterface' in body
