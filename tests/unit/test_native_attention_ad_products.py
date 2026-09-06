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
    ('reverse', 'paired', '', '0.2'),
])
def test_unimplemented_attention_products_fail_closed(mode,flag,wrt,dropout):
    result = run(source(mode,wrt,dropout),flag)
    assert result.returncode != 0


def test_interfaces_are_registered_on_the_public_attention_op():
    text=(Path(__file__).resolve().parents[2]/'src/compiler/ir/TesseraOps.td').read_text()
    body=text.split('def Tessera_FlashAttnOp :',1)[1].split('let summary',1)[0]
    assert 'Tessera_AdjointInterface' in body and 'Tessera_TangentInterface' in body


def test_paired_attention_persists_forward_lse_without_backward_recompute():
    result = run(source('reverse'), 'paired')
    assert result.returncode == 0, result.stderr
    forward, backward = result.stdout.split('func.func @attention__bwd', 1)
    assert forward.count('tessera_attn.checkpoint_forward') == 1
    assert 'tessera_attn.checkpoint_forward' not in backward
    assert 'tessera.flash_attn ' not in result.stdout
    assert 'tessera.flash_attn:lse' in forward and 'tessera.flash_attn:lse' in backward
    assert 'tessera_attn.checkpoint_backward' in backward
    # Q, K, V, dO, persisted LSE: backward consumes its fifth argument.
    assert '%arg4' in backward.split('tessera_attn.checkpoint_backward', 1)[1].split(':', 1)[0]


def test_two_attention_results_keep_distinct_lse_residual_slots():
    text = source('reverse')
    shape = 'tensor<1x2x4x8xf32>'
    start = text.index('        %o =')
    end = text.index('        return')
    second = text[start:end].replace('%o =', '%second =').replace('causal = true', 'causal = false')
    text = text[:end] + second + text[end:]
    text = text.replace(f'-> {shape}\n      attributes', f'-> ({shape}, {shape})\n      attributes')
    text = text.replace(f'return %o : {shape}', f'return %o, %second : {shape}, {shape}')
    result = run(text, 'paired')
    assert result.returncode == 0, result.stderr
    forward, backward = result.stdout.split('func.func @attention__bwd', 1)
    assert forward.count('tessera_attn.checkpoint_forward') == 2
    assert 'tessera_attn.checkpoint_forward' not in backward
    products = backward.split('tessera_attn.checkpoint_backward')[1:]
    assert len(products) == 2
    assert '%arg6' in products[0].split(':', 1)[0]
    assert '%arg5' in products[1].split(':', 1)[0]
    assert 'causal = false' in products[0] and 'causal = true' in products[1]


@pytest.mark.parametrize('backward',[False,True])
def test_generated_attention_exports_to_physical_checkpoint(backward):
    from tessera.compiler.scheduled_checkpoint import lower_generated_checkpoint
    if find_tessera_opt() is None:
        pytest.skip('native compiler unavailable')
    text = source('reverse').replace('module {','module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {',1)
    artifact=lower_generated_checkpoint(text,backward=backward)
    assert artifact.backward is backward
    assert artifact.dims==(1,2,1,4,6,8,8)
    assert 'tessera.attention_ad_pair' in artifact.graph_ir
    assert 'tile.attention_backward_kernel' in artifact.tile_ir if backward else 'tile.attention_kernel' in artifact.tile_ir


def test_checkpoint_export_refuses_unsupported_role():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler unavailable')
    result = subprocess.run([str(tool),'--tessera-autodiff-paired=checkpoint-product=jvp'],input=source('reverse'),text=True,capture_output=True)
    assert result.returncode != 0
    assert 'forward/backward role' in result.stderr


@pytest.mark.parametrize('indices',['0','1','0, 1','0, 1, 2'])
def test_automatic_score_jvp_has_same_generation_checkpoint(indices):
    result=run(source('forward', ', tessera.autodiff.wrt_indices = ['+indices+']'),'forward')
    assert result.returncode==0,result.stderr
    assert 'tessera_attn.checkpoint_jvp' in result.stdout
    assert 'tensor<1x2x4xf32>' in result.stdout


def test_score_jvp_rejects_mixed_lse_generation():
    import re
    result=run(source('forward'),'forward')
    assert result.returncode==0,result.stderr
    # Swap O and LSE in the product: equal provenance labels cannot hide an
    # invalid typed producer relationship.
    line=next(line for line in result.stdout.splitlines() if 'tessera_attn.checkpoint_jvp' in line)
    operands=re.search(r'checkpoint_jvp ([^{]+)',line)[1].strip().split(', ')
    operands[3],operands[4]=operands[4],operands[3]
    corrupted=result.stdout.replace(line,line.replace(re.search(r'checkpoint_jvp ([^{]+)',line)[1],', '.join(operands)+' '))
    tool=find_tessera_opt()
    checked=subprocess.run([str(tool)],input=corrupted,text=True,capture_output=True,timeout=30)
    assert checked.returncode!=0


def test_score_jvp_rejects_equal_shape_lse_from_another_forward():
    import re
    result=run(source('forward'),'forward')
    assert result.returncode==0,result.stderr
    forward=next(line for line in result.stdout.splitlines() if 'tessera_attn.checkpoint_forward' in line)
    duplicate='    %other_output, %other_lse ='+forward.split('=',1)[1]
    line=next(line for line in result.stdout.splitlines() if 'tessera_attn.checkpoint_jvp' in line)
    args=re.search(r'checkpoint_jvp ([^{]+)',line)[1]
    operands=args.strip().split(', ')
    operands[4]='%other_lse'
    corrupted=result.stdout.replace(forward,forward+'\n'+duplicate).replace(line,line.replace(args,', '.join(operands)+' '))
    checked=subprocess.run([str(find_tessera_opt())],input=corrupted,text=True,capture_output=True,timeout=30)
    assert checked.returncode!=0
    assert 'same forward' in checked.stderr
