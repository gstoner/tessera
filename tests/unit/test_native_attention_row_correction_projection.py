"""Original attention Graph -> native AD correction product, not Python reconstruction."""
import re

import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_saved_lse_half_backward import source

pytestmark = pytest.mark.compiler_route


def project(text, *, prune=False):
    tool = find_tessera_opt()
    assert tool is not None, 'production tessera-opt required'
    options = '--tessera-autodiff-paired=checkpoint-product=row_correction'
    if prune:
        options += ' prune-checkpoint-gradients=true'
    return run_tessera_opt(tool, text, options)


@pytest.mark.parametrize('storage', ['f16', 'bf16'])
@pytest.mark.parametrize('wide_output', [False, True])
@pytest.mark.parametrize('bias,saved', [(False, False), (False, True), (True, False), (True, True)])
def test_native_ad_projects_correction_without_rounded_output(storage, wide_output, bias, saved):
    result = project(source(storage, bias=bias, saved=saved,
                            output='f32' if wide_output else storage))
    body = result[result.index('  func.func @attention_row_correction_checkpoint'):]
    assert 'tessera_attn.checkpoint_row_correction' in body
    assert 'tessera_attn.checkpoint_backward' not in body
    names = ['dO', 'q', 'k', 'v'] + (['bias'] if bias else []) + ['lse'] + (['row_seed'] if saved else [])
    assert 'tessera.argument_bindings = [' + ', '.join('"' + name + '"' for name in names) + ']' in body
    assert 'tessera.result_bindings = ["row_correction"]' in body
    assert body.count('bufferization.writable = false') == len(names)
    assert '-> tensor<1x2x3xf32>' in body
    # The complete original generated pair remains a witness, not another
    # executable derivative or a substitution of rounded saved O into delta.
    assert 'tessera.attention_ad_pair' in result
    assert 'tessera_attn.checkpoint_forward' in result
    assert 'tessera_attn.checkpoint_backward' in result
    verify = run_tessera_opt(find_tessera_opt(), result, '--canonicalize')
    assert 'tessera_attn.checkpoint_row_correction' in verify


def test_native_projection_preserves_reordered_frontend_and_requested_activity():
    text = source('bf16', bias=True, saved=True)
    def reorder(match):
        parts = match[1].split(', ')
        assert len(parts) == 4
        return 'func.func @attention(' + ', '.join(parts[i] for i in (3, 2, 0, 1)) + ')'
    text = re.sub(r'func.func @attention\(([^)]*)\)', reorder, text)
    text = text.replace('tessera.autodiff = "reverse"',
                        'tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [0, 2]')
    result = project(text, prune=True)
    assert 'tessera.attention_argument_indices = array<i64: 2, 3, 1, 0>' in result
    assert 'tessera.row_correction_gradient_activity = array<i64: 1, 0, 0, 1>' in result


def test_explicit_correction_export_does_not_change_legacy_f32_policy():
    with pytest.raises(RuntimeError, match='static half checkpoint inputs'):
        project(source('f32', saved=True))


def test_correction_projection_retains_initial_static_envelope():
    text = source('f16', saved=True).replace('tessera.arch = "sm_120"',
        'tessera.arch = "sm_120", tessera.attention_sequence_bounds = array<i64: 7, 11>')
    with pytest.raises(RuntimeError, match='static half checkpoint inputs'):
        project(text)


@pytest.mark.parametrize('reorder', [False, True])
def test_native_projection_preserves_source_argument_layout_metadata(reorder):
    text = source("f16", saved=True).replace(
        "%q: tensor<1x2x3x4xf16>",
        "%q: tensor<1x2x3x4xf16> {tessera.layout = \"col_major\"}")
    if reorder:
        text = re.sub(r'func.func @attention\(([^)]*)\)',
            lambda match: 'func.func @attention(' + ', '.join(
                match[1].split(', ')[i] for i in (2, 0, 1)) + ')', text)
    result = project(text)
    body = result[result.index("  func.func @attention_row_correction_checkpoint"):]
    query = re.search(r'%arg1: tensor<1x2x3x4xf16> \{([^}]+)\}', body)
    assert query is not None
    assert 'tessera.layout = "col_major"' in query[1]
