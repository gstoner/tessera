"""Independent timing evidence retains raw samples and rejects fabricated summaries."""
from copy import deepcopy
import pytest
from benchmarks.repeat_rocm_prefetch import summarize
from benchmarks.measure_rocm_prefetch import immediate_wait


def packets():
    row = dict(blocks=32,width=64,rounds=7,oracle='exact',
        hip_event_ms={'prefetch':[1.,1.,1.], 'immediate_wait':[2.,2.,2.], 'nonblocking_wait':[1.,1.,1.]},
        median_ms={'prefetch':1., 'immediate_wait':2., 'nonblocking_wait':1.})
    return [dict(process_id=i, trials=3, compiler_sha256='c',source_sha256='s',recorder_sha256='r',
                 images={'image':'a'},device='gfx1151',rows=[deepcopy(row)]) for i in range(5)]


def test_fixed_count_independent_ratios_are_derived():
    result = summarize(packets())[0]
    assert result['matched_median_ratio'] == 2
    assert result['matched_runs_favoring_nonblocking'] == 5
    with pytest.raises(ValueError,match='five'):
        summarize(packets()[:4])


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),-1.,0.])
def test_invalid_raw_timing_cannot_attest_a_win(value):
    p=packets()
    p[0]['rows'][0]['hip_event_ms']['prefetch'][0]=value
    with pytest.raises(ValueError,match='samples'):
        summarize(p)


def test_repeated_process_or_edited_identity_or_summary_is_rejected():
    for edit in [lambda p:p[1].update(process_id=0),lambda p:p[1].update(compiler_sha256='other'),
                 lambda p:p[1]['rows'][0]['median_ms'].update(prefetch=.5)]:
        p=packets();edit(p)
        with pytest.raises(ValueError):
            summarize(p)


def test_matched_wait_control_changes_only_the_explicit_threshold():
    source='%loaded = llvm.load %ptr : !llvm.ptr<1> -> f32'
    drained=immediate_wait(source)
    control=drained.replace('vmcnt(0)','vmcnt(63)')
    assert control.replace('vmcnt(63)','vmcnt(0)') == drained
    with pytest.raises(ValueError):
        immediate_wait(source+source)


def test_matched_isa_rejects_a_changed_compute_instruction():
    from benchmarks.repeat_rocm_prefetch import matched_instructions
    a='s_waitcnt vmcnt(0) // 0: FF\nv_add_f32 v0, v1, v2 // 4: AA'
    b=a.replace('vmcnt(0)','vmcnt(63) expcnt(7) lgkmcnt(63)')
    assert matched_instructions(a,b)['all_other_instructions_identical']
    with pytest.raises(ValueError,match='beyond'):
        matched_instructions(a,b.replace('v_add_f32','v_mul_f32'))
