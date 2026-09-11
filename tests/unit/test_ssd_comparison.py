import copy
import pytest
from benchmarks.compare_ssd_variants import summarize


def evidence():
    return [{name:dict(backend='nvidia',architecture='sm_120',compiler_sha256='compiler',shape=[32,2,16,4],
                      clock='CUDA events',execution='native_gpu',cooperative=name=='cooperative',
                      rows=[dict(chunk=8,binding_digest=name,image_sha256=name,device_event_ms=[time]*7,
                                 max_abs_errors=[0.,0.,0.])]) for name,time in [('serial',2.),('cooperative',1.)]}
            for _ in range(9)]


def test_ssd_paired_bound_and_single_outlier():
    pairs = evidence()
    pairs[0]['cooperative']['rows'][0]['device_event_ms'] = [100.]*7
    result = summarize(pairs)
    assert result['median_speedup_lower_bound'] == 2.
    assert not result['promotion_eligible']
    assert result['confidence'] > .95


@pytest.mark.parametrize('field,value',[('device_event_ms',[True]*7),('device_event_ms',[float('nan')]*7),
    ('image_sha256','changed'),('chunk',4),('max_abs_errors',[True]*3)])
def test_ssd_comparison_refuses_corrupt_or_mismatched_evidence(field,value):
    pairs = copy.deepcopy(evidence())
    pairs[-1]['cooperative']['rows'][0][field] = value
    with pytest.raises(ValueError):
        summarize(pairs)


def test_native_selector_binds_actual_candidate_only_after_calibration(monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from tessera.compiler.ssd_performance import bind_measured_ssd
    from tessera.compiler.profiler_rocm_evidence import build_rocm_profiler_packet
    from test_profiler_rocm_evidence import _timing,_capture,_image
    logical = SimpleNamespace(compiler_digest='compiler',schedule_ir='chunk_size = 8 : i64')
    specs = [SimpleNamespace(shape=(32,2,4)),None,SimpleNamespace(shape=(32,2,16))]
    def artifact(cooperative):
        name = 'cooperative' if cooperative else 'serial'
        return SimpleNamespace(logical=logical,adjoint=False,cooperative=cooperative,
            package=SimpleNamespace(backend='rocm',chip='gfx1151',binding_digest=name,image=name.encode()),
            validate=lambda:specs,bind=lambda:name)
    incumbent,candidate = artifact(False),artifact(True)
    pairs = evidence()
    for pair in pairs:
        for name,packet in pair.items():
            packet.update(backend='rocm',architecture='gfx1151',clock='HIP events')
            packet['rows'][0]['image_sha256'] = hashlib.sha256(name.encode()).hexdigest()
    comparison = dict(pairs=pairs,promotion_eligible=True,median_speedup_lower_bound=10000.)
    bound,decision = bind_measured_ssd(incumbent,candidate,comparison)
    assert bound == 'serial' and not decision.admitted
    calibrations = []
    for i,pair in enumerate(pairs):
        for name in ('serial','cooperative'):
            timing = _timing()
            timing['sample_id'] = f'{i}-{name}'
            clean,probe = _image(1,'clean'),_image(1,'probe')
            for image in (clean,probe):
                image.update(calibration_sample_id=timing['sample_id'],semantic_sha256=hashlib.sha256(logical.schedule_ir.encode()).hexdigest(),duration_ns=pair[name]['rows'][0]['device_event_ms'][0]*1e6)
            clean['image_sha256'] = pair[name]['rows'][0]['image_sha256']
            calibrations.append(build_rocm_profiler_packet(timing=timing,capture=_capture(),uninstrumented=clean,instrumented=probe,source=dict(source_commit='a'*40,worktree_dirty=False)))
    bound,decision = bind_measured_ssd(incumbent,candidate,comparison,calibrations)
    assert bound == 'cooperative' and decision.admitted
    # Altering the eligibility bit cannot bypass the native environment gate.
    calibrations[0]['timing']['execution_environment'] = 'wsl2'
    calibrations[0]['eligible_for_promotion'] = True
    with pytest.raises(ValueError,match='WSL'):
        bind_measured_ssd(incumbent,candidate,comparison,calibrations)
    for clock in calibrations[0]['timing']['clocks'].values():
        clock['eligible_for_promotion'] = False
    bound,decision = bind_measured_ssd(incumbent,candidate,comparison,calibrations)
    assert bound == 'serial' and not decision.admitted
    comparison['pairs'][0]['cooperative']['rows'][0]['image_sha256'] = 'foreign'
    with pytest.raises(ValueError,match='changed'):
        bind_measured_ssd(incumbent,candidate,comparison)
