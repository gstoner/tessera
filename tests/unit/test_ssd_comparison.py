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
    for i,pair in enumerate(pairs):
        for name,packet in pair.items():
            packet.update(backend='rocm',architecture='gfx1151',clock='HIP events',run_id=f'{i}-{name}')
            packet['rows'][0]['image_sha256'] = hashlib.sha256(name.encode()).hexdigest()
    comparison = dict(pairs=pairs,promotion_eligible=True,median_speedup_lower_bound=10000.,source=dict(source_commit='a'*40))
    bound,decision = bind_measured_ssd(incumbent,candidate,comparison)
    assert bound == 'serial' and not decision.admitted
    calibrations = []
    for i,pair in enumerate(pairs):
        for name in ('serial','cooperative'):
            timing = _timing()
            timing['sample_id'] = f'{i}-{name}'
            timing['environment']['run_id'] = f'{i}-{name}'
            clean,probe = _image(1,'clean'),_image(1,'probe')
            for image in (clean,probe):
                image.update(calibration_sample_id=timing['sample_id'],semantic_sha256=hashlib.sha256(logical.schedule_ir.encode()).hexdigest(),duration_ns=pair[name]['rows'][0]['device_event_ms'][0]*1e6)
            clean['image_sha256'] = pair[name]['rows'][0]['image_sha256']
            calibrations.append(build_rocm_profiler_packet(timing=timing,capture=_capture(),uninstrumented=clean,instrumented=probe,source=dict(source_commit='a'*40,worktree_dirty=False)))
    bound,decision = bind_measured_ssd(incumbent,candidate,comparison,calibrations)
    assert bound == 'cooperative' and decision.admitted
    # Even a fast, fully calibrated candidate must pass the numerical gate.
    bad = copy.deepcopy(comparison)
    bad['pairs'][-1]['cooperative']['rows'][0]['max_abs_errors'][2] = 1e6
    with pytest.raises(ValueError, match='absolute admission tolerance'):
        bind_measured_ssd(incumbent,candidate,bad,calibrations)
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


def test_ssd_admits_a_wsl_device_clock_witness_calibration():
    """Owner direction 2026-09-25: SSD promotion on gfx1151 no longer needs a
    rocprofiler packet (KFD) when every process carries the device-clock
    witness; a disagreeing witness still refuses."""
    import hashlib
    from types import SimpleNamespace
    from tessera.compiler.ssd_performance import bind_measured_ssd
    from tessera.compiler.profiler_rocm_evidence import build_rocm_profiler_packet
    from test_profiler_rocm_evidence import _wsl_witness_timing, _no_kfd_capture, _image
    logical = SimpleNamespace(compiler_digest='compiler',schedule_ir='chunk_size = 8 : i64')
    specs = [SimpleNamespace(shape=(32,2,4)),None,SimpleNamespace(shape=(32,2,16))]
    def artifact(cooperative):
        name = 'cooperative' if cooperative else 'serial'
        return SimpleNamespace(logical=logical,adjoint=False,cooperative=cooperative,
            package=SimpleNamespace(backend='rocm',chip='gfx1151',binding_digest=name,image=name.encode()),
            validate=lambda:specs,bind=lambda:name)
    incumbent,candidate = artifact(False),artifact(True)
    pairs = evidence()
    for i,pair in enumerate(pairs):
        for name,packet in pair.items():
            packet.update(backend='rocm',architecture='gfx1151',clock='HIP events',run_id=f'{i}-{name}')
            packet['rows'][0]['image_sha256'] = hashlib.sha256(name.encode()).hexdigest()
    comparison = dict(pairs=pairs,promotion_eligible=True,median_speedup_lower_bound=10000.,source=dict(source_commit='a'*40))
    def calibrations(event_ns):
        out = []
        for i,pair in enumerate(pairs):
            for name in ('serial','cooperative'):
                timing = _wsl_witness_timing(event_ns=event_ns,
                    image_sha256=pair[name]['rows'][0]['image_sha256'])
                timing['sample_id'] = f'{i}-{name}'
                timing['environment']['run_id'] = f'{i}-{name}'
                clean,probe = _image(1,'clean'),_image(1,'probe')
                for image in (clean,probe):
                    image.update(calibration_sample_id=timing['sample_id'],semantic_sha256=hashlib.sha256(logical.schedule_ir.encode()).hexdigest(),duration_ns=pair[name]['rows'][0]['device_event_ms'][0]*1e6)
                clean['image_sha256'] = pair[name]['rows'][0]['image_sha256']
                out.append(build_rocm_profiler_packet(timing=timing,capture=_no_kfd_capture(),uninstrumented=clean,instrumented=probe,source=dict(source_commit='a'*40,worktree_dirty=False)))
        return out
    bound,decision = bind_measured_ssd(incumbent,candidate,comparison,calibrations(10_100))
    assert bound == 'cooperative' and decision.admitted
    with pytest.raises(ValueError, match='disagree'):
        calibrations(20_000)
    # A calibration naming another process's run is refused (review).
    stolen = calibrations(10_100)
    stolen[0]['timing']['environment']['run_id'] = 'someone-else'
    with pytest.raises(ValueError, match='measured process run'):
        bind_measured_ssd(incumbent,candidate,comparison,stolen)
    # Calibrations from two source commits cannot be mixed.
    mixed = calibrations(10_100)
    mixed[0]['source']['source_commit'] = 'b'*40
    bound,decision = bind_measured_ssd(incumbent,candidate,comparison,mixed)
    assert bound == 'serial' and 'source commit' in decision.reason
    # ...and a comparison that states no commit refuses outright (review).
    unstated = {k: v for k, v in comparison.items() if k != 'source'}
    bound,decision = bind_measured_ssd(incumbent,candidate,unstated,calibrations(10_100))
    assert bound == 'serial' and 'source commit' in decision.reason


@pytest.mark.parametrize('variant', ['serial', 'cooperative'])
@pytest.mark.parametrize('output', range(3))
def test_ssd_absolute_correctness_gate_boundary(variant, output):
    import math
    from tessera.compiler.ssd_performance import SSD_MAX_ABS_ERROR
    pairs = evidence()
    errors = pairs[-1][variant]['rows'][0]['max_abs_errors']
    errors[output] = SSD_MAX_ABS_ERROR
    summarize(pairs)
    errors[output] = math.nextafter(SSD_MAX_ABS_ERROR, math.inf)
    with pytest.raises(ValueError, match='absolute admission tolerance'):
        summarize(pairs)
