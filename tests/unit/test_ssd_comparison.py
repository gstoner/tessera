import copy
import pytest
from benchmarks.compare_ssd_variants import summarize


def evidence():
    return [{name:dict(backend='nvidia',architecture='sm_120',compiler_sha256='compiler',shape=[32,2,16,4],
                      clock='CUDA events',execution='native_gpu',cooperative=name=='cooperative',
                      rows=[dict(chunk=8,binding_digest=name,image_sha256=name,device_event_ms=[time]*7,
                                 launches_per_window=10,max_abs_errors=[0.,0.,0.])])
                 for name,time in [('serial',2.),('cooperative',1.)]}
            for _ in range(9)]


def _interleaved(timing):
    """Stamp a fixture timing sample the way the current recorder does: the
    interleaved window protocol, and the launch count (``batch_size``, 10 in
    the fixtures) in the device clock's provenance."""
    from tessera.compiler.ssd_performance import SSD_CALIBRATION_WINDOW_PROTOCOL
    timing['environment']['window_protocol'] = SSD_CALIBRATION_WINDOW_PROTOCOL
    timing['clocks']['device_wall_clock_ns']['provenance'] = {'launches_per_window': timing['batch_size']}
    return timing


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
            timing = _interleaved(_timing())
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
                timing = _interleaved(_wsl_witness_timing(event_ns=event_ns,
                    image_sha256=pair[name]['rows'][0]['image_sha256']))
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


def _witness_admission(package_chip, calibration_chip, *, candidate_chip=None, stamp=None):
    """One SSD admission over nine witness-calibrated pairs, where the rows and
    packages name ``package_chip`` and every calibration ``calibration_chip``.
    ``candidate_chip`` overrides the cooperative package's chip; ``stamp``
    replaces the current-protocol stamp on every calibration timing."""
    import hashlib
    from types import SimpleNamespace
    from tessera.compiler.ssd_performance import bind_measured_ssd
    from tessera.compiler.profiler_rocm_evidence import build_rocm_profiler_packet
    from test_profiler_rocm_evidence import _wsl_witness_timing, _no_kfd_capture, _image
    logical = SimpleNamespace(compiler_digest='compiler',schedule_ir='chunk_size = 8 : i64')
    specs = [SimpleNamespace(shape=(32,2,4)),None,SimpleNamespace(shape=(32,2,16))]
    def artifact(cooperative):
        name = 'cooperative' if cooperative else 'serial'
        chip = (candidate_chip or package_chip) if cooperative else package_chip
        return SimpleNamespace(logical=logical,adjoint=False,cooperative=cooperative,
            package=SimpleNamespace(backend='rocm',chip=chip,binding_digest=name,image=name.encode()),
            validate=lambda:specs,bind=lambda:name)
    pairs = evidence()
    for i,pair in enumerate(pairs):
        for name,packet in pair.items():
            packet.update(backend='rocm',architecture=package_chip,clock='HIP events',run_id=f'{i}-{name}')
            packet['rows'][0]['image_sha256'] = hashlib.sha256(name.encode()).hexdigest()
    comparison = dict(pairs=pairs,source=dict(source_commit='a'*40))
    calibrations = []
    for i,pair in enumerate(pairs):
        for name in ('serial','cooperative'):
            row = pair[name]['rows'][0]
            timing = (stamp or _interleaved)(
                _wsl_witness_timing(image_sha256=row['image_sha256'],architecture=calibration_chip))
            timing['sample_id'] = f'{i}-{name}'
            timing['environment']['run_id'] = f'{i}-{name}'
            clean,probe = _image(1,'clean',calibration_chip),_image(1,'probe',calibration_chip)
            for image in (clean,probe):
                image.update(calibration_sample_id=timing['sample_id'],
                             semantic_sha256=hashlib.sha256(logical.schedule_ir.encode()).hexdigest(),
                             duration_ns=row['device_event_ms'][0]*1e6)
            clean['image_sha256'] = row['image_sha256']
            calibrations.append(build_rocm_profiler_packet(timing=timing,capture=_no_kfd_capture(),
                uninstrumented=clean,instrumented=probe,source=dict(source_commit='a'*40,worktree_dirty=False)))
    return bind_measured_ssd(artifact(False),artifact(True),comparison,calibrations)


@pytest.mark.parametrize('chip', ['gfx1151', 'gfx1201'])
def test_ssd_admits_a_witness_calibration_on_its_own_chip(chip):
    """Sync GFX1201-SSD-CALIBRATION-2026-09-26: both calibrated chips admit
    through the same route when calibrations name the package chip."""
    bound,decision = _witness_admission(chip, chip)
    assert bound == 'cooperative' and decision.admitted


@pytest.mark.parametrize('package_chip,calibration_chip', [('gfx1201','gfx1151'),('gfx1151','gfx1201')])
def test_ssd_calibration_never_transfers_between_chips(package_chip, calibration_chip):
    with pytest.raises(ValueError, match='does not match the measured package chip'):
        _witness_admission(package_chip, calibration_chip)


def test_ssd_admission_refuses_a_rocm_chip_without_a_calibration_route():
    import hashlib
    from types import SimpleNamespace
    from tessera.compiler.ssd_performance import admit_ssd_candidate
    logical = SimpleNamespace(compiler_digest='compiler',schedule_ir='chunk_size = 8 : i64')
    specs = [SimpleNamespace(shape=(32,2,4)),None,SimpleNamespace(shape=(32,2,16))]
    def artifact(cooperative):
        name = 'cooperative' if cooperative else 'serial'
        return SimpleNamespace(logical=logical,adjoint=False,cooperative=cooperative,
            package=SimpleNamespace(backend='rocm',chip='gfx1100',binding_digest=name,image=name.encode()),
            validate=lambda:specs)
    pairs = evidence()
    for pair in pairs:
        for name,packet in pair.items():
            packet.update(backend='rocm',architecture='gfx1100',clock='HIP events')
            packet['rows'][0]['image_sha256'] = hashlib.sha256(name.encode()).hexdigest()
    decision = admit_ssd_candidate(artifact(False),artifact(True),dict(pairs=pairs),())
    assert not decision.admitted and 'no native calibration adapter' in decision.reason


def _legacy(timing):
    """The committed pre-interleaving gfx1151 packets: no ``window_protocol``."""
    timing['clocks']['device_wall_clock_ns']['provenance'] = {'launches_per_window': timing['batch_size']}
    return timing


@pytest.mark.parametrize('chip', ['gfx1151', 'gfx1201'])
def test_ssd_admission_refuses_a_legacy_window_protocol(chip):
    """Pre-PR review item 4 (GFX1201-SSD-CALIBRATION-2026-09-26): a packet
    recorded before interleaving would pass every other gate, so admission
    refuses it by name rather than admitting a power-state-biased ratio."""
    bound,decision = _witness_admission(chip, chip, stamp=_legacy)
    assert bound == 'serial' and not decision.admitted
    assert decision.reason.startswith('SSD_CALIBRATION_WINDOW_PROTOCOL_LEGACY')
    # Any other protocol name is legacy too; only the current one admits.
    def other(timing):
        _interleaved(timing)['environment']['window_protocol'] = 'plain_first_then_bracketed'
        return timing
    bound,decision = _witness_admission(chip, chip, stamp=other)
    assert not decision.admitted and 'WINDOW_PROTOCOL_LEGACY' in decision.reason


@pytest.mark.parametrize('field', ['batch_size', 'provenance', 'missing_provenance'])
def test_ssd_admission_refuses_a_mismatched_launch_count(field):
    """The calibration's launches (batch and device-clock provenance) must
    equal the row's ``launches_per_window``: the fixed per-window bracket
    offset makes a calibration at one window length say nothing about another."""
    def mismatched(timing):
        _interleaved(timing)
        if field == 'batch_size':
            timing['batch_size'] = 1000
            timing['clocks']['device_wall_clock_ns']['provenance']['launches_per_window'] = 1000
        elif field == 'provenance':
            timing['clocks']['device_wall_clock_ns']['provenance']['launches_per_window'] = 1000
        else:
            del timing['clocks']['device_wall_clock_ns']['provenance']['launches_per_window']
        return timing
    bound,decision = _witness_admission('gfx1151', 'gfx1151', stamp=mismatched)
    assert bound == 'serial' and not decision.admitted
    assert decision.reason.startswith('SSD_CALIBRATION_LAUNCHES_MISMATCH')


def test_ssd_admission_refuses_a_row_without_a_launch_count():
    """A row recorded before ``launches_per_window`` existed cannot be matched."""
    import tessera.compiler.ssd_performance as ssd
    assert ssd._calibration_protocol_refusal(
        _interleaved({'environment': {}, 'batch_size': 10,
                      'clocks': {'device_wall_clock_ns': {}}}),
        {}).startswith('SSD_CALIBRATION_LAUNCHES_MISMATCH')


@pytest.mark.parametrize('package_chip,candidate_chip', [('gfx1151','gfx1201'),('gfx1201','gfx1151')])
def test_ssd_admission_refuses_a_cross_chip_candidate(package_chip, candidate_chip):
    """Pre-existing gap: backends were compared but not ``package.chip``."""
    with pytest.raises(ValueError, match='different semantic parents or targets'):
        _witness_admission(package_chip, package_chip, candidate_chip=candidate_chip)


_BASELINES = __import__('pathlib').Path(__file__).resolve().parents[2] / 'benchmarks' / 'baselines'


@pytest.mark.parametrize('packet,admitted_protocol', [
    ('gfx1201_ssd_calibrated_pairs_20260926', True),
    ('gfx1151_ssd_calibrated_pairs_interleaved_20260926', True),
    ('gfx1151_ssd_calibrated_pairs_20260926', False),
])
def test_committed_calibrations_carry_the_protocol_admission_reads(packet, admitted_protocol):
    """The committed packets, read as data: gfx1201 and the gfx1151 re-record
    were recorded interleaved with matching launches; the first gfx1151
    packet predates the stamp and stays history (validates, never admits).
    The end-to-end replays need tessera-opt and the device, so they are
    committed as each packet's ``replay.json`` rather than run here."""
    import json
    from tessera.compiler.profiler_rocm_evidence import validate_rocm_profiler_packet
    from tessera.compiler.ssd_performance import _calibration_protocol_refusal
    root = _BASELINES / packet
    refusals = []
    for i in range(9):
        for name in ('serial', 'cooperative'):
            row = json.loads((root / f'{i}-{name}.json').read_text())['rows'][0]
            calibration = json.loads((root / f'{i}-{name}-calibration.json').read_text())
            validate_rocm_profiler_packet(calibration)
            refusals.append(_calibration_protocol_refusal(calibration['timing'], row))
    if admitted_protocol:
        assert refusals == [None] * 18
    else:
        assert all(r and r.startswith('SSD_CALIBRATION_WINDOW_PROTOCOL_LEGACY') for r in refusals)
