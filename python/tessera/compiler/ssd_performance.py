"""Exact-artifact SSD selection; incomplete evidence retains the incumbent."""
import hashlib
import math
import statistics
import re
from dataclasses import dataclass


# The recorder checks rtol=1e-5, atol=1e-6. Maxima alone do not retain
# per-element reference magnitudes, so admission uses the sufficient absolute
# component. Do not infer a relative-error allowance from these summaries.
SSD_MAX_ABS_ERROR = 1e-6

#: The only calibration window protocol admission accepts: plain (the
#: comparison row) and marker-bracketed windows interleaved in alternating
#: order, each behind one span-reset + synchronize gap. The recorder stamps it
#: into ``timing.environment`` and the launch count into ``timing.batch_size``
#: and the device clock's provenance, inside the part ``timing_sha256`` covers;
#: admission validates the stored packet first, so an edit that was not
#: resealed is refused. The digests are unkeyed SHA-256: they catch accidental
#: or careless edits, not a deliberate reseal, and the row's own launch count
#: (in the comparison) is not digest-covered. The launch count is checked for
#: consistency (calibration, device clock, row, and one value across all
#: rows), not against the measured durations, which are stored per launch. The
#: earlier protocol (every plain window first, then the marker compiled, then
#: the bracketed windows) compared two GPU power states on gfx1201 (sync
#: GFX1201-SSD-CALIBRATION-2026-09-26), so a packet without this stamp stays
#: readable history but never admits.
SSD_CALIBRATION_WINDOW_PROTOCOL = 'interleaved_alternating_plain_bracketed'


def _calibration_protocol_refusal(timing, row):
    """A named refusal when ``timing`` was not recorded under the current
    window protocol with its row's launch count, else ``None``."""
    protocol = (timing.get('environment') or {}).get('window_protocol')
    if protocol != SSD_CALIBRATION_WINDOW_PROTOCOL:
        return ('SSD_CALIBRATION_WINDOW_PROTOCOL_LEGACY: calibration window protocol '
                f'{protocol!r} is not {SSD_CALIBRATION_WINDOW_PROTOCOL!r}; re-record '
                'under the interleaved protocol')
    launches = row.get('launches_per_window')
    stamped = timing.get('batch_size')
    provenance = ((timing.get('clocks') or {}).get('device_wall_clock_ns') or {}).get('provenance') or {}
    counts = (launches, stamped, provenance.get('launches_per_window'))
    if any(type(v) is not int or v <= 0 for v in counts) or len(set(counts)) != 1:
        return ('SSD_CALIBRATION_LAUNCHES_MISMATCH: launches per window disagree '
                f'(row {launches!r}, calibration batch {stamped!r}, device clock '
                f'{provenance.get("launches_per_window")!r})')
    return None


def summarize(pairs):
    if len(pairs) != 9:
        raise ValueError('SSD comparison requires nine prespecified process pairs')
    identity = None
    variants: dict[str, tuple[int,str,str]] = {}
    ratios = []
    for pair in pairs:
        if set(pair) != {'serial','cooperative'}:
            raise ValueError('SSD comparison requires both variants')
        times = {}
        for name,packet in pair.items():
            if packet.get('execution') != 'native_gpu' or packet.get('cooperative') is not (name == 'cooperative'):
                raise ValueError('SSD variant execution disagrees')
            if any(type(v) is not int or v <= 0 for v in packet['shape']):
                raise ValueError('SSD shapes must contain positive integers')
            key = (packet['backend'],packet['architecture'],packet['compiler_sha256'],tuple(packet['shape']),packet['clock'])
            if identity is not None and identity != key:
                raise ValueError('SSD comparison identity changed')
            identity = key
            if len(packet['rows']) != 1:
                raise ValueError('SSD comparison requires one chunk policy')
            row = packet['rows'][0]
            if type(row['chunk']) is not int or row['chunk'] <= 0:
                raise ValueError('SSD chunk must be a positive integer')
            variant = (row['chunk'],row['binding_digest'],row['image_sha256'])
            if name in variants and variants[name] != variant:
                raise ValueError('SSD artifact changed across runs')
            variants[name] = variant
            samples = row['device_event_ms']
            if len(samples) != 7 or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in samples):
                raise ValueError('SSD event samples must be finite positive numbers')
            errors = row['max_abs_errors']
            if len(errors) != 3 or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in errors):
                raise ValueError('SSD correctness evidence is malformed')
            if any(v > SSD_MAX_ABS_ERROR for v in errors):
                raise ValueError('SSD correctness evidence exceeds the absolute admission tolerance')
            times[name] = statistics.median(samples)
        if variants['serial'][0] != variants['cooperative'][0]:
            raise ValueError('SSD checkpoint policies disagree')
        ratios.append(times['serial']/times['cooperative'])
    # For nine independent pairs P(Binomial(9, .5) <= 1)=10/512.
    # The second order statistic is a conservative one-sided 95% median bound.
    lower = sorted(ratios)[1]
    return dict(schema=1,objective='resident device event window including submission gaps',
        paired_run_speedups=ratios,median_speedup=statistics.median(ratios),
        median_speedup_lower_bound=lower,confidence=1-10/512,
        run_count=9,variants=variants,promotion_eligible=False,
        missing_gates=['device clock calibration','production selector exact-artifact admission'],
        identity=identity)



@dataclass(frozen=True)
class SSDAdmission:
    admitted: bool
    reason: str
    lower_bound: float | None = None


def admit_ssd_candidate(incumbent, candidate, comparison, calibrations=()):
    """Check the actual forward artifacts and independently recompute policy.

    Each process needs calibration of its exact measured artifact. CUDA uses
    either the compiler-built ``%globaltimer`` marker witness (sm_120; sync
    NVIDIA-GLOBALTIMER-MARKER-2026-09-26) or a launch-inclusive Nsight
    timeline; gfx1151 and gfx1201 use native ROCm calibration, and every
    calibration must name the package's exact chip (evidence never transfers
    between parts; sync GFX1201-SSD-CALIBRATION-2026-09-26). A device-clock
    calibration (ROCm or NVIDIA) must also carry the current window protocol
    and its row's launch count; a legacy packet is refused with
    ``SSD_CALIBRATION_WINDOW_PROTOCOL_LEGACY``.
    """
    incumbent_specs = incumbent.validate()
    candidate.validate()
    if incumbent.adjoint or candidate.adjoint or incumbent.cooperative or not candidate.cooperative:
        raise ValueError('SSD selection requires serial/cooperative forward artifacts')
    if (incumbent.logical != candidate.logical or incumbent.package.backend != candidate.package.backend
            or incumbent.package.chip != candidate.package.chip):
        # The chip is part of the target: evidence never transfers between
        # architectures, so a candidate built for another chip cannot ride an
        # incumbent's measurements (pre-PR review, 2026-09-26).
        raise ValueError('SSD candidates have different semantic parents or targets')
    report = summarize(comparison['pairs'])
    identity = report['identity']
    shape = incumbent_specs[0].shape
    states = incumbent_specs[2].shape[2]
    if (identity[0] != incumbent.package.backend or identity[1] != incumbent.package.chip
            or identity[2] != incumbent.logical.compiler_digest or tuple(identity[3]) != (shape[0],shape[1],states,shape[2])):
        raise ValueError('SSD measurement target/compiler/shape disagrees')
    for name,artifact in [('serial',incumbent),('cooperative',candidate)]:
        chunk,binding,image = report['variants'][name]
        if (binding != artifact.package.binding_digest or image != hashlib.sha256(artifact.package.image).hexdigest()
                or re.search(r'chunk_size = '+str(chunk)+r' : i64',artifact.logical.schedule_ir) is None):
            raise ValueError('SSD measurement artifact disagrees')
    lower = report['median_speedup_lower_bound']
    if lower <= 1.05:
        return SSDAdmission(False,'paired speedup bound does not exceed five percent',lower)
    if identity[0] == 'nvidia':
        from .profiler_nvidia_evidence import NVIDIA_DEVICE_CLOCK_PACKET_SCHEMA_VERSION
        # The route is chosen by what every calibration IS (its schema), never
        # by a flag; a mixture of the two routes is refused, not averaged.
        schemas = {c.get('schema') for c in calibrations} if calibrations else set()
        if schemas == {NVIDIA_DEVICE_CLOCK_PACKET_SCHEMA_VERSION}:
            return _admit_nvidia_device_clock(incumbent,comparison,calibrations,report,lower)
        if NVIDIA_DEVICE_CLOCK_PACKET_SCHEMA_VERSION in schemas:
            return SSDAdmission(False,'CUDA calibrations mix the device-clock and Nsight routes',lower)
        return _admit_cuda_windows(comparison,calibrations,lower)
    from .profiler_rocm_evidence import (
        ROCM_PROFILER_ARCHITECTURES, ROCmProfilerPacketError, build_rocm_profiler_packet,
        validate_rocm_profiler_packet)
    if identity[0] != 'rocm' or identity[1] not in ROCM_PROFILER_ARCHITECTURES:
        return SSDAdmission(False,'target has no native calibration adapter',lower)
    chip = identity[1]
    if identity[4] != 'HIP events' or len(calibrations) != 18:
        return SSDAdmission(False,'each measured process requires native HIP clock calibration',lower)
    commits = {(c.get('source') or {}).get('source_commit') for c in calibrations}
    stated = (comparison.get('source') or {}).get('source_commit')
    # The comparison must state its commit (the recorder always writes it), or
    # calibrations from any one stale commit would pass (review).
    if stated is None or commits != {stated}:
        return SSDAdmission(False,'calibrations do not share one source commit with the comparison',lower)
    # One launch count across every row (review): the bracket offset's share
    # of a window depends on its length, so pairs recorded at different counts
    # are not one measurement.
    counts = {pair[name]['rows'][0].get('launches_per_window')
              for pair in comparison['pairs'] for name in ('serial', 'cooperative')}
    if len(counts) != 1:
        return SSDAdmission(False, 'SSD_CALIBRATION_LAUNCHES_MISMATCH: rows were recorded at '
                            f'different launch counts {sorted(map(str, counts))}', lower)
    seen = set()
    for index,pair in enumerate(comparison['pairs']):
        for offset,name in enumerate(('serial','cooperative')):
            packet = calibrations[2*index+offset]
            # Validate the stored packet before anything reads it: its digests
            # must match what it says (review; a stamp edited without a reseal
            # used to reach the protocol check untouched).
            try:
                validate_rocm_profiler_packet(packet)
            except ROCmProfilerPacketError as exc:
                raise ValueError(f'stored SSD calibration does not validate: {exc}') from exc
            timing = packet['timing']
            if timing['sample_id'] in seen:
                raise ValueError('SSD calibration sample was reused across process runs')
            seen.add(timing['sample_id'])
            # Bound to the measured process, not only to an equal duration
            # (review): the calibration names the row's run_id.
            run_id = pair[name].get('run_id')
            if not run_id or (timing.get('environment') or {}).get('run_id') != run_id:
                raise ValueError('SSD calibration does not name the measured process run')
            refusal = _calibration_protocol_refusal(timing, pair[name]['rows'][0])
            if refusal is not None:
                return SSDAdmission(False,refusal,lower)
            images = packet['instrumentation_comparison']
            clean,probe = images['uninstrumented'],images['instrumented']
            if any(type(im.get('duration_ns')) not in (float,int) or not math.isfinite(im['duration_ns']) or im['duration_ns'] <= 0 for im in (clean,probe)):
                raise ValueError('SSD calibrated durations must be finite positive numbers')
            expected = statistics.median(pair[name]['rows'][0]['device_event_ms'])*1e6
            semantic = hashlib.sha256(incumbent.logical.schedule_ir.encode()).hexdigest()
            if (clean['image_sha256'] != report['variants'][name][2] or clean['semantic_sha256'] != semantic
                    or clean['clock_source'] != 'hip_event' or not math.isclose(clean['duration_ns'],expected,rel_tol=1e-9)):
                raise ValueError('SSD calibration does not describe the measured image and duration')
            rebuilt = build_rocm_profiler_packet(timing=timing,capture=packet['capture'],
                uninstrumented=clean,instrumented=probe,source=packet['source'],maximum_instrumentation_overhead=1.05)
            # The rebuilt architecture is derived from the timing target and
            # both images; it and the stored claim must name the package chip.
            if rebuilt['architecture'] != chip or packet.get('architecture') != chip:
                raise ValueError(f'SSD calibration architecture {packet.get("architecture")!r} '
                                 f'does not match the measured package chip {chip!r}')
            if not rebuilt['eligible_for_promotion']:
                return SSDAdmission(False,'native calibration refuses promotion: '+', '.join(rebuilt['ineligibility_reasons']),lower)
    return SSDAdmission(True,'exact-artifact paired measurements and native calibration admitted',lower)


def select_ssd_candidate(incumbent, candidate, comparison, calibrations=()):
    decision = admit_ssd_candidate(incumbent,candidate,comparison,calibrations)
    return (candidate if decision.admitted else incumbent),decision


def bind_measured_ssd(incumbent, candidate, comparison, calibrations=()):
    selected,decision = select_ssd_candidate(incumbent,candidate,comparison,calibrations)
    return selected.bind(),decision



def _admit_nvidia_device_clock(incumbent, comparison, calibrations, report, lower):
    """The ROCm device-clock loop's NVIDIA twin: every process's CUDA-event
    duration calibrated by its own ``%globaltimer`` marker packet, each packet
    re-derived from its inputs, named to its row's run and to the measured
    image, and on the package's exact architecture."""
    from .profiler_nvidia_evidence import (
        NVIDIADeviceClockPacketError, build_nvidia_device_clock_packet,
        validate_nvidia_device_clock_packet)
    identity = report['identity']
    chip = identity[1]
    if identity[4] != 'CUDA events' or len(calibrations) != 18:
        return SSDAdmission(False,'each measured process requires CUDA-event device-clock calibration',lower)
    commits = {(c.get('source') or {}).get('source_commit') for c in calibrations}
    stated = (comparison.get('source') or {}).get('source_commit')
    if stated is None or commits != {stated}:
        return SSDAdmission(False,'calibrations do not share one source commit with the comparison',lower)
    # One physical GPU across all eighteen processes (review): each packet
    # names the part it ran on, and a set assembled from two cards of the same
    # model would otherwise pass every per-packet check.
    uuids = {(((c.get('timing') or {}).get('environment') or {}).get('device_identity') or {}).get('uuid')
             for c in calibrations}
    if len(uuids) != 1 or None in uuids or '' in uuids:
        return SSDAdmission(False,'DEVICE_CLOCK_PART_MISMATCH: calibrations do not name one '
                            f'device UUID ({sorted(map(str, uuids))})',lower)
    # One launch count across every row, as on the ROCm route: the bracket
    # offset's share of a window depends on its length (measured on sm_120:
    # ~10-16 us per window, 2026-09-26).
    counts = {pair[name]['rows'][0].get('launches_per_window')
              for pair in comparison['pairs'] for name in ('serial', 'cooperative')}
    if len(counts) != 1:
        return SSDAdmission(False, 'SSD_CALIBRATION_LAUNCHES_MISMATCH: rows were recorded at '
                            f'different launch counts {sorted(map(str, counts))}', lower)
    semantic = hashlib.sha256(incumbent.logical.schedule_ir.encode()).hexdigest()
    seen = set()
    for index,pair in enumerate(comparison['pairs']):
        for offset,name in enumerate(('serial','cooperative')):
            packet = calibrations[2*index+offset]
            try:
                validate_nvidia_device_clock_packet(packet)
            except NVIDIADeviceClockPacketError as exc:
                raise ValueError(f'stored SSD calibration does not validate: {exc}') from exc
            timing = packet['timing']
            if timing['sample_id'] in seen:
                raise ValueError('SSD calibration sample was reused across process runs')
            seen.add(timing['sample_id'])
            run_id = pair[name].get('run_id')
            if not run_id or (timing.get('environment') or {}).get('run_id') != run_id:
                raise ValueError('SSD calibration does not name the measured process run')
            refusal = _calibration_protocol_refusal(timing, pair[name]['rows'][0])
            if refusal is not None:
                return SSDAdmission(False,refusal,lower)
            images = packet['instrumentation_comparison']
            clean,probe = images['uninstrumented'],images['instrumented']
            if any(type(im.get('duration_ns')) not in (float,int) or not math.isfinite(im['duration_ns']) or im['duration_ns'] <= 0 for im in (clean,probe)):
                raise ValueError('SSD calibrated durations must be finite positive numbers')
            expected = statistics.median(pair[name]['rows'][0]['device_event_ms'])*1e6
            if (clean['image_sha256'] != report['variants'][name][2] or clean['semantic_sha256'] != semantic
                    or clean['clock_source'] != 'cuda_event' or not math.isclose(clean['duration_ns'],expected,rel_tol=1e-9)):
                raise ValueError('SSD calibration does not describe the measured image and duration')
            rebuilt = build_nvidia_device_clock_packet(timing=timing,uninstrumented=clean,instrumented=probe,
                source=packet['source'],maximum_instrumentation_overhead=1.05)
            if rebuilt['architecture'] != chip or packet.get('architecture') != chip:
                raise ValueError(f'SSD calibration architecture {packet.get("architecture")!r} '
                                 f'does not match the measured package chip {chip!r}')
            if not rebuilt['eligible_for_promotion']:
                return SSDAdmission(False,'native calibration refuses promotion: '+', '.join(rebuilt['ineligibility_reasons']),lower)
    return SSDAdmission(True,'exact-artifact paired measurements and %globaltimer device-clock calibration admitted',lower)


def _admit_cuda_windows(comparison, calibrations, lower):
    from .profiler_cuda_window import build_cuda_window_calibration
    if len(calibrations) != 18:
        return SSDAdmission(False,'each measured process requires CUDA activity-window calibration',lower)
    samples,captures = set(),set()
    runs: set[str] = set()
    for i,pair in enumerate(comparison['pairs']):
        for j,name in enumerate(('serial','cooperative')):
            packet = calibrations[2*i+j]
            # Rebuild every eligibility gate from raw windows and intervals.
            if packet['sample_id'] in samples or packet['capture_sha256'] in captures:
                raise ValueError('CUDA calibration capture reused across process runs')
            run_ids = (packet['clean']['run_id'],packet['profiled']['run_id'])
            if any(run in runs for run in run_ids):
                raise ValueError('CUDA process run reused across calibrations')
            runs.update(run_ids)
            samples.add(packet['sample_id'])
            captures.add(packet['capture_sha256'])
            if packet['clean'] != pair[name]:
                raise ValueError('CUDA calibration does not describe the measured process')
            rebuilt = build_cuda_window_calibration(**{key:packet[key] for key in (
                'clean','profiled','kernels','source','capture_device','capture_sha256','sample_id')})
            if not rebuilt['eligible_for_promotion']:
                return SSDAdmission(False,'native calibration refuses promotion: '+', '.join(rebuilt['ineligibility_reasons']),lower)
    return SSDAdmission(True,'exact-artifact paired measurements and CUDA calibration admitted',lower)
