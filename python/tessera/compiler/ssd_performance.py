"""Exact-artifact SSD selection; incomplete evidence retains the incumbent."""
import hashlib
import math
import statistics
import re
from dataclasses import dataclass


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
    a launch-inclusive Nsight timeline; gfx1151 uses native ROCm calibration.
    """
    incumbent_specs = incumbent.validate()
    candidate.validate()
    if incumbent.adjoint or candidate.adjoint or incumbent.cooperative or not candidate.cooperative:
        raise ValueError('SSD selection requires serial/cooperative forward artifacts')
    if incumbent.logical != candidate.logical or incumbent.package.backend != candidate.package.backend:
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
        return _admit_cuda_windows(comparison,calibrations,lower)
    if identity[0] != 'rocm' or identity[1] != 'gfx1151':
        return SSDAdmission(False,'target has no native calibration adapter',lower)
    if identity[4] != 'HIP events' or len(calibrations) != 18:
        return SSDAdmission(False,'each measured process requires native HIP clock calibration',lower)
    from .profiler_rocm_evidence import build_rocm_profiler_packet
    seen = set()
    for index,pair in enumerate(comparison['pairs']):
        for offset,name in enumerate(('serial','cooperative')):
            packet = calibrations[2*index+offset]
            timing = packet['timing']
            if timing['sample_id'] in seen:
                raise ValueError('SSD calibration sample was reused across process runs')
            seen.add(timing['sample_id'])
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
            if not rebuilt['eligible_for_promotion']:
                return SSDAdmission(False,'native calibration refuses promotion: '+', '.join(rebuilt['ineligibility_reasons']),lower)
    return SSDAdmission(True,'exact-artifact paired measurements and native calibration admitted',lower)


def select_ssd_candidate(incumbent, candidate, comparison, calibrations=()):
    decision = admit_ssd_candidate(incumbent,candidate,comparison,calibrations)
    return (candidate if decision.admitted else incumbent),decision


def bind_measured_ssd(incumbent, candidate, comparison, calibrations=()):
    selected,decision = select_ssd_candidate(incumbent,candidate,comparison,calibrations)
    return selected.bind(),decision



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
