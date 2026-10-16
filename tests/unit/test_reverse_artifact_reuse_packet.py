"""Integrity and measurement scope of the frozen same-image adapter A/B."""
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / 'benchmarks/baselines/sm120_reverse_artifact_reuse_20261016'


def test_reverse_adapter_packet_retains_bound_sources_and_receipts():
    manifest = json.loads((PACKET / 'manifest.json').read_text())
    for relative, digest in manifest['sha256'].items():
        assert hashlib.sha256((PACKET / relative).read_bytes()).hexdigest() == digest
    packet = json.loads(gzip.decompress((PACKET / 'adapter-ab.json.gz').read_bytes()))
    assert packet['source_before'] == packet['source_after']
    assert packet['source_before']['status'] == []
    assert packet['source_before']['commit'] == manifest['measured_source']
    assert packet['control_sha256'] == hashlib.sha256(gzip.decompress(
        (PACKET / 'parent_adapter.py.gz').read_bytes())).hexdigest()
    assert packet['compiler_and_concrete_frontend_forbidden_during_timing'] is True
    assert 'no kernel speedup claim' in packet['scope']
    assert 'RTX 5070' in packet['device'] and packet['device'].endswith('12.0')


def test_reverse_adapter_packet_has_complete_positive_separate_timing_rows():
    packet = json.loads(gzip.decompress((PACKET / 'adapter-ab.json.gz').read_bytes()))
    rows = packet['rows']
    assert len(rows) == 12
    assert {(r['storage'], r['sq'], r['sk'], r['lane']) for r in rows} == {
        (storage, sq, sk, lane)
        for storage, (sq, sk), lane in itertools.product(
            ('f16', 'bf16', 'f32'), ((3, 5), (7, 129)), ('host', 'resident'))}
    for row in rows:
        assert row['shared_native_owner'] is True
        assert hashlib.sha256((ROOT / row['input_file']).read_bytes()).hexdigest() == row['input_sha256']
        for arm in ('parent', 'candidate'):
            for metric in ('completed_ms', 'forward_event_ms', 'backward_event_ms'):
                values = row['samples'][arm][metric]
                assert len(values) == 2 * packet['rounds'] * packet['reps']
                assert all(np.isfinite(x) and x > 0 for x in values)
                assert row['medians'][arm][metric] == statistics.median(values)
        assert row['completed_call_parent_over_candidate'] == (
            row['medians']['parent']['completed_ms'] / row['medians']['candidate']['completed_ms'])
