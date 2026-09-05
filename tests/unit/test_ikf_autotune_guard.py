"""IKF-1: explain-run timestamps cannot become dispatch evidence."""
import pytest

from tessera.compiler.emit.autotune import MeasureCache, MeasureRecord, record_is_admissible

KEY = ('test_device', 'x86', 'matmul', (16, 16, 16), 'f32', 'device')


@pytest.mark.parametrize('level', [2, 3, -1, 4, True, '1', None])
def test_instrumented_or_invalid_level_is_refused_at_every_entry(level):
    record = MeasureRecord('candidate', 1.0, evidence={'instr_level': level})
    assert not record_is_admissible(record)
    cache = MeasureCache()
    with pytest.raises(ValueError, match='instrumented evidence'):
        cache.put(KEY, record)
    assert cache.size == 0
    # Build a valid serialized key using the owning serializer, then inject
    # the inadmissible evidence as a foreign/persisted producer might.
    cache.put(KEY, MeasureRecord('candidate', 1.0))
    payload = cache.to_dict()
    payload['records'][0]['evidence'] = {'instr_level': level}
    loaded = MeasureCache()
    assert loaded.load_dict(payload) == 0
    assert loaded.size == 0


@pytest.mark.parametrize('evidence', [{}, {'instr_level': 0}, {'instr_level': 1}])
def test_provider_and_legacy_measurements_remain_admissible(evidence):
    record = MeasureRecord('candidate', 1.0, evidence=evidence)
    assert record_is_admissible(record)
    cache = MeasureCache()
    cache.put(KEY, record)
    restored = MeasureCache()
    assert restored.load_dict(cache.to_dict()) == 1
    assert restored.get(KEY).evidence == evidence


def test_mutated_cached_evidence_cannot_bypass_admission():
    cache = MeasureCache()
    record = MeasureRecord('candidate', 1.0)
    cache.put(KEY, record)
    record.evidence['instr_level'] = 2
    assert cache.get(KEY) is None
