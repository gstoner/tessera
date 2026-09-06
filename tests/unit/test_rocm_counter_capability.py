from benchmarks.record_rocm_counter_capability import classify


def test_counter_absence_is_not_an_attribution_result():
    assert classify(dict(returncode=0, stdout='No pmc counters supported', stderr='')) == 'unsupported'
    assert classify(dict(returncode=0, stdout='', stderr='no counter metrics found')) == 'unsupported'
    assert classify(dict(returncode=0, stdout='', stderr='')) == 'unverified'
    assert classify(dict(returncode=0, stdout='Counter: SQ_WAVES', stderr='')) == 'unverified'
    assert classify(dict(returncode=1, stdout='', stderr='failure')) == 'probe_failed'
