"""Navigation contracts, not inferred status from historical prose."""
from pathlib import Path
import importlib.util
import pytest

ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('plan_check',ROOT/'scripts/check_compiler_plan.py')
assert _spec and _spec.loader
checker=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


def test_compiler_plan_routing_and_log_are_consistent():
    checker.check(ROOT)


def test_duplicate_task_id_refuses():
    with pytest.raises(ValueError,match='duplicate task'):
        checker.records('## F4\n### W4.1\n### W4.1\n')


def test_historical_prose_is_not_a_task_registry():
    assert checker.records('Old Next: W4.1 was incomplete.\n')=={}


def test_new_log_entry_requires_owner_change():
    plan=(ROOT/checker.PLAN).read_text(); log=(ROOT/checker.LOG).read_text()
    extra='''\n### 2026-09-08 — Test increment

Owner: [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a)

PRs: Uncommitted.

Outcome: A test.

Remaining: At owner.

Evidence: Test fixture.

<!-- entry-fields:end -->
'''
    with pytest.raises(ValueError,match='requires an update'):
        checker.check_transition(plan,log,plan,log+extra)
    changed=plan.replace('Gate: Broaden scoped readers','Gate: Extend scoped readers')
    checker.check_transition(plan,log,changed,log+extra)


def test_routing_relationship_is_not_a_readiness_state():
    with pytest.raises(ValueError,match='malformed routing'):
        checker.routes('## Routing index\n| OLD-1 | [owner](#owner) | complete |\n')


def test_historical_correction_does_not_require_fake_task_progress():
    plan=(ROOT/checker.PLAN).read_text(); log=(ROOT/checker.LOG).read_text()
    checker.check_transition(plan,log,plan,log.replace('Original recorded scope','Corrected recorded scope',1))


def test_fragment_anchor_survives_title_change():
    assert 'w24a' in checker.anchors('### W2.4a\n\nOld descriptive title\n')
    assert 'w24a' in checker.anchors('### W2.4a\n\nRevised descriptive title\n')


def test_unrelated_non_utf8_fixture_is_not_a_navigation_source(tmp_path):
    fixture=tmp_path/'fixture.md'
    fixture.write_bytes(b'price: \xa3')
    tracked={(tmp_path/'INTEGRATED_COMPILER_PLAN.md').resolve()}
    assert checker.link_source(fixture,tracked)==''
    fixture.write_bytes(b'\xa3 [plan](INTEGRATED_COMPILER_PLAN.md)')
    with pytest.raises(ValueError,match='navigation source must be UTF-8'):
        checker.link_source(fixture,tracked)
