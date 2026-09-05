"""Exact coverage and contamination tests for opt-in median route evidence."""
import math

import pytest

from tessera.compiler.apple_route_selector import (
    aggregate_stable_route_reports,
    median_speedup_confidence_interval,
    speedup_confidence_interval,
)
from tests.unit.test_apple_route_selector import _report, _stable_row


def test_exact_tail_probability_and_insufficient_runs():
    assert median_speedup_confidence_interval([0.2] * 4) is None
    for n in range(5, 50):
        low, high = median_speedup_confidence_interval(list(range(n)))
        k = int(low) + 1
        assert high == n - k
        assert sum(math.comb(n, j) for j in range(k)) / 2**n <= .05
        assert sum(math.comb(n, j) for j in range(k + 1)) / 2**n > .05


def test_one_extreme_run_does_not_move_eight_run_median_bound():
    values = [.4] * 7 + [.06]
    assert speedup_confidence_interval(values)[0] < .4
    assert median_speedup_confidence_interval(values) == (.4, .4)
    assert median_speedup_confidence_interval([.04] * 7 + [.99]) == (.04, .04)
    assert median_speedup_confidence_interval([float('nan')] * 8) is None


@pytest.mark.parametrize('bad_time,expected', [(940, 'promote_candidate'), (1100, 'retain_incumbent_unstable_candidate')])
def test_median_policy_keeps_every_run_safety_floor(bad_time, expected):
    reports = [_report(_stable_row('mps', 1000, 1000),
                       _stable_row('simdgroup_matrix', t, t))
               for t in [600] * 7 + [bad_time]]
    ledger = aggregate_stable_route_reports(reports, cross_run_estimator='median_order_statistic')
    assert ledger['promotion_rules']['cross_run_estimator'] == 'median_order_statistic'
    assert all(row['status'] == expected for row in ledger['decisions'])


def test_unknown_policy_is_refused():
    with pytest.raises(ValueError, match='unsupported cross-run estimator'):
        aggregate_stable_route_reports([], cross_run_estimator='trim_until_win')


def test_consumer_rejects_forged_median_bound_and_unknown_policy():
    from tessera.compiler.apple_route_selector import promotion_rule_violations
    reports = [_report(_stable_row('mps', 1000, 1000),
                       _stable_row('simdgroup_matrix', 600, 600)) for _ in range(8)]
    ledger = aggregate_stable_route_reports(reports, cross_run_estimator='median_order_statistic')
    row = ledger['decisions'][0]
    rules = ledger['promotion_rules']
    assert promotion_rule_violations(row, rules, source_report_count=8) == []
    row['route_evidence'][row['selected_route']]['speedup_lower_confidence_bound'] = .99
    assert 'median_bound_does_not_match_runs' in promotion_rule_violations(row, rules)
    rules['cross_run_estimator'] = 'unknown'
    assert 'unsupported_cross_run_estimator' in promotion_rule_violations(row, rules)
    rules.pop('minimum_speedup_lower_confidence_bound')
    rules['cross_run_speedup_spread_is_diagnostic_only'] = False
    assert 'unsupported_cross_run_estimator' in promotion_rule_violations(row, rules)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), True])
def test_nonfinite_or_boolean_threshold_cannot_bypass_gate(invalid):
    from tessera.compiler.apple_route_selector import promotion_rule_violations
    reports = [_report(_stable_row('mps', 1000, 1000),
                       _stable_row('simdgroup_matrix', 600, 600)) for _ in range(8)]
    ledger = aggregate_stable_route_reports(reports, cross_run_estimator='median_order_statistic')
    ledger['promotion_rules']['minimum_speedup_lower_confidence_bound'] = invalid
    assert 'no_stability_rule_declared' in promotion_rule_violations(
        ledger['decisions'][0], ledger['promotion_rules'])
