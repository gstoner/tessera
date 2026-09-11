import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from benchmarks.lattice_reasoning_core.core import _apple_gpu_metric_row
from benchmarks.dlop_longtail_core.core import run_core

ROOT = Path(__file__).resolve().parents[2]


def test_native_marker_does_not_turn_host_clock_into_kernel_clock():
    row = _apple_gpu_metric_row('probe', 'fp32', '2',
                               lambda: (np.ones(2), 'metal_runtime'), np.ones(2), {})
    assert row.metrics['observed_native_execution'] is True
    assert row.profile.cpu_wall_ms is not None
    assert row.profile.kernel_elapsed_ms is None
    assert row.metrics['promotion_eligible'] is False


def test_dlop_counts_are_estimates_not_dispatch_measurements():
    for row in run_core():
        assert row.metrics['dispatch_count_source'] == 'static_decomposition_estimate'
        assert row.metrics['observed_dispatches'] is None
        assert row.metrics['promotion_eligible'] is False


def test_retired_synthetic_attention_cannot_emit_a_latency_row():
    result = subprocess.run([sys.executable, str(ROOT / 'benchmarks/Tessera_SuperBench/benches/kernel/attention_placeholder.py')], capture_output=True, text=True)
    assert result.returncode != 0
    assert not result.stdout
    assert 'retired' in result.stderr


@pytest.mark.parametrize('name', ['grid_ai', 'visual_complex'])
def test_library_benchmarks_reject_empty_measurement(name):
    if name == 'grid_ai':
        from benchmarks.grid_ai_core.core import GridAICoreBenchmark as Benchmark
    else:
        from benchmarks.visual_complex_core.core import VisualComplexCoreBenchmark as Benchmark
    with pytest.raises(ValueError, match='positive'):
        Benchmark(reps=0)


@pytest.mark.parametrize('result', [
    {'ok': True, 'execution_kind': 'reference_cpu', 'output': 1.0},
    {'ok': False, 'execution_kind': 'native_gpu', 'output': 1.0},
    {'ok': True, 'execution_kind': 'native_gpu', 'output': float('nan')},
])
def test_policy_timing_rejects_fallback_failure_and_nan(result):
    from benchmarks.rl.benchmark_policy_losses import _require_native_apple
    with pytest.raises(RuntimeError):
        _require_native_apple(result)


def test_dlop_cli_uses_requested_seed_and_qualifies_summary(monkeypatch, capsys):
    import json
    from benchmarks.dlop_longtail_core import benchmark_dlop_longtail as cli
    seeds = []
    def run(cfg):
        seeds.append(cfg.seed)
        return run_core(cfg)
    monkeypatch.setattr(cli, 'run_core', run)
    assert cli.main(['--seed', '731']) == 0
    report = json.loads(capsys.readouterr().out)
    assert seeds == [731] and report['seed'] == 731
    assert report['report']['promotion_eligible'] is False
    assert report['report']['dispatch_count_source'] == 'static_decomposition_estimate'
