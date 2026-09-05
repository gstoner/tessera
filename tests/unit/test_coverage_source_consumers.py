"""Coverage consumers work in clean checkouts without generated snapshots."""
import importlib.util
from pathlib import Path
import sys

from tessera.compiler import compiler_progress


def test_stub_report_write_renders_coverage_when_csv_is_absent(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'stub_surface_report.py'
    spec = importlib.util.spec_from_file_location('stub_surface_report_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'GEN', tmp_path / 'generated')
    monkeypatch.setattr(module, 'AUDIT', tmp_path)
    monkeypatch.setattr(sys, 'argv', [str(script), '--write'])
    assert not (module.GEN / 'test_coverage.csv').exists()
    assert module.main() == 0
    text = (tmp_path / 'stub_surface.md').read_text()
    assert 'test_coverage.csv not found' not in text
    assert '? needs-direct-test' not in text
    assert ' ops:' in text
    assert '`needs_direct_test`' in text
    assert 'live test-coverage renderer' in text


def test_compiler_progress_names_live_coverage_producer():
    direct = compiler_progress._test_row([
        {'is_thinly_tested': '0', 'bucket': 'direct'},
        {'is_thinly_tested': '1', 'bucket': 'needs_direct_test'},
    ])
    dashboard = next(row for row in compiler_progress._dashboard_map_rows()
                     if row.item == 'test_coverage')
    for row in (direct, dashboard):
        assert row.source == 'tessera.compiler.generated_docs:test_coverage (live renderer)'
    assert (direct.ready, direct.total) == (1, 2)
