"""Provenance failures must prevent publishing coverage evidence."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _module(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('coverage_evidence_test', scripts / 'coverage_evidence.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_export_binds_revision_and_refuses_tree_changes(tmp_path, monkeypatch):
    module = _module(monkeypatch)
    from tessera.compiler.generated_docs import GeneratedDoc
    doc = GeneratedDoc('test_coverage', 'test', tmp_path / 'unused.md', lambda: '# Coverage\n',
                       csv_path=tmp_path / 'unused.csv', render_csv=lambda: 'op,refs\na,1\n',
                       also_gate_md=True)
    monkeypatch.setattr(module.gd, 'get', lambda name: doc)
    state = {'head': 'a' * 40, 'files': {'source.py': 'digest'}}
    monkeypatch.setattr(module, 'snapshot', lambda root: state)
    monkeypatch.setenv('GITHUB_SHA', state['head'])
    path = module.export(tmp_path / 'good', tmp_path)
    manifest = json.loads(path.read_text())
    assert manifest['source_commit'] == state['head']
    assert set(manifest['artifacts']) == {'test_coverage.csv', 'test_coverage.md'}
    assert all(len(value) == 64 for value in manifest['artifacts'].values())
    snapshots = iter([state, dict(state, head='changed')])
    monkeypatch.setattr(module, 'snapshot', lambda root: next(snapshots))
    with pytest.raises(ValueError, match='changed during generation'):
        module.export(tmp_path / 'bad', tmp_path)
    assert not (tmp_path / 'bad' / 'manifest.json').exists()


def test_export_rejects_wrong_ci_checkout(tmp_path, monkeypatch):
    module = _module(monkeypatch)
    monkeypatch.setattr(module, 'snapshot', lambda root: {'head': 'wrong'})
    monkeypatch.setenv('GITHUB_SHA', 'expected')
    with pytest.raises(ValueError, match='differs from GITHUB_SHA'):
        module.export(tmp_path, tmp_path)


def test_export_never_reuses_stale_manifest(tmp_path, monkeypatch):
    module = _module(monkeypatch)
    (tmp_path / 'manifest.json').write_text('old evidence')
    with pytest.raises(ValueError, match='must be empty'):
        module.export(tmp_path, tmp_path)
    assert (tmp_path / 'manifest.json').read_text() == 'old evidence'
