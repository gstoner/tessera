"""The validation guard must reject wrong trees even when a command passes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / 'scripts/validation_tree.py'
spec = importlib.util.spec_from_file_location('validation_tree', GUARD)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def repo(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'code.py').write_text('original')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                    'commit', '-qm', 'fixture'], cwd=tmp_path, check=True)
    return tmp_path


def test_guard_rejects_missing_expected_code(tmp_path):
    root = repo(tmp_path / 'repo')
    expected = module.snapshot(root)
    (root / 'code.py').write_text('new code')
    expected = module.snapshot(root)
    (root / 'code.py').write_text('original')
    manifest = tmp_path / 'expected.json'
    manifest.write_text(json.dumps(expected))
    result = subprocess.run([sys.executable, str(GUARD), '--root', str(root),
                             '--expect', str(manifest), '--', sys.executable, '-c', 'pass'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'source mismatch: code.py' in result.stderr


def test_guard_rejects_mutation_during_passing_gate(tmp_path):
    root = repo(tmp_path / 'repo')
    result = subprocess.run([sys.executable, str(GUARD), '--root', str(root), '--',
                             sys.executable, '-c', 'from pathlib import Path; Path("code.py").write_text("changed")'],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert 'tree changed' in result.stderr


def test_guard_records_stable_passing_gate(tmp_path):
    root = repo(tmp_path / 'repo')
    receipt = tmp_path / 'receipt.json'
    result = subprocess.run([sys.executable, str(GUARD), '--root', str(root),
                             '--receipt', str(receipt), '--', sys.executable, '-c', 'pass'],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(receipt.read_text())['stable']
