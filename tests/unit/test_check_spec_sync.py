"""Guard the pre-push structural-sync pre-flight (`scripts/check_spec_sync.py`).

That script is the fast local mirror of two full-pytest-only gates
(`test_python_api_spec_lists_current_runtime_op_catalog` +
`test_no_unregistered_generated_markdown`). These tests keep the script honest:
its two checks must agree with the live op-catalog / generated-docs registry, so
the pre-flight can't silently rot and let a real drift through to CI.
"""

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_checker():
    path = ROOT / "scripts" / "check_spec_sync.py"
    spec = importlib.util.spec_from_file_location("check_spec_sync", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_op_catalog_in_spec_passes_on_clean_tree():
    chk = _load_checker()
    assert chk._check_op_catalog_in_spec() == []


def test_generated_markdown_registered_passes_on_clean_tree():
    chk = _load_checker()
    assert chk._check_generated_markdown_registered() == []


def test_main_exit_zero_on_clean_tree():
    chk = _load_checker()
    assert chk.main() == 0


def test_checker_detects_undocumented_op(monkeypatch):
    chk = _load_checker()
    from tessera.compiler import op_catalog as cat
    fake = cat.OpSpec("zzz_unlisted_op", "tessera.zzz_unlisted", 1, 1)
    patched = dict(cat.OP_SPECS)
    patched["zzz_unlisted_op"] = fake
    monkeypatch.setattr(cat, "OP_SPECS", patched)
    assert "zzz_unlisted_op" in chk._check_op_catalog_in_spec()
