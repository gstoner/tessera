from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _validation_env_module():
    path = ROOT / "scripts" / "validation_env.py"
    spec = importlib.util.spec_from_file_location("validation_env", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["validation_env"] = module
    spec.loader.exec_module(module)
    return module


def test_validation_prefers_explicit_python_env(tmp_path):
    mod = _validation_env_module()
    root = tmp_path / "repo"
    home = tmp_path / "home"
    root.mkdir()
    home.mkdir()

    assert mod.choose_python(env={"PYTHON": "/custom/python"}, root=root, home=home) == "/custom/python"


def test_validation_prefers_user_venv_before_repo_venv(tmp_path):
    mod = _validation_env_module()
    root = tmp_path / "repo"
    home = tmp_path / "home"
    user_python = home / "venv" / "bin" / "python"
    repo_python = root / ".venv" / "bin" / "python"
    user_python.parent.mkdir(parents=True)
    repo_python.parent.mkdir(parents=True)
    user_python.write_text("#!/bin/sh\n", encoding="utf-8")
    repo_python.write_text("#!/bin/sh\n", encoding="utf-8")
    user_python.chmod(0o755)
    repo_python.chmod(0o755)

    assert mod.choose_python(env={}, root=root, home=home) == str(user_python)


def test_validation_script_runs_environment_bootstrap_before_expensive_steps():
    text = (ROOT / "scripts" / "validate.sh").read_text(encoding="utf-8")

    assert "$HOME/venv/bin/python" in text
    assert "scripts/validation_env.py" in text
    assert text.index("scripts/validation_env.py") < text.index("scripts/check_versions.py")
