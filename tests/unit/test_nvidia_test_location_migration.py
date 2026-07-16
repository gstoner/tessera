"""Static guard for the NVIDIA-TEST-6-HIGH node-ID relocation contract."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "tests/device/nvidia/node_migrations.json"


def _node_path(node_id: str) -> Path:
    return ROOT / node_id.split("::", 1)[0]


def _expanded_mappings(data: dict) -> dict[str, str]:
    mappings = dict(data["mappings"])
    for migration in data.get("parameterized_migrations", []):
        for parameter in migration["parameters"]:
            old = f"{migration['source_file']}::{migration['test_name']}[{parameter}]"
            new = f"{migration['destination_file']}::{migration['test_name']}[{parameter}]"
            assert old not in mappings, old
            mappings[old] = new
    return mappings


def test_nvidia_moe_transport_node_migration_has_no_old_file_duplicate():
    data = json.loads(MIGRATIONS.read_text(encoding="utf-8"))
    mappings = _expanded_mappings(data)
    assert data["migration"] == "NVIDIA-TEST-6-HIGH"
    assert len(mappings) == 286
    for new_node in mappings.values():
        assert _node_path(new_node).is_file(), new_node
    assert len(set(mappings.values())) == len(mappings)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *sorted({str(_node_path(node).relative_to(ROOT)) for node in mappings.values()}),
            "--collect-only",
            "-q",
            "--no-header",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    collected = {
        line for line in proc.stdout.splitlines() if line.startswith("tests/")
    }
    assert collected == set(mappings.values())

    old_paths = sorted({_node_path(node) for node in mappings if _node_path(node).is_file()})
    if old_paths:
        old_proc = subprocess.run(
            [sys.executable, "-m", "pytest", *[str(path.relative_to(ROOT)) for path in old_paths], "--collect-only", "-q", "--no-header"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert old_proc.returncode == 0, old_proc.stderr
        old_collected = {line for line in old_proc.stdout.splitlines() if line.startswith("tests/")}
        assert not old_collected.intersection(mappings), old_collected.intersection(mappings)
