"""A before/after comparison must actually execute the requested lowerers."""
import os
from pathlib import Path
import subprocess

import pytest

from benchmarks.record_allocation_lifetime_comparison import selected_nvidia_compiler
from tessera.compiler import nvidia_native


@pytest.mark.parametrize('previous', [None, '/preexisting/nvidia-opt'])
def test_distinct_native_compilers_are_consumed_and_environment_restored(tmp_path, monkeypatch, previous):
    monkeypatch.setenv('TESSERA_OPT', '/unchanged/core-opt')
    if previous is None:
        monkeypatch.delenv('TESSERA_NVIDIA_OPT', raising=False)
    else:
        monkeypatch.setenv('TESSERA_NVIDIA_OPT', previous)
    for name in ('before', 'after'):
        compiler = tmp_path / name
        compiler.write_text('#!/bin/sh\nprintf "%s\\n" "' + name + '"\n')
        compiler.chmod(0o755)
        with selected_nvidia_compiler(compiler):
            consumed = nvidia_native._tool('tessera-nvidia-opt')
            assert consumed == compiler
            assert subprocess.check_output([str(consumed)], text=True).strip() == name
            assert os.environ['TESSERA_OPT'] == '/unchanged/core-opt'
        assert os.environ.get('TESSERA_NVIDIA_OPT') == previous
    with pytest.raises(RuntimeError), selected_nvidia_compiler(Path('/missing/compiler')):
        pass
    assert os.environ.get('TESSERA_NVIDIA_OPT') == previous
