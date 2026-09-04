import pytest
from tessera import _apple_gpu_dispatch as dispatch


@pytest.mark.parametrize('initial', [False, True])
def test_capture_restores_nested_and_exception_state(monkeypatch, initial):
    state = [initial]
    def setter(value):
        state[0] = bool(value.value)
    monkeypatch.setattr(dispatch, 'bind_registered',
                        lambda name: setter if name.endswith('set_enabled') else lambda: int(state[0]))
    with pytest.raises(ValueError, match='injected'):
        with dispatch.dispatch_telemetry_capture():
            assert state[0]
            with dispatch.dispatch_telemetry_capture():
                assert state[0]
            assert state[0]
            raise ValueError('injected')
    assert state[0] is initial


def test_capture_refuses_missing_runtime(monkeypatch):
    monkeypatch.setattr(dispatch, 'bind_registered', lambda name: None)
    with pytest.raises(RuntimeError, match='unavailable'):
        with dispatch.dispatch_telemetry_capture():
            pytest.fail('must not collect evidence without a working capture')


def test_metal4_timeout_sets_error_channel():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] / 'src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm').read_text()
    body = source.split('static bool mtl4_encode_and_wait(', 1)[1].split('static bool mtl4_write_timestamp', 1)[0]
    assert 'g_last_gpu_error_kind = 1;' in body
    assert 'g_last_gpu_error_msg = "Metal 4 command buffer timed out' in body


def test_order_tracer_attributes_transition_after_teardown(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    source = tmp_path / 'test_order.py'
    source.write_text('''import types, sys
state = [False]
def test_leaker():
    handle = types.SimpleNamespace(tessera_apple_gpu_dispatch_telemetry_enabled=lambda: state[0])
    sys.modules['tessera._apple_gpu_dispatch'] = types.SimpleNamespace(_handle=handle, _dylib_path='fake-image')
    state[0] = True
def test_downstream():
    assert state[0]
''')
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / 'transitions.jsonl'
    env = dict(os.environ, PYTHONPATH=str(root), TESSERA_TELEMETRY_TRACE=str(output))
    result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'scripts.trace_apple_telemetry', str(source), '-q'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]['test'].endswith('test_leaker')
    assert records[0]['after'] == {'enabled': True, 'image': 'fake-image'}
