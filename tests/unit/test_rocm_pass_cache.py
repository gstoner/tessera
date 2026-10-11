"""Native replay reuse must preserve compiler identity and rejection checks."""
from pathlib import Path
import pytest
from tessera.compiler import rocm_pass_cache as cache

@pytest.fixture(autouse=True)
def clean_cache():
    cache.clear()
    yield
    cache.clear()

@pytest.fixture
def replay(monkeypatch, tmp_path):
    tool = tmp_path / "compiler"
    tool.write_text("v1")
    monkeypatch.setattr(cache, "_identity",
                        lambda p: (str(p.resolve()), p.read_text(), ()))
    calls = []
    def execute(tool, source, option):
        calls.append((source, option))
        return tool.read_text() + source + option
    return tool, calls, execute

def test_exact_replay_reuses_native_output(replay):
    tool, calls, execute = replay
    first = cache.run(tool, "IR", "--canonicalize", execute=execute)
    assert cache.run(tool, "IR", "--canonicalize", execute=execute) == first
    assert len(calls) == 1

@pytest.mark.parametrize("change", ["source", "pass", "tool", "environment", "runner"])
def test_changed_inputs_do_not_reuse_output(replay, monkeypatch, change):
    tool, calls, execute = replay
    cache.run(tool, "IR", "--canonicalize", execute=execute)
    source, option = "IR", "--canonicalize"
    if change == "source":
        source = "changed IR"
    elif change == "pass":
        option = "--tessera-schedule-to-tile"
    elif change == "tool":
        tool.write_text("v2")
    elif change == "environment":
        monkeypatch.setattr(cache, "_identity", lambda p: (str(p), "v1", (("ENV", "new"),)))
    else:
        original = execute
        execute = lambda *args: original(*args)
    cache.run(tool, source, option, execute=execute)
    assert len(calls) == 2

def test_failures_are_never_cached(replay):
    tool, calls, _ = replay
    def fail(*args):
        calls.append(args)
        raise RuntimeError("native verification refused")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="refused"):
            cache.run(tool, "bad IR", "--canonicalize", execute=fail)
    assert len(calls) == 2

def test_changed_tool_during_run_is_not_cached(replay):
    tool, calls, _ = replay
    def execute(tool, source, option):
        calls.append(source)
        tool.write_text(tool.read_text() + "changed")
        return source
    for _ in range(2):
        cache.run(tool, "IR", "--canonicalize", execute=execute)
    assert len(calls) == 2
    assert not cache._outputs

def test_output_byte_and_entry_limits(replay, monkeypatch):
    tool, _, execute = replay
    monkeypatch.setattr(cache, "_LIMIT_BYTES", 80)
    monkeypatch.setattr(cache, "_LIMIT_ENTRIES", 2)
    for i in range(8):
        cache.run(tool, str(i), "--canonicalize", execute=execute)
    assert len(cache._outputs) <= 2 and cache._bytes <= 80
    cache.run(tool, "x" * 100, "--canonicalize", execute=execute)
    assert len(cache._outputs) <= 2 and cache._bytes <= 80

def test_impure_pass_is_not_admitted(replay):
    tool, calls, execute = replay
    with pytest.raises(ValueError, match="pure ancestry"):
        cache.run(tool, "IR", "--emit-side-effecting-file", execute=execute)
    assert not calls

def test_real_tool_environment_and_contents_participate(tmp_path, monkeypatch):
    tool = tmp_path / "compiler"
    tool.write_text("v1")
    first = cache._identity(tool)
    monkeypatch.setenv("TESSERA_CACHE_TEST_ENV", "changed")
    assert cache._identity(tool) != first
    second = cache._identity(tool)
    tool.write_text("v2")
    assert cache._identity(tool) != second

def test_working_directory_is_part_of_replay_identity(tmp_path, monkeypatch):
    from tessera.compiler import rocm_native
    tool = tmp_path / "compiler"
    tool.write_text("same binary")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    # The binary and resolved dependencies may be identical in both dirs.
    monkeypatch.setattr(rocm_native, "_tool_digest", lambda tool: "same contents")
    calls = []
    def execute(tool, source, option):
        calls.append(str(Path.cwd()))
        return str(Path.cwd())
    monkeypatch.chdir(first)
    assert cache.run(tool, "IR", "--canonicalize", execute=execute) == str(first)
    assert cache.run(tool, "IR", "--canonicalize", execute=execute) == str(first)
    monkeypatch.chdir(second)
    assert cache.run(tool, "IR", "--canonicalize", execute=execute) == str(second)
    assert cache.run(tool, "IR", "--canonicalize", execute=execute) == str(second)
    assert calls == [str(first), str(second)]
