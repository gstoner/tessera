import json
from pathlib import Path

from tessera.cli import mlir, prof, runtime


def _write_model(tmp_path: Path) -> Path:
    model = tmp_path / "my_model.py"
    model.write_text(
        "\n".join(
            [
                "import tessera as ts",
                "",
                "@ts.jit",
                "def loss(x):",
                "    return ts.ops.softmax(x)",
                "",
                "class TinyModel:",
                "    pass",
            ]
        )
    )
    return model


def test_tessera_mlir_dumps_graph_ir_to_stdout(tmp_path, capsys):
    model = _write_model(tmp_path)

    assert mlir.main([str(model), "--emit=graph-ir", "--debug"]) == 0

    out = capsys.readouterr().out
    assert "tessera-mlir emit=graph-ir" in out
    assert "func.func @loss" in out
    assert "tessera.graph.module" in out
    assert "debug: discovered func loss" in out


def test_tessera_mlir_writes_tile_ir(tmp_path):
    model = _write_model(tmp_path)
    output = tmp_path / "tile.mlir"

    assert mlir.main([str(model), "--emit=tile-ir", "--output", str(output)]) == 0

    text = output.read_text()
    assert "tessera-mlir emit=tile-ir" in text
    assert "tessera.tile.entry_stub" in text
    assert 'sym_name = "loss"' in text


def test_tessera_prof_reports_requested_metrics(tmp_path, capsys):
    model = _write_model(tmp_path)

    assert prof.main([str(model), "--metrics=flops,bandwidth,occupancy"]) == 0

    out = capsys.readouterr().out
    assert "Op" in out
    assert "FLOPs(G)" in out
    assert "Bandwidth(GB/s)" in out
    assert "model.inspect" in out


def test_tessera_prof_writes_chrome_trace(tmp_path, capsys):
    model = _write_model(tmp_path)
    trace = tmp_path / "trace.json"

    assert prof.main([str(model), "--trace", str(trace)]) == 0

    payload = json.loads(trace.read_text())
    assert "traceEvents" in payload
    assert payload["traceEvents"][0]["name"] == "model.inspect"
    telemetry = payload["traceEvents"][0]["args"]["telemetry"]
    assert telemetry["schema"] == "tessera.telemetry.v1"
    assert telemetry["op"] == "model.inspect"
    assert telemetry["kernel_id"] == model.stem
    assert str(trace) in capsys.readouterr().out


def test_tessera_runtime_smoke_writes_telemetry(tmp_path):
    output = tmp_path / "runtime.json"

    assert runtime.main(["--output", str(output), "--bytes", "16"]) == 0

    payload = json.loads(output.read_text())
    assert payload["schema"] == "tessera.telemetry.v1"
    assert payload["runtime_status"] == "success"
    assert payload["mapped_bytes"] == 16


def test_pyproject_registers_console_scripts():
    text = Path("pyproject.toml").read_text()
    assert 'tessera-mlir = "tessera.cli.mlir:main"' in text
    assert 'tessera-prof = "tessera.cli.prof:main"' in text
    assert 'tessera-runtime-smoke = "tessera.cli.runtime:main"' in text


def test_guides_document_debugging_and_profiling_commands():
    debug_doc = Path("docs/guides/Tessera_Debugging_Tools_Guide.md").read_text()
    prof_doc = Path("docs/guides/Tessera_Profiling_And_Autotuning_Guide.md").read_text()
    api_doc = Path("docs/spec/PYTHON_API_SPEC.md").read_text()

    assert "tessera-mlir my_model.py --emit=graph-ir --debug" in debug_doc
    assert "tessera-prof my_model.py --metrics=flops,bandwidth,occupancy" in prof_doc
    assert "tessera-prof my_model.py --trace=trace.json" in api_doc
