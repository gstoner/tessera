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


def test_tessera_mlir_emits_metadata_diagnostics_trace_and_graphviz(tmp_path, capsys):
    model = _write_model(tmp_path)

    assert mlir.main([str(model), "--emit=metadata", "--target=apple_cpu"]) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["schema"] == "tessera.mlir.metadata.v1"
    assert metadata["mode"] == "source_inspection"
    assert metadata["target"] == "apple_cpu"
    assert metadata["symbols"][0]["name"] == "loss"

    assert mlir.main([str(model), "--emit=diagnostics", "--mode=compile_artifact"]) == 0
    diagnostics = json.loads(capsys.readouterr().out)
    assert diagnostics["diagnostics"][0]["code"] == "W_COMPILE_ARTIFACT_STATIC_ONLY"

    assert mlir.main([str(model), "--emit=trace"]) == 0
    trace = json.loads(capsys.readouterr().out)
    assert trace["traceEvents"][0]["name"] == "source.inspect"

    assert mlir.main([str(model), "--emit=graphviz"]) == 0
    assert "digraph tessera_source" in capsys.readouterr().out


def test_tessera_mlir_emit_all_writes_artifact_bundle(tmp_path, capsys):
    model = _write_model(tmp_path)
    artifacts_dir = tmp_path / "debug_artifacts"

    assert mlir.main([str(model), "--emit=all", "--artifacts-dir", str(artifacts_dir), "--target=nvidia_sm90"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "tessera.mlir.debug_bundle.v1"
    assert payload["target"] == "nvidia_sm90"
    assert (artifacts_dir / "graph-ir.mlir").exists()
    assert (artifacts_dir / "metadata.json").exists()
    assert (artifacts_dir / "graphviz.dot").exists()


def test_tessera_mlir_compile_artifact_mode_uses_jit_symbol(tmp_path, capsys):
    model = _write_model(tmp_path)
    artifacts_dir = tmp_path / "compiled_debug"

    assert mlir.main([
        str(model),
        "--mode=compile_artifact",
        "--symbol=loss",
        "--emit=metadata",
    ]) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["mode"] == "compile_artifact"
    assert metadata["symbol"] == "loss"
    assert metadata["artifact_metadata"]["artifact_hashes"]["graph"]

    assert mlir.main([
        str(model),
        "--mode=compile_artifact",
        "--symbol=loss",
        "--emit=all",
        "--artifacts-dir",
        str(artifacts_dir),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "compile_artifact"
    assert (artifacts_dir / "graph-ir.mlir").exists()
    assert (artifacts_dir / "trace.json").exists()


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


def test_tessera_prof_emits_json_and_autotune_artifact(tmp_path, capsys):
    model = _write_model(tmp_path)
    artifact = tmp_path / "schedule.json"
    cache = tmp_path / "tuning.db"

    assert prof.main([
        str(model),
        "--emit=json",
        "--autotune",
        "--autotune-method=on_device",
        "--compile-target=sm90",
        "--shapes=128,128,128",
        "--max-trials=1",
        "--cache",
        str(cache),
        "--artifact",
        str(artifact),
    ]) == 0

    payload = json.loads(capsys.readouterr().out.split("\nartifact:")[0])
    schedule = json.loads(artifact.read_text())
    assert payload["mode"] == "source_inspection"
    assert payload["schedule_artifact"]["measurements"]["status"] == "unmeasured"
    assert schedule["target_features"]["family"] == "nvidia"
    assert cache.exists()


def test_tessera_prof_emits_advanced_profiler_plan(tmp_path, capsys):
    model = _write_model(tmp_path)
    manifest_path = tmp_path / "model_analyzer.json"
    result_path = tmp_path / "model_analyzer_result.json"

    assert prof.main([
        str(model),
        "--emit=json",
        "--compile-target=sm90",
        "--advanced-plan",
        "--model-analyzer-manifest",
        str(manifest_path),
        "--model-analyzer-result",
        str(result_path),
        "--trace-features=runtime_api,device_activity,intra_kernel,model_analyzer",
        "--kernels",
        "matmul",
        "flash_attn",
        "--analyzer-mode=quick",
        "--batch-sizes=1,4",
        "--instance-counts=1,2",
    ]) == 0

    stdout = capsys.readouterr().out
    payload = json.loads(stdout.split("\nmodel_analyzer_manifest:")[0])
    plan = payload["advanced_profiler_plan"]
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())
    providers = {cap["feature"]: cap["provider"] for cap in plan["capabilities"]}
    assert plan["target"] == "nvidia"
    assert plan["kernels"] == ["matmul", "flash_attn"]
    assert providers["runtime_api"] == "cupti-callback-api"
    assert providers["device_activity"] == "cupti-activity-api"
    assert providers["intra_kernel"] == "cupti-pc-sampling+compiler-instrumentation"
    assert plan["analyzer_sweep"]["batch_sizes"] == [1, 4]
    assert plan["intra_kernel_probes"][0]["phase"] == "prologue"
    assert manifest["schema"] == "tessera.compiler.model_analyzer_manifest.v1"
    assert manifest["runner"]["status"] == "planned"
    assert manifest["telemetry"]["required_features"] == ["runtime_api", "device_activity"]
    assert result["schema"] == "tessera.compiler.model_analyzer_result.v1"
    assert result["trial_count"] == 8
    assert result["best"]["status"] == "planned_estimate"
    assert str(manifest_path) in stdout
    assert str(result_path) in stdout


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
    assert 'tessera-autotune = "tessera.cli.autotune:main"' in text
    assert 'tessera-runtime-smoke = "tessera.cli.runtime:main"' in text


def test_guides_document_debugging_and_profiling_commands():
    debug_doc = Path("docs/guides/Tessera_Debugging_Tools_Guide.md").read_text()
    prof_doc = Path("docs/guides/Tessera_Profiling_And_Autotuning_Guide.md").read_text()
    api_doc = Path("docs/spec/PYTHON_API_SPEC.md").read_text()

    assert "tessera-mlir my_model.py --emit=graph-ir --debug" in debug_doc
    assert "tessera-prof my_model.py --metrics=flops,bandwidth,occupancy" in prof_doc
    assert "tessera-prof my_model.py --trace=trace.json" in api_doc
