"""An explicit package compiler also owns native member image verification."""
import pytest
from tessera.compiler.native_tensor_program import compile_native_tensor_program, restore_runtime_artifact, runtime_artifact
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.unit.test_native_tensor_program_pipeline import traced_source, relu_product

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


@pytest.mark.parametrize("operation", ["package", "validate", "restore"])
def test_selected_compiler_owns_images_despite_conflicting_backend_environment(operation, monkeypatch, tmp_path):
    tool = find_tessera_opt()
    assert tool is not None
    source = traced_source(relu_product, "f16")
    baseline = compile_native_tensor_program(source, tool=tool)
    assert baseline is not None
    artifact = runtime_artifact(baseline, argument_names=("a", "b"))
    monkeypatch.setenv("TESSERA_NVIDIA_OPT", str(tmp_path / "unrelated-missing-compiler"))
    try:
        if operation == "package":
            current = compile_native_tensor_program(source, tool=tool)
        elif operation == "restore":
            current = restore_runtime_artifact(artifact)
        else:
            baseline.validate()
            current = baseline
    except RuntimeError as error:
        raise AssertionError("selected compiler was not forwarded to native image lowering") from error
    assert current is not None
    assert current.compiler.resolve() == tool.resolve()
    assert current.compiler_identity == baseline.compiler_identity
    assert current.frame == baseline.frame
    assert current.images == baseline.images
