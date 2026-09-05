"""Native checkpoint ownership survives serialization and rejects semantic drift."""

from dataclasses import replace

import pytest

from tessera.compiler import nvidia_native as native
from tessera.compiler import scheduled_checkpoint as checkpoint
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")


def artifact(backward=False):
    names = ("do", "q", "k", "v", "row_lse", "dq", "dk", "dv") if backward else ("q", "k", "v", "o", "row_lse")
    return checkpoint.lower_scheduled_checkpoint(names, (1, 2, 1, 3, 4, 4, 3), 0.5, True, backward=backward)


@pytest.mark.parametrize("backward", [False, True])
def test_package_consumes_native_checkpoint_without_graph(backward, monkeypatch):
    scheduled = artifact(backward)
    calls = []
    monkeypatch.setattr(
        native,
        "_compile_tile_ir",
        lambda text, entry: (calls.append(text) or text, "// PTX", {}, "compiler", "toolchain", (), "cold"),
    )
    monkeypatch.setattr(checkpoint, "_graph_text", lambda *a: pytest.fail("reconstructed Graph"))
    package = native.package_scheduled_checkpoint(
        replace(scheduled, graph_ir="discarded"), pipeline_name="tessera-nvidia-pipeline-sm120"
    )
    assert calls == [scheduled.tile_ir]
    assert package.descriptor.provenance["schedule_digest"] == scheduled.schedule_digest
    assert not hasattr(native, "emit_attention_tile_ir")


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("tile_ir", "corrupted"),
        ("scale", 0.25),
        ("causal", False),
        ("dims", (1, 2, 1, 3, 5, 4, 3)),
        ("entry", "another_entry"),
        ("schedule_digest", "0" * 64),
        ("names", ("x", "k", "v", "o", "row_lse")),
    ],
)
def test_checkpoint_rejects_tampered_artifact_before_target_compile(backward, field, value, monkeypatch):
    scheduled = artifact(backward)
    monkeypatch.setattr(native, "_compile_tile_ir", lambda *a: pytest.fail("compiled tampered artifact"))
    with pytest.raises(ValueError):
        native.package_scheduled_checkpoint(
            replace(scheduled, **{field: value}), pipeline_name="tessera-nvidia-pipeline-sm120"
        )


@pytest.mark.parametrize("backward", [False, True])
def test_checkpoint_rejects_swapped_tile_pointers(backward):
    scheduled = artifact(backward)
    # Preserve types, attributes and hash; only corrupt the executable operand binding.
    import re

    text, count = re.subn(r"(tile.attention(?:_backward)?_kernel )%arg0, %arg1,", r"\1%arg1, %arg0,", scheduled.tile_ir)
    assert count == 1
    with pytest.raises(ValueError, match="replay"):
        replace(scheduled, tile_ir=text).validate()


@pytest.mark.parametrize("backward", [False, True])
def test_native_schedule_rejects_changed_checkpoint_policy(backward):
    scheduled = artifact(backward)
    with pytest.raises(RuntimeError, match="contract changed"):
        run_tessera_opt(
            find_tessera_opt(),
            scheduled.schedule_ir.replace("causal = true", "causal = false"),
            "--tessera-schedule-to-tile",
        )


@pytest.mark.parametrize("backward", [False, True])
def test_registered_checkpoint_rejects_wrong_lse_shape(backward):
    scheduled = artifact(backward)
    malformed = scheduled.graph_ir.replace("tensor<1x2x3xf32>", "tensor<1x2x4xf32>")
    with pytest.raises(RuntimeError, match="checkpoint"):
        run_tessera_opt(find_tessera_opt(), malformed, "--tessera-graph-to-schedule")


@pytest.mark.parametrize("field,value", [("causal", 1), ("backward", 0), ("scale", True)])
def test_checkpoint_descriptor_rejects_boolean_number_confusion(field, value):
    with pytest.raises(ValueError):
        replace(artifact(), **{field: value}).validate()


def test_checkpoint_rejects_input_output_binding_alias():
    with pytest.raises(RuntimeError, match="unique"):
        checkpoint.lower_scheduled_checkpoint(("q", "k", "v", "q", "row_lse"),
                                             (1, 2, 1, 3, 4, 4, 3), 0.5, True)
