from __future__ import annotations

import copy

import pytest

from tessera.compiler.scheduled_spectral import (
    lower_scheduled_spectral,
    validate_scheduled_spectral_metadata,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt

# These lower through the production `tessera-opt`, which the CI unit lane does
# not build. The library correctly RAISES rather than silently degrading, so
# the tests must skip there rather than fail.
_needs_opt = pytest.mark.skipif(
    find_tessera_opt() is None, reason="tessera-opt not built"
)


@pytest.mark.parametrize(
    ("op_name", "shapes", "hop", "child_count"),
    [
        ("tessera.spectral_filter", ((2, 17), (2, 17)), None, 0),
        ("tessera.dct", ((2, 17),), None, 1),
        ("tessera.spectral_conv", ((2, 13), (2, 7)), None, 2),
        ("tessera.stft", ((2, 43), (17,)), 6, 1),
        ("tessera.istft", ((2, 5, 9), (17,)), 6, 1),
    ],
)
@_needs_opt
def test_compound_contract_binds_physical_and_child_identity(
    op_name, shapes, hop, child_count
):
    artifact = lower_scheduled_spectral(
        target="rocm", op_name=op_name, input_shapes=shapes, hop=hop
    )
    metadata = artifact.to_metadata()
    validate_scheduled_spectral_metadata(metadata, input_shapes=shapes)

    assert metadata["schema"] == "tessera.scheduled_spectral.v2"
    assert metadata["architecture"] == "gfx1151"
    assert metadata["complex_layout"] == "interleaved_f32x2"
    assert metadata["workspace_policy"] == "persistent_artifact_workspace"
    assert metadata["workspace_bytes"] > 0
    assert metadata["mutation_lineage"] == "inputs_immutable_output_fresh_v1"
    assert len(metadata["child_ffts"]) == child_count
    assert metadata["child_fft_digests"] == [
        child["schedule_digest"] for child in metadata["child_ffts"]
    ]


@_needs_opt
def test_compound_contract_rejects_tampering():
    artifact = lower_scheduled_spectral(
        target="rocm",
        op_name="tessera.stft",
        input_shapes=((43,), (17,)),
        hop=6,
    ).to_metadata()
    tampered = copy.deepcopy(artifact)
    tampered["workspace_bytes"] += 8
    with pytest.raises(ValueError, match="workspace_bytes"):
        validate_scheduled_spectral_metadata(
            tampered, input_shapes=((43,), (17,))
        )


@pytest.mark.parametrize("target", ["rocm_gfx1200", "rocm_gfx1250"])
def test_compound_contract_fails_closed_without_architecture_evidence(target):
    with pytest.raises(ValueError, match="architecture-owned profiles"):
        lower_scheduled_spectral(
            target=target,
            op_name="tessera.spectral_filter",
            input_shapes=((8,), (8,)),
        )
