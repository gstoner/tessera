"""Artifact transport retains native AD source/program witnesses and all members."""
import base64
from dataclasses import replace
import json
import re

import pytest

from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tests.unit.test_native_source_attention_reverse_outline import source

pytestmark = pytest.mark.compiler_route


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_source_attention_reverse_transport_never_constructs_graph(storage, monkeypatch):
    from tessera.compiler import graph_ir
    text = source(storage, "permuted")

    def forbidden(*args, **kwargs):
        pytest.fail("native source-attention reverse transport reconstructed Graph")

    monkeypatch.setattr(graph_ir.GraphIRFunction, "__init__", forbidden)
    monkeypatch.setattr(graph_ir.IROp, "__init__", forbidden)
    program = export_native_source_attention_reverse(text)
    assert program.source_ir == program.manifest["source_graph_ir"]
    stages = program.lower_members()
    assert len(stages) == 4
    assert "tessera.normalization_stats" in stages[3][0]
    assert "tile.row_region_kernel" in stages[3][2]
    assert "tessera_nvidia.row_region_kernel" not in stages[3][4]


@pytest.mark.parametrize("fault", ["lifetime", "seed_order", "gradient_order", "source", "compiler"])
def test_source_attention_reverse_rejects_changed_native_program(fault):
    program = export_native_source_attention_reverse(source())
    if fault == "compiler":
        changed = replace(program, compiler_identity="0" * 64)
        with pytest.raises(RuntimeError, match="compiler changed"):
            changed.validate()
        return
    if fault == "source":
        changed = replace(program, source_ir=program.source_ir + "\n")
    else:
        plan = program.manifest
        if fault == "lifetime":
            plan["buffers"][5]["last_read"] = 1
        elif fault == "seed_order":
            plan["cotangent_indices"].reverse()
        else:
            plan["outputs"].reverse()
        encoded = base64.b64encode(json.dumps(plan).encode()).decode()
        altered = re.sub(r'(tessera.native.source_attention_reverse_program_json = ")[^"]+(")',
                         lambda match: match[1] + encoded + match[2], program.outlined_ir)
        changed = replace(program, outlined_ir=altered)
    with pytest.raises(ValueError):
        changed.validate()
