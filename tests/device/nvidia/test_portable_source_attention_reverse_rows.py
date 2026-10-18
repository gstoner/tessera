"""Actual five-member source program replay in a fresh process without tools."""
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tessera.compiler.prepared_source_attention_reverse import PreparedSourceAttentionReverse
from tests.device.nvidia.test_source_attention_reverse import source, oracle

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_source_rows_fresh_process_replay_has_no_compiler_or_subprocess(storage, tmp_path):
    assert rt._nvidia_device_name() == "sm_120", "owning RTX5070 required"
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(988)
    shapes = ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3), (1, 2, 3, 3), (1, 2, 3))
    values = tuple(rng.normal(0, .2, shape).astype(np.float32 if i == 4 else dtype)
                   for i, shape in enumerate(shapes))
    with PreparedSourceAttentionReverse(export_native_source_attention_reverse(
            source(values[:3], "normal", (0, 1, 2), rows=True))) as owner:
        expected = owner(*values)
        for actual, wanted in zip(expected, oracle(*values), strict=True):
            np.testing.assert_allclose(actual.astype(np.float32), wanted.astype(np.float32), rtol=.03, atol=.002)
        artifact = owner.runtime_artifact()
        assert artifact.metadata["native_program"]["schema"] == "tessera.nvidia.source_attention_reverse_program.v2"
        (tmp_path / "artifact.json").write_text(artifact.to_json())
        np.savez(tmp_path / "roots.npz", **{str(i): x.astype(np.float32) for i, x in enumerate(values)})
    script = """
from pathlib import Path
import sys, subprocess, numpy as np, ml_dtypes
from tessera import runtime as rt
root=Path(sys.argv[1]); storage=sys.argv[2]
data=np.load(root/"roots.npz")
dtype=np.float16 if storage=="f16" else ml_dtypes.bfloat16
values=tuple(data[str(i)].astype(np.float32 if i==4 else dtype) for i in range(5))
artifact=rt.RuntimeArtifact.from_json((root/"artifact.json").read_text())
def forbidden(*args,**kwargs):
    raise AssertionError("portable source rows invoked an external process")
subprocess.run=subprocess.Popen=subprocess.check_output=forbidden
result=rt.launch(artifact,values)
assert result.get("ok") and result.get("execution_kind")=="native_gpu",result
assert len(result["component_receipts"])==5
np.savez(root/"outputs.npz",**{str(i):x.astype(np.float32) for i,x in enumerate(result["output"])})
"""
    child = subprocess.run([sys.executable, "-c", script, str(tmp_path), storage],
        env={**os.environ, "PATH": "/unavailable/tools", "TESSERA_OPT": "/unavailable/compiler",
             "TESSERA_NVIDIA_OPT": "/unavailable/compiler"}, capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr
    replay = np.load(tmp_path / "outputs.npz")
    for i, actual in enumerate(expected):
        np.testing.assert_array_equal(replay[str(i)], actual.astype(np.float32))

