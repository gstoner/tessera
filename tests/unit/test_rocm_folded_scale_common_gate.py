"""Folded vector scales retain per-element exceptional recovery behind a common gate."""
import re
import subprocess

import pytest

pytestmark = pytest.mark.compiler_route


@pytest.mark.parametrize("vector_scales", [False, True])
def test_vector_scale_common_branch_keeps_scalar_recovery(compiler_toolchain, vector_scales):
    from tests.device.rocm.test_folded_scale_native import SOURCE

    tool = compiler_toolchain.require_tessera_opt("--lower-tile-to-rocm")
    source = SOURCE.replace("bounds(%m, %n) : !fc",
        f"bounds(%m, %n) {{vector_scales = {str(vector_scales).lower()}}} : !fc")
    result = subprocess.run([str(tool), "--lower-tile-to-rocm=arch=gfx1201"],
                            input=source, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    # Vector path branches once around eight original scalar recovery gates.
    vector_branches = re.findall(r"scf\.if [^\n]* -> \(vector<8xf32>\)", result.stdout)
    scalar_branches = re.findall(r"scf\.if [^\n]* -> \(f32\)", result.stdout)
    assert len(vector_branches) == (2 if vector_scales else 0)
    assert len(scalar_branches) == 8
    assert "arith.extf" in result.stdout and "f32 to f64" in result.stdout
