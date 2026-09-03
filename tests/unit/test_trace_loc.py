"""Tracer ops carry ``loc(...)`` in the canonical render -- Decision #13.

The tracer records the USER's op call site (the first frame outside the tessera
package) and the parser-bound render emits it as MLIR ``loc("file":line:col)``.
The paren (golden-text) render is untouched. When tessera-opt is available the
location must survive a real parse -- that is the end-to-end claim.
"""

import inspect
import os
import re
import subprocess

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.trace import to_graph_ir_module, trace


def _mm(a, b):
    return ts.ops.matmul(a, b)  # the loc must point at THIS line


def _traced():
    t = trace(_mm, np.zeros((8, 16), np.float32), np.zeros((16, 4), np.float32))
    return to_graph_ir_module(t, name="f")


def test_recorded_op_carries_user_source_span():
    m = _traced()
    (op,) = m.functions[0].body
    assert op.op_name == "tessera.matmul"
    span = op.source_span
    assert span is not None and span.source_name
    assert os.path.samefile(span.source_name, __file__)
    lines, start = inspect.getsourcelines(_mm)
    assert start <= span.line < start + len(lines)
    assert "ts.ops.matmul" in lines[span.line - start]
    assert span.col >= 1


def test_canonical_render_emits_loc_and_paren_render_does_not():
    m = _traced()
    canonical = m.to_mlir(canonical=True)
    assert re.search(
        r'tessera\.matmul .* loc\("[^"]*test_trace_loc\.py":\d+:\d+\)', canonical
    ), canonical
    assert "loc(" not in m.to_mlir()


def test_loc_survives_tessera_opt_parse():
    from tests._support.compiler_tool import tessera_opt_path

    tool = os.environ.get("TESSERA_OPT") or tessera_opt_path()
    if not tool:
        pytest.skip("tessera-opt not available on this host")
    text = _traced().to_mlir(canonical=True)
    proc = subprocess.run([str(tool), "-", "-mlir-print-debuginfo"], input=text,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "test_trace_loc.py" in proc.stdout


def test_control_flow_op_carries_user_source_span():
    def f(x, w):
        return ts.control.fori_loop(0, 4, lambda i, c: ts.ops.silu(ts.ops.matmul(c, w)), x)

    t = trace(f, np.zeros((1, 8), np.float32), np.zeros((8, 8), np.float32))
    (cf,) = t.body
    assert cf.op_name == "tessera.control_for"
    assert cf.source_span is not None and cf.source_span.source_name
    assert os.path.samefile(cf.source_span.source_name, __file__)
    # Ops recorded inside the loop body are user ops too and carry a span.
    assert all(op.source_span is not None for op in cf.kwargs["_body"])


# ── render-level: the path a loc names is host-independent ────────────────── #

from tessera.compiler.graph_ir import IROp, SourceSpan  # noqa: E402


def _render(span):
    op = IROp(result="r", op_name="tessera.relu", operands=["%x"],
              operand_types=["tensor<4xf32>"], result_type="tensor<4xf32>",
              source_span=span)
    return op.to_mlir(canonical=True)


def test_in_repo_loc_is_repo_relative_so_canonical_digests_are_host_independent():
    text = _render(SourceSpan(line=3, col=5, source_name=__file__))
    assert ' loc("tests/unit/test_trace_loc.py":3:5)' in text, text
    assert "/Users/" not in text and "/home/" not in text


def test_out_of_repo_loc_keeps_its_absolute_path():
    text = _render(SourceSpan(line=9, col=2, source_name="/tmp/elsewhere/user.py"))
    assert ' loc("/tmp/elsewhere/user.py":9:2)' in text, text


def test_span_without_a_file_emits_no_loc_at_all():
    text = _render(SourceSpan(line=1, col=1))
    assert "loc(" not in text
    assert "loc(" not in _render(None)
