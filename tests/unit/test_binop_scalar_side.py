"""``scalar_side`` — the operand-order contract for a lifted BinOp literal.

``graph_ir._OpExtractor._try_map_binop`` lifts a literal operand out of a
``BinOp`` into the ``scalar`` attribute, leaving a one-operand IROp, and records
which side it came from in ``scalar_side``. That record is load-bearing for the
non-commutative ops the extractor lowers — ``tessera.sub`` and ``tessera.div``:
without it, ``2.0 - x`` and ``x - 2.0`` emit IROps that are identical, and a
consumer that reads ``scalar`` as the right operand computes ``x - 2.0`` for
both. Sign-flipped for ``sub``, reciprocal for ``div``, silently.

``scalar_side`` was a Decision #29 violation until 2026-08-19: nothing in
``python/``, ``src/``, or ``tools/`` read it. The consumer is now
``runtime._apple_gpu_dispatch_mpsgraph_binary``, whose opcode set is mostly
non-commutative (sub, div, pow, mod, floor_div, atan2, and all six
comparisons). The other two ``scalar`` consumers — ``matmul_pipeline``'s and
``runtime``'s CPU executors — implement the scalar form for ``add``/``mul``
only, which commute, so operand order cannot change their result.

What each section pins:

* ``TestConsumerContract``   — the dispatcher honours the side, treats absence
  as "right" (the definition the hand-written ``{"scalar": s}`` call sites in
  ``runtime.py`` rely on), and rejects anything else rather than guessing.
* ``TestProducerEmitsSide``  — the extractor still states the side on both
  branches, so the declaration keeps a producer.
* ``TestProducerConsumerRoundTrip`` — the extractor's own kwargs, fed verbatim
  into the dispatcher exactly as ``_apple_gpu_execute_artifact`` forwards them,
  match numpy. This is the assertion that would have caught the original bug;
  the two halves were each self-consistent and only the join was wrong.
* ``TestEndToEnd``           — ``@tessera.jit`` values for both spellings.
* ``TestTracerFailsClosed``  — the successor frontend (E2E-REAL-6 retires
  ``_OpExtractor``) cannot inherit this bug, because it refuses scalar operands
  outright rather than lifting them to an unordered attribute.
"""

import sys

import numpy as np
import pytest

import tessera
import tessera.runtime as runtime_mod
from tessera import ops
from tessera.compiler import trace as trace_mod
from tessera.compiler.graph_ir import GraphIRBuilder
from tessera.runtime import _apple_gpu_dispatch_mpsgraph_binary as dispatch_binary

X = np.array([1.0, 2.0, 4.0], dtype=np.float32)
S = 2.0

#: op short name → numpy truth, for the non-commutative half of
#: ``_APPLE_GPU_BINARY_OPCODES``. Every one of these gives a different answer
#: for ``S <op> x`` than for ``x <op> S``.
NONCOMMUTATIVE = {
    "sub": lambda a, b: a - b,
    "div": lambda a, b: a / b,
    "pow": np.power,
    "mod": lambda a, b: a - b * np.floor(a / b),
    "floor_div": lambda a, b: np.floor(a / b),
    "atan2": np.arctan2,
    "lt": lambda a, b: (a < b).astype(np.float32),
    "le": lambda a, b: (a <= b).astype(np.float32),
    "gt": lambda a, b: (a > b).astype(np.float32),
    "ge": lambda a, b: (a >= b).astype(np.float32),
}


def _build(body: str, name: str = "f"):
    """Lower ``def name(x): <body>`` through the AST frontend."""
    source = f"def {name}(x):\n    {body}\n"
    namespace: dict = {}
    exec("from tessera import ops\n" + source, namespace)  # noqa: S102
    builder = GraphIRBuilder()
    return builder.lower(namespace[name], source_text=source), builder


def _live_kernel_available() -> bool:
    """Whether the compiled binary kernel implements the ops under test here.

    On Darwin the MPSGraph/MSL implementation covers the whole opcode table. On
    every other host `_load_apple_gpu_runtime` compiles the portable stub
    (`apple_gpu_runtime_stub.cpp`), whose switch implements opcodes 0-8 and
    whose `default:` arm assigns `out[i] = x` — so `mod`, `floor_div`, the six
    comparisons, and the logical/bitwise ops silently return the LEFT operand
    instead of computing anything. That is a pre-existing defect in the stub,
    independent of the operand-ordering contract under test here, and it is why
    the live-kernel lane is Darwin-gated rather than probed: probing for it
    would mean asserting the very thing these tests assert.
    """
    return sys.platform == "darwin" and runtime_mod._apple_gpu_mpsgraph_binary_f32() is not None


@pytest.fixture(params=["host_reference", "live_kernel"])
def lane(request, monkeypatch):
    """Run each contract assertion through both dispatcher lanes.

    The ordering swap happens before the lane split, so both must agree. The
    host-reference lane is forced by making the symbol lookup miss, which is
    the dispatcher's own documented fallback and is deterministic on every
    host — so operand ordering stays covered in CI even where the compiled
    kernel is not.
    """
    if request.param == "host_reference":
        monkeypatch.setattr(runtime_mod, "_apple_gpu_mpsgraph_binary_f32", lambda: None)
    elif not _live_kernel_available():
        pytest.skip(
            "compiled binary kernel unavailable or incomplete on this host "
            "(non-Darwin uses apple_gpu_runtime_stub.cpp, which implements "
            "opcodes 0-8 only and returns operand A for the rest)"
        )
    return request.param


class TestConsumerContract:
    """``runtime._apple_gpu_dispatch_mpsgraph_binary`` — the one consumer."""

    @pytest.mark.parametrize("short", sorted(NONCOMMUTATIVE))
    def test_left_side_scalar_is_the_left_operand(self, short, lane):
        """The negative fixture: a left-side scalar on a NON-commutative op.

        Every one of these returned the right-side answer before the fix.
        """
        got = dispatch_binary(
            f"tessera.{short}", [X], {"scalar": S, "scalar_side": "left"}, np
        )
        np.testing.assert_allclose(
            np.asarray(got), NONCOMMUTATIVE[short](S, X), rtol=1e-6, atol=1e-6
        )

    @pytest.mark.parametrize("short", sorted(NONCOMMUTATIVE))
    def test_left_and_right_actually_differ(self, short, lane):
        """Guards the fixtures themselves: if these two agreed, the parametrized
        cases above would pass under a consumer that ignored the side."""
        left = np.asarray(
            dispatch_binary(f"tessera.{short}", [X], {"scalar": S, "scalar_side": "left"}, np)
        )
        right = np.asarray(
            dispatch_binary(f"tessera.{short}", [X], {"scalar": S, "scalar_side": "right"}, np)
        )
        assert not np.allclose(left, right)

    @pytest.mark.parametrize("short", sorted(NONCOMMUTATIVE))
    def test_right_side_scalar_is_the_right_operand(self, short, lane):
        got = dispatch_binary(
            f"tessera.{short}", [X], {"scalar": S, "scalar_side": "right"}, np
        )
        np.testing.assert_allclose(
            np.asarray(got), NONCOMMUTATIVE[short](X, S), rtol=1e-6, atol=1e-6
        )

    @pytest.mark.parametrize("short", sorted(NONCOMMUTATIVE))
    def test_absent_side_means_right(self, short, lane):
        """``runtime.py``'s own ``{"scalar": s}`` call sites (softcap, clamp,
        the unary-with-scalar helper, ...) omit the key and mean the right
        operand. Absence must stay pinned to that reading."""
        got = dispatch_binary(f"tessera.{short}", [X], {"scalar": S}, np)
        np.testing.assert_allclose(
            np.asarray(got), NONCOMMUTATIVE[short](X, S), rtol=1e-6, atol=1e-6
        )

    def test_unknown_side_is_rejected_not_guessed(self, lane):
        with pytest.raises(ValueError, match="scalar_side must be"):
            dispatch_binary("tessera.sub", [X], {"scalar": S, "scalar_side": "middle"}, np)

    def test_two_real_operands_ignore_the_side(self, lane):
        """With both operands materialized, order is carried by the dataflow and
        the attribute is irrelevant (it describes the lifted form only)."""
        y = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        got = dispatch_binary(
            "tessera.sub", [X, y], {"scalar": 99.0, "scalar_side": "left"}, np
        )
        np.testing.assert_allclose(np.asarray(got), X - y, rtol=1e-6)


class TestProducerEmitsSide:
    """``_OpExtractor._try_map_binop`` — the producer keeps the declaration."""

    @pytest.mark.parametrize(
        "body,op_name,side",
        [
            ("return 2.0 - x", "tessera.sub", "left"),
            ("return x - 2.0", "tessera.sub", "right"),
            ("return 2.0 / x", "tessera.div", "left"),
            ("return x / 2.0", "tessera.div", "right"),
            ("return 2.0 + x", "tessera.add", "left"),
            ("return x + 2.0", "tessera.add", "right"),
            ("return 2.0 * x", "tessera.mul", "left"),
            ("return x * 2.0", "tessera.mul", "right"),
        ],
    )
    def test_side_is_recorded_on_both_branches(self, body, op_name, side):
        fn_ir, _ = _build(body)
        assert len(fn_ir.body) == 1
        op = fn_ir.body[0]
        assert op.op_name == op_name
        assert op.kwargs["scalar"] == 2.0
        # Stated on BOTH branches rather than leaning on the absent-means-right
        # default: the extractor is not the only producer of a bare ``scalar``.
        assert op.kwargs["scalar_side"] == side

    def test_left_and_right_are_distinguishable_in_the_ir(self):
        """The regression itself: these two lowered to indistinguishable IR."""
        left, _ = _build("return 2.0 - x", name="l")
        right, _ = _build("return x - 2.0", name="r")
        assert left.body[0].kwargs != right.body[0].kwargs


class TestProducerConsumerRoundTrip:
    """The join. ``_apple_gpu_execute_artifact`` forwards an op's ``kwargs``
    verbatim to the lane handler, so the extractor's dict is exactly what the
    dispatcher sees."""

    @pytest.mark.parametrize(
        "body,truth",
        [
            ("return 2.0 - x", lambda x: 2.0 - x),
            ("return x - 2.0", lambda x: x - 2.0),
            ("return 2.0 / x", lambda x: 2.0 / x),
            ("return x / 2.0", lambda x: x / 2.0),
            ("return 2.0 + x", lambda x: 2.0 + x),
            ("return 2.0 * x", lambda x: 2.0 * x),
        ],
    )
    def test_extracted_kwargs_dispatch_to_the_right_answer(self, body, truth, lane):
        fn_ir, _ = _build(body)
        op = fn_ir.body[0]
        got = dispatch_binary(op.op_name, [X], dict(op.kwargs), np)
        np.testing.assert_allclose(np.asarray(got), truth(X), rtol=1e-6, atol=1e-6)


class TestEndToEnd:
    @pytest.mark.parametrize("target", ["apple_gpu", "apple_cpu"])
    @pytest.mark.parametrize(
        "spelling,truth",
        [
            ("2.0 - x", lambda x: 2.0 - x),
            ("x - 2.0", lambda x: x - 2.0),
            ("2.0 / x", lambda x: 2.0 / x),
            ("x / 2.0", lambda x: x / 2.0),
        ],
    )
    def test_jit_value(self, target, spelling, truth):
        source = f"def f(x):\n    return {spelling}\n"
        namespace: dict = {}
        exec(source, namespace)  # noqa: S102
        # ``@jit`` refuses a function whose source it cannot inspect rather than
        # silently running eager Python, so hand it the source explicitly.
        fn = tessera.jit(target=target, source=source)(namespace["f"])
        np.testing.assert_allclose(np.asarray(fn(X)), truth(X), rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("target", ["apple_gpu", "apple_cpu"])
    def test_left_side_scalar_as_an_intermediate(self, target):
        """Not just the whole return value — the lifted op feeding another op."""
        source = "def f(x):\n    y = 2.0 - x\n    return ops.relu(y)\n"
        namespace: dict = {}
        exec("from tessera import ops\n" + source, namespace)  # noqa: S102
        fn = tessera.jit(target=target, source=source)(namespace["f"])
        np.testing.assert_allclose(
            np.asarray(fn(X)), np.maximum(0.0, 2.0 - X), rtol=1e-6, atol=1e-6
        )


class TestTracerFailsClosed:
    """E2E-REAL-6 retires ``_OpExtractor`` in favour of the tracer. The tracer
    cannot inherit this bug, and these pin why: it never lifts a scalar into an
    unordered attribute in the first place. Both spellings are refused, so the
    successor frontend has no wrong-side path to fall into — if it ever grows
    infix-scalar support, it must add ordered operands, and these tests turn
    into the reminder to check the side then."""

    @pytest.mark.parametrize("fn", [lambda x: 2.0 - x, lambda x: x - 2.0,
                                    lambda x: 2.0 / x, lambda x: x / 2.0])
    def test_infix_scalar_binop_is_refused(self, fn):
        with pytest.raises(TypeError, match="unsupported operand type"):
            trace_mod.trace(fn, X)

    @pytest.mark.parametrize("fn", [lambda x: ops.sub(2.0, x), lambda x: ops.sub(x, 2.0),
                                    lambda x: ops.div(2.0, x), lambda x: ops.div(x, 2.0)])
    def test_positional_scalar_operand_is_refused(self, fn):
        with pytest.raises(trace_mod.TesseraTraceError, match="non-Tracer positional operand"):
            trace_mod.trace(fn, X)
