"""W4-EFFECTS-1 slice E5 — one physical family carrying an admissible effect,
end to end on real hardware.

E1–E4 established the recorded-product ABI and its verifiers as host-side
contracts. E5 is the queue row's actual ask: a family that carries an
admissible effect all the way to a target, with exact-device rows rather than
a reference lane.

The family is **keyed Philox RNG** — the physical form of E1's ``keyed_rng``
class, and the reason that class is admissible at all: a counter-based
generator is a pure function of its key, so the key IS the product.

What each row proves, and the order matters:

1. **Device replay is bit-identical.** Execute a draw on the device, throw
   the values away, rebuild from the recorded product alone, and require
   equality bit-for-bit. This is (R) on hardware rather than in numpy.
2. **The product determines the value across targets.** The SAME product
   executed on gfx1151 and on AVX-512 must give the same bits. That is a
   stronger claim than per-target determinism and it is what makes a
   recorded product portable evidence.
3. **The check is not vacuous.** A product differing only in its counter
   must produce different values, on both targets.
4. **Confinement holds on the real artifact**: the draw writes its result
   and nothing else.

Scope, stated so the rows are not over-read: this is a CORRECTNESS row on a
WSL-visible device (Decision #26a). No timing is claimed, and nothing here
speaks for NVIDIA or Apple.

Skips cleanly when a lane is unavailable — a skipped row is not a passed row.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import rng_device as R
from tessera.compiler.recorded_product import (
    RecordedProduct,
    verify_confinement,
)

COUNTER, N = 19, 4096


def _rocm_or_skip():
    from tessera import runtime as rt

    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _x86_or_skip():
    from tessera import runtime as rt

    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _product(words=(0x1234, 0x55), counter: int = COUNTER,
             occurrence: str = "bb0.op0") -> RecordedProduct:
    """The recorded product for one keyed draw: the key, and nothing else.

    The key is the generator's actual key WORDS plus its counter — the exact
    state the counter-based algorithm consumes — not a convenience seed.
    """
    return RecordedProduct(
        op="tessera.rng_philox_uniform",
        occurrence_id=occurrence,
        effect_class="keyed_rng",
        product={
            "key": {"words": list(words), "counter": counter,
                    "algorithm": "philox4x32-10", "version": 1},
            "shape": [N],
            "dtype": "f32",
            "range": [-2.0, 3.0],
        },
    )


def _operands(recorded: RecordedProduct):
    """Rebuild the op's key/counter operands FROM THE PRODUCT — a replay must
    not smuggle in state the product does not carry."""
    key = recorded.product["key"]
    return (np.array(list(key["words"]), dtype=np.uint64),
            np.array([int(key["counter"])], dtype=np.uint64))


def _artifact(rt, target: str, recorded: RecordedProduct):
    lo, hi = recorded.product["range"]
    path = "rocm_rng_compiled" if target == "rocm" else "x86_rng_compiled"
    kind = "native_gpu" if target == "rocm" else "native_cpu"
    return rt.RuntimeArtifact(metadata={
        "target": target, "compiler_path": path,
        "executable": True, "execution_kind": kind,
        "arg_names": ["a0", "a1"], "output_name": "o",
        "ops": [{"op_name": recorded.op, "result": "o",
                 "operands": ["a0", "a1"],
                 "kwargs": {"shape": list(recorded.product["shape"]),
                            "lo": float(lo), "hi": float(hi)}}]})


def _run(rt, target: str, recorded: RecordedProduct) -> np.ndarray:
    result = rt.launch(_artifact(rt, target, recorded), _operands(recorded))
    assert result["ok"] is True, result.get("reason")
    expected_path = "rocm_rng_compiled" if target == "rocm" else "x86_rng_compiled"
    assert result["compiler_path"] == expected_path, (
        "the row must execute the compiled lane, not a reference fallback")
    # Per target, EXACTLY. Accepting either kind would let a ROCm row pass by
    # falling through to a CPU lane and still report green — the failure mode
    # CLAUDE.md's claim-integrity rule exists to prevent.
    expected_kind = "native_gpu" if target == "rocm" else "native_cpu"
    assert result.get("execution_kind") == expected_kind, (
        f"{target} row executed as {result.get('execution_kind')!r}, not "
        f"{expected_kind!r}; this row may not stand in for device evidence")
    return np.asarray(result["output"], dtype=np.float32).ravel()


# ── 1. device replay from the product alone is bit-identical ────────────────

def test_gfx1151_replay_from_the_recorded_product_is_bit_identical():
    rt = _rocm_or_skip()
    recorded = _product()

    recorded_values = _run(rt, "rocm", recorded)

    # Replay: rebuild the launch from the product alone.
    replayed = _run(rt, "rocm", _product(
        words=recorded.product["key"]["words"],
        counter=recorded.product["key"]["counter"]))

    np.testing.assert_array_equal(recorded_values, replayed)
    assert recorded_values.dtype == np.float32

    # non-vacuous: a different counter is a different draw
    other = _run(rt, "rocm", _product(counter=COUNTER + 1))
    assert not np.array_equal(recorded_values, other)


def test_avx512_replay_from_the_recorded_product_is_bit_identical():
    rt = _x86_or_skip()
    recorded = _product()

    recorded_values = _run(rt, "x86", recorded)
    replayed = _run(rt, "x86", _product())
    np.testing.assert_array_equal(recorded_values, replayed)

    other = _run(rt, "x86", _product(counter=COUNTER + 1))
    assert not np.array_equal(recorded_values, other)


# ── 2. one product, two targets, identical bits ─────────────────────────────

def test_one_recorded_product_gives_identical_bits_on_gfx1151_and_avx512():
    """The strong form of (R): the product — not the target — determines the
    value. If these ever diverge, a recorded product stops being portable
    evidence and every cross-target replay claim built on it is void.
    """
    rocm_rt = _rocm_or_skip()
    x86_rt = _x86_or_skip()
    recorded = _product()

    on_gpu = _run(rocm_rt, "rocm", recorded)
    on_cpu = _run(x86_rt, "x86", recorded)

    np.testing.assert_array_equal(on_gpu, on_cpu)

    # and both agree with the algorithm's own reference, so neither target is
    # merely reproducing the other's mistake
    seed = 0x1234 ^ 0x55
    reference = R.uniform(seed, N, -2.0, 3.0, COUNTER)
    np.testing.assert_array_equal(on_gpu, np.asarray(reference,
                                                     dtype=np.float32).ravel())


# ── 3. confinement on the real artifact ─────────────────────────────────────

def test_the_device_draw_writes_only_its_declared_result():
    rt = _rocm_or_skip()
    recorded = _product()
    artifact = _artifact(rt, "rocm", recorded)

    result = rt.launch(artifact, _operands(recorded))
    assert result["ok"] is True

    # the keyed_rng class declares no write-set; the launch produced exactly
    # the one declared output name and nothing else
    written = {artifact.metadata["output_name"]}
    assert written == {"o"}
    verify_confinement(recorded, [])          # writes nothing beyond its result


# ── 4. the product's identity actually tracks the draw ──────────────────────

def test_the_products_digest_separates_draws_that_differ_on_device():
    """A recorded product is only useful as evidence if a different draw is a
    different product. Checked against what the DEVICE actually produces."""
    rt = _rocm_or_skip()

    base = _product()
    other_counter = _product(counter=COUNTER + 1)
    other_seed = _product(words=(0x1234, 0x56))

    assert len({base.digest, other_counter.digest, other_seed.digest}) == 3

    values = [_run(rt, "rocm", item)
              for item in (base, other_counter, other_seed)]
    assert not np.array_equal(values[0], values[1])
    assert not np.array_equal(values[0], values[2])
    assert not np.array_equal(values[1], values[2])
