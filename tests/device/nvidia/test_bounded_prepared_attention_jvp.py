"""Exact SM120 bounded host preparation through the native capacity owner."""
from pathlib import Path
import os

import ml_dtypes
import numpy as np
import pytest
import tessera as ts

from tessera import runtime as rt
from tessera.compiler import native_attention_jvp_runtime as runtime
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_public_typed_attention_jvp import reference
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.compiler_route]


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_bounded_prepared_host_reuses_capacity_owner(storage, saved, bias, subset, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261017)
    def inputs(sq, sk):
        values = {name: rng.normal(0, .4, shape).astype(dtype) for name, shape in {
            "q": (1, 2, sq, 16), "k": (1, 1, sk, 16), "v": (1, 1, sk, 16),
        }.items()}
        if bias:
            values["bias"] = rng.normal(0, .1, (1, 2, 1, 1)).astype(np.float32)
        return values
    traced = inputs(1, 1)
    active = (("bias", "q") if bias else ("v",)) if subset else tuple(traced)
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=active)(
        function(saved, bias, True))
    program = fn.compile_native_attention_jvp(
        **traced, compiler=Path(os.environ["TESSERA_OPT"]),
        llvm_bin=Path(os.environ["TESSERA_LLVM_BIN"]), sequence_bounds=(7, 9))
    names = [f"primal_{i}" for i in range(len(program.input_indices))]
    names += [f"tangent_{program.input_indices[i]}" for i in program.active]
    metadata = dict(program_json=program.to_json(), program_digest=program.program_digest,
                    arg_names=names)
    frontend = ("bias", "v", "q", "k") if bias else ("q", "k", "v")
    identity = (program.program_digest, program.tangent.binding_digest,
                program.pair.forward.image.image_digest)
    def forbidden(*args, **kwargs):
        pytest.fail("bounded prepared replay attempted compilation")
    retained = []
    runtime.clear_prepared()
    try:
        handle = None
        with monkeypatch.context() as guard:
            guard.setattr("subprocess.run", forbidden)
            for sq, sk in ((1, 1), (3, 4), (7, 9), (7, 1), (1, 9), (3, 4)):
                values = inputs(sq, sk)
                seeds = {n: rng.normal(0, .1, x.shape).astype(x.dtype) if n in active
                         else np.zeros_like(x) for n, x in values.items()}
                result = runtime.execute(metadata, tuple(values[n] for n in frontend) +
                                         tuple(seeds[n] for n in active))
                expected = reference(*(values[n] for n in ("q", "k", "v")),
                    *(seeds[n] for n in ("q", "k", "v")), causal=True,
                    bias=values.get("bias"), dbias=seeds.get("bias"))
                for actuals, targets in zip(result, expected, strict=True):
                    actuals = actuals if saved else (actuals,)
                    targets = targets if saved else targets[:1]
                    for index, (actual, target) in enumerate(zip(actuals, targets, strict=True)):
                        assert actual.dtype == (np.dtype("float32") if index else dtype)
                        compare(actual, target)
                owner = runtime.prepared(metadata)
                if handle is None:
                    handle = owner.handle
                assert owner.handle == handle
                assert identity == (owner.program.program_digest,
                    owner.program.tangent.binding_digest,
                    owner.program.pair.forward.image.image_digest)
                snapshots = tuple(tuple(x.copy() for x in group) if saved else group.copy()
                                  for group in result)
                retained.append((result, snapshots))
                for prior, snapshots in retained:
                    for actual, snapshot in zip(prior, snapshots, strict=True):
                        for a, b in zip(actual if saved else (actual,),
                                        snapshot if saved else (snapshot,), strict=True):
                            np.testing.assert_array_equal(a, b)
            for sq, sk in ((8, 4), (3, 10), (0, 4), (3, 0)):
                values = inputs(sq, sk)
                with pytest.raises(ValueError, match="sequence|capacity|envelope"):
                    runtime.execute(metadata, tuple(values[n] for n in frontend) +
                                    tuple(np.zeros_like(values[n]) for n in active))
            # Capacity refusal must leave the same owner usable.
            values = inputs(3, 4)
            runtime.execute(metadata, tuple(values[n] for n in frontend) +
                            tuple(np.zeros_like(values[n]) for n in active))
            assert runtime.prepared(metadata).handle == handle
    finally:
        runtime.clear_prepared()
