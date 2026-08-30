"""Host-free contracts on the emitted dense-Krylov CUDA source.

There is no CUDA on the machines that run this suite, so these assert on the
GENERATED TEXT. They pin the memory-access shape of `tsr_matvec`, which is the
O(n^2) term of every CG/GMRES iteration; nothing here is a device measurement,
and no wall-clock claim can be made from it (code review 2026-08-29, P3).
"""

from __future__ import annotations

import re

from tessera.compiler.emit import nvidia_solver_krylov as kv


def _device_fn(source: str, name: str) -> str:
    start = source.index(f"void {name}(")
    open_brace = source.index("{", start)
    depth = 0
    for i in range(open_brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace:i + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def test_matvec_columns_are_walked_by_lane_not_by_one_thread():
    """One thread per row makes adjacent lanes read `a[row*n + col]` for
    consecutive ROWS — addresses n*sizeof(T) apart, so one memory transaction
    per lane instead of one per 32. A warp per row with lane-strided columns
    makes the 32 lanes of a load touch 32 consecutive elements of one row.

    Reasoned from the emitted access pattern; NOT device-measured. The
    transaction-count argument is sound, the wall-clock effect is not available
    on a host without CUDA.
    """
    body = _device_fn(kv._source(), "tsr_matvec")
    assert "for (int col = lane; col < n; col += 32)" in body, (
        "column loop is not lane-strided — the loads are uncoalesced")
    assert "for (int col = 0; col < n; ++col)" not in body


def test_matvec_reduction_is_a_fixed_full_mask_shuffle_tree():
    """The per-row sum must stay reproducible for a fixed launch geometry, the
    same contract `tsr_grid_sum` states. A fixed 5-step butterfly over a fixed
    32-lane warp is; a size-dependent or atomic reduction is not."""
    body = _device_fn(kv._source(), "tsr_matvec")
    assert "for (int off = 16; off; off >>= 1)" in body
    assert "__shfl_down_sync(0xffffffffu, sum, off)" in body
    assert "atomicAdd" not in body, "an atomic makes the row sum non-reproducible"
    assert "if (lane == 0) y[row] = sum;" in body


def test_matvec_row_index_is_warp_uniform():
    """The full-mask shuffles require all 32 lanes of the warp to reach them,
    so `row` must be derived from a warp id, not a thread id — otherwise lanes
    of one warp take different trip counts and the shuffle is undefined."""
    body = _device_fn(kv._source(), "tsr_matvec")
    assert re.search(r"long warp = .*blockDim\.x \+ threadIdx\.x\) >> 5", body)
    assert re.search(r"long warps = .*gridDim\.x \* blockDim\.x\) >> 5", body)
    assert "for (long row = warp; row < n; row += warps)" in body
    # The precondition the shuffles rely on: every launch site uses 256 threads.
    assert "const int threads = 256;" in kv._source()


def test_matvec_still_widens_low_precision_storage_before_multiplying():
    """The module contract: f16/bf16 operands convert to f32 before the
    multiply so the Krylov convergence claim is unambiguous."""
    body = _device_fn(kv._source(), "tsr_matvec")
    assert "fmaf(tsr_load(arow, col), x[col], sum)" in body
