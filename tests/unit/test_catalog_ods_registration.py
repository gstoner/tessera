"""Every catalogued Graph op must be an op ``tessera-opt`` registers.

``op_catalog`` is what the Python front end accepts and emits as Graph IR;
``TesseraOps.td`` is what MLIR can parse and verify. A catalogued name with no
ODS registration is Graph IR the parser rejects ("unregistered operation ...
found in dialect ('tessera') that does not allow unknown operations"), so it
can never cross the strict validation gate (gstoner/tessera#1005) or reach a
parser-bound lowering. Measured 2026-10-24 by probing the Mac build's
``tessera-opt``: 143 of 352 catalogued names were unregistered.

**Ratchet.** ``UNREGISTERED`` names each gap with its family:

* ``reduce-kind`` / ``dotted-spelling`` / ``renamed`` -- a registered op with a
  matching name or kind exists (``tessera.reduce``, ``tessera.clifford.*``,
  ``tessera.ebm.*``, ``tessera.paged_kv_read``, ...). These are *candidates*:
  equivalence is not verified here, and Python consumers across every backend
  key on the catalogued spelling, so each is closed by its own change with
  device evidence for the lanes it touches.
* ``undeclared: <family>`` -- no ODS op exists. Closing one means declaring
  it with a verifier *and* a consumer (Decision #29), or moving the op out of
  the Graph IR front end.

An exempt name must still be unregistered -- once it registers, this gate
fails until the entry is deleted -- and the list may not grow past
``UNREGISTERED_CEILING``. A new catalogued op lands with its ODS declaration.

``tessera-opt`` is resolved the way the driver resolves it; a host without a
runnable build skips, since it cannot evaluate this contract.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from tessera.compiler import driver, op_catalog

#: Catalogued Graph name WITHOUT the ``tessera.`` prefix -> "family: note".
#: Bare on purpose: ``test_coverage_audit.py`` counts a literal
#: ``"tessera.<op>"`` string in a test file as a direct test reference, so the
#: prefixed form would credit these explicitly UNREGISTERED ops with direct
#: test evidence they do not have (see ``test_apple_gpu_no_lane_residual.py``).
#: Do not re-add the prefix; ``_t()`` reassembles it at runtime.
UNREGISTERED: dict[str, str] = {
    "adadelta": "undeclared: optimizer",
    "adagrad": "undeclared: optimizer",
    "amax": "reduce-kind: candidate: frontend spelling of reduce {kind = max} (same kind; consumers key on this name)",
    "amin": "reduce-kind: candidate: frontend spelling of reduce {kind = min} (same kind; consumers key on this name)",
    "arange": "undeclared: shape-index",
    "argsort": "undeclared: shape-index",
    "atan2": "undeclared: other",
    "bsmm": "undeclared: sparse",
    "cat": "undeclared: shape-index",
    "check_cauchy_riemann": "undeclared: complex",
    "chunk": "undeclared: shape-index",
    "clamp": "undeclared: other",
    "clifford_codiff": "undeclared: clifford",
    "clifford_conjugate": "undeclared: clifford",
    "clifford_exp": "undeclared: clifford",
    "clifford_ext_deriv": "undeclared: clifford",
    "clifford_geometric_product": "dotted-spelling: candidate: registered as clifford.geometric_product; equivalence unverified",
    "clifford_grade_involution": "undeclared: clifford",
    "clifford_grade_projection": "dotted-spelling: candidate: registered as clifford.grade_project; equivalence unverified",
    "clifford_hodge_star": "undeclared: clifford",
    "clifford_inner": "dotted-spelling: candidate: registered as clifford.inner_product; equivalence unverified",
    "clifford_left_contraction": "undeclared: clifford",
    "clifford_log": "undeclared: clifford",
    "clifford_norm": "dotted-spelling: candidate: registered as clifford.norm; equivalence unverified",
    "clifford_norm_squared": "undeclared: clifford",
    "clifford_reverse": "dotted-spelling: candidate: registered as clifford.reverse; equivalence unverified",
    "clifford_rotor_sandwich": "dotted-spelling: candidate: registered as clifford.rotor_sandwich; equivalence unverified",
    "clifford_vec_deriv": "undeclared: clifford",
    "clifford_wedge": "dotted-spelling: candidate: registered as clifford.outer_product; equivalence unverified",
    "clip": "undeclared: other",
    "complex_abs": "undeclared: complex",
    "complex_arg": "undeclared: complex",
    "complex_conjugate": "undeclared: complex",
    "complex_div": "undeclared: complex",
    "complex_exp": "undeclared: complex",
    "complex_log": "undeclared: complex",
    "complex_mul": "undeclared: complex",
    "complex_pow": "undeclared: complex",
    "complex_sqrt": "undeclared: complex",
    "conformal_energy_on_sphere": "undeclared: complex",
    "conformal_jacobian": "undeclared: complex",
    "conv3d_ndhwc": "undeclared: other",
    "count_nonzero": "undeclared: reduction",
    "cross_ratio": "undeclared: complex",
    "dbar": "undeclared: complex",
    "dequantize_fp6": "undeclared: quantize",
    "dequantize_nvfp4": "undeclared: quantize",
    "det": "undeclared: linalg",
    "dynamic_update_slice": "undeclared: shape-index",
    "dz": "undeclared: complex",
    "ebm_energy_quadratic": "dotted-spelling: candidate: registered as ebm.energy_quadratic; equivalence unverified",
    "ebm_inner_step": "dotted-spelling: candidate: registered as ebm.inner_step; equivalence unverified",
    "ebm_refinement": "dotted-spelling: candidate: registered as ebm.refinement; equivalence unverified",
    "ebm_self_verify": "dotted-spelling: candidate: registered as ebm.self_verify; equivalence unverified",
    "eigh": "undeclared: linalg",
    "einsum": "undeclared: other",
    "entmax15": "undeclared: other",
    "factorized_matmul": "undeclared: other",
    "flip": "undeclared: shape-index",
    "game_boltzmann_value": "undeclared: game",
    "game_coalition_excess": "undeclared: game",
    "game_coalition_marginal": "undeclared: game",
    "game_mex": "undeclared: game",
    "game_semivalue": "undeclared: game",
    "group_norm": "undeclared: other",
    "gumbel_softmax": "renamed: candidate: registered as arch.gumbel_softmax; equivalence unverified",
    "index_select": "undeclared: shape-index",
    "index_update": "undeclared: shape-index",
    "instance_norm": "undeclared: other",
    "inv": "undeclared: linalg",
    "is_concyclic": "undeclared: complex",
    "kron": "undeclared: linalg",
    "kv_cache.read": "renamed: candidate: registered as paged_kv_read; equivalence unverified",
    "laplacian_2d": "undeclared: other",
    "linear_general": "undeclared: other",
    "logdet": "undeclared: linalg",
    "logsumexp": "undeclared: reduction",
    "loss.asymmetric_bce": "undeclared: loss",
    "loss.binary_brier": "undeclared: loss",
    "loss.brier": "undeclared: loss",
    "loss.contrastive_divergence": "undeclared: loss",
    "loss.ddpm_noise_pred": "undeclared: loss",
    "loss.denoising_score_matching": "undeclared: loss",
    "loss.implicit_score_matching": "undeclared: loss",
    "loss.load_balance_loss": "undeclared: loss",
    "loss.ordinal_logistic": "undeclared: loss",
    "loss.persistent_cd": "undeclared: loss",
    "loss.score_matching": "undeclared: loss",
    "loss.soft_binned_ece": "undeclared: loss",
    "loss.vlb": "undeclared: loss",
    "loss.z_loss": "undeclared: loss",
    "masked_categorical": "undeclared: other",
    "matrix_power": "undeclared: linalg",
    "max": "reduce-kind: candidate: frontend spelling of reduce {kind = max} (same kind; consumers key on this name)",
    "mean": "reduce-kind: candidate: frontend spelling of reduce {kind = mean} (same kind; consumers key on this name)",
    "memory_index_score": "undeclared: memory",
    "memory_index_select": "undeclared: memory",
    "memory_index_select_ste": "undeclared: memory",
    "min": "reduce-kind: candidate: frontend spelling of reduce {kind = min} (same kind; consumers key on this name)",
    "mobius": "undeclared: complex",
    "mobius_from_three_points": "undeclared: complex",
    "moe": "undeclared: other",
    "nonzero": "undeclared: reduction",
    "norm": "undeclared: reduction",
    "pack": "undeclared: shape-index",
    "pad": "undeclared: shape-index",
    "perturbed_argmax": "undeclared: other",
    "prod": "undeclared: reduction",
    "qkv_projection": "undeclared: other",
    "quantize_fp6": "undeclared: quantize",
    "quantize_nvfp4": "undeclared: quantize",
    "quantized_matmul": "undeclared: quantize",
    "rearrange": "undeclared: shape-index",
    "repeat": "undeclared: shape-index",
    "rmsprop": "undeclared: optimizer",
    "roll": "undeclared: shape-index",
    "scatter": "undeclared: shape-index",
    "scatter_add": "undeclared: shape-index",
    "scatter_reduce": "undeclared: shape-index",
    "sddmm": "undeclared: sparse",
    "segment_reduce": "undeclared: sparse",
    "shampoo": "undeclared: optimizer",
    "sigmoid_safe": "undeclared: other",
    "soft_top_k": "undeclared: shape-index",
    "softcap": "undeclared: other",
    "solve": "undeclared: linalg",
    "sort": "undeclared: shape-index",
    "sparsemax": "undeclared: other",
    "split": "undeclared: shape-index",
    "spmm_coo": "undeclared: sparse",
    "spmm_csr": "undeclared: sparse",
    "stack": "undeclared: shape-index",
    "std": "undeclared: reduction",
    "stereographic": "undeclared: complex",
    "take": "undeclared: shape-index",
    "tile": "undeclared: shape-index",
    "tile_view": "undeclared: shape-index",
    "top_k": "undeclared: shape-index",
    "trace": "undeclared: linalg",
    "unpack": "undeclared: shape-index",
    "var": "undeclared: reduction",
    "vec": "undeclared: linalg",
    "weight_norm": "undeclared: other",
}
UNREGISTERED_CEILING = 143

_FAMILIES = ("reduce-kind:", "dotted-spelling:", "renamed:", "undeclared:")
_PREFIX = "tessera."


def _t(bare: str) -> str:
    """Dotted Graph op name from a bare leaf (see ``UNREGISTERED``)."""
    return _PREFIX + bare


def _tracked() -> set[str]:
    return {_t(bare) for bare in UNREGISTERED}


def _catalog_names() -> list[str]:
    return sorted({spec.graph_name for spec in op_catalog.OP_SPECS.values()})


def _unregistered(tool: str, names: list[str]) -> set[str]:
    """Names tessera-opt rejects as unregistered, from one split-input run."""
    chunks = [
        f'func.func @probe() {{\n  "{name}"() : () -> ()\n  return\n}}\n'
        for name in names
    ]
    proc = subprocess.run(
        [tool, "-", "--split-input-file", "-o", "/dev/null"],
        input="// -----\n".join(chunks), capture_output=True, text=True,
        timeout=120, check=False)
    found = set(re.findall(
        r"unregistered operation '([^']+)' found in dialect", proc.stderr))
    if not found and proc.returncode != 0 and "unregistered" not in proc.stderr:
        # Every probe is a verifier failure for a registered op; anything else
        # (a crash, a usage error) means the probe itself did not run.
        assert "error:" in proc.stderr, proc.stderr[:2000]
    return found


@pytest.fixture(scope="module")
def unregistered() -> set[str]:
    tool = driver._resolve_tessera_opt()
    if tool is None:
        pytest.skip("no runnable tessera-opt; this host cannot evaluate catalog/ODS registration")
    return _unregistered(tool, _catalog_names())


def test_every_catalogued_graph_op_is_registered_or_tracked(unregistered):
    untracked = sorted(unregistered - _tracked())
    assert not untracked, (
        f"catalogued Graph ops tessera-opt does not register: {untracked}. "
        f"Declare each in TesseraOps.td with a verifier and a consumer "
        f"(Decision #29), or keep it out of the Graph IR front end.")


def test_tracked_gaps_are_still_gaps(unregistered):
    closed = sorted(_tracked() - unregistered)
    assert not closed, (
        f"now registered, so delete from UNREGISTERED and lower "
        f"UNREGISTERED_CEILING: {closed}")


def test_gap_list_only_shrinks_and_names_its_family():
    stale = sorted(_tracked() - set(_catalog_names()))
    assert not stale, f"no longer catalogued: {stale}"
    assert len(UNREGISTERED) <= UNREGISTERED_CEILING
    unlabelled = sorted(n for n, why in UNREGISTERED.items() if not why.startswith(_FAMILIES))
    assert not unlabelled, f"every gap names its family: {unlabelled}"


def test_probe_tells_registered_from_unregistered(unregistered):
    assert _t("relu") not in unregistered
    assert _t("matmul") not in unregistered
    assert _t("mean") in unregistered
