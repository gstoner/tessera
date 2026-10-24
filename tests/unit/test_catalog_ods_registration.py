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

#: catalogued Graph name -> "family: note" (see module docstring).
UNREGISTERED: dict[str, str] = {
    "tessera.adadelta": "undeclared: optimizer",
    "tessera.adagrad": "undeclared: optimizer",
    "tessera.amax": "reduce-kind: candidate: frontend spelling of tessera.reduce {kind = max} (same kind; consumers key on this name)",
    "tessera.amin": "reduce-kind: candidate: frontend spelling of tessera.reduce {kind = min} (same kind; consumers key on this name)",
    "tessera.arange": "undeclared: shape-index",
    "tessera.argsort": "undeclared: shape-index",
    "tessera.atan2": "undeclared: other",
    "tessera.bsmm": "undeclared: sparse",
    "tessera.cat": "undeclared: shape-index",
    "tessera.check_cauchy_riemann": "undeclared: complex",
    "tessera.chunk": "undeclared: shape-index",
    "tessera.clamp": "undeclared: other",
    "tessera.clifford_codiff": "undeclared: clifford",
    "tessera.clifford_conjugate": "undeclared: clifford",
    "tessera.clifford_exp": "undeclared: clifford",
    "tessera.clifford_ext_deriv": "undeclared: clifford",
    "tessera.clifford_geometric_product": "dotted-spelling: candidate: registered as tessera.clifford.geometric_product; equivalence unverified",
    "tessera.clifford_grade_involution": "undeclared: clifford",
    "tessera.clifford_grade_projection": "dotted-spelling: candidate: registered as tessera.clifford.grade_project; equivalence unverified",
    "tessera.clifford_hodge_star": "undeclared: clifford",
    "tessera.clifford_inner": "dotted-spelling: candidate: registered as tessera.clifford.inner_product; equivalence unverified",
    "tessera.clifford_left_contraction": "undeclared: clifford",
    "tessera.clifford_log": "undeclared: clifford",
    "tessera.clifford_norm": "dotted-spelling: candidate: registered as tessera.clifford.norm; equivalence unverified",
    "tessera.clifford_norm_squared": "undeclared: clifford",
    "tessera.clifford_reverse": "dotted-spelling: candidate: registered as tessera.clifford.reverse; equivalence unverified",
    "tessera.clifford_rotor_sandwich": "dotted-spelling: candidate: registered as tessera.clifford.rotor_sandwich; equivalence unverified",
    "tessera.clifford_vec_deriv": "undeclared: clifford",
    "tessera.clifford_wedge": "dotted-spelling: candidate: registered as tessera.clifford.outer_product; equivalence unverified",
    "tessera.clip": "undeclared: other",
    "tessera.complex_abs": "undeclared: complex",
    "tessera.complex_arg": "undeclared: complex",
    "tessera.complex_conjugate": "undeclared: complex",
    "tessera.complex_div": "undeclared: complex",
    "tessera.complex_exp": "undeclared: complex",
    "tessera.complex_log": "undeclared: complex",
    "tessera.complex_mul": "undeclared: complex",
    "tessera.complex_pow": "undeclared: complex",
    "tessera.complex_sqrt": "undeclared: complex",
    "tessera.conformal_energy_on_sphere": "undeclared: complex",
    "tessera.conformal_jacobian": "undeclared: complex",
    "tessera.conv3d_ndhwc": "undeclared: other",
    "tessera.count_nonzero": "undeclared: reduction",
    "tessera.cross_ratio": "undeclared: complex",
    "tessera.dbar": "undeclared: complex",
    "tessera.dequantize_fp6": "undeclared: quantize",
    "tessera.dequantize_nvfp4": "undeclared: quantize",
    "tessera.det": "undeclared: linalg",
    "tessera.dynamic_update_slice": "undeclared: shape-index",
    "tessera.dz": "undeclared: complex",
    "tessera.ebm_energy_quadratic": "dotted-spelling: candidate: registered as tessera.ebm.energy_quadratic; equivalence unverified",
    "tessera.ebm_inner_step": "dotted-spelling: candidate: registered as tessera.ebm.inner_step; equivalence unverified",
    "tessera.ebm_refinement": "dotted-spelling: candidate: registered as tessera.ebm.refinement; equivalence unverified",
    "tessera.ebm_self_verify": "dotted-spelling: candidate: registered as tessera.ebm.self_verify; equivalence unverified",
    "tessera.eigh": "undeclared: linalg",
    "tessera.einsum": "undeclared: other",
    "tessera.entmax15": "undeclared: other",
    "tessera.factorized_matmul": "undeclared: other",
    "tessera.flip": "undeclared: shape-index",
    "tessera.game_boltzmann_value": "undeclared: game",
    "tessera.game_coalition_excess": "undeclared: game",
    "tessera.game_coalition_marginal": "undeclared: game",
    "tessera.game_mex": "undeclared: game",
    "tessera.game_semivalue": "undeclared: game",
    "tessera.group_norm": "undeclared: other",
    "tessera.gumbel_softmax": "renamed: candidate: registered as tessera.arch.gumbel_softmax; equivalence unverified",
    "tessera.index_select": "undeclared: shape-index",
    "tessera.index_update": "undeclared: shape-index",
    "tessera.instance_norm": "undeclared: other",
    "tessera.inv": "undeclared: linalg",
    "tessera.is_concyclic": "undeclared: complex",
    "tessera.kron": "undeclared: linalg",
    "tessera.kv_cache.read": "renamed: candidate: registered as tessera.paged_kv_read; equivalence unverified",
    "tessera.laplacian_2d": "undeclared: other",
    "tessera.linear_general": "undeclared: other",
    "tessera.logdet": "undeclared: linalg",
    "tessera.logsumexp": "undeclared: reduction",
    "tessera.loss.asymmetric_bce": "undeclared: loss",
    "tessera.loss.binary_brier": "undeclared: loss",
    "tessera.loss.brier": "undeclared: loss",
    "tessera.loss.contrastive_divergence": "undeclared: loss",
    "tessera.loss.ddpm_noise_pred": "undeclared: loss",
    "tessera.loss.denoising_score_matching": "undeclared: loss",
    "tessera.loss.implicit_score_matching": "undeclared: loss",
    "tessera.loss.load_balance_loss": "undeclared: loss",
    "tessera.loss.ordinal_logistic": "undeclared: loss",
    "tessera.loss.persistent_cd": "undeclared: loss",
    "tessera.loss.score_matching": "undeclared: loss",
    "tessera.loss.soft_binned_ece": "undeclared: loss",
    "tessera.loss.vlb": "undeclared: loss",
    "tessera.loss.z_loss": "undeclared: loss",
    "tessera.masked_categorical": "undeclared: other",
    "tessera.matrix_power": "undeclared: linalg",
    "tessera.max": "reduce-kind: candidate: frontend spelling of tessera.reduce {kind = max} (same kind; consumers key on this name)",
    "tessera.mean": "reduce-kind: candidate: frontend spelling of tessera.reduce {kind = mean} (same kind; consumers key on this name)",
    "tessera.memory_index_score": "undeclared: memory",
    "tessera.memory_index_select": "undeclared: memory",
    "tessera.memory_index_select_ste": "undeclared: memory",
    "tessera.min": "reduce-kind: candidate: frontend spelling of tessera.reduce {kind = min} (same kind; consumers key on this name)",
    "tessera.mobius": "undeclared: complex",
    "tessera.mobius_from_three_points": "undeclared: complex",
    "tessera.moe": "undeclared: other",
    "tessera.nonzero": "undeclared: reduction",
    "tessera.norm": "undeclared: reduction",
    "tessera.pack": "undeclared: shape-index",
    "tessera.pad": "undeclared: shape-index",
    "tessera.perturbed_argmax": "undeclared: other",
    "tessera.prod": "undeclared: reduction",
    "tessera.qkv_projection": "undeclared: other",
    "tessera.quantize_fp6": "undeclared: quantize",
    "tessera.quantize_nvfp4": "undeclared: quantize",
    "tessera.quantized_matmul": "undeclared: quantize",
    "tessera.rearrange": "undeclared: shape-index",
    "tessera.repeat": "undeclared: shape-index",
    "tessera.rmsprop": "undeclared: optimizer",
    "tessera.roll": "undeclared: shape-index",
    "tessera.scatter": "undeclared: shape-index",
    "tessera.scatter_add": "undeclared: shape-index",
    "tessera.scatter_reduce": "undeclared: shape-index",
    "tessera.sddmm": "undeclared: sparse",
    "tessera.segment_reduce": "undeclared: sparse",
    "tessera.shampoo": "undeclared: optimizer",
    "tessera.sigmoid_safe": "undeclared: other",
    "tessera.soft_top_k": "undeclared: shape-index",
    "tessera.softcap": "undeclared: other",
    "tessera.solve": "undeclared: linalg",
    "tessera.sort": "undeclared: shape-index",
    "tessera.sparsemax": "undeclared: other",
    "tessera.split": "undeclared: shape-index",
    "tessera.spmm_coo": "undeclared: sparse",
    "tessera.spmm_csr": "undeclared: sparse",
    "tessera.stack": "undeclared: shape-index",
    "tessera.std": "undeclared: reduction",
    "tessera.stereographic": "undeclared: complex",
    "tessera.take": "undeclared: shape-index",
    "tessera.tile": "undeclared: shape-index",
    "tessera.tile_view": "undeclared: shape-index",
    "tessera.top_k": "undeclared: shape-index",
    "tessera.trace": "undeclared: linalg",
    "tessera.unpack": "undeclared: shape-index",
    "tessera.var": "undeclared: reduction",
    "tessera.vec": "undeclared: linalg",
    "tessera.weight_norm": "undeclared: other",
}
UNREGISTERED_CEILING = 143

_FAMILIES = ("reduce-kind:", "dotted-spelling:", "renamed:", "undeclared:")


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
    untracked = sorted(unregistered - set(UNREGISTERED))
    assert not untracked, (
        f"catalogued Graph ops tessera-opt does not register: {untracked}. "
        f"Declare each in TesseraOps.td with a verifier and a consumer "
        f"(Decision #29), or keep it out of the Graph IR front end.")


def test_tracked_gaps_are_still_gaps(unregistered):
    closed = sorted(set(UNREGISTERED) - unregistered)
    assert not closed, (
        f"now registered, so delete from UNREGISTERED and lower "
        f"UNREGISTERED_CEILING: {closed}")


def test_gap_list_only_shrinks_and_names_its_family():
    stale = sorted(set(UNREGISTERED) - set(_catalog_names()))
    assert not stale, f"no longer catalogued: {stale}"
    assert len(UNREGISTERED) <= UNREGISTERED_CEILING
    unlabelled = sorted(n for n, why in UNREGISTERED.items() if not why.startswith(_FAMILIES))
    assert not unlabelled, f"every gap names its family: {unlabelled}"


def test_probe_tells_registered_from_unregistered(unregistered):
    assert "tessera.relu" not in unregistered
    assert "tessera.matmul" not in unregistered
