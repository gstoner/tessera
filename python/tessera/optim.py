"""Functional optimizer, schedule, and gradient-transform references.

S10 keeps training-step semantics inside Tessera rather than relying on
PyTorch, Optax, or Flax. These functions operate over nested Python containers
of numpy arrays: dicts, lists, tuples, and leaves that coerce with
``np.asarray``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import warnings

import numpy as np


Tree = Any


_DTYPE_ALIASES = {
    "f64": "fp64",
    "float64": "fp64",
    "f32": "fp32",
    "float32": "fp32",
    "f16": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
}

_NUMPY_DTYPES = {
    "fp64": np.float64,
    "fp32": np.float32,
    "tf32": np.float32,
    "bf16": np.float32,  # numpy reference stores bf16 as fp32.
    "fp16": np.float16,
    "fp8_e4m3": np.float32,
    "fp8_e5m2": np.float32,
    "fp4_e2m1": np.float32,
    "nvfp4": np.float32,
}


def _asarray(x: Any) -> np.ndarray:
    if hasattr(x, "_data"):
        x = x._data
    if hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


def _normalize_dtype(dtype: str | None, *, default: str = "fp32") -> str:
    if dtype is None:
        return default
    normalized = _DTYPE_ALIASES.get(str(dtype), str(dtype))
    if normalized not in _NUMPY_DTYPES:
        raise ValueError(f"Unsupported optimizer dtype {dtype!r}")
    return normalized


def _np_dtype(dtype: str | None, *, default: str = "fp32"):
    return _NUMPY_DTYPES[_normalize_dtype(dtype, default=default)]


def _compute_array(x: Any, compute_dtype: str | None) -> np.ndarray:
    arr = _asarray(x)
    # Treat the default as "at least fp32" for reference math: fp16/bf16-style
    # storage promotes to fp32, while fp64 test/reference inputs keep fp64.
    if _normalize_dtype(compute_dtype) == "fp32" and arr.dtype == np.float64:
        return arr.astype(np.float64, copy=False)
    return arr.astype(_np_dtype(compute_dtype), copy=False)


def _state_array(x: Any, state_dtype: str | None) -> np.ndarray:
    return _asarray(x).astype(_np_dtype(state_dtype), copy=False)


def _cast_like_param(x: Any, param: Any, cast_updates_to_param_dtype: bool) -> np.ndarray:
    arr = np.asarray(x)
    if not cast_updates_to_param_dtype:
        return arr
    return arr.astype(_asarray(param).dtype, copy=False)


def _master_tree(params: Tree, state: dict[str, Any] | None, master_dtype: str | None) -> Tree:
    if master_dtype is None:
        return params
    if state is not None and "master_params" in state:
        return state["master_params"]
    return tree_map(lambda p: _asarray(p).astype(_np_dtype(master_dtype), copy=True), params)


def _attach_master_state(state: dict[str, Any], master_params: Tree, master_dtype: str | None) -> dict[str, Any]:
    if master_dtype is None:
        return state
    out = dict(state)
    out["master_params"] = master_params
    out["master_dtype"] = _normalize_dtype(master_dtype)
    return out


def attach_master_params(
    state: dict[str, Any],
    params: Tree,
    *,
    master_dtype: str,
) -> dict[str, Any]:
    """Turn state from an fp32 run into state a mixed-precision run accepts.

    Enabling `master_dtype` mid-run is legitimate, but it cannot happen
    silently: the optimizer needs a master copy of the weights, and the only
    thing available at the moment of the switch is the low-precision
    parameter storage. Upcasting it is exactly right ONCE, there — nothing
    has accumulated yet — and exactly wrong on every step afterwards, where
    it would throw away the master weights the run has been maintaining.
    This function is the one place that upcast is allowed, so that the same
    operation on step 5000 stays an error.

    Mirrors `migrate_adafactor_state`, which exists for the same reason: a
    representation change a checkpoint cannot infer for itself.
    """
    if "master_params" in state:
        return dict(state)
    return _attach_master_state(
        dict(state),
        tree_map(lambda p: _asarray(p).astype(_np_dtype(master_dtype), copy=True), params),
        master_dtype,
    )


def _resolve_state(
    state: "dict[str, Any] | None",
    *,
    optimizer: str,
    required: "tuple[str, ...]",
    fresh: Callable[[], dict[str, Any]],
    master_dtype: str | None = None,
) -> dict[str, Any]:
    """Resolve optimizer state, or refuse with a diagnostic that names the slot.

    `None` is the documented fresh-start value and stays the ONLY one. An
    empty or partial dict is deliberately NOT treated as a fresh start: in
    practice it is state that got dropped between steps -- a checkpoint that
    saved parameters but not slots, a tree rebuilt by name, a dict
    comprehension that filtered. Silently restarting there discards the
    accumulated moments and degrades training with no error at all, which is
    the failure mode worth refusing (#21a: a key that selects semantics fails
    closed).

    What this replaces is a raw `KeyError('velocity')` from inside a tree map
    -- an exception that names neither the optimizer, nor the contract, nor
    the fix (MSW-3 / correctness-audit finding M-4).

    `master_params` is required **exactly when `master_dtype` is set** — pass
    it as `master_dtype` rather than listing it in `required`. The first
    version of this helper excluded the slot unconditionally, reasoning that
    demanding it would refuse valid fp32 state. That conflated two different
    situations (review on #693). With no `master_dtype` the slot legitimately
    does not exist; with one set it is mandatory, because `_master_tree`
    falls back to **upcasting the rounded parameter storage** when the slot
    is absent and calls the result master weights. Resuming a bf16 run from
    state that lost that slot would silently discard exactly the accumulated
    precision mixed-precision training exists to keep — the same silent
    restart this contract refuses for the moments, and a worse one.

    Enabling `master_dtype` on a run that did not have it is a real
    migration rather than an error, so it gets an explicit door:
    `attach_master_params`.
    """
    if state is None:
        return fresh()
    required = tuple(required) + (("master_params",) if master_dtype is not None else ())
    missing = [slot for slot in required if slot not in state]
    if missing:
        raise ValueError(
            f"{optimizer}: optimizer state is missing {missing!r}. "
            f"Pass state=None to start fresh -- a dict that is empty or has "
            f"lost slots is treated as dropped state, not as a fresh start, "
            f"because silently restarting would discard the accumulated "
            f"moments and quietly degrade training. "
            f"Slots present: {sorted(state)!r}; required: {list(required)!r}."
            + (
                " 'master_params' is required because master_dtype is set: "
                "without it the master weights would be rebuilt by upcasting "
                "the rounded parameter storage, silently discarding the "
                "accumulated precision. To ENABLE mixed precision on a run "
                "that did not have it, pass the state through "
                "optim.attach_master_params(state, params, master_dtype=...) "
                "once."
                if master_dtype is not None and "master_params" in missing
                else ""
            )
        )
    return state


def tree_map(fn: Callable[[Any], Any], tree: Tree) -> Tree:
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, v) for v in tree)
    if isinstance(tree, list):
        return [tree_map(fn, v) for v in tree]
    return fn(tree)


def tree_map2(fn: Callable[[Any, Any], Any], a: Tree, b: Tree) -> Tree:
    if isinstance(a, dict):
        return {k: tree_map2(fn, a[k], b[k]) for k in a}
    if isinstance(a, tuple):
        return tuple(tree_map2(fn, x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, list):
        return [tree_map2(fn, x, y) for x, y in zip(a, b, strict=True)]
    return fn(a, b)


def tree_map3(fn: Callable[[Any, Any, Any], Any], a: Tree, b: Tree, c: Tree) -> Tree:
    if isinstance(a, dict):
        return {k: tree_map3(fn, a[k], b[k], c[k]) for k in a}
    if isinstance(a, tuple):
        return tuple(tree_map3(fn, x, y, z) for x, y, z in zip(a, b, c, strict=True))
    if isinstance(a, list):
        return [tree_map3(fn, x, y, z) for x, y, z in zip(a, b, c, strict=True)]
    return fn(a, b, c)


def zeros_like_tree(tree: Tree, dtype: str | None = "fp32") -> Tree:
    return tree_map(lambda x: np.zeros_like(_asarray(x), dtype=_np_dtype(dtype)), tree)


def tree_l2_norm(tree: Tree) -> float:
    total = 0.0

    def add(x):
        nonlocal total
        arr = _asarray(x).astype(np.float64, copy=False)
        total += float(np.sum(arr * arr))
        return x

    tree_map(add, tree)
    return math.sqrt(total)


def sgd(
    params: Tree,
    grads: Tree,
    lr: float,
    *,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> Tree | tuple[Tree, dict[str, Any]]:
    """Plain SGD update."""
    del state_dtype  # SGD has no optimizer slots but accepts the common dtype policy.
    base_params = _master_tree(params, None, master_dtype)
    new_master = tree_map2(
        lambda p, g: _compute_array(p, compute_dtype) - float(lr) * _compute_array(g, compute_dtype),
        base_params,
        grads,
    )
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    if master_dtype is None:
        return new_params
    return new_params, {"master_params": new_master, "master_dtype": _normalize_dtype(master_dtype)}


def moment_free(
    params: Tree,
    grads: Tree,
    *,
    lr: float,
    threshold: float = 0.0,
    weight_decay: float = 0.0,
    compute_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> Tree | tuple[Tree, dict[str, Any]]:
    """Stateless sign-threshold update for zeroth-order pseudo-gradients.

    Each element applies ``-lr * sign(g)`` only when ``abs(g) > threshold``.
    There are no first- or second-moment slots; optional weight decay is
    decoupled.  This is the EGGROLL W3 moment-free path and is deliberately a
    composition of existing elementwise semantics rather than a new Graph op.
    """
    if not math.isfinite(float(lr)) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if not math.isfinite(float(weight_decay)) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    base_params = _master_tree(params, None, master_dtype)

    def apply(p, g):
        p_arr = _compute_array(p, compute_dtype)
        grad = _compute_array(g, compute_dtype)
        direction = np.where(np.abs(grad) > float(threshold), np.sign(grad), 0.0)
        if weight_decay:
            p_arr = p_arr * (1.0 - float(lr) * float(weight_decay))
        return p_arr - float(lr) * direction

    new_master = tree_map2(apply, base_params, grads)
    new_params = tree_map2(
        lambda p_new, p_orig: _cast_like_param(
            p_new, p_orig, cast_updates_to_param_dtype
        ),
        new_master,
        params,
    )
    if master_dtype is None:
        return new_params
    return new_params, {
        "master_params": new_master,
        "master_dtype": _normalize_dtype(master_dtype),
    }


def momentum(
    params: Tree,
    grads: Tree,
    state: dict[str, Tree] | None = None,
    *,
    lr: float,
    momentum: float = 0.9,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Tree]]:
    """SGD with classical momentum — `def:momentum_two`.

    MSW-3 audit. The source gives FOUR distinct momentum recursions, all
    sharing the same parameter update `Theta_n = Theta_{n-1} - gamma_n m_n`
    and differing only in how the gradient enters:

        def:momentum        m_n = alpha m_{n-1} + (1 - alpha) g
        def:momentum_two    m_n = alpha m_{n-1} + g              <- this one
        def:momentum_three  m_n = alpha m_{n-1} + (1 - alpha) gamma g
        def:momentum_four   m_n = alpha m_{n-1} + gamma g

    Tessera implements the second: an unnormalized accumulation, with the
    learning rate applied at the update rather than folded into the moment.
    The distinction is not cosmetic -- the four differ in the effective step
    size by a factor of `(1 - alpha)`, which is 10x at the default
    `alpha = 0.9`, so reading a hyper-parameter from a paper that uses a
    different convention silently mis-scales training by an order of
    magnitude.

    `test_recorded_momentum_formulation_is_the_one_implemented` pins this
    against all four, so the claim fails rather than rots if the recursion
    changes.
    """
    base_params = _master_tree(params, state, master_dtype)
    state = _resolve_state(
        state, optimizer="momentum", master_dtype=master_dtype, required=("velocity",),
        fresh=lambda: {"velocity": zeros_like_tree(params, state_dtype)})
    velocity = state["velocity"]
    new_velocity = tree_map2(
        lambda v, g: _state_array(float(momentum) * _compute_array(v, compute_dtype) + _compute_array(g, compute_dtype), state_dtype),
        velocity,
        grads,
    )
    new_master = tree_map2(lambda p, v: _compute_array(p, compute_dtype) - float(lr) * _compute_array(v, compute_dtype), base_params, new_velocity)
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    return new_params, _attach_master_state({"velocity": new_velocity}, new_master, master_dtype)


def nesterov(
    params: Tree,
    grads: Tree,
    state: dict[str, Tree] | None = None,
    *,
    lr: float,
    momentum: float = 0.9,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Tree]]:
    """Nesterov momentum — the `def:momentum_two` accumulation, look-ahead update.

    Shares `momentum`'s recursion `m_n = alpha m_{n-1} + g` (see the audit
    note there) and differs in the applied direction: `g + alpha m_n` rather
    than `m_n`.
    """
    base_params = _master_tree(params, state, master_dtype)
    state = _resolve_state(
        state, optimizer="nesterov", master_dtype=master_dtype, required=("velocity",),
        fresh=lambda: {"velocity": zeros_like_tree(params, state_dtype)})
    velocity = state["velocity"]
    new_velocity = tree_map2(
        lambda v, g: _state_array(float(momentum) * _compute_array(v, compute_dtype) + _compute_array(g, compute_dtype), state_dtype),
        velocity,
        grads,
    )
    update = tree_map2(
        lambda g, v: _compute_array(g, compute_dtype) + float(momentum) * _compute_array(v, compute_dtype),
        grads,
        new_velocity,
    )
    new_master = tree_map2(lambda p, u: _compute_array(p, compute_dtype) - float(lr) * _compute_array(u, compute_dtype), base_params, update)
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    return new_params, _attach_master_state({"velocity": new_velocity}, new_master, master_dtype)


def adamw(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    """AdamW with decoupled weight decay."""
    base_params = _master_tree(params, state, master_dtype)
    state = _resolve_state(
        state, optimizer="adamw", master_dtype=master_dtype, required=("m", "v", "step"),
        fresh=lambda: {"m": zeros_like_tree(params, state_dtype),
                       "v": zeros_like_tree(params, state_dtype), "step": 0})
    step = int(state["step"]) + 1
    m = tree_map2(
        lambda m_, g: _state_array(beta1 * _compute_array(m_, compute_dtype) + (1.0 - beta1) * _compute_array(g, compute_dtype), state_dtype),
        state["m"],
        grads,
    )
    v = tree_map2(
        lambda v_, g: _state_array(beta2 * _compute_array(v_, compute_dtype) + (1.0 - beta2) * (_compute_array(g, compute_dtype) ** 2), state_dtype),
        state["v"],
        grads,
    )
    b1_corr = 1.0 - beta1 ** step
    b2_corr = 1.0 - beta2 ** step

    def update_param(p, m_, v_):
        p_arr = _compute_array(p, compute_dtype)
        update = (_compute_array(m_, compute_dtype) / b1_corr) / (np.sqrt(_compute_array(v_, compute_dtype) / b2_corr) + eps)
        if weight_decay:
            p_arr = p_arr * (1.0 - lr * weight_decay)
        return p_arr - lr * update

    new_master = tree_map3(update_param, base_params, m, v)
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    return new_params, _attach_master_state({"m": m, "v": v, "step": step}, new_master, master_dtype)


def adam(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    """Adam without decoupled weight decay."""
    # Validated under the PUBLIC name before delegating (review on #693).
    # `adam` forwards to `adamw`, so without this the caller is told that
    # `adamw` rejected their state -- an optimizer they never called -- which
    # defeats the point of a diagnostic that names the optimizer. `fresh=dict`
    # is safe here: a None state short-circuits before the check, and the
    # real fresh state is built by the delegate.
    _resolve_state(state, optimizer="adam", master_dtype=master_dtype,
                   required=("m", "v", "step"), fresh=dict)
    return adamw(
        params,
        grads,
        state,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=0.0,
        compute_dtype=compute_dtype,
        state_dtype=state_dtype,
        master_dtype=master_dtype,
        cast_updates_to_param_dtype=cast_updates_to_param_dtype,
    )


_ADAFACTOR_V_REPRESENTATION = "debiased_v1"


def migrate_adafactor_state(
    state: dict[str, Any], beta2: float, *, state_dtype: str = "fp32"
) -> dict[str, Any]:
    """Convert a pre-bias-correction Adafactor state to the debiased form.

    Call this ONCE on a checkpoint written before `adafactor_decay` existed.
    Such a state holds the raw EMA ``v_raw = (1 - b2**t) * v_debiased``, so it
    is divided by that factor -- exactly inverting the bias it was written
    with. A state already carrying the marker is returned unchanged.
    """
    if state.get("v_representation") == _ADAFACTOR_V_REPRESENTATION:
        return state
    step = int(state.get("step", 0))
    migrated = dict(state)
    migrated["v_representation"] = _ADAFACTOR_V_REPRESENTATION
    bias = 1.0 - float(beta2) ** step
    if step <= 0 or bias <= 0.0:
        return migrated
    migrated["v"] = _adafactor_tree_map_unary(
        lambda s: _adafactor_scale_state(s, 1.0 / bias, state_dtype=state_dtype),
        state["v"],
    )
    return migrated


def _warn_if_adafactor_state_unmarked(state: dict[str, Any]) -> None:
    """Flag a state whose second-moment representation is ambiguous.

    ``state["v"]`` carries the DEBIASED estimate since the bias correction
    landed; before that it carried the raw EMA. A state without the marker
    could be either, and the two cannot be told apart from the values.

    This deliberately does NOT rescale. Auto-migrating on a missing marker
    would silently rewrite every hand-built state dict -- which is a worse
    failure than the one it fixes, since a test or caller that assembles its
    own state has no legacy bias to remove. Callers restoring a genuine
    pre-correction checkpoint should call `migrate_adafactor_state` once.
    """
    if state.get("v_representation") == _ADAFACTOR_V_REPRESENTATION:
        return
    if int(state.get("step", 0)) <= 0:
        return
    warnings.warn(
        "Adafactor state carries no 'v_representation' marker at step "
        f"{int(state.get('step', 0))}. Since the bias correction landed, "
        "state['v'] holds the DEBIASED second moment; a checkpoint written "
        "before that holds the raw EMA and will be read as if it were already "
        "debiased, erasing most of its history. If this state came from an "
        "older checkpoint, pass it through optim.migrate_adafactor_state "
        "once; if it was built by hand, set "
        "state['v_representation'] = optim._ADAFACTOR_V_REPRESENTATION.",
        RuntimeWarning,
        stacklevel=3,
    )


def adafactor_effective_decay(beta2: float, step: int | None) -> float:
    """The decay an Adafactor update actually applies, given an optional step.

    One implementation of a rule that was previously written out three times
    and disagreed with itself (fixed 2026-09-02): the eager `ops.adafactor`
    used the nominal decay when no step was supplied, while the compiled x86
    and ROCm forward executors and the VJP state contract each substituted
    `step=1`. `adafactor_decay(b2, 1)` is exactly 0, so those three silently
    discarded the incoming second moment on any call that omitted a step —
    turning a stateful optimizer into a stateless one, which is precisely the
    failure `ops.adafactor` documents as the reason ABSENT must not mean 1.

    `step is None` therefore means "the caller applied no bias correction" and
    returns the nominal decay unchanged. A supplied step is 1-based.
    """
    if step is None:
        return float(beta2)
    index = int(step)
    if index < 1:
        raise ValueError(
            f"adafactor step is the 1-based update index; got {index}. Pass "
            "None for an update that applied no bias correction."
        )
    return adafactor_decay(float(beta2), index)


def adafactor_decay(beta2: float, step: int) -> float:
    """Bias-corrected second-moment decay for the Adafactor update at ``step``.

    Adafactor (Shazeer & Stern 2018) does **not** run a fixed second-moment
    decay.  A raw EMA ``v_t = b2*v_{t-1} + (1-b2)*g^2`` started from
    ``v_0 = 0`` is biased low by ``1 - b2**t``, so the first updates are
    inflated by ``1/sqrt(1 - b2**t)`` — 31.6x at the default ``b2 = 0.999``,
    and still >2x after 1000 steps.

    This returns the step-dependent decay

        b2_t = b2 * (1 - b2**(t-1)) / (1 - b2**t)

    for which the recursion carries the *debiased* estimate directly:

        v_t = b2_t*v_{t-1} + (1 - b2_t)*g_t^2  ==  EMA_t / (1 - b2**t)

    (at ``t = 1``, ``b2_1 = 0`` and ``v_1 = g_1^2`` exactly).  So this is
    algebraically the explicit ``1 - beta2**step`` correction that ``adamw``
    already applies above, expressed as a decay rate the way the paper's own
    ``1 - t**-0.8`` schedule is.  Expressing it as a decay rate is what makes
    it landable: every physical Adafactor kernel (AVX-512, gfx1151, sm_120)
    already takes ``beta2`` as a scalar, so the correction is applied
    host-side and needs no kernel ABI change.  Unlike ``1 - t**-0.8`` it also
    preserves the caller's ``beta2`` as the asymptotic decay instead of
    silently discarding it (Decision #21a).

    ``step`` is 1-based: it is the index of the update being computed, i.e.
    ``state["step"] + 1``.
    """
    b2 = float(beta2)
    t = int(step)
    if not 0.0 <= b2 < 1.0:
        raise ValueError(
            f"adafactor beta2 must lie in [0, 1); got {beta2!r}"
        )
    if t < 1:
        raise ValueError(
            f"adafactor step is 1-based and must be >= 1; got {step!r}"
        )
    prev = 1.0 - b2 ** (t - 1)
    current = 1.0 - b2**t
    if current <= 0.0:
        return 0.0
    return b2 * prev / current


def adafactor(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    beta2: float = 0.999,
    eps: float = 1e-30,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    """Adafactor reference update.

    Matrix leaves store factored row/column second moments; lower-rank leaves
    fall back to full second moments.
    """
    base_params = _master_tree(params, state, master_dtype)
    state = _resolve_state(
        state, optimizer="adafactor", master_dtype=master_dtype, required=("v", "step"),
        fresh=lambda: {
            "v": tree_map(lambda p: _adafactor_zero_state(_asarray(p), state_dtype=state_dtype), params),
            "step": 0,
            "v_representation": _ADAFACTOR_V_REPRESENTATION,
        })
    # `v` changed meaning when the bias correction landed: the step-dependent
    # decay makes the recursion carry the DEBIASED estimate directly, where it
    # previously carried the raw EMA. A checkpoint written before that holds
    # the raw form, and reading it as debiased silently erases most of the
    # accumulated history -- at step 2 with the default beta2 the effective
    # decay is ~0.5, so a stored 0.001*g1^2 is treated as the whole prior
    # estimate. The schema had no marker to tell the two apart, so it gets one,
    # and a state without it is migrated rather than misread.
    _warn_if_adafactor_state_unmarked(state)
    # The tracked step is finally used: without it the zero-initialized second
    # moment is biased low and the first updates are inflated by
    # 1/sqrt(1 - beta2**step).  See `adafactor_decay`.
    decay = adafactor_decay(beta2, int(state["step"]) + 1)
    new_v = _adafactor_tree_map(
        lambda s, g: _adafactor_update_state(s, _compute_array(g, compute_dtype), decay, state_dtype=state_dtype),
        state["v"],
        grads,
    )
    updates = _adafactor_tree_map(
        lambda s, g: _adafactor_update_from_state(s, _compute_array(g, compute_dtype), eps, compute_dtype=compute_dtype),
        new_v,
        grads,
    )
    new_master = tree_map2(lambda p, u: _compute_array(p, compute_dtype) - float(lr) * _compute_array(u, compute_dtype), base_params, updates)
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    return new_params, _attach_master_state(
        {
            "v": new_v,
            "step": int(state["step"]) + 1,
            "v_representation": _ADAFACTOR_V_REPRESENTATION,
        },
        new_master,
        master_dtype,
    )


def _is_adafactor_slot(x: Any) -> bool:
    return isinstance(x, dict) and "factored" in x


def _adafactor_tree_map(fn: Callable[[Any, Any], Any], slot_tree: Tree, grad_tree: Tree) -> Tree:
    if _is_adafactor_slot(slot_tree):
        return fn(slot_tree, grad_tree)
    if isinstance(slot_tree, dict):
        return {k: _adafactor_tree_map(fn, slot_tree[k], grad_tree[k]) for k in slot_tree}
    if isinstance(slot_tree, tuple):
        return tuple(_adafactor_tree_map(fn, s, g) for s, g in zip(slot_tree, grad_tree, strict=True))
    if isinstance(slot_tree, list):
        return [_adafactor_tree_map(fn, s, g) for s, g in zip(slot_tree, grad_tree, strict=True)]
    return fn(slot_tree, grad_tree)


def _adafactor_zero_state(arr: np.ndarray, *, state_dtype: str = "fp32"):
    if arr.ndim >= 2:
        return {
            "row": np.zeros(arr.shape[:-1], dtype=_np_dtype(state_dtype)),
            "col": np.zeros(arr.shape[-1], dtype=_np_dtype(state_dtype)),
            "factored": True,
        }
    return {"v": np.zeros_like(arr, dtype=_np_dtype(state_dtype)), "factored": False}


def _adafactor_tree_map_unary(fn: Callable[[Any], Any], slot_tree: Tree) -> Tree:
    if _is_adafactor_slot(slot_tree):
        return fn(slot_tree)
    if isinstance(slot_tree, dict):
        return {k: _adafactor_tree_map_unary(fn, v) for k, v in slot_tree.items()}
    if isinstance(slot_tree, tuple):
        return tuple(_adafactor_tree_map_unary(fn, s) for s in slot_tree)
    if isinstance(slot_tree, list):
        return [_adafactor_tree_map_unary(fn, s) for s in slot_tree]
    return fn(slot_tree)


def _adafactor_scale_state(state, scale: float, *, state_dtype: str = "fp32"):
    """Scale a slot's second moments, preserving the factored/full shape."""
    dtype = _np_dtype(state_dtype)
    if state["factored"]:
        return {
            "row": (np.asarray(state["row"], dtype=dtype) * dtype(scale)).astype(dtype, copy=False),
            "col": (np.asarray(state["col"], dtype=dtype) * dtype(scale)).astype(dtype, copy=False),
            "factored": True,
        }
    return {
        "v": (np.asarray(state["v"], dtype=dtype) * dtype(scale)).astype(dtype, copy=False),
        "factored": False,
    }


def _adafactor_update_state(state, grad: np.ndarray, beta2: float, *, state_dtype: str = "fp32"):
    grad2 = grad.astype(_np_dtype(state_dtype), copy=False) ** 2
    if state["factored"]:
        return {
            "row": _state_array(beta2 * state["row"] + (1.0 - beta2) * grad2.mean(axis=-1), state_dtype),
            "col": _state_array(beta2 * state["col"] + (1.0 - beta2) * grad2.mean(axis=tuple(range(grad.ndim - 1))), state_dtype),
            "factored": True,
        }
    return {"v": _state_array(beta2 * state["v"] + (1.0 - beta2) * grad2, state_dtype), "factored": False}


def _adafactor_update_from_state(state, grad: np.ndarray, eps: float, *, compute_dtype: str = "fp32"):
    grad = _compute_array(grad, compute_dtype)
    if state["factored"]:
        row = np.maximum(_compute_array(state["row"], compute_dtype), eps)
        col = np.maximum(_compute_array(state["col"], compute_dtype), eps)
        scale = row[..., None] * col / max(float(np.mean(row)), eps)
        return grad / (np.sqrt(scale) + eps)
    return grad / (np.sqrt(np.maximum(_compute_array(state["v"], compute_dtype), eps)) + eps)


def lion(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    base_params = _master_tree(params, state, master_dtype)
    state = _resolve_state(
        state, optimizer="lion", master_dtype=master_dtype, required=("m", "step"),
        fresh=lambda: {"m": zeros_like_tree(params, state_dtype), "step": 0})
    update = tree_map2(
        lambda m, g: beta1 * _compute_array(m, compute_dtype) + (1.0 - beta1) * _compute_array(g, compute_dtype),
        state["m"],
        grads,
    )
    new_m = tree_map2(
        lambda m, g: _state_array(beta2 * _compute_array(m, compute_dtype) + (1.0 - beta2) * _compute_array(g, compute_dtype), state_dtype),
        state["m"],
        grads,
    )

    def apply(p, u):
        p_arr = _compute_array(p, compute_dtype)
        if weight_decay:
            p_arr = p_arr * (1.0 - lr * weight_decay)
        return p_arr - lr * np.sign(_asarray(u))

    new_master = tree_map2(apply, base_params, update)
    new_params = tree_map2(lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype), new_master, params)
    return new_params, _attach_master_state({"m": new_m, "step": int(state["step"]) + 1}, new_master, master_dtype)


def muon(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    momentum: float = 0.95,
) -> tuple[Tree, dict[str, Any]]:
    state = _resolve_state(
        state, optimizer="muon", required=("velocity",),
        fresh=lambda: {"velocity": zeros_like_tree(params)})
    velocity = state["velocity"]
    new_velocity = tree_map2(lambda v, g: momentum * _asarray(v) + _asarray(g), velocity, grads)
    updates = tree_map(_orthogonalize_if_matrix, new_velocity)
    return sgd(params, updates, lr), {"velocity": new_velocity}


def _orthogonalize_if_matrix(x: Any) -> np.ndarray:
    arr = _asarray(x).astype(np.float32, copy=False)
    if arr.ndim < 2:
        norm = np.linalg.norm(arr)
        return arr / (norm + 1e-12)
    mat = arr.reshape(arr.shape[0], -1)
    u, _, vh = np.linalg.svd(mat, full_matrices=False)
    return (u @ vh).reshape(arr.shape)


def lamb(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-6,
    weight_decay: float = 0.0,
) -> tuple[Tree, dict[str, Any]]:
    # Validated under the PUBLIC name before delegating (review on #693).
    # `lamb` forwards to `adamw`, so without this the caller is told that
    # `adamw` rejected their state -- an optimizer they never called -- which
    # defeats the point of a diagnostic that names the optimizer. `fresh=dict`
    # is safe here: a None state short-circuits before the check, and the
    # real fresh state is built by the delegate.
    _resolve_state(state, optimizer="lamb", master_dtype=None,
                   required=("m", "v", "step"), fresh=dict)
    next_params, adam_state = adamw(
        params,
        grads,
        state,
        lr=1.0,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=0.0,
    )
    adam_update = tree_map2(lambda p, p_next: _asarray(p) - _asarray(p_next), params, next_params)

    def apply(p, u):
        p_arr = _asarray(p)
        update = _asarray(u) + weight_decay * p_arr
        p_norm = np.linalg.norm(p_arr)
        u_norm = np.linalg.norm(update)
        trust = 1.0 if p_norm == 0.0 or u_norm == 0.0 else p_norm / u_norm
        return p_arr - lr * trust * update

    return tree_map2(apply, params, adam_update), adam_state



# --- MSW-3 optimizer breadth -------------------------------------------------
#
# Each of these is transcribed from a numbered definition in Jentzen, Kuckuck &
# von Wurstemberger, *Mathematical Introduction to Deep Learning* (arXiv
# 2310.20360v3), and the docstring names the label. Two transcription details
# recur and are easy to get silently wrong, so they are stated once here:
#
#   * **eps sits OUTSIDE the square root** in Adagrad and RMSprop -- the
#     definitions read `(eps + M^(1/2))^-1`, not `(M + eps)^(-1/2)` and not
#     `M^(1/2) + eps` applied to a shifted M. Adadelta is the exception: there
#     eps is inside, appearing in BOTH the numerator and denominator of a
#     single square root over a ratio.
#   * **the bias adjustment is a PRODUCT of the decay factors**,
#     `1 - prod(beta_k)`, not `1 - beta^n`. They coincide only for a constant
#     beta, which is the common case but not the definition; these take a
#     scalar beta and so use the closed form, and say so.


def adagrad(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-2,
    eps: float = 1e-8,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    """Adagrad, per `def:determ_adagrad` eq. (1).

        M_n     = sum_{k<n} |g_k|^2
        Theta_n = Theta_{n-1} - gamma_n (eps + M_n^(1/2))^-1 g(Theta_{n-1})

    The accumulator is a plain running SUM with no decay, which is what
    separates Adagrad from RMSprop below: the effective step size is
    monotonically non-increasing and can stall on long runs. That is a
    property of the method, not a defect to be tuned away here.
    """
    base_params = _master_tree(params, state, master_dtype)
    state = _resolve_state(
        state, optimizer="adagrad", master_dtype=master_dtype, required=("m",),
        fresh=lambda: {"m": zeros_like_tree(params, state_dtype)})
    m = tree_map2(
        lambda m_, g: _state_array(
            _compute_array(m_, compute_dtype) + _compute_array(g, compute_dtype) ** 2,
            state_dtype),
        state["m"], grads)
    new_master = tree_map3(
        lambda p, m_, g: _compute_array(p, compute_dtype) - float(lr) * (
            _compute_array(g, compute_dtype)
            / (float(eps) + np.sqrt(_compute_array(m_, compute_dtype)))),
        base_params, m, grads)
    new_params = tree_map2(
        lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype),
        new_master, params)
    return new_params, _attach_master_state({"m": m}, new_master, master_dtype)


def rmsprop(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    beta: float = 0.9,
    eps: float = 1e-8,
    bias_adjusted: bool = False,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    """RMSprop, per `def:determ_RMSprop` eq. (1)-(2).

        M_n     = beta M_{n-1} + (1 - beta) |g|^2
        Theta_n = Theta_{n-1} - gamma_n (eps + M_n^(1/2))^-1 g

    `bias_adjusted=True` selects `def:determ_RMSprop_bias` eq. (1)-(2)
    instead, which divides the second moment by `1 - prod_{k<=n} beta_k`
    before the root:

        Theta_n = Theta_{n-1} - gamma_n (eps + (M_n / (1 - prod beta_k))^(1/2))^-1 g

    Both variants live in one function because they share the accumulator
    exactly and differ only in that divisor -- two functions would be two
    copies of the same recursion (#31), and the `step` slot they need is
    identical.

    `beta` is a scalar here, so `prod_{k<=n} beta_k` is `beta**n`. The
    definition admits a per-step sequence `(beta_n)`; if that is ever wanted,
    the state must carry the running product rather than the step count,
    because the closed form stops being valid.
    """
    base_params = _master_tree(params, state, master_dtype)
    state = _resolve_state(
        state, optimizer="rmsprop", master_dtype=master_dtype, required=("m", "step"),
        fresh=lambda: {"m": zeros_like_tree(params, state_dtype), "step": 0})
    step = int(state["step"]) + 1
    m = tree_map2(
        lambda m_, g: _state_array(
            float(beta) * _compute_array(m_, compute_dtype)
            + (1.0 - float(beta)) * _compute_array(g, compute_dtype) ** 2,
            state_dtype),
        state["m"], grads)
    correction = (1.0 - float(beta) ** step) if bias_adjusted else 1.0

    def update(p, m_, g):
        second = _compute_array(m_, compute_dtype) / correction
        return _compute_array(p, compute_dtype) - float(lr) * (
            _compute_array(g, compute_dtype) / (float(eps) + np.sqrt(second)))

    new_master = tree_map3(update, base_params, m, grads)
    new_params = tree_map2(
        lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype),
        new_master, params)
    return new_params, _attach_master_state({"m": m, "step": step}, new_master, master_dtype)


def adadelta(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    beta: float = 0.9,
    delta: float = 0.9,
    eps: float = 1e-6,
    lr: float = 1.0,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    """Adadelta, per `def:determ_adadelta` eq. (1)-(4).

        M_n     = beta M_{n-1} + (1 - beta) |g|^2
        Theta_n = Theta_{n-1} - ((eps + Delta_{n-1}) / (eps + M_n))^(1/2) g
        Delta_n = delta Delta_{n-1} + (1 - delta) |Theta_n - Theta_{n-1}|^2

    Note the shape of the step: ONE square root over a ratio, with eps in
    both the numerator and the denominator. It is not `sqrt(Delta + eps) /
    (sqrt(M) + eps)`, and the difference is not cosmetic -- eps in the
    numerator is what gives the very first step (where `Delta_0 = 0`) a
    non-zero size.

    **`lr` is NOT in the definition.** Adadelta is deliberately learning-rate
    free: the `Delta` accumulator makes the step self-scaling, which is the
    whole point of the method. `lr=1.0` reproduces the definition exactly and
    is the default; other values are a Tessera extension, not the source's
    method, and are recorded as such rather than presented as a parameter of
    Adadelta.
    """
    base_params = _master_tree(params, state, master_dtype)
    state = _resolve_state(
        state, optimizer="adadelta", master_dtype=master_dtype, required=("m", "delta"),
        fresh=lambda: {"m": zeros_like_tree(params, state_dtype),
                       "delta": zeros_like_tree(params, state_dtype)})
    m = tree_map2(
        lambda m_, g: _state_array(
            float(beta) * _compute_array(m_, compute_dtype)
            + (1.0 - float(beta)) * _compute_array(g, compute_dtype) ** 2,
            state_dtype),
        state["m"], grads)

    def step_size(d_prev, m_now, g):
        ratio = (float(eps) + _compute_array(d_prev, compute_dtype)) / (
            float(eps) + _compute_array(m_now, compute_dtype))
        return float(lr) * np.sqrt(ratio) * _compute_array(g, compute_dtype)

    delta_step = tree_map3(step_size, state["delta"], m, grads)
    new_master = tree_map2(
        lambda p, s: _compute_array(p, compute_dtype) - s, base_params, delta_step)
    new_delta = tree_map2(
        lambda d_prev, s: _state_array(
            float(delta) * _compute_array(d_prev, compute_dtype)
            + (1.0 - float(delta)) * s ** 2, state_dtype),
        state["delta"], delta_step)
    new_params = tree_map2(
        lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype),
        new_master, params)
    return new_params, _attach_master_state(
        {"m": m, "delta": new_delta}, new_master, master_dtype)


def _inverse_fourth_root(a: np.ndarray, *, eps: float) -> np.ndarray:
    """`A^(-1/4)` for a symmetric positive-definite `A`, via its spectrum.

    Shampoo's preconditioners are sums of Gram matrices plus `eps*I`, so they
    are symmetric PSD by construction and `eigh` is both correct and cheaper
    than a general matrix function. Eigenvalues are floored at `eps` rather
    than at zero: a zero eigenvalue is a division by zero in a method whose
    whole purpose is to divide by this matrix, and the definition's
    `L_0 = eps*I` says the floor is `eps`.
    """
    w, v = np.linalg.eigh((a + a.T) * 0.5)
    w = np.maximum(w, float(eps))
    return (v * (w ** -0.25)) @ v.T


def _shampoo_matrix(x: Any) -> np.ndarray:
    """The `d1 x d2` view Shampoo preconditions, or a refusal.

    Validated on every call rather than only when state is created, so a
    resumed run cannot smuggle in a rank the method cannot express.
    """
    arr = _asarray(x)
    if arr.ndim == 0 or arr.ndim > 2:
        raise ValueError(
            f"shampoo: parameter of rank {arr.ndim} (shape {arr.shape}) is not "
            "a matrix. Shampoo preconditions a d1 x d2 parameter on both sides; "
            "rank-1 is accepted as the d x 1 matrix it already is, but rank 0 "
            "and rank >= 3 have no canonical (d1, d2) split, and choosing one "
            "would silently change the method. Reshape explicitly, or use a "
            "diagonal method such as adagrad/rmsprop for this parameter.")
    return arr.reshape(arr.shape[0], -1)


def shampoo(
    params: Tree,
    grads: Tree,
    state: dict[str, Any] | None = None,
    *,
    lr: float = 1e-3,
    eps: float = 1e-4,
    compute_dtype: str = "fp32",
    state_dtype: str = "fp32",
    master_dtype: str | None = None,
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    """Shampoo, per `def:determ_Shampoo` eq. (1)-(4).

        L_0 = eps I,  R_0 = eps I
        L_n = L_{n-1} + G G*,   R_n = R_{n-1} + G* G
        Theta_n = Theta_{n-1} - gamma_n L_n^(-1/4) G R_n^(-1/4)

    Full-matrix preconditioning on both sides, so this is defined for a
    parameter that IS a matrix. A rank-1 parameter is handled as the `d x 1`
    matrix it already is -- the same definition, not a special case -- and
    rank 0 or rank >= 3 is refused (#21a) rather than silently flattened,
    because there is no canonical choice of which axes become `d_1` and
    `d_2`.

    The two preconditioners are kept as SEPARATE state trees (`left`,
    `right`) rather than one tree of `{left, right}` pairs. The pair form
    reads better but is wrong here: `tree_map2` treats a dict as a pytree
    node, so a per-parameter dict is indistinguishable from a nesting level
    and the mapper tries to index the gradient by `"left"`.
    """
    base_params = _master_tree(params, state, master_dtype)
    state = _resolve_state(
        state, optimizer="shampoo", master_dtype=master_dtype,
        required=("left", "right"),
        fresh=lambda: {
            "left": tree_map(
                lambda p: _state_array(float(eps) * np.eye(_shampoo_matrix(p).shape[0]), state_dtype),
                params),
            "right": tree_map(
                lambda p: _state_array(float(eps) * np.eye(_shampoo_matrix(p).shape[1]), state_dtype),
                params),
        })

    left = tree_map2(
        lambda l_, g: _state_array(
            _compute_array(l_, compute_dtype)
            + (lambda m: m @ m.T)(_shampoo_matrix(g).astype(_np_dtype(compute_dtype))),
            state_dtype),
        state["left"], grads)
    right = tree_map2(
        lambda r_, g: _state_array(
            _compute_array(r_, compute_dtype)
            + (lambda m: m.T @ m)(_shampoo_matrix(g).astype(_np_dtype(compute_dtype))),
            state_dtype),
        state["right"], grads)

    def precondition(g, l_, r_):
        mat = _shampoo_matrix(g).astype(_np_dtype(compute_dtype))
        return (_inverse_fourth_root(_compute_array(l_, compute_dtype), eps=eps)
                @ mat
                @ _inverse_fourth_root(_compute_array(r_, compute_dtype), eps=eps))

    steps = tree_map3(precondition, grads, left, right)
    new_master = tree_map2(
        lambda p, s: _compute_array(p, compute_dtype) - float(lr) * s.reshape(_asarray(p).shape),
        base_params, steps)
    new_params = tree_map2(
        lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype),
        new_master, params)
    return new_params, _attach_master_state(
        {"left": left, "right": right}, new_master, master_dtype)


def midpoint_sgd(
    params: Tree,
    grad_fn: Callable[[Tree], Tree],
    state: dict[str, Any] | None = None,
    *,
    lr: float,
    compute_dtype: str = "fp32",
    cast_updates_to_param_dtype: bool = True,
) -> tuple[Tree, dict[str, Any]]:
    """Explicit midpoint SGD, per `def:midpointSGD` eq. (1).

        Theta_n = Theta_{n-1} - gamma_n g(Theta_{n-1} - (gamma_n / 2) g(Theta_{n-1}))

    **This takes a gradient FUNCTION, not a gradient**, and that asymmetry
    with every other optimizer here is forced by the method rather than
    chosen. The midpoint step evaluates the gradient a second time at a probe
    point that does not exist until the first gradient is known, so a
    `(params, grads)` signature cannot express it: anything that fits that
    mould is not this method. Bending it to match -- reusing `grads` at the
    probe point -- would silently degrade it to plain SGD with a half-step,
    which is the failure worth refusing rather than hiding.

    It is stateless: `state` is accepted and returned empty so the call shape
    matches its siblings, and is validated so a caller who passes real state
    is told it is unused rather than having it silently dropped.
    """
    if state:
        raise ValueError(
            f"midpoint_sgd is stateless but was given state with slots "
            f"{sorted(state)!r}. The midpoint method carries nothing between "
            "steps; state here is almost certainly meant for a different "
            "optimizer, and dropping it silently would lose it.")
    probe = tree_map2(
        lambda p, g: _compute_array(p, compute_dtype)
        - 0.5 * float(lr) * _compute_array(g, compute_dtype),
        params, grad_fn(params))
    stepped = tree_map2(
        lambda p, g: _compute_array(p, compute_dtype) - float(lr) * _compute_array(g, compute_dtype),
        params, grad_fn(probe))
    # Cast back to the parameter's storage dtype like every other optimizer
    # here (review on #695). Without it, fp16 parameters came back fp32 after
    # one step -- the tree silently widens, which contradicts the storage-dtype
    # contract and is invisible until memory or a dtype assertion notices.
    new_params = tree_map2(
        lambda p_new, p_orig: _cast_like_param(p_new, p_orig, cast_updates_to_param_dtype),
        stepped, params)
    return new_params, {}


def constant_lr(value: float) -> Callable[[int], float]:
    return lambda step: float(value)


def cosine_lr(step: int, *, init_value: float, end_value: float = 0.0, decay_steps: int) -> float:
    t = min(max(int(step), 0), int(decay_steps)) / max(1, int(decay_steps))
    return float(end_value + 0.5 * (init_value - end_value) * (1.0 + math.cos(math.pi * t)))


def cosine_warmup_lr(step: int, *, peak_value: float, warmup_steps: int, decay_steps: int, end_value: float = 0.0) -> float:
    step = int(step)
    if step < warmup_steps:
        return float(peak_value * step / max(1, warmup_steps))
    return cosine_lr(step - warmup_steps, init_value=peak_value, end_value=end_value, decay_steps=max(1, decay_steps - warmup_steps))


def linear_warmup_lr(step: int, *, peak_value: float, warmup_steps: int) -> float:
    return float(peak_value * min(int(step), int(warmup_steps)) / max(1, int(warmup_steps)))


def polynomial_lr(step: int, *, init_value: float, end_value: float, decay_steps: int, power: float = 1.0) -> float:
    t = min(max(int(step), 0), int(decay_steps)) / max(1, int(decay_steps))
    return float((init_value - end_value) * ((1.0 - t) ** power) + end_value)


def inverse_sqrt_lr(step: int, *, init_value: float, warmup_steps: int = 1) -> float:
    """Inverse-square-root decay with a linear warmup, peaking at ``init_value``.

    The previous form returned ``init_value * sqrt(warmup)/sqrt(step)`` for
    every step, which is *larger* than ``init_value`` for the whole warmup —
    the opposite of warming up, and 63x the nominal peak on step 1 at the usual
    warmup_steps=4000. The rate now ramps linearly to ``init_value`` at
    ``warmup_steps`` and decays as sqrt(warmup/step) after it, so the schedule
    is continuous at the boundary and ``init_value`` is its maximum.
    """
    step = max(1, int(step))
    warmup = max(1, int(warmup_steps))
    if step < warmup:
        return float(init_value * step / warmup)
    return float(init_value * math.sqrt(warmup) / math.sqrt(step))


def cyclical_lr(
    step: int,
    *,
    base_value: float,
    max_value: float,
    step_size: int,
    mode: str = "triangular",
) -> float:
    """Triangular cyclical learning-rate schedule."""
    cycle = math.floor(1.0 + int(step) / (2.0 * max(1, step_size)))
    x = abs(int(step) / max(1, step_size) - 2.0 * cycle + 1.0)
    amplitude = max(0.0, 1.0 - x)
    if mode == "triangular2":
        amplitude /= 2.0 ** (cycle - 1)
    elif mode != "triangular":
        raise ValueError("cyclical_lr mode must be 'triangular' or 'triangular2'")
    return float(base_value + (max_value - base_value) * amplitude)


def chained_schedule(*schedules: Callable[[int], float]) -> Callable[[int], tuple[float, ...]]:
    """Return a schedule that evaluates each child schedule at the same step."""
    return lambda step: tuple(float(schedule(step)) for schedule in schedules)


def clip_grad_norm(grads: Tree, max_norm: float, norm_type: float = 2.0) -> tuple[Tree, float]:
    """Scale `grads` so their `norm_type` norm is at most `max_norm`.

    `norm_type` selects semantics, so it fails closed on a value this cannot
    compute rather than silently substituting the L2 norm (Decision #21a) —
    which previously made `norm_type=1.0` clip by, and report, the L2 norm.
    """
    if norm_type == float("inf"):
        max_abs = {"value": 0.0}
        tree_map(lambda g: _update_max_abs(g, max_abs), grads)
        total = max_abs["value"]
    elif float(norm_type) == 2.0:
        total = tree_l2_norm(grads)
    elif float(norm_type) > 0.0:
        p = float(norm_type)
        acc = {"value": 0.0}

        def _accumulate(g):
            acc["value"] += float(np.sum(np.abs(_asarray(g), dtype=np.float64) ** p))
            return g

        tree_map(_accumulate, grads)
        total = float(acc["value"] ** (1.0 / p))
    else:
        raise ValueError(
            f"clip_grad_norm norm_type must be positive or inf; got {norm_type!r}")
    scale = min(1.0, float(max_norm) / (total + 1e-12))
    return tree_map(lambda g: _asarray(g) * scale, grads), total


def _update_max_abs(g: Any, scope: dict[str, Any]):
    scope["value"] = max(float(scope["value"]), float(np.max(np.abs(_asarray(g)))))
    return g


def clip_grad_value(grads: Tree, clip_value: float) -> Tree:
    c = abs(float(clip_value))
    return tree_map(lambda g: np.clip(_asarray(g), -c, c), grads)


def centralize_grad(grads: Tree) -> Tree:
    def centralize(g):
        arr = _asarray(g)
        if arr.ndim <= 1:
            return arr
        axes = tuple(range(arr.ndim - 1))
        return arr - arr.mean(axis=axes, keepdims=True)

    return tree_map(centralize, grads)


def add_decoupled_weight_decay(grads: Tree, params: Tree, weight_decay: float) -> Tree:
    return tree_map2(lambda g, p: _asarray(g) + float(weight_decay) * _asarray(p), grads, params)


def ema_update(ema_params: Tree, params: Tree, decay: float) -> Tree:
    return tree_map2(lambda e, p: float(decay) * _asarray(e) + (1.0 - float(decay)) * _asarray(p), ema_params, params)


def polyak_avg(avg_params: Tree, params: Tree, step: int) -> Tree:
    step = int(step)
    return tree_map2(lambda a, p: (_asarray(a) * step + _asarray(p)) / (step + 1), avg_params, params)


@dataclass
class OptaxStyleChain:
    """Small transform chain for update trees.

    Each transform receives ``(updates, params)`` and returns a new update tree.
    """

    transforms: tuple[Callable[[Tree, Tree], Tree], ...]

    def __call__(self, updates: Tree, params: Tree) -> Tree:
        out = updates
        for transform in self.transforms:
            out = transform(out, params)
        return out


def chain(*transforms: Callable[[Tree, Tree], Tree]) -> OptaxStyleChain:
    return OptaxStyleChain(tuple(transforms))


def _grad_tree_from_params(params: Tree) -> Tree:
    def grad_leaf(p):
        grad = getattr(p, "grad", None)
        if grad is None:
            return np.zeros_like(_asarray(p), dtype=np.float32)
        return _asarray(grad)

    return tree_map(grad_leaf, params)


def _assign_tree(dst: Tree, src: Tree) -> None:
    if isinstance(dst, dict):
        for key in dst:
            _assign_tree(dst[key], src[key])
        return
    if isinstance(dst, tuple):
        for d, s in zip(dst, src, strict=True):
            _assign_tree(d, s)
        return
    if isinstance(dst, list):
        for d, s in zip(dst, src, strict=True):
            _assign_tree(d, s)
        return
    if hasattr(dst, "_data") and hasattr(dst._data, "_data"):
        dst._data._data[...] = np.asarray(src, dtype=dst._data._data.dtype)
        return
    if isinstance(dst, np.ndarray):
        dst[...] = np.asarray(src, dtype=dst.dtype)
        return
    raise TypeError(f"Cannot assign optimizer update into {type(dst).__name__}")


class Optimizer:
    """Stateful wrapper around Tessera's functional optimizer updates.

    The wrapper is intentionally small: parameters remain owned by modules or
    callers, optimizer state is a Python tree, and ``step()`` mutates parameter
    storage in place for ergonomic training-loop compatibility.
    """

    update_fn: Callable[..., tuple[Tree, dict[str, Any]]]

    def __init__(self, params: Tree, **kwargs: Any) -> None:
        self.params = list(params) if not isinstance(params, (dict, list, tuple)) else params
        self.kwargs = dict(kwargs)
        self.state: dict[str, Any] | None = None

    def step(self, grads: Tree | None = None) -> Tree:
        if grads is None:
            grads = _grad_tree_from_params(self.params)
        new_params, self.state = self.update_fn(self.params, grads, self.state, **self.kwargs)
        _assign_tree(self.params, new_params)
        return self.params

    def zero_grad(self) -> None:
        def clear(p):
            if hasattr(p, "zero_grad"):
                p.zero_grad()
            return p

        tree_map(clear, self.params)


class AdamW(Optimizer):
    """Stateful AdamW optimizer over module parameters or array trees."""

    update_fn = staticmethod(adamw)

    def __init__(
        self,
        params: Tree,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        **dtype_policy: Any,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            **dtype_policy,
        )


class Adam(Optimizer):
    """Stateful Adam optimizer over module parameters or array trees."""

    update_fn = staticmethod(adam)

    def __init__(
        self,
        params: Tree,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        **dtype_policy: Any,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            **dtype_policy,
        )


__all__ = [
    "Adam",
    "AdamW",
    "OptaxStyleChain",
    "Optimizer",
    "adafactor",
    "adafactor_effective_decay",
    "adam",
    "adamw",
    "add_decoupled_weight_decay",
    "centralize_grad",
    "chain",
    "chained_schedule",
    "clip_grad_norm",
    "clip_grad_value",
    "constant_lr",
    "cosine_lr",
    "cosine_warmup_lr",
    "cyclical_lr",
    "ema_update",
    "inverse_sqrt_lr",
    "lamb",
    "linear_warmup_lr",
    "lion",
    "momentum",
    "muon",
    "moment_free",
    "nesterov",
    "polyak_avg",
    "polynomial_lr",
    "sgd",
    "tree_l2_norm",
    "tree_map",
    "tree_map2",
    "zeros_like_tree",
]
