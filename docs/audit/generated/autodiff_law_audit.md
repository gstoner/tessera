# Autodiff law audit — AD-LAW-1

> **Generated** by `python/tessera/compiler/law_audit.py` from
> `tessera.autodiff.laws.run_law_sweep()`. Do not hand-edit; regenerate
> via `scripts/check_generated_docs.sh --write`. Design authority:
> [`AUTODIFF_NEXTGEN_PLAN.md`](../compiler/AUTODIFF_NEXTGEN_PLAN.md) §4.

Law 3 (adjoint, `⟨Jv,u⟩ = ⟨v,Jᵀu⟩`) is complete for the transpose
relationship between a paired JVP/VJP; it cannot certify the derivative
itself (a matched-wrong pair passes). Law 1 (chain vs finite
differences) is the derivative-correctness complement. `no_spec` rows
are real unswept debt, not exclusions.

## `tensor` registry

| Law | Status | Ops |
|---|---|---:|
| adjoint | no_spec | 237 |
| adjoint | pass | 54 |
| adjoint | vjp_only | 17 |
| chain | pass | 52 |

## `geometric` registry

| Law | Status | Ops |
|---|---|---:|
| adjoint | no_spec | 16 |

## Checked tensor ops

`abs`, `absolute`, `acos`, `add`, `amax`, `amin`, `asin`, `atan`, `atan2`, `clamp`, `cos`, `cosh`, `cross_entropy_loss`, `cumsum`, `erf`, `erfc`, `exp`, `expm1`, `gelu`, `gemm`, `kl_divergence`, `layer_norm`, `log`, `log1p`, `log_softmax`, `logsumexp`, `matmul`, `maximum`, `mean`, `minimum`, `mse_loss`, `mul`, `prod`, `reciprocal`, `relu`, `reshape`, `rmsnorm`, `sigmoid`, `sigmoid_safe`, `sign`, `silu`, `sin`, `sinh`, `softcap`, `softmax`, `softmax_safe`, `softplus`, `sqrt`, `std`, `sum`, `tan`, `tanh`, `transpose`, `var`
