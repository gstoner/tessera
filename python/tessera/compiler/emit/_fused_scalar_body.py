"""Shared scalar kernel body for the ``FusedRegion`` compiled lanes (x86 C + ROCm
HIP). The per-element math is arch-agnostic — the same ``expf``/``tanhf``/
``rsqrtf`` snippets compile in host C (clang/cc, x86 lane) and HIP device code
(hipcc, ROCm lane) — so both backends synthesize the *identical* body from here
and stay locked to the same ``fusion_core`` numpy reference. Only the wrapper
differs (a plain C function that loops over rows vs a ``__global__`` one thread
per row); each backend supplies its own.

The body assumes these names are in scope: ``const float* A, B, bias, residual``,
``float* row`` (= ``out + m*N``), and ``int M, N, K, m``. It writes ``row[0..N]``
for row ``m`` (matmul + prologue on A + epilogue chain + optional residual +
optional row reduction), matching ``FusedRegion.reference`` element for element.

``EpilogueOp.emit(target)`` / ``ReductionOp.emit(target)`` deliberately raise for
non-Metal targets, so the op-name → C tables live here (Decision #21: no silent
wrong-language emit).
"""
from __future__ import annotations

from typing import Any


def pointwise_snippet(op: str, var: str) -> str:
    """C/HIP statement applying pointwise epilogue op ``op`` to lvalue ``var``
    (f32). Mirrors ``EPILOGUE_OPS[op].ref`` numerically. ``bias`` is handled by
    the caller (it reads ``bias[n]``); the rest are activations valid in a
    prologue too."""
    if op == "relu":
        return f"{var} = {var} > 0.0f ? {var} : 0.0f;"
    if op == "gelu":  # tanh approximation, clamped — identical to fusion_core._gelu
        return (
            f"{{ float _t = 0.7978845608028654f*({var}+0.044715f*{var}*{var}*{var});"
            f" _t = _t < -30.0f ? -30.0f : (_t > 30.0f ? 30.0f : _t);"
            f" {var} = 0.5f*{var}*(1.0f+tanhf(_t)); }}"
        )
    if op == "silu":
        return f"{var} = {var} / (1.0f + expf(-{var}));"
    if op == "sigmoid":
        return f"{var} = 1.0f / (1.0f + expf(-{var}));"
    if op == "tanh":
        return f"{var} = tanhf({var});"
    raise ValueError(f"no scalar snippet for pointwise op {op!r}")


def reduction_snippet(name: str, eps: float) -> str:
    """C/HIP block reducing the length-``N`` row ``row`` in place (f32). Mirrors
    ``REDUCTION_OPS[name].ref``."""
    if name == "rmsnorm":
        return (
            "        { float _ss = 0.0f;\n"
            "          for (int n = 0; n < N; ++n) _ss += row[n]*row[n];\n"
            f"          float _inv = 1.0f/sqrtf(_ss/(float)N + {eps!r}f);\n"
            "          for (int n = 0; n < N; ++n) row[n] = row[n]*_inv; }\n"
        )
    if name == "softmax":
        return (
            "        { float _mx = -INFINITY;\n"
            "          for (int n = 0; n < N; ++n) if (row[n] > _mx) _mx = row[n];\n"
            "          float _sm = 0.0f;\n"
            "          for (int n = 0; n < N; ++n) { row[n] = expf(row[n]-_mx); _sm += row[n]; }\n"
            "          for (int n = 0; n < N; ++n) row[n] = row[n]/_sm; }\n"
        )
    if name == "layer_norm":
        return (
            "        { float _mean = 0.0f;\n"
            "          for (int n = 0; n < N; ++n) _mean += row[n];\n"
            "          _mean /= (float)N;\n"
            "          float _var = 0.0f;\n"
            "          for (int n = 0; n < N; ++n) { float _d = row[n]-_mean; _var += _d*_d; }\n"
            f"          float _inv = 1.0f/sqrtf(_var/(float)N + {eps!r}f);\n"
            "          for (int n = 0; n < N; ++n) row[n] = (row[n]-_mean)*_inv; }\n"
        )
    raise ValueError(f"no scalar snippet for reduction op {name!r}")


def row_compute_body(region: Any) -> str:
    """The per-row compute body for row ``m`` (``float* row = out + m*N`` in scope):
    matmul + prologue(A) + epilogue chain + optional residual + optional row
    reduction. Shared verbatim by the x86 C function and the ROCm HIP kernel."""
    prologue = "".join(
        f"                {pointwise_snippet(op, 'a')}\n" for op in region.prologue
    )
    epi_lines = []
    for op in region.epilogue:
        if op == "bias":
            epi_lines.append("            v = v + bias[n];")
        else:
            epi_lines.append(f"            {pointwise_snippet(op, 'v')}")
    epilogue = "\n".join(epi_lines)
    residual = ("            v = v + residual[(long)m*N + n];\n"
                if region.has_residual else "")
    reduction = (reduction_snippet(region.reduction, region.eps)
                 if region.reduction else "")
    return (
        "        for (int n = 0; n < N; ++n) {\n"
        "            float v = 0.0f;\n"
        "            for (int k = 0; k < K; ++k) {\n"
        "                float a = A[(long)m*K + k];\n"
        f"{prologue}"
        "                v += a * B[(long)k*N + n];\n"
        "            }\n"
        f"{epilogue}\n"
        f"{residual}"
        "            row[n] = v;\n"
        "        }\n"
        f"{reduction}"
    )
