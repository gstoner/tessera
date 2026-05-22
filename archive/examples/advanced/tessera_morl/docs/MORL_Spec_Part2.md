## Scalarization
**Linear:** `S_lin(x; w) = ⟨w, x⟩`  
**(ε-)Tchebycheff:** `S_tch(x; w, z*) = - max_m w_m ⋅ |x_m - z*_m| - ε ⋅ Σ_m w_m |x_m - z*_m|`

```tessera
// morl/scalarize.tsr
package morl

@kernel scalarize_linear(x: tensor<*xMxf32>, w: tensor<Mxf32>, out: tensor<*xf32>) {
  parallel_for i in 0..N {
    s = 0.0
    for m in 0..M { s += x[i,m] * w[m] }
    out[i] = s
  }
}

@kernel scalarize_tchebycheff(x: tensor<*xMxf32>, w: tensor<Mxf32>, z: tensor<Mxf32>,
                              eps: f32, out: tensor<*xf32>) {
  parallel_for i in 0..N {
    maxv = -INF; sumv = 0.0
    for m in 0..M {
      d = abs(x[i,m] - z[m])
      term = w[m] * d
      maxv = max(maxv, term)
      sumv += term
    }
    out[i] = -(maxv + eps * sumv)
  }
}
```

## PCGrad (pairwise)
Given per-objective gradients `{g_m}`, project `g_i` onto the orthogonal complement of `g_j` if `⟨g_i, g_j⟩ < 0`.
```tessera
// morl/pcgrad.tsr
package morl

@kernel pcgrad_pairwise(grads: tensor<MxDxf32>, out: tensor<Dxf32>) {
  // grads[m, d]
  g = zeros<D>()
  for m in 0..M {
    g_m = grads[m,:]
    for n in 0..M {
      if (n == m) continue
      dot = dot(grads[m,:], grads[n,:])
      if (dot < 0) {
        proj = dot / (norm2(grads[n,:]) + 1e-8)
        g_m = g_m - proj * grads[n,:]
      }
    }
    g += g_m
  }
  out[:] = g / M
}
```
<<MERGE_END: MORL_Spec>>
