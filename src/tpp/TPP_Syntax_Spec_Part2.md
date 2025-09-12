[MERGE_START: TPP_Syntax_Spec.md]

## 8. Examples (Normative IR Sketches)

### 8.1 Shallow-Water Step (periodic)
```mlir
// Types
%mesh = arith.constant 0 : index  // placeholder
%u : !tpp.field<f32, nhwc, grid(latlon, res_km=25), time(dt=300.0, unit="s")>

// Step
%u1, %v1, %h1 = tpp.time.step %state {
  %Hx = tpp.grad %h { axis = "x", scheme = "central", order = 2, halo = (1,0) }
  %Hy = tpp.grad %h { axis = "y", scheme = "central", order = 2, halo = (0,1) }
  %u2 = arith.subf %u, arith.mulf(%g, %Hx) : f32
  %v2 = arith.subf %v, arith.mulf(%g, %Hy) : f32
  %u3 = tpp.bc.enforce %u2 { bc = #tpp.bc<periodic> }
  %v3 = tpp.bc.enforce %v2 { bc = #tpp.bc<periodic> }
  %h2 = tpp.bc.enforce %h  { bc = #tpp.bc<periodic> }
  tpp.yield %u3, %v3, %h2 : !tpp.field<...>, !tpp.field<...>, !tpp.field<...>
} { scheme = RK4, dt = 300.0 }
```

### 8.2 Ensemble Map/Reduce
```mlir
%Ys = tpp.ensemble.map %Ens (%θ: tensor<...>) {
  %y = tpp.physics.step %θ { steps = 24 } : tensor<...> -> tensor<...>
  tpp.yield %y : tensor<...>
}
%mean = tpp.ensemble.reduce %Ys { op = mean }
%p90  = tpp.ensemble.reduce %Ys { op = "quantile(q=0.9)" }
```

### 8.3 RNG & Quasi-Random
```mlir
%noise = tpp.rng.counter %ctr, %key { algo = philox } : (i64, i64) -> tensor<1024xf32>
%sobol = tpp.quasi.random %i { seq = sobol, dims = 8, scramble = owen } : (i64) -> tensor<8xf32>
```

---

## 9. Pass Pipeline (Alias)
```
-tpp-space-time = (
  -tessera-normalize,
  -tpp-legalize-space-time,
  -tpp-halo-infer,
  -tpp-fuse-stencil-time,
  -tpp-async-prefetch,
  -tpp-vectorize,
  -tpp-distribute-halo,
  -canonicalize, -cse,
  -lower-tpp-to-target-ir
)
```

---

## 10. Mapping to Target-IR (Guidance)
- **NVIDIA**: cp.async.bulk.tensor / TMA + mbarrier; WGMMA for fused matvecs.
- **AMD**: LDS double-buffering + `ds_read_b128` streams; MFMA tiles.
- **CPU**: AVX-512/AMX; cache-blocked halos + prefetch.

---

## 11. Autodiff Notes (JVP/VJP)
- Provide JVP for all first-order primitives (`grad/div/stencil.apply`, `time.step`).
- For `pde.solve` (iterative), supply implicit differentiation via fixed-point (optional unroll).

---

## 12. Conformance
- LIT tests included in `/tpp/test/` validate: halo inference, BC enforcement, fusion, RNG determinism, and pipeline alias.

---

## 13. Change Log
- v0.1: initial types/attrs, key ops, pass stubs, tests, docs scaffold.

[MERGE_END: TPP_Syntax_Spec.md]
