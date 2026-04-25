[MERGE_START: TPP_Syntax_Spec.md]

# Tessera Tensor Processing Primitives (TPP) — Syntax & Semantics (v0.1)
Status: Draft (compile-ready stubs included in `/tpp/`)

This document is the **normative** BNF-inspired syntax for the `tessera.tpp` dialect, followed by
well-defined semantics and lowering notes. See `/tpp/docs/TPP_Space_Time_Guide.md` for tutorials.

---

## 1. Design Goals
- Space–time first-class (meshes, grids, time) for weather, world models, physics.
- Tile-first lowering with halos, async prefetch, and distributed exchanges.
- Physics-aware verifiers (units, conservation) and solver-friendly ops.
- Deterministic parallel RNG + ensemble constructs.

---

## 2. Lexical Conventions
- **Identifiers**: `[A-Za-z_][A-Za-z0-9_./-]*`
- **Literals**:
  - integers: `[-+]?[0-9]+`
  - floats: `[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)? | [-+]?[0-9]+([eE][-+]?[0-9]+)`
  - strings: `"(?:[^"\\]|\\.)*"`
- **Attr dictionaries**: `{ key = value, ... }`
- **Types** are `!tpp.*`, **attributes** are `#tpp.*`.

---

## 3. Core Types (BNF)
```
<type> ::= !tpp.field "<" <elem> "," <layout> "," <space> ["," <time>] ">"
        |  !tpp.mesh "<" <topology> ["," <coords>] ["," <levels>] ">"
        |  !tpp.ensemble "<" "count" ":" <int> ["," "seed" ":" <int>] ["," <perturb>] ">"
        |  !tpp.handle "<" <string> ">"
        |  !tpp.token
        |  <builtin-type>                    // f32, memref<...>, tensor<...>, index, etc.

<elem>      ::= "f16" | "bf16" | "f32" | "f64" | "i8" | "i16" | "i32" | "i64"
<layout>    ::= "nhwc" | "nchw" | "xyz" | "yxz" | "blocked(" <ints> ")"
<space>     ::= "grid(" <grid_kind> ["," <kvpairs>] ")"
<grid_kind> ::= "latlon" | "cubed_sphere" | "ico" | "cartesian"
<coords>    ::= "coord:" ( "latlon" | "gnomonic" | "cartesian" )
<levels>    ::= "levels:" <int>
<perturb>   ::= "perturb:" <dist-spec>
<dist-spec> ::= "gaussian(" "sigma" "=" <float> ["," "mean" "=" <float>] ")"
              | "lognormal(" "sigma" "=" <float> ["," "mu" "=" <float>] ")"
<time>      ::= "time(" "dt" "=" <float> ["," "unit" "=" <string>] ")"
<ints>      ::= <int> { "," <int> }*
<kvpairs>   ::= <kvpair> { "," <kvpair> }*
<kvpair>    ::= <ident> "=" ( <int> | <float> | <string> )
```

**Examples**
```
!tpp.field<f32, nhwc, grid(latlon, res_km=25), time(dt=300.0, unit="s")>
!tpp.mesh<icosahedral, coord:latlon, levels:137>
!tpp.ensemble<count:50, seed:1337, perturb:gaussian(sigma=0.1)>
```

---

## 4. Attributes (BNF)
```
<attr> ::= #tpp.units "<" <string> ">"
         | #tpp.bc "<" <bc-kind> { "," <face> "=" <bc-kind> }* ">"
         | #tpp.conservation "<" <cons-kind> ">"
         | #tpp.space_time "{" "blocking" ":" "(" <ints> ")" 
                            ["," "halo" ":" "(" <ints> ")"]
                            ["," "sliding" ":" "time(" "k" ")" ] "}"
<bc-kind>    ::= "periodic" | "dirichlet" | "neumann" | "radiation" | "clamp"
<face>       ::= "N" | "S" | "E" | "W" | "top" | "bot"
<cons-kind>  ::= "mass" | "energy" | "momentum"
```

**Intent**: enable compile-time dimensional checks and codegen hints for halo and blocking.

---

## 5. Operations (BNF)

### 5.1 Field/Boundary
```
tpp.bc.enforce(%field : <type>) { bc = #tpp.bc<...> } : (<tpp.field>) -> (<tpp.field>)
```

### 5.2 Differential & Stencil
```
tpp.grad      (%x : <tpp.field>) { axis = <axis>, scheme = <scheme>, order = <int>, halo = "(" <ints> ")" }
             : (<tpp.field>) -> (<tpp.field>)

tpp.div       (%ux, %uy [,%uz] : <tpp.field>...) { scheme=<scheme>, order=<int>, halo="(" <ints> ")" }
             : (<tpp.field>...) -> (<tpp.field>)

tpp.stencil.apply (%x : <tpp.field>, %k : tensor<...>) { axis = <axis>?, order=<int>?, halo="(" <ints> ")"? }
             : (<tpp.field>, tensor<...>) -> (<tpp.field>)

<axis>   ::= "x" | "y" | "z" | "xy" | "xyz"
<scheme> ::= "central" | "upwind" | "weno" | "eno"
```

### 5.3 Multigrid & Smoothing
```
tpp.restrict (%x : <tpp.field>) { ratio = "(" <ints> ")" } : (<tpp.field>) -> (<tpp.field>)
tpp.prolong  (%x : <tpp.field>) { ratio = "(" <ints> ")" } : (<tpp.field>) -> (<tpp.field>)
tpp.smooth   (%x : <tpp.field>, %b : <tpp.field>) { method = <smethod>, iters = <int> }
            : (<tpp.field>, <tpp.field>) -> (<tpp.field>)

<smethod> ::= "jacobi" | "gauss_seidel" | "chebyshev"
```

### 5.4 Temporal
```
tpp.time.step (%state : tuple<...>) { scheme = <time_scheme>, dt = <float> }
  { ... region ... }
  : (<types>) -> (<types>)

<time_scheme> ::= "RK2" | "RK4" | "semi_lagrangian" | "BDF2"

tpp.time.convolution (%x, %k : tensor<...>) { causal = <bool>, radius = <int> }
             : (tensor<...>, tensor<...>) -> (tensor<...>)

tpp.time.rolling (%x : tensor<...>) { window = <int>, reduce = <reduce_kind> }
             : (tensor<...>) -> (tensor<...>)

<reduce_kind> ::= "mean" | "max" | "var" | "sum"
```

### 5.5 Spectral / Spherical
```
tpp.fft.plan () { shape = "(" <ints> ")", type = <fft_type>, real = <bool> } : () -> (!tpp.handle<"fft.plan">)
tpp.fft.exec (%plan, %x) : (!tpp.handle<"fft.plan">, tensor<...>) -> (tensor<...>)

tpp.sht.fwd (%x : tensor<...>) { lmax = <int>, mmax = <int>, layout = <layout> } : ... -> ...
tpp.sht.inv (%X : tensor<...>) { lmax = <int>, mmax = <int>, layout = <layout> } : ... -> ...
<fft_type> ::= "c2c" | "r2c" | "c2r" | "dct" | "dst"
```

### 5.6 Solvers & Constraints
```
tpp.pde.solve (%A, %b) { method = <lin>, precond = <pc>?, tol = <float>, iters = <int>? }
  : (tensor<...>, tensor<...>) -> (tensor<...>)

<lin> ::= "CG" | "BiCGStab" | "GMRES"
<pc>  ::= "ILU" | "Jacobi" | "AMG"

tpp.nl.solve (%F, %x0) { method = <nlin>, line_search = <ls>? } : (...) -> (...)
<nlin> ::= "Newton" | "LBFGS"
<ls>   ::= "wolfe" | "armijo"

tpp.project (%x) { conserve = #tpp.conservation<...>, domain = !tpp.mesh<...> } : (...) -> (...)
tpp.enforce.positivity (%x) { eps = <float> } : (...) -> (...)
```

### 5.7 World-Model Primitives
```
tpp.slot_attention (%x) { slots = <int>, iters = <int> } : (tensor<...>) -> (tensor<...>, tensor<...>)
tpp.memory.bank   (%x) { size = <int>, update = <update_kind> } : (tensor<...>) -> (!tpp.handle<"mem">)
<update_kind> ::= "ema(beta=<float>)" | "knn(K=<int>)"

tpp.coord.warp (%x, %flow) { mode = <warp_mode>, boundary = <bc> } : (...) -> (...)
<warp_mode> ::= "backwarp" | "bilinear"
```

### 5.8 RNG & Ensembles
```
tpp.rng.counter (%ctr, %key) { algo = <rng_algo> } : (i64, i64) -> (tensor<...>)
<rng_algo> ::= "philox" | "threefry"

tpp.quasi.random (%idx) { seq = <seq_kind>, dims = <int>, scramble = <scramble>? } : (i64) -> (tensor<...>)
<seq_kind>  ::= "sobol" | "halton"
<scramble>  ::= "owen"

tpp.ensemble.map    (%E, ^bb(%θ): ...) : (!tpp.ensemble<...>, ...) -> tensor<...>
tpp.ensemble.reduce (%Ys) { op = <reduce_kind> | "quantile(q=<float>)" } : (...) -> (...)
```

### 5.9 IO
```
tpp.io.read (%path) { fmt = <fmt>, fields = "[" <ids> "]", chunk = "(" <ints> ")" }
  : (memref<...>) -> (!tpp.handle<"dataset">)
<fmt> ::= "netcdf" | "zarr" | "grib2"
<ids> ::= <ident> { "," <ident> }*
```

---

## 6. Verifiers (Normative)
- **Units**: ops must match `#tpp.units`; incompatible dimensions are illegal.
- **Halo**: `tpp.grad/stencil/div` require halo ≥ stencil radius.
- **BC**: `tpp.bc.enforce` must reference valid faces for the mesh topology.
- **Solvers**: `tpp.pde.solve` requires square A; tolerance `tol>0`.
- **Ensemble**: `count>0`; RNG shapes consistent with outputs.

---

## 7. Lowering Overview (Informative)
1. `-tpp-legalize-space-time`: normalize types/attrs; annotate halos.
2. `-tpp-halo-infer`: compute `(hx,hy,hz)` from stencils/BC.
3. `-tpp-fuse-stencil-time`: fuse spatial loops within `tpp.time.step` region.
4. `-tpp-async-prefetch`: emit cp.async/TMA (NVIDIA) or ds_read streams (AMD).
5. `-tpp-vectorize`: map to WGMMA/MFMA/AMX tiles where applicable.
6. `-tpp-distribute-halo`: insert async sends/recvs overlapping compute.
7. `-lower-tpp-to-target-ir`: lower to Tessera Target-IR → backends.

[MERGE_END: TPP_Syntax_Spec.md]
