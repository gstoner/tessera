---
title: Full Code Review — 2026-08-29 (logic · mathematical correctness · algorithms · performance)
last_updated: 2026-08-29
scope: Python numeric core (autodiff · RNG/quantization · losses/optimizers · fusion + four backend emitters) · MLIR passes (linalg lowering · autodiff · tiling · analysis · legality · solver dialects)
status: 102 findings confirmed by an independent refutation pass; all four severity tiers have fixes committed, with 3 P3 rows still owing device verification
audit_role: snapshot
---

# Tessera full code review — logic, mathematical correctness, algorithms, performance

> **Status.** All four tiers have fixes committed: P0 (9), P1 (36), P2 (42), P3 (16).
> Three P3 rows are fixed in source but **not device-verified** — see that section.
> Route work through
> [`README.md`](README.md)'s authority chain; the generated dashboards in
> `../generated/` remain the primary status evidence (Decision #26).
>
> **A source-level fix is not a device claim.** Every batch was written and
> tested on hosts that do not have all four targets, so the per-backend todos
> carry what each queue still owes. Read those before treating a row here as
> proven on your hardware.
>
> | Severity | Closed in | Outcome |
> |---|---|---|
> | P0 · 9 | PR #635 | all 9 fixed |
> | P1 · 36 | PRs #636, #637, #638, the P3 batch, and the deferred-P1 batch | all 36 fixed. The last 2 — Adafactor bias correction and the rank-4 dropout per-instance seed — were deferred 2026-08-29 with ABI reasoning that **did not survive re-examination**: neither needed a kernel ABI change. Both fixed 2026-08-30; see their rows. The dropout row's severity was also corrected downward (the executing gfx1151 lane was already iid across batch/head; the defect was in the Tile-IR contract). The rank-4 dropout fix is **not yet lit-verified on the ROCm box** |
> | P2 · 42 | PR #640 | 39 fixed; 3 needed no change — two had already been fixed by the P0 batch and the review quoted the pre-fix body, one (`AutodiffPass.cpp:199`) was closed by the P0 seed-type fix |
> | P3 · 16 | the P3 batch | all 16 fixed in source; 13 host-measured, **3 device-unverified** (see the P3 section); 2 of the 16 were correctness defects, not improvements |
>
> The three P2 rows that needed no change are worth keeping visible rather than
> deleting: they are the cost of reviewing a moving tree, and each was
> established by re-running the finding's own reproduction, not by assuming.
>
> Findings were produced by review agents and each was then re-checked by an
> independent agent instructed to refute it; only what survived is listed as
> confirmed. The refuted section is kept because the disproofs are useful —
> most are cases where an ODS verifier already rejects the input a reviewer
> assumed reachable.


Generated 2026-08-28. 17 review units, each swept across four dimensions by a dedicated agent, then every finding adversarially re-checked by a second agent that tried to refute it.

**121 raw findings → 102 confirmed, 17 refuted, 2 uncertain.**

Confirmed by severity: P0 9 · P1 36 · P2 41 · P3 16.  
By dimension: logic 56 · math 24 · algorithm 7 · performance 15.


## P0 — wrong result or crash on plausible mainstream input (9)

### `python/tessera/autodiff/vjp.py:3741` — group_norm VJP crashes when affine weight present

*Autodiff — reverse mode (VJP) · logic*

**What is wrong:** vjp_group_norm reshapes the per-channel weight (size C) to grouped.shape (size N*C*prod(spatial)) via w.reshape(grouped.shape), which raises ValueError for any real input, so every backward through an affine GroupNorm crashes.

**Evidence:** Reproduced: x shape (2,4,3,3), num_groups=2, weight shape (4,) -> the _VJPS entry for 'group_norm', called with (dout, x, 2, w), raises 'ValueError: cannot reshape array of size 4 into shape (2,2,2,3,3)'. Affine weight is the default in nn.GroupNorm-style usage; the only non-crashing case is N==1 with no spatial dims.

**Fix:** Reshape the weight to (1, num_groups, C//num_groups, *[1]*n_spatial) and let broadcasting apply it: do_grouped = do.reshape(grouped.shape) * w.reshape(1, num_groups, c//num_groups, *([1]*(x_arr.ndim-2))).

**Independently verified:** Reproduced on current source: the _VJPS entry for 'group_norm', called with (dout, x of shape (2,4,3,3), 2, w of shape (4,)), raises ValueError('cannot reshape array of size 4 into shape (2,2,2,3,3)') at vjp.py:3741. The forward (nn/functional.py group_norm) applies a per-channel weight, so the path is reachable; no guard prevents it.

### `python/tessera/autodiff/tape.py:271` — Aliased op output doubles all upstream gradients

*Autodiff — tape / grad / transforms · logic*

**What is wrong:** Tape.backward has no guard for an entry whose output object IS one of its inputs (entry.output_id == desc.array_id). The accumulation `cotan[desc.array_id] = cotan[desc.array_id] + g` reads dout from a key and adds the passthrough cotangent back into the same key, so every upstream gradient is doubled — silently, with no diagnostic. Real ops do return their input object: `ops.clamp(x)` with both bounds None returns `x` itself (python/tessera/__init__.py:4279-4280), a realistic path when clamp bounds are config-driven and happen to be unset.

**Evidence:** Reproduced on this repo: f(x) = reduce(clamp(mul(x, 2.0)), 'sum'); grad(f)(np.array([1.,2.])) returns [4., 4.] where the analytic gradient is [2., 2.] (control without the clamp returns [2., 2.]). Mechanism: mul's output y and clamp's output are the same object, so three tape entries share output_id/array_id id(y); the clamp entry reads cotan[id(y)]=[1,1], its VJP returns [1,1], and accumulation writes cotan[id(y)]=[2,2], which the mul entry then consumes.

**Fix:** In the accumulation loop, detect desc.array_id == entry.output_id and replace (not add to) the cotangent for that key, or fail closed with a diagnostic naming the op; alternatively forbid identity-returning ops by copying at the ops layer, but the tape must not silently double either way.

**Independently verified:** Independently reproduced. `ops.clamp(y)` with both bounds None returns the input object verbatim (python/tessera/__init__.py, `clamp`: `if min is None and max is None: return a`), so the clamp TapeEntry has `output_id == inputs[0].array_id`. Instrumented tape dump: mul out=4862599664 / clamp out=4862599664 ins=[4862599664], and the final cotangent map shows cotan[id(y)] = [2.,2.] where it must be [1.,1.], giving grad = [4.,4.] instead of the analytic [2.,2.] (control without the clamp = [2.,2.]). Mechanism is exactly as claimed: line 271 `cotan[desc.array_id] = cotan[desc.array_id] + g` re-adds the passthrough cotangent into the same key that `dout` was read from at line 202. No guard exists anywhere in Tape.backward (`grep` for the aliasing check finds only the unrelated target-on-tape scan at line 149), and no test pins this. Silent — no diagnostic.

### `python/tessera/quantization.py:40` — Asymmetric auto-scale never derives zero_point

*RNG & quantization · math*

**What is wrong:** In quantize_int8 (and identically quantize_int4, line 58), the asymmetric path derives scale from the data range (max-min)/255 when scale is None, but leaves zero_point at the caller default 0. The affine map q = round(x/s + zp) then only covers [-128*s, 127*s] around zero, so any tensor whose range is not centered at 0 is silently saturated - the exact case (non-negative activations) asymmetric quantization exists for.

**Evidence:** quantize_int8(np.array([0.0, 10.0]), symmetric=False) gives s = 10/255 = 0.0392, q = round(x/s + 0) = [0, 255] -> clip(-128,127) -> [0, 127]; dequantize_int8 returns [0.0, 4.98] instead of [0.0, 10.0] - a 2x error on the max element and half the int8 code space wasted. Any ReLU/softmax-output tensor hits this.

**Fix:** When scale is None and symmetric=False, also derive zero_point = int(clip(round(-128 - x_min/s), -128, 127)) (and -8-based for int4); or fail closed with a diagnostic if the caller supplies neither scale nor zero_point, per Decision #21a.

**Independently verified:** Traced quantization.py:40-44 (and the identical int4 path at 58-62) and reproduced it: quantize_int8(np.array([0.0,10.0]), symmetric=False) returns q=[0,127], scale=0.0392, zero_point=0, and dequantize_int8 gives [0.0, 4.98] instead of [0.0, 10.0]; quantize_int8([5.0,10.0], symmetric=False) is worse - both elements saturate to 127 and dequant to [2.49, 2.49], destroying all information. The derived scale assumes the full qmin..qmax span (range/255) while zero_point stays at the caller default 0, so only the [-128*s, 127*s] window around zero is representable. No caller contract prevents this: zero_point has a default, fake_quantize(x, symmetric=False) forwards the same defaults, and both functions are exported from tessera/__init__.py. No test pins the asymmetric auto-scale path (test_s7_s8_s9.py only exercises symmetric qparams). Notably the compiled lane reproduces the same bug - runtime.py::_intquant_params derives the asymmetric scale from (max-min)/(qmax-qmin) but returns zero_point straight from kwargs (default 0) - so a reference-vs-compiled differential test would agree while both are wrong; that is corroboration, not a refutation.

### `python/tessera/compiler/fusion_core.py:1460` — Attention discovery silently drops non-constant multiply

*Fusion core & emitter seams · logic*

**What is wrong:** discover_attention_regions consumes a scale-position mul op into the fused region unconditionally, but only reads its factor from a constant attr. A mul whose factor is a graph value (tensor scale, mask multiply) leaves region.scale=1.0 while the op is still appended to `chain` and marked consumed — the multiply's semantics are discarded and the fused kernel computes softmax(Q.K^T).V with no scaling.

**Evidence:** Lines 1456-1462: `if ops[j].name in _SCALE_NAMES:` then `factor = attrs.get('scale', attrs.get('factor'))`; when factor is not an int/float, scale stays 1.0 yet `chain.append(j); consumed.add(j)` still run. Consumer runtime.py:33830-33850 executes the region and `consumed.update(chain_idx)`, so the mul op never executes. The F4 oracle cannot catch it: verify_synthesized_attention compares the kernel to region.reference, both built from the same scale=1.0 region. Input: graph `s=matmul(Q,K,transpose_b=True); s2=mul(s, scale_tensor); p=softmax(s2); o=matmul(p,V)` where the 1/sqrt(d) scale is a value rather than an attr — output is wrong by exp-scale distortion of every softmax row, silently. Violates Decision #21a (semantic key silently defaulted).

**Fix:** Only append/consume the mul when the factor is a recognized scalar constant; otherwise `continue` (abandon the region) so the multiply executes unfused.

**Independently verified:** Reproduced end-to-end. fusion_core.py:1456-1462 appends/consumes the scale-position op unconditionally; only the attr read is guarded (`isinstance(factor,(int,float))`). The tracer emits `mul` as a *binary* op with two operands and no `scale` kwarg (trace.py:731, _SHAPE_RULES 'mul': _broadcast_shape), so the value-factor form is the normal traced form, not an exotic one. Direct run: discover_attention_regions([matmul(Q,K,transpose_b),mul(s,sc),softmax,matmul(p,V)]) -> chain [0,1,2,3] with scale=1.0. Feeding the same op list to runtime._apple_gpu_try_synthesized_fusion on this Mac returned consumed={0,1,2,3} and the produced value matched the UNSCALED reference (allclose True) and NOT the 0.125-scaled reference (allclose False). No guard, caller contract, or test prevents it; the F4 oracle cannot catch it because region.reference is built from the same scale=1.0 region. Silent wrong numerics, Decision #21a-class.

### `python/tessera/compiler/emit/rocm_hip.py:282` — Missing barrier race in paged-attention softmax

*Emitter — Apple MSL / ROCm HIP / x86 · logic*

**What is wrong:** In the generated HIP paged_attn kernel, after the max tree-reduction every thread reads `m=red[0]` and then immediately overwrites `red[t]=z` (its partial exp-sum) with no __syncthreads() in between. Thread 0 can write red[0]=z before a slower thread has read the max, so that thread uses a partial sum as the softmax max — a classic shared-memory read/overwrite data race producing wrong attention output.

**Evidence:** Kernel text: `...red[t]=fmaxf(red[t],red[t+s]);__syncthreads();}}m=red[0];float z=0.f;for(int j=t;j<T;j+=256)z+=expf(scores[j]-m);red[t]=z;__syncthreads();`. blockDim is 256 = 8 independent wave32s on gfx1151; for decode-sized T (T<=256) the exp loop is at most one iteration, so thread 0 reaches `red[0]=z` almost immediately while other waves may not yet have loaded red[0]. Contrast the same construct done correctly in this repo: apple_msl.py _TILED_REDUCTIONS["softmax"] inserts `threadgroup_barrier` between `float _mx = tg_red[0];` and the subsequent tg_red overwrite (apple_msl.py:248-253). Every earlier red[] reuse in the same HIP kernel is separated by a barrier; only this one is not.

**Fix:** Insert `__syncthreads();` immediately after `m=red[0];` (before `red[t]=z`) in _synthesize_paged_attention_direct_hip.

**Independently verified:** Traced the emitted kernel text in _synthesize_paged_attention_direct_hip (rocm_hip.py:282). Sequence is: max tree-reduce ... `__syncthreads();}} m=red[0]; float z=0.f; for(int j=t;j<T;j+=256) z+=expf(scores[j]-m); red[t]=z; __syncthreads();`. Between the max reduction's last barrier and `red[t]=z` there is no barrier, so thread 0's write to red[0] races with the reads of red[0] by threads 1..255 (the only conflicting address; red[t>0] is not read until after the next barrier). blockDim=256 spans 8 independent wave32s on gfx1151, and nothing orders them after a barrier. The other two red[] reuses in the same kernel ARE barrier-separated (the per-j score reduce ends with `if(t==0)scores[j]=...; __syncthreads();` before the next `red[t]=x`), so this one is the outlier, and the corroborating construct in apple_msl.py _TILED_REDUCTIONS["softmax"] does insert `threadgroup_barrier` right after `float _mx = tg_red[0];`. Consequence is real: a thread that reads the clobbered red[0] gets a partial exp-sum as its softmax max, and m is used per-thread again in the output loop (`expf(scores[j]-m)/z`), so that thread's output columns are scaled wrong. Kernel is on a live path (paged_kv.py::_paged_attention_rocm). No guard, no test pins this. Caveat: whether the race actually manifests on gfx1151 needs device execution — the defect (missing __syncthreads) is provable from source, the manifestation is not.

### `src/transforms/lib/AutodiffPass.cpp:187` — Null ShapedType deref crashes on scalar loss

*C++ — MLIR autodiff passes · logic*

**What is wrong:** `auto shapedType = dyn_cast<ShapedType>(lossValue.getType()); shapedType.hasStaticShape()` calls a method on a null type interface when the loss is a plain scalar (f32/f64), which is undefined behavior/segfault. The `else if (!isa<ShapedType>...)` branch at line 198 that was written to handle scalars is unreachable.

**Evidence:** The pass's own precondition (line 158) demands 'a single scalar return (the loss seed)'. A function returning a bare `f32` loss makes `dyn_cast<ShapedType>` return a null interface; `hasStaticShape()` then dispatches through a null impl pointer. Only a rank-0 `tensor<f32>` loss avoids the crash. On the fleet's NDEBUG MLIR builds this is silent UB rather than an assert.

**Fix:** Guard with `shapedType && shapedType.hasStaticShape()` so scalar losses reach the non-shaped branch.

**Independently verified:** Reproduced empirically. `func @scalar_loss(%a: f32, %b: f32) -> f32 attributes {tessera.autodiff = "reverse"}` run through `./build/tools/tessera-opt/tessera-opt --tessera-autodiff` segfaults (exit 139) with the stack frame `#3 mlir::ShapedType::hasStaticShape() const` called directly from `AutodiffPass::runOnOperation()`. `dyn_cast<ShapedType>` on a bare `f32` yields a null interface and `hasStaticShape()` dispatches `(*this).hasRank()` through the null `Concept *impl` (mlir/IR/BuiltinTypeInterfaces.h.inc:648). Nothing guards it: `runOnOperation` (lines 133-174) validates only the marker attribute and the single-operand return — unlike AutodiffForwardPass, it applies no argument/result type admission. The `else if (!isa<ShapedType>...)` branch at line 198 is therefore dead code, as claimed. Fix `shapedType && shapedType.hasStaticShape()` is correct.

### `src/transforms/lib/TileToX86Pass.cpp:379` — fused_epilogue silently drops activation

*C++ — tiling & Tile IR lowering · logic*

**What is wrong:** LowerFusedEpilogueToX86 applies the epilogue only inside `if (hasBias)` and only when the bias type is static, and maps epilogueKind==1 (Relu) to `tessera_x86_epilogue_bias_fp32` which applies no activation. A fused_epilogue with has_bias=false, a dynamic bias dim, or epilogue=Relu is replaced by the raw GEMM result (or bias-only result) with the activation silently discarded — no diagnostic, violating the repo's Decision #21/#21a fail-closed rule for semantic attributes.

**Evidence:** Input: `tessera.fused_epilogue %A, %B, %bias {epilogue = 2 /*Gelu*/, has_bias = false}` on 64x64 bf16 tensors. hasBias is false (line 377-378), so the entire epilogue block (lines 379-399) is skipped; the op is replaced at line 405 by the plain GEMM output — GELU is never applied, every negative element is wrong (e.g. input -3.0 should become ~-0.004, stays -3.0). Same silent loss for `{epilogue = 1 /*Relu*/, has_bias = true}`: line 389 routes Relu to bias_fp32, so negative outputs are not clamped to 0.

**Fix:** Emit a stable diagnostic and signalPassFailure/notifyMatchFailure for any epilogue configuration without a matching C symbol (Relu, no-bias activation, dynamic bias), instead of falling through to the bare GEMM.

**Independently verified:** Traced and stands. src/transforms/lib/TileToX86Pass.cpp:376-405 emits the epilogue call only inside `if (hasBias)` and only for a static bias dim, and line 387-390 routes everything except epilogueKind==2 to `tessera_x86_epilogue_bias_fp32`. I read that C symbol (src/compiler/codegen/tessera_x86_backend/src/kernels/epilogue.cpp:7-14): it does `C[i*N+j] += bias[j]` and applies no activation, so Relu(1) and Silu(3) are silently dropped, and with has_bias=false the whole block is skipped and line 405 replaces the op with the bare GEMM. The ODS admits all of these (TesseraOps.td:55-67 EpilogueKind {None,Relu,Gelu,Silu}; :1509-1519 `has_bias` DefaultValuedAttr false), and I confirmed the pass reads `epilogue` as a plain IntegerAttr (tessera-opt accepted `{epilogue = 2 : i32}` on the registered op). No diagnostic anywhere on the drop path — contrary to Decision #21/#21a. Two corrections to the reviewer's framing, neither of which saves the code: (a) has_bias=false is not free-form — FusedEpilogueOp::verify (TesseraOps.cpp:2820-2827) requires a 0-element bias operand, so the input is `tensor<0xf32>` + has_bias=false, still 3 operands, still matched, still silently dropped; (b) the only in-tree producer of the op, FuseMatmulBiasGELU (CanonicalizeTesseraIR.cpp:45-50), always emits epilogue=2/has_bias=true, so today the wrong paths are latent rather than live — this is a fail-open acceptance bug, not a currently-firing P0.

### `src/transforms/lib/TileToX86Pass.cpp:231` — f16 matmul routed to bf16 kernel

*C++ — tiling & Tile IR lowering · logic*

**What is wrong:** LowerMatmulToX86 (and LowerFusedEpilogueToX86, line 323) accepts f16 element types but unconditionally calls `tessera_x86_{amx,avx512}_gemm_bf16`, whose ABI (aPtr,bPtr,cPtr,M,N,K,beta) carries no dtype flag. The f16 bit pattern is reinterpreted as bf16 (different exponent/mantissa split: 5/10 vs 8/7), producing garbage on every element. Additionally the rhs element type is never checked against lhs (line 226-231 tests only lhsElem), so a mixed-dtype matmul is admitted into the same pattern.

**Evidence:** Input: `tessera.matmul %A, %B : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf32>`. f16 value 1.0 = 0x3C00; read as bf16, 0x3C00 = 2^-7 * 1.5 ≈ 0.01172. The GEMM computes with every operand off by ~2 orders of magnitude — silently, since all IR types are self-consistent f16 so no verifier catches it.

**Fix:** Restrict the pattern to isBF16() on both lhs and rhs elements (or add/dispatch an f16 kernel symbol), and check rhsTy.getElementType() == lhsElem.

**Independently verified:** Traced and stands. TileToX86Pass.cpp:229-231 explicitly admits isF16(), and lines 276-277 (and 367-368 in the fused-epilogue twin) select `tessera_x86_{amx,avx512}_gemm_bf16` unconditionally; the ABI declared at line 278-279 and in tessera/x86/target.h:34-36 is (const uint16_t*,const uint16_t*,float*,M,N,K,beta) with no dtype selector, and the kernel body decodes each uint16 as bf16 (`bits = uint32(value) << 16`, avx512_gemm_bf16.cpp:10-14). So f16 0x3C00 (=1.0) is consumed as 2^-7 = 0.0078125 — every element wrong (the reviewer's 0.01172 is slightly off, the conclusion is not). I looked for an upstream guard and found none: the `tessera-lower-to-x86` pipeline (pipeline_registry.py:369-380) contains no dtype conversion, and `tessera-compute-legalize` (DtypeLegalizePass.cpp) only stamps numeric_policy accum, never rewrites element types; IRContractLegalityPass validates numeric_policy dicts, not tensor element types. The secondary claim also holds: only lhsElem is tested and rhsMemTy is built from lhsElem (lines 252-253), and I verified the MatmulOp verifier (TesseraOps.cpp:163-192) does not require lhs/rhs element-type equality — a bf16×f16 matmul parses clean.

### `src/solvers/clifford/lib/Passes/GradeFusion.cpp:86` — GradeFusion drops projection semantics for shared products

*C++ — spectral / Clifford / TPP solvers · logic*

**What is wrong:** GradeFusionPattern replaces EVERY grade op with the raw geo_product result and stores the UNION of all consumers' grades on the product, without checking the product's other users. Any geo_product with (a) two grade consumers requesting different grades, or (b) one non-grade consumer, computes wrong values after ExpandProductTable applies the union/output restriction.

**Evidence:** IR `%p = geo_product(a,b); %s = grade(0,%p); %b2 = grade(2,%p)`: both grade ops are replaced by %p annotated output_grades={0,2}, so the grade-0 consumer receives nonzero bivector coefficients (e.g. a=b=e1: correct grade(0)=<1,0,...,0> but delivered value also carries any grade-2 terms of a general product). Conversely `use(%p); grade(2,%p)` annotates %p with output_grades={2} and ExpandProductTable zeroes all non-grade-2 coefficients that use(%p) legitimately needs. Both are silent wrong-value miscompiles on mainstream GA code (extracting scalar and bivector parts of one product is the canonical rotor idiom).

**Fix:** Only attach output_grades and fold when all users of the geo_product result are grade ops requesting the same grade set; otherwise keep the grade op (or materialize a per-consumer masked copy).

**Independently verified:** Traced GradeFusion.cpp:47-88. The pattern reads only the grade op and its defining geo_product; it never inspects src->getResult(0).getUsers(). It unions the requested grades into tessera.clifford.output_grades on the shared product and then rewriter.replaceOp(op, src->getResult(0)). ExpandProductTable.cpp:121-131,164-165 turns that union into wantGrade[] and emits every table entry whose result grade is in the union, so the single expanded tensor carries all unioned grades. CL_GradeProjectionOp is documented as 'Project a multivector onto a grade subset', so a grade-0 consumer must see zeros at grades 1..n. Case (a) is pinned as intentional by src/solvers/clifford/test/ir/passes/grade_fusion_multi_consumer.mlir (grade(0,%gp) and grade(2,%gp) both replaced by %gp with output_grades=[0,2], CHECK-NOT grade) — but that fixture's premise ('a downstream ExpandProductTable still emits the correct (joint) set') is exactly the bug: the grade-0 consumer receives nonzero bivector coefficients. Case (b) is worse and unpinned: a non-grade user of the product gets its non-requested grades zeroed by the output restriction. No guard, no user check, no verifier prevents either; RotorSandwichFold does not interpose. Real silent wrong-value miscompile on the canonical rotor idiom.


## P1 — wrong result on a realistic edge case, or large hot-path cost (36)

### `python/tessera/autodiff/vjp.py:1375` — flash_attn VJP silently ignores dropout mask

*Autodiff — reverse mode (VJP) · math*

**What is wrong:** The forward (tessera/__init__.py flash_attn) applies a Bernoulli mask scaled by 1/(1-p) to the attention weights when dropout_p>0, but the VJP's dropout branch is a bare 'pass' and differentiates the no-dropout function — even when seed is provided and the mask is exactly reproducible (as vjp_dropout already does). Gradients are silently wrong for any training run with attention dropout.

**Evidence:** Measured: Q,K,V shape (1,4,3), dropout_p=0.5, seed=7 — max relative error between the central-difference dQ (through the seeded forward) and the VJP's dQ is 0.76. No diagnostic is emitted, violating the fail-closed rule (Decision #21a / stable-diagnostic convention).

**Fix:** When dropout_p>0: if seed is given, replay the mask with np.random.default_rng(seed).binomial(1,1-p,P.shape)/(1-p) and thread it through the P/dP terms; otherwise raise the same 'not differentiable without a reproducible seed' error vjp_dropout raises.

**Independently verified:** Forward flash_attn applies a seeded reproducible mask via np.random.default_rng(seed) when dropout_p>0; the VJP's dropout branch (vjp.py:1375-1378) is a bare pass and seed is swallowed by **_. Reproduced: seeded finite-difference dQ vs analytic dQ shows 0.71 max relative error with no diagnostic. vjp_dropout (vjp.py:1127) shows the repo's own convention is seeded replay or fail-closed; flash_attn does neither. The docstring documents the limitation but that does not make the silent wrong gradient correct.

### `python/tessera/autodiff/vjp.py:413` — pad VJP wrong for reflect/edge/wrap modes

*Autodiff — reverse mode (VJP) · math*

**What is wrong:** vjp_pad ignores the mode kwarg and always applies the constant-mode adjoint (a plain interior slice). For mode='reflect'/'edge'/'wrap' — all accepted by the forward ops.pad — the cotangent mass in the padded regions must be scatter-added back to its interior source positions; slicing silently drops it.

**Evidence:** Reproduced: x=arange(5), pad_width=[(2,2)], mode='reflect', dout=ones(9): central-difference gradient is [1,2,3,2,1]; vjp_pad returns [1,1,1,1,1]. Silent no-op on an unsupported semantic path, contra the repo's fail-closed convention.

**Fix:** Either implement the adjoint per mode (scatter-add padded cotangent back through the reflection/edge/wrap index map, e.g. via np.add.at over the source-index array) or raise a stable diagnostic for mode != 'constant'.

**Independently verified:** ops.pad forwards mode='reflect'/'edge'/'wrap' to np.pad, but vjp_pad (vjp.py:403-417) ignores mode and always slices. Reproduced exactly as claimed: x=arange(5), reflect pad (2,2), dout=ones(9) gives numeric gradient [1,2,3,2,1] vs analytic [1,1,1,1,1]. Cotangent mass in padded regions is silently dropped.

### `python/tessera/autodiff/vjp.py:1792` — modified_delta_attention norm-term divisor clamped wrongly

*Autodiff — reverse mode (VJP) · math*

**What is wrong:** For delta = update/(1+||update||_F), the correct pullback is dupdate = ddelta/denom - update*(update.ddelta)/(norm*denom^2). The code divides the projection term by safe_norm = np.maximum(norm, 1.0), which clamps every norm below 1 up to 1, under-scaling the term by the factor norm whenever 0 < ||update|| < 1 (the common small-update regime). The guard should only replace norm==0.

**Evidence:** Measured on (1,1,3,2) Q/K/V drawn ~N(0,0.01) (||update||~0.01-0.1): dK max abs error 9.6e-5 against a max-gradient scale of 8.9e-3 (~1% and growing as ||update||->0), while dQ (which bypasses the term) matches to 3e-14.

**Fix:** safe_norm = np.where(norm > 0.0, norm, 1.0) (or np.maximum(norm, tiny)) so the divisor equals norm away from zero; keep the existing np.where(norm>0, ...) zero-guard.

**Independently verified:** Re-derived: for delta=u/(1+n) with n=||u||, dupdate = ddelta/(1+n) - u*(ddelta.u)/(n*(1+n)^2); the divisor must be n, and the code's safe_norm=np.maximum(norm,1.0) (vjp.py:1792) clamps every n<1 to 1. Isolated formula check: correct formula matches numerics to 8e-11 while the code's formula errs 0.07. Full-op repro at update norms 0.3-0.5: dK max abs error 0.073 against gradient scale 0.34 (~21% relative), while dQ matches to 1.5e-15. The existing np.where(norm>0) already handles the zero case, so the maximum clamp serves no purpose.

### `python/tessera/autodiff/vjp.py:4374` — nesterov velocity cotangent off by factor (1+m)/m

*Autodiff — reverse mode (VJP) · math*

**What is wrong:** For the forward the docstring itself states (new_velocity = m*v + g; look_ahead = g + m*new_velocity; new_params = p - lr*look_ahead, matching optim.nesterov), d new_params/d velocity = -lr*m^2, but the code returns d_velocity = -lr*m*(1+m)*dout — wrong by factor (1+m)/m (2.11x at m=0.9). d_grads = -lr*(1+m)*dout is correct.

**Evidence:** Measured against optim.nesterov with lr=0.1, m=0.9, dout=ones: central-difference d_velocity = [-0.0805,-0.0805] (= -lr*m^2), analytic VJP returns [-0.171,-0.171] (= -lr*m*(1+m)).

**Fix:** d_velocity = -float(lr) * m * m * do.

**Independently verified:** Re-derived from optim.nesterov: look_ahead = g(1+m) + m^2*v, so d(new_params)/dv = -lr*m^2. Finite difference through optim.nesterov (lr=0.1, m=0.9) gives ~-0.081 = -lr*m^2; vjp.py:4374 returns -0.171 = -lr*m*(1+m). The unit test (test_sprint_collectives_optim_memory_cumextrema.py) only checks d_grads, not d_velocity, so nothing pins the wrong value — and the native x86/ROCm backward helper _momentum_vjp in test_autodiff_training_series_target_binding.py:283 uses the correct -lr*m^2 formula, meaning the numpy reference contradicts the certified native path.

### `python/tessera/autodiff/jvp.py:98` — id-keyed tangent map allows stale-id aliasing

*Autodiff — forward mode (JVP) · logic*

**What is wrong:** _JVPTrace.tangents maps id(ndarray) -> tangent but never holds a reference to the primal array. When an intermediate op output is garbage-collected mid-trace and CPython/numpy reuses its address for a new, unrelated array, the new array's id() matches the dead entry and record_op treats the constant as active with the dead value's tangent — a silently wrong (or shape-mismatched, crashing) tangent. The dict also grows unboundedly over a trace.

**Evidence:** bind() stores self.tangents[id(value)] = tangent without retaining value; record_op looks up self.tangents.get(id(primal)) (line 129). Scenario: inside jvp(fn, ...), fn computes tmp = ops.relu(x) (output bound at id A), rebinds/drops tmp so the array is freed, then allocates a same-shape constant c = np.ones_like(x) which lands at address A; ops.mul(x, c) now finds a tangent for c and propagates relu's tangent through a constant. numpy reuses freed buffers of equal size routinely, so this is reachable in ordinary multi-statement traced functions.

**Fix:** Keep the primal arrays alive in the trace (e.g. store (value, tangent) pairs, or an id -> value keep-alive list alongside the tangent dict) so an id can never be reused while the trace is active.

**Independently verified:** Reproduced end-to-end through the public API. `_JVPTrace.bind` (jvp.py:98) stores `self.tangents[id(value)] = tangent` and nothing anywhere retains `value`; `record_op` looks up `id(primal)` (jvp.py:129). ndarray PyObjects are fixed-size, so CPython's obmalloc reuses a freed one's address immediately. Script: `def f(x): tmp = ts.ops.relu(x); idA = id(tmp); del tmp; c = np.full_like(x, 2.0); return ts.ops.mul(x, c)` — printed `id reuse: True` (0x11f69a430 both times), and `jvp(f,(x,),(ones,))` returned tangent [3., 4., 5.] where the correct answer is [2., 2., 2.] (relu's stale tangent was propagated through the constant `c`). No guard, no keep-alive list, no dict clearing exists on the path. The unbounded-growth half is also true but minor.

### `python/tessera/autodiff/jvp.py:2571` — amax/amin tie-count squeeze() collapses unrelated size-1 dims

*Autodiff — forward mode (JVP) · logic*

**What is wrong:** jvp_amax/jvp_amin (and their max/min delegates) divide the reduced tangent by counts.squeeze() when keepdims=False and axis is not None. squeeze() with no argument removes ALL size-1 axes, not just the reduced one, so any input with a genuine size-1 dimension broadcasts the division into a wrong-shaped, wrong-valued tangent.

**Evidence:** x of shape (3, 1, 5), axis=2, keepdims=False: numerator np.sum(mask*dx, axis=2) has shape (3, 1); counts has shape (3, 1, 1) and counts.squeeze() has shape (3,). (3,1)/(3,) broadcasts to shape (3, 3) — the returned tangent is (3,3) while the primal is (3,1), and every element is a cross-mixed quotient. Size-1 dims (kept batch/head axes) are common, so this fires on realistic inputs. Same defect verbatim in jvp_amin (line 2584).

**Fix:** Use counts.squeeze(axis=axis) (squeeze only the reduced axis/axes), or compute tan with keepdims=True and squeeze both numerator and counts together at the end.

**Independently verified:** Re-derived and reproduced. For x.shape=(3,1,5), axis=2, keepdims=False: numerator `np.sum(mask*dx, axis=2)` is (3,1); `counts` is (3,1,1); `counts.squeeze()` is (3,) — squeeze() with no argument drops the genuine size-1 dim too. (3,1)/(3,) broadcasts to (3,3). Verified through the public trace: `jvp(lambda a: ts.ops.amax(a, axis=2), (x,), (ones,))` returns primal shape (3,1) with tangent shape (3,3), values cross-mixed. `amax` IS a registered/wrapped op, so this is on the live trace path. jvp_amin (2584) and the max/min delegates (2592/2597) are identical. The law spec for amax/amin uses (3,4) inputs with no size-1 dims, so the existing sweep cannot catch it (`amax adjoint pass`).

### `python/tessera/autodiff/jvp.py:2559` — prod/cumprod tangent is zero at zero-valued inputs

*Autodiff — forward mode (JVP) · math*

**What is wrong:** jvp_prod computes the tangent with the ratio trick tan = sum((p/safe_x) * dx) where safe_x replaces zeros by 1. When the slice contains exactly one zero, p = 0 so every term vanishes and the tangent is 0 — but the true derivative w.r.t. the zero entry is the product of the remaining entries, which is nonzero. jvp_cumprod (line 4064) has the same defect for every position at or after a zero.

**Evidence:** x = [0., 2., 3.], dx = [1., 0., 0.], axis=None: analytic d(prod) = 2*3 = 6; the rule returns p=0, ratios=[1/1e-30->dx/1, 0, 0]*p... concretely tan = sum((0/safe)*dx) = 0. Zeros are routine in DL tensors (ReLU outputs, masks, padded regions), so jacfwd through prod at such points is silently wrong, not merely a measure-zero corner. The correct single-zero case needs the sub-product excluding the zero entry (as JAX's prod JVP does).

**Fix:** Handle the zero count explicitly per slice: with no zeros use the ratio trick; with exactly one zero, the tangent is (product of nonzeros) * dx at the zero position; with two or more zeros the tangent is 0.

**Independently verified:** Re-derived and reproduced on the live trace path (prod/cumprod are registered ops). `jvp(lambda a: ts.ops.prod(a), ([0.,2.,3.],), ([1.,0.,0.],))` returns tangent 0.0; central finite difference gives 6.0. `jvp(lambda a: ts.ops.cumprod(a, axis=0), ...)` returns [0.,0.,0.] where the analytic answer is [1.,2.,6.]. Decisively not a declared convention: the repo's own reverse-mode counterpart `vjp_prod` (vjp.py:2407-2421) explicitly special-cases it — 'if x has a unique zero at i, grad_i = prod(x \ {x_i})' — so forward and reverse mode disagree at zeros. The law sweep passes only because `law_inputs.py:164/258` deliberately sample `0.5 + |normal|`, i.e. the zero regime is never probed.

### `python/tessera/autodiff/jvp.py:3284` — jvp_pad crashes for every non-constant pad mode

*Autodiff — forward mode (JVP) · logic*

**What is wrong:** jvp_pad passes constant_values to np.pad unconditionally for both primal and tangent, but numpy rejects constant_values for any mode other than 'constant'. The canonical forward (ops.pad, __init__.py:4417) guards this and only forwards constant_values when mode=='constant'; the JVP does not, so forward-mode through pad with mode='reflect'/'edge'/'wrap' raises ValueError.

**Evidence:** Verified on this host: np.pad(np.ones(3), 1, mode='reflect', constant_values=0) raises ValueError: unsupported keyword arguments for mode 'reflect': {'constant_values'}. Any traced call ops.pad(x, pw, mode='reflect') therefore succeeds untraced and crashes under jvp/jacfwd.

**Fix:** Mirror the forward's guard: pass constant_values only when mode == 'constant' (tangent uses constant_values=0 in that case; other modes are linear and pad the tangent with the same mode).

**Independently verified:** Reproduced. `ops.pad` (__init__.py:4417-4425) guards `constant_values` behind `if mode == 'constant'`; jvp_pad (3284/3286) passes it unconditionally to both np.pad calls. Ran it: `f = lambda x: ts.ops.pad(x, 1, mode='reflect')` returns [1,0,1,2,3,2] untraced, and `jvp(f,(x,),(ones,))` raises `ValueError: unsupported keyword arguments for mode 'reflect': {'constant_values'}`. `pad` is a registered/wrapped op so this is live. The pad law spec uses the default constant mode, so `pad chain pass` does not cover it. The proposed fix (mirror the forward's guard) is correct — the non-constant modes are linear in x, so padding the tangent with the same mode is right.

### `python/tessera/autodiff/transforms.py:222` — jacrev silently returns zero Jacobian for untaped outputs

*Autodiff — tape / grad / transforms · logic*

**What is wrong:** The structural-resolution branch (`if not any(entry.output_id == id(out) ...)`) conflates 'fn returned a constant' with 'fn's tail math ran in raw numpy on top of taped ops'. In the second case the tape is non-empty and gradients exist, but jacrev returns an all-zero Jacobian with no error — a fail-open silent no-op, exactly what Decision #21a forbids — while grad() on the same function raises TesseraAutodiffError with a precise message. Tuple-returning fns hit the same branch (id(out) of the tuple never equals any per-component output_id) and also get silent zeros.

**Evidence:** Reproduced: jacrev(lambda x: np.tanh(ops.matmul(x, np.eye(3))))(np.array([.1,.2,.3])) returns a 3x3 zero matrix (true Jacobian is diag(1 - tanh^2) ≈ diag(0.99, 0.96, 0.91)); grad on the same expression raises 'backward target is not a tape-recorded output'. Additionally _returns_this_parameter treats any same-shape memory-sharing view as identity, so a square-permuted view (e.g. a reversed or transposed buffer) would yield an identity Jacobian instead of the permutation.

**Fix:** Take the structural branch only when the tape is EMPTY (true constant/identity); when entries exist but id(out) is not among them, raise the same diagnostic Tape.backward raises for a non-taped target.

**Independently verified:** Reproduced all three sub-claims. (a) `jacrev(lambda x: np.tanh(ops.matmul(x, np.eye(3))))([.1,.2,.3])` returns a 3x3 all-zero matrix; the true Jacobian is diag(0.990, 0.961, 0.915). The tape is NOT empty (matmul is recorded), but `id(out)` is the raw-numpy tanh result, so the structural branch at line 222 fires and returns the pre-zeroed buffers. `grad` on the same expression raises TesseraAutodiffError('backward target is not a tape-recorded output'), so the two paths disagree — jacrev fails open. (b) Tuple output: `jacrev(lambda v: (ops.mul(v,2.), ops.mul(v,3.)))` returns an all-zero 2x2x2 array, same branch. (c) `_returns_this_parameter` (transforms.py:141-146) accepts any same-shape memory-sharing view: `jacrev(lambda v: np.asarray(v)[::-1])` returns the identity instead of the anti-diagonal permutation. The proposed fix (take the branch only when the tape is empty) does not break the pinning tests: test_autodiff.py:669-696 covers `lambda v: v` (empty tape) and `lambda v: np.array([5.,6.])` (empty tape), both still on the structural path.

### `python/tessera/autodiff/tape.py:364` — record_custom_vjp_call mangles scalar positional arguments

*Autodiff — tape / grad / transforms · logic*

**What is wrong:** The docstring contract says non-array positionals 'are passed to forward' unchanged, but the code only passes through _NON_ARRAY descriptors; a python int/float positional is described by _describe's literal branch as a float64 0-d ndarray (tape.py:539-541) and that array — not the original scalar — is handed to the forward. An int used as an axis/index crashes only when a tape is active; an int used arithmetically silently computes in float64 and is recorded as a differentiable input the VJP must answer for. Taped and untaped execution of the same call diverge.

**Evidence:** Reproduced: record_custom_vjp_call('mysum', lambda x, axis: np.sum(x, axis=axis), vjp, x, 1) returns [3., 3.] with no tape, but inside `with tape():` raises TypeError 'only integer scalar arrays can be converted to a scalar index' because the forward receives np.asarray(1, dtype=np.float64).

**Fix:** In the pass-through loop, treat desc.is_literal args like _NON_ARRAY for the forward call (append the original value), and only record them if the custom VJP actually declares a slot for them.

**Independently verified:** Reproduced verbatim. `_describe(1)` hits the literal branch (tape.py:539-541) and returns an InputDesc with `array=np.asarray(1, dtype=np.float64)` — it is NOT `_NON_ARRAY`, so the pass-through loop at lines 360-365 takes the else branch and hands the float64 0-d array to the forward. Measured: `record_custom_vjp_call('mysum', lambda x, axis: np.sum(x, axis=axis), vjp, x, 1)` returns [3., 7.] with no tape (untaped path uses `_to_forward_arg`, preserving the int) but raises TypeError 'only integer scalar arrays can be converted to a scalar index' inside `with tape():`. That is a genuine taped/untaped divergence and a direct contradiction of the function's own docstring ('Non-array positional arguments are passed to forward but are not recorded as differentiable inputs'). Caveat on blast radius, not on the claim: the sole in-repo caller is `custom_root` (implicit.py:683), whose `*params` are meant to be differentiable arrays — so the defect is reachable only when a caller passes a scalar/index-style parameter, which the contract as written permits.

### `python/tessera/autodiff/algebra.py:1117` — TaylorModel.mul enclosure not outward-rounded per accumulation

*Autodiff — laws & algebra oracles · math*

**What is wrong:** TaylorModel.mul accumulates per-term corner bounds with round-to-nearest additions (`lo[i+j] = lo[i+j] + l`) and applies only ONE final np.nextafter per coefficient, so the claimed containment certificate ('the exact value lies in this interval, full stop') is unsound whenever a coefficient sums 3+ product terms with cancellation. An intermediate add's rounding error is half an ulp of the INTERMEDIATE magnitude, which can vastly exceed the one final-ulp outward step when large terms cancel. `add` widens after every addition; `mul` must do the same (or use directed rounding).

**Evidence:** Coefficient 2 of a product with terms t1=1e16, t2=3, t3=-1e16 (exact sum 3): fl(1e16+3) rounds to 1e16+4 (ulp at 1e16 is 2), then adding -1e16 leaves lo ≈ 4; the final nextafter(4, -inf) moves ~4.4e-16, so lo ≈ 4 > exact 3 — the exact coefficient escapes the interval by O(1). enclosure_check (laws.py:984) only exercises (x+c)·x, whose coefficients have at most 2 terms (one add, covered by the single nextafter), so the sweep cannot see this; but the docstring sells the interval as arbiter-grade evidence.

**Fix:** In mul, widen after each accumulation: `lo[i+j], hi[i+j] = self._widen(lo[i+j] + l, hi[i+j] + h)`, and drop the single trailing nextafter pass (or keep it — it is then harmlessly conservative).

**Independently verified:** Defect stands, but the reported mechanism and evidence are both wrong and severity is overstated. DISPROOF of the stated evidence: I ran the reviewer's own example (a=[1e16,3,-1e16], b=[1,1,1], order 2) through the real code — coefficient 2 comes back as [-2.0, 8.0], which contains the exact value 3. The claim 'applies only ONE final np.nextafter per coefficient' is factually false: algebra.py:1115 calls `self._widen(min(corners), max(corners))` on EVERY term before accumulation, so there are N+1 outward steps, not 1, and that per-term full-ulp slack absorbs the half-ulp accumulation error in the reviewer's scenario. WHAT DOES STAND: mul is still not rigorously outward-rounded. Randomized search over 400k order-4 products (exact Fraction reference) found 2 genuine containment escapes, e.g. a=[-9.85e15, 3.26e7, -0.28, -7.08e-9, 7.54e7], b=[6.92e7, -8.06e29, -2.12e17, 8.76e15, -4.91e15] where coefficient 4's exact value falls outside [lo,hi] — driven by exactly the mechanism named (terms spanning 1e15..1e31, so ulp of the running sum swamps the per-term slack). Escapes are ~1 ulp, not O(1). Reachability is much narrower than claimed: 200k randomized programs built only from lift/add/mul (depths 3-12, magnitudes 1e-20..1e30, order 4) produced ZERO escapes — the escapes need hand-built dense point-interval models, because after the first mul the intervals are non-degenerate and their width dominates. So 'unsound whenever a coefficient sums 3+ product terms with cancellation' is false; it is a rare, sub-ulp hole. The proposed fix (widen after each accumulation) is correct and cheap, and the class docstring's unconditional 'full stop' certificate claim does need it, so the finding is actionable — at P3, not P1.

### `python/tessera/autodiff/implicit.py:329` — FD rmatvec rebuilds full Jacobian every GMRES iteration

*Autodiff — implicit / jets / remat · algorithm*

**What is wrong:** rmatvec computes Jᵀr by looping over all n_in basis columns, each column costing 2 residual evaluations, with no caching across calls. root_vjp's default path runs GMRES on A_op.T, which calls rmatvec twice per inner iteration (Arnoldi step + true-residual check), so the identical n columns of J are recomputed from scratch every iteration. The 'matrix-free' default is therefore strictly dominated by the dense oracle it is supposed to improve on: materializing J once costs 2n residual evals total, while the GMRES path costs ~4n evals per iteration.

**Evidence:** n = 100 solution size, 30 GMRES iterations (typical for tol=1e-8): gmres path = 30 iters x 2 rmatvec x 100 columns x 2 F-evals = 12,000 residual evaluations vs 200 for linear_solver='dense' (materialize + solve). The asymptotic cost is O(n * iters) F-evals vs O(n). Fix: memoize the FD columns (or materialize J lazily on first rmatvec) when provenance is 'numerical_oracle'; the compiler-linearization path is unaffected.

**Fix:** Cache the n FD Jacobian columns on first rmatvec call (build J once, reuse J.T @ r on subsequent calls), or route the numerical-oracle transpose solve through the dense path.

**Independently verified:** Traced and measured. python/tessera/autodiff/implicit.py:322-332 rmatvec loops all n_in basis columns, each calling matvec (2 F-evals, lines 312-314), with no memoization; OperatorTangent.T (operator.py:133-141) just swaps fwd/adj so gmres_solve's apply() IS rmatvec, and gmres calls apply twice per inner iteration (lines 158 and 180) plus once at line 143. Measured with an instrumented residual, n=12: root_vjp(linear_solver='gmres') = 626 F-evals (25 matvecs x 24) vs linear_solver='dense' = 50, identical gradients to 3e-15. Cost is (2*iters+1) x 2n vs 2n, so the FD default is strictly dominated in residual evaluations by the dense oracle it ships alongside, exactly as claimed. The 'O(n) memory' rationale in the code comment (lines 316-318) is additionally undercut on the default path: custom_root defaults certify=True and certify_root already calls A_op.materialize() (operator.py:325), building the full dense n x n Jacobian before the 'matrix-free' solve runs. No test, guard, or caller contract avoids this; both proposed fixes (memoize the FD columns, or route the numerical_oracle transpose through the dense path) are valid.

### `python/tessera/quantization.py:112` — Observer zero-point clipped without nudging range to zero

*RNG & quantization · math*

**What is wrong:** CalibrationObserver.calculate_qparams computes zp = clip(round(qmin - min_val/scale), qmin, qmax) but never widens [min_val, max_val] to include 0. When min_val > 0 (or max_val < 0), the true zero-point lies outside [qmin, qmax], the clip destroys the affine map, and quantization with the returned qparams saturates most of the range.

**Evidence:** observe([2.0, 4.0]); calculate_qparams(8, symmetric=False) -> scale = 2/255 = 0.00784, raw zp = round(-128 - 255.1) = -383, clipped to -128. Quantizing x=4 with those qparams: round(4/0.00784 - 128) = 382 -> clip -> 127 -> dequant (127+128)*0.00784 = 2.0, i.e. every value in [2,4] collapses to <= 2.0. Strictly-positive calibration data (sigmoid outputs, variance statistics) is realistic.

**Fix:** Follow the standard TFLite/PyTorch practice: use min(min_val, 0) and max(max_val, 0) when computing the asymmetric scale, which guarantees the zero-point is representable and the clip is a no-op.

**Independently verified:** Re-derived and reproduced at quantization.py:100-113. CalibrationObserver().observe([2.0,4.0]).calculate_qparams(8, symmetric=False) returns scale=0.007843, zp=-128 (raw zp = round(-128 - 2/0.007843) = -383, clipped). Feeding those qparams back through quantize_int8(..., symmetric=False) on [2,3,4] gives q=[127,127,127] -> dequant [2.0,2.0,2.0]: the entire calibrated range collapses to its minimum. The zp convention matches quantize_int8/dequantize_int8 (q = round(x/s)+zp, x = (q-zp)*s), so the two are consistent and the failure is genuinely in the qparam derivation, not a convention mismatch. The code never clamps min_val/max_val toward 0, which is the standard fix (min(min,0)/max(max,0)) that makes the clip a no-op. Only affects ranges excluding zero (strictly positive or strictly negative calibration data) - realistic for post-ReLU/sigmoid activations. The only in-repo test (test_s7_s8_s9.py:94) uses the default symmetric=True path, so nothing pins this as intentional.

### `python/tessera/optim.py:427` — Adafactor missing second-moment bias correction

> **FIXED 2026-08-30 — the deferral's premise was wrong; no kernel ABI change
> was needed.** The deferral said the flat `ts.ops.adafactor` "carries no step
> counter in its signature, so it cannot compute `1 - beta2**t` at all". Two
> things falsify that. (a) The flat **`adam`/`adamw`** ABI right next to it
> already takes `step: int = 1` as a kwarg and the ROCm/x86 executors already
> read `int(kwargs.get("step", 1))` — so a step kwarg on the flat adafactor is
> the house pattern, not a new ABI. (b) The correction is exactly expressible
> as a step-dependent **decay rate**,
> `b2_t = b2*(1 - b2**(t-1))/(1 - b2**t)`, for which the recursion carries the
> debiased estimate directly (`v_t == EMA_t/(1 - b2**t)`, `b2_1 = 0`) — and
> every physical Adafactor kernel already takes `beta2` as a **scalar**, so the
> correction is applied host-side and `tessera_x86_avx512_adafactor_*` /
> `adafactor_row|col|mean|update` / `sm120_adafactor_*` are untouched.
> Landed as one shared `optim.adafactor_decay` consumed by the tree form, the
> flat op, the analytic VJP, the x86/ROCm/NVIDIA forward and backward
> executors, and the backward state contract (which now records nominal
> `beta2`, `step`, and `beta2_effective`). Pinned by six tests in
> `tests/unit/test_s10_optim.py`;
> `test_flat_adafactor_full_and_factored_match_tree_reference` was strengthened
> to step 3, since at step 1 the corrected decay is 0 and the carried state
> would not be exercised at all.


*Losses, optimizers, RL, nn.functional · math*

**What is wrong:** adafactor uses a fixed beta2=0.999 EMA for the factored/full second moment with no bias correction and no decay schedule, so early-step updates are inflated by 1/sqrt(1-beta2^t). The state dict tracks 'step' but the update math never uses it; real Adafactor (Shazeer & Stern) uses the beta2_t = 1 - t^-0.8 schedule (or an explicit 1-beta2^t correction) precisely to avoid this.

**Evidence:** From a fresh state at step 1, row/col/v = (1-0.999)*g^2 = 0.001*g^2, so update = g/(sqrt(0.001)*|g~|) ~= 31.6x the normalized gradient — the first ~1000 steps apply an effective learning rate up to ~32x lr, destabilizing training starts. Compare adamw at line 308-313, which divides by b2_corr = 1-beta2^step for exactly this reason.

**Fix:** Divide the second-moment estimate by (1 - beta2**step) using the already-tracked step, or adopt the paper's per-step decay rate beta2_t = 1 - step**-0.8.

**Independently verified:** Re-derived and executed. `_adafactor_update_state` (optim.py:416-424) applies a fixed `beta2` EMA and `_adafactor_update_from_state` (427-434) divides by sqrt of it with no step term; `step` is incremented at line 387 but never passed into either helper. Algebraically the (1-beta2) factor does not cancel in the factored path: scale = row*col/mean(row) = (1-b2)^2*r*c / ((1-b2)*mean(r)) = (1-b2)*r*c/mean(r), so sqrt(scale) carries sqrt(1-b2). Measured: `adafactor({'w':ones(4,4)}, grads=0.1, lr=1.0)` from fresh state gives a step-1 update magnitude of 31.622774 == 1/sqrt(1-0.999), vs the intended ~1.0 normalized update. Shazeer & Stern's beta2_t = 1 - t^-0.8 gives beta2_1 = 0 (v_1 = g^2 exactly, update 1.0), so the paper's schedule does not have this inflation. adamw at optim.py:308-313 does apply b2_corr = 1-beta2**step for exactly this reason. No test pins the buggy value: tests/unit/test_s10_optim.py:41 asserts only state shapes/factored flag; the ROCm compiled tests differential-check against this same reference, so they mirror rather than justify it.

### `python/tessera/optim.py:561` — inverse_sqrt_lr exceeds init_value during warmup

*Losses, optimizers, RL, nn.functional · math*

**What is wrong:** inverse_sqrt_lr returns init_value*sqrt(warmup_steps)/sqrt(step) with no clamp of step to warmup_steps and no warmup ramp, so for every step < warmup_steps the lr is LARGER than init_value — the opposite of warmup. The standard inverse-sqrt-with-warmup semantic (fairseq/T5) is init*sqrt(warmup/max(step, warmup)), which is flat at init through warmup then decays.

**Evidence:** inverse_sqrt_lr(1, init_value=1e-3, warmup_steps=4000) returns 1e-3*sqrt(4000) ~= 0.063 — 63x the nominal peak at the very first step; every call with warmup_steps > 1 produces a decaying blow-up over the whole warmup window instead of a ramp.

**Fix:** Clamp the denominator: return init_value*math.sqrt(max(1, warmup_steps))/math.sqrt(max(step, warmup_steps)) (optionally with a linear ramp for step < warmup_steps).

**Independently verified:** optim.py:559-561 is `step=max(1,step); return init*sqrt(max(1,warmup))/sqrt(step)` — no clamp of the denominator to warmup_steps and no ramp branch (contrast cosine_warmup_lr:545-546 and linear_warmup_lr:551, which both ramp). Executed inverse_sqrt_lr(step, init_value=1e-3, warmup_steps=4000): step 1 -> 0.0632 (63.2x init), 10 -> 0.02, 100 -> 0.0063, 1000 -> 0.002, 4000 -> 0.001, 8000 -> 7.07e-4. So lr strictly exceeds init_value for every step < warmup_steps and decays across the whole 'warmup' window — the inverse of the fairseq/T5 semantic the parameter name asserts. Nothing pins this: the only test (tests/unit/test_s10_optim.py:72) uses warmup_steps=1, the exact value at which the defect is invisible, and docs/audit/generated/test_coverage.csv marks the symbol 'structural_only'. No docstring or spec grants an alternative contract.

### `python/tessera/optim.py:589` — clip_grad_norm silently ignores non-2, non-inf norm_type

*Losses, optimizers, RL, nn.functional · logic*

**What is wrong:** clip_grad_norm branches only on norm_type == inf; every other value (1.0, 3.0, ...) silently falls through to tree_l2_norm and clips by the L2 norm while reporting the L2 total. A semantic parameter is silently defaulted instead of failing closed (Decision #21a) — the caller gets a wrong clip scale and a wrong reported norm with no diagnostic.

**Evidence:** clip_grad_norm(grads, max_norm=1.0, norm_type=1.0) on grads = [3.0, 4.0] returns total = 5.0 (L2) and scale 0.2, whereas the requested L1 norm is 7.0 and the correct scale is 1/7; no error or warning is emitted.

**Fix:** Compute the general p-norm (sum(|g|**p)**(1/p)) for finite norm_type, or raise ValueError for unsupported values.

**Independently verified:** optim.py:588-596: the only branch is `norm_type == float('inf')`; every other value falls to `tree_l2_norm(grads)`. Executed on grads {a:[3.0], b:[4.0]}: norm_type=2.0 -> 5.0, norm_type=1.0 -> 5.0 (true L1 = 7.0), norm_type=3.0 -> 5.0 (true L3 = 4.497), inf -> 4.0. No warning, no exception; the returned `total` and the derived `scale` are both wrong for any finite p != 2. The parameter is in the public signature with a float type and no docstring restricting it, and no test exercises a non-default norm_type (test_s10_optim.py:80 uses the default; test_rocm_grad_clip_compiled.py only cross-checks L2), so nothing establishes the silent fallthrough as intentional.

### `python/tessera/losses.py:210` — focal_loss silently wraps negative/out-of-range target indices

*Losses, optimizers, RL, nn.functional · logic*

**What is wrong:** focal_loss fancy-indexes flat_probs[arange, idx] with no range check and no ignore_index support, so negative targets (including the -100 padding convention that cross_entropy_loss in this same module documents and handles) silently index from the end and produce a finite but wrong loss; targets >= C raise IndexError instead of the module's stable ValueError.

**Evidence:** focal_loss(logits(B,50257), targets containing -100) picks class 50157's probability for the padded positions and averages it into the loss — no error, wrong value. cross_entropy_loss (lines 91-94) validates the identical situation and masks ignore_index; a user switching CE -> focal on padded batches gets silently corrupted training signal.

**Fix:** Validate 0 <= idx < C and support ignore_index with the same valid-mask/mean-over-valid handling as cross_entropy_loss.

**Independently verified:** losses.py:204-214 has no ignore_index parameter and no range check before `flat_probs[np.arange(idx.size), idx]` at line 210. Executed with C=50257 and targets [1,-100,2,3]: focal_loss returns 11.286 with no error, and I verified the padded row's contribution is probs[1, 50157] = 2.571e-06 (identical to probs[1][-100]) — i.e. it silently scored class 50157 for the padded position. cross_entropy_loss on the same input returns 10.759 after correctly masking. With C=10 and target -100 it raises a bare numpy IndexError, not the module's ValueError, and target 99 likewise raises IndexError, whereas cross_entropy_loss:92-93 raises 'target class index out of range'. Both halves of the claim reproduce; -100 is the module's own documented padding convention (cross_entropy_loss default ignore_index at line 73), so the CE->focal swap hazard is real.

### `python/tessera/losses.py:242` — kl_divergence NaN when p_log_probs contains -inf

*Losses, optimizers, RL, nn.functional · math*

**What is wrong:** kl_divergence computes p * (p_log - log(max(q, eps))) with p = exp(p_log); an entry with p_log = -inf gives p = 0 and 0 * (-inf) = NaN, which propagates through the sum and reduction. The mathematically correct contribution of a zero-probability entry is 0 (the epsilon floor protects q but not p).

**Evidence:** p_log_probs produced by log_softmax over masked logits (vocab masking sets logits to -inf — routine in distillation/constrained decoding) contain -inf; kl_divergence(p_log, q) then returns NaN for the whole batch instead of the finite KL over the support.

**Fix:** Guard the zero-support entries: loss = np.where(np.isfinite(p_log), p * (p_log - np.log(np.maximum(q, eps))), 0.0).

**Independently verified:** losses.py:241-243: `p = np.exp(p_log); loss = p * (p_log - log(max(q, eps)))`. The epsilon guard at line 242 floors q only; p_log is used raw. For p_log = -inf, p = 0 and 0 * (-inf - finite) = 0 * -inf = NaN, which the np.sum at line 243 propagates across the whole axis and then the reduction. Executed: kl_divergence([[-1.3133, -0.3133, -inf]], [[0.3,0.6,0.1]]) -> nan. The correct value under the standard 0*log0 = 0 convention is the finite KL over the support. -inf entries are reachable through the module's own `_log_softmax` (line 34-35) on masked logits, and no shape/finiteness guard rejects them — the validation at lines 237-240 checks only shape equality and epsilon. js_divergence (246-265) floors both operands and is unaffected, which shows the guard was simply omitted here.

### `python/tessera/compiler/fusion_core.py:1115` — Attention F4 probe ignores transpose flags

*Fusion core & emitter seams · logic*

**What is wrong:** verify_synthesized_attention generates natural-oriented probes Q,K,V of shape (8,16) but never orients them per region.q_transposed/k_transposed, so for any transposed region region._natural double-flips the probe into a Q/K head-dim mismatch and the verifier raises instead of returning a verdict. The sibling verify_synthesized_matmul (lines 1088-1094) explicitly fixed this exact double-flip; attention was left unfixed.

**Evidence:** For a region with k_transposed=True (produced by discover_attention_regions line 1447 for any graph whose score matmul has no transpose_b flag, i.e. K materialized pre-transposed), run_fused_attention (apple_msl.py:1697-1710) computes Kn=(16,8) so Dk=8 != D=16 and raises ValueError; the same happens in region.reference at line 1123. In runtime.py:33846 the exception is swallowed by the enclosing `except Exception: continue`, silently disabling the fused attention lane for that whole region class; through candidate.verify_candidate the exception propagates and crashes arbitrate/run_arbitrated.

**Fix:** Mirror the matmul verifier: transpose the Q/K probes into the region's raw orientation (and use a non-square probe so a mismatch cannot alias) before calling the runner and reference.

**Independently verified:** Reproduced. verify_synthesized_attention (fusion_core.py:1114-1118) builds square (8,16) Q/K/V and passes them raw; apple_msl.run_fused_attention:1697-1710 applies region._natural first, so for k_transposed=True Kn=(16,8) vs D=16 and it raises ValueError unconditionally (the check precedes the Metal-symbol branch, so even a non-Metal host raises). Direct call: `verify_synthesized_attention(AttentionRegion(k_transposed=True))` -> ValueError 'Q/K head_dim mismatch: Q (8, 16), K (16, 8)'. Runtime effect confirmed with a pre-transposed-K graph (matmul->scale->softmax->matmul, no transpose_b): with a cleared verdict cache _apple_gpu_try_synthesized_fusion returned consumed=set() (lane silently disabled by runtime.py:33851 `except Exception: continue`); after warming the cache with an untransposed region of the same (scale,causal) the identical graph fused (consumed={0,1,2,3}) - exactly the order-dependence claimed. verify_candidate (candidate.py:327) has no try/except, so the same exception propagates out of arbitrate. The sibling verify_synthesized_matmul does orient its probe (lines 1091-1094), so the asymmetry is real.

### `python/tessera/compiler/fusion_core.py:879` — Attention fuse gate dead-ends the online-softmax lane

*Fusion core & emitter seams · logic*

**What is wrong:** attention_cost/should_fuse_attention hard-fail any region with Nk > SYNTH_MAX_N (1024), but the F3b selector (select_attention_lowering) and run_fused_attention explicitly support the online-softmax kernel for Nk beyond that cap whenever Dv <= SYNTH_MAX_D (256) — the documented large-context case (see the SYNTH_MAX_D comment at lines 71-76). Because the runtime prepass gates on should_fuse_attention before ever calling run_fused_attention, long-sequence attention never reaches the online kernel from this path.

**Evidence:** fusion_core.py:879-881 returns FusionCost(False, ...) for Nk>1024; runtime.py:33844 does `if not should_fuse_attention(...): continue` before run_fused_attention, whose own selector (apple_msl.py:1714-1724 -> select_attention_lowering) would choose 'online' for e.g. M=64, Nk=4096, D=Dv=128. Result: exactly the shape the online kernel was built for (long-context decode) falls back to 3 dispatches + a 2*M*Nk*4-byte score-matrix DRAM round-trip (4 MB at M=64, Nk=4096, plus per-dispatch overhead) instead of one fused kernel.

**Fix:** Make attention_cost feasible when Nk <= SYNTH_MAX_N or Dv <= SYNTH_MAX_D (matching the selector's feasibility), or gate directly on select_attention_lowering(...).variant != 'reference'.

**Independently verified:** Verified by execution: should_fuse_attention(AttentionRegion(), M=64, Nk=4096, D=128, Dv=128) -> False with reason 'Nk=4096 exceeds per-thread stack cap 1024', while select_attention_lowering(64,4096,128,128) -> AttnLoweringCost(variant='online', feasible=True). runtime.py:33844 gates on should_fuse_attention *before* calling run_fused_attention, and run_fused_attention (apple_msl.py:1684) is the only entry to synthesize_attention_online_msl / _ATTN_ONLINE_ENTRY anywhere in python/ - its only production caller is runtime.py:33848. So the online-softmax kernel is unreachable from the runtime prepass for exactly the Nk range it was built for, and the region falls back to per-op dispatch with the score-matrix DRAM round-trip. Not intentional: SYNTH_MAX_D/online landed 2026-07-04 while the pinning assertion `too_big = attention_cost(..., SYNTH_MAX_N+1); assert not too_big.fusible` (test_fusion_synthesis.py:955) dates from 2026-06-14, and select_attention_lowering's own docstring says it replaces the hard Nk<=SYNTH_MAX_N branch. Fixing the gate will require updating that pre-existing test.

### `python/tessera/compiler/emit/nvidia_cuda.py:1137` — Missing barrier when reusing shared reduction scratch

*Emitter — NVIDIA CUDA · logic*

**What is wrong:** Several emitted block-reduction kernels read the reduction result (scratch[0]) and then rewrite the same shared buffer for the next phase with no __syncthreads()/barrier in between, a data race under the CUDA memory model. Instances: tsr_softmax_kernel f32 (line 1137: 'mx = scratch[0]' then line 1139 'scratch[tid] = sum' with no barrier between) and its f16 sibling (lines 1176-1178); _synthesize_norm_cuda's tsr_sum (lines 1284-1288) whose LayerNorm path calls it twice back-to-back, so the second call's 's[t]=v' (thread 0 writes s[0]) races with other warps still reading the mean from s[0]; the resident 'softmax' kernel (line 3063, 'm=q[0]' then 'q[t]=z'); and paged_attn (line 3067, 'm=red[0]' then 'red[t]=z'). All are broadcast values consumed by every thread, so a stale/overwritten read corrupts the whole row.

**Evidence:** run_row_norm(x, 'layer_norm', eps) with K=4096, block=256 (8 warps): after tsr_sum(sum,s) returns, warp 0's thread 0 races through its K/256-element deviation loop and re-enters tsr_sum(dev,s), writing s[0]=dev0, while a warp stalled on a global-memory load has not yet executed 'return s[0]' — it then reads dev0 instead of the row sum, producing a wrong mean and wrong normalized output for that row. Identical write-after-read hazard on scratch[0]/q[0]/red[0] in the softmax kernels: any thread that has not yet read the row max when thread 0 stores its partial sum gets a corrupted max. Nondeterministic wrong results on mainstream softmax/layer_norm/paged-attention lanes; only warp-scheduling luck hides it today.

**Fix:** Insert __syncthreads() at the top of tsr_sum/tsr_block_sum (before s[t]=v) or after every broadcast read of element 0 before the buffer is rewritten (softmax kernels need one between 'mx = scratch[0]' and 'scratch[tid] = sum', and between reading the sum and the output loop is already safe); in paged_attn add __syncthreads() between 'm=red[0]' and 'red[t]=z'.

**Independently verified:** Traced every cited kernel; the write-after-read hazard on shared element 0 is real in all five instances. (a) nvidia_cuda.py:1132-1139 f32 tsr_softmax_kernel: the max-reduction's final __syncthreads() is inside the loop, then every thread does `mx = scratch[0]`, runs a K-element global-memory pass, and writes `scratch[tid] = sum` — the barrier at 1139 is AFTER the write, so nothing orders thread 0's store to scratch[0] against another warp's still-pending load of scratch[0]. (b) identical in the f16 sibling (1176-1178: `mx=s[0]` ... `s[t]=sum;__syncthreads();`). (c) _synthesize_norm_cuda's tsr_sum (1284-1288) opens with `s[t]=v;__syncthreads();` and closes with `return s[0]` after its loop's last barrier; the LayerNorm path (1293/1295) calls it twice back-to-back with only a K-element deviation loop between, so the second call's `s[0]=dev0` from thread 0 races the first call's `return s[0]` in any warp that has not yet issued that LDS — this is the strongest instance because the intervening work is a long global-load pass. (d) resident `softmax` at 3063 (`m=q[0]` ... `q[t]=z`) and (e) `paged_attn` at 3067 (`m=red[0]` ... `red[t]=z`) repeat the pattern. I checked the one place where the pattern is NOT a race for completeness: paged_attn's score loop is safe, because thread 0's `red[0]` read is followed by a `__syncthreads()` before the next iteration's `red[t]=v`. No guard, launch contract, or test makes the racing schedule impossible; the CUDA memory model makes an unsynchronized shared write-after-read UB, and compute-sanitizer racecheck flags exactly this shape. Practically it is hard to observe (the racing thread needs to starve a peer across a global-latency-bound loop for a single LDS), so the P1 severity is arguable, but the missing barrier and the proposed fix (a __syncthreads() at the top of tsr_sum / between the broadcast read and the buffer rewrite) are both correct.

### `python/tessera/compiler/emit/apple_msl.py:322` — Tiled kernel KeyError on valid layer_norm reduction

*Emitter — Apple MSL / ROCm HIP / x86 · logic*

**What is wrong:** synthesize_matmul_epilogue_msl_tiled looks up `_TILED_REDUCTIONS[region.reduction]`, but _TILED_REDUCTIONS (line 226) only defines rmsnorm and softmax while fusion_core.REDUCTION_OPS — the set FusedRegion.__post_init__ validates against — also contains layer_norm. A valid FusedRegion(reduction="layer_norm") crashes with an uncaught KeyError instead of falling back to the reference or raising EmitError.

**Evidence:** Reachable on any Mac with the Metal runtime: run_fused_region f32 path takes the tiled branch for SYNTH_MAX_N < N <= SYNTH_MAX_N_TILED (1024 < N <= 8192, apple_msl.py:1039-1042) with no reduction filter; _run_fused_region_f16 (line 943-945) and _run_fused_region_bf16 (line 893-895) hit the same lookup whenever N > 1024; AppleMSLEmitter.emit(variant=TILED) hits it for any N. E.g. FusedRegion(epilogue=(), reduction="layer_norm") with N=2048 f32 raises KeyError inside run_fused_region — a crash, not the documented reference fallback. The coopmat-reduce path guards membership explicitly (coopmat_reduce_eligible, line 544); the tiled path has no such guard.

**Fix:** Add a layer_norm cooperative block to _TILED_REDUCTIONS, or gate the tiled branches on `region.reduction in _TILED_REDUCTIONS` (falling back to the stack kernel/reference) and raise EmitError from the emitter.

**Independently verified:** Reproduced host-free on this machine: `FusedRegion((), reduction='layer_norm')` constructs fine (fusion_core.REDUCTION_OPS has a layer_norm entry with a valid MSL block, and __post_init__ only rejects reductions outside REDUCTION_OPS), then `synthesize_matmul_epilogue_msl_tiled(r)` raises `KeyError: 'layer_norm'` at apple_msl.py:322, and `AppleMSLEmitter().emit(r, variant=TILED, dims=...)` raises the same KeyError — not EmitError. _TILED_REDUCTIONS (line 226) defines only rmsnorm and softmax. The launch paths are unguarded exactly as claimed: run_fused_region's f32 tiled branch (1039-1045) tests only `not region.has_residual and SYNTH_MAX_N < N <= SYNTH_MAX_N_TILED`, and _run_fused_region_f16/_bf16 (943-946, 893-896) branch on `is_tiled = 0 if N <= SYNTH_MAX_N else 1` with no reduction filter — so on a Mac with the tiled symbol present, N>1024 + layer_norm crashes instead of the documented 'numbers are correct either way' reference fallback. The coopmat path guards membership (coopmat_reduce_eligible, 541-546) and synthesize_matmul_reduction_coopmat_msl raises a clear ValueError, confirming the intended pattern. tests/unit/test_fusion_synthesis.py::_TILED_CASES covers only softmax/rmsnorm — nothing pins layer_norm as deliberately unsupported.

### `python/tessera/compiler/emit/apple_msl.py:1703` — Mixed-dtype attention reinterprets Q/K bits, wrong output

*Emitter — Apple MSL / ROCm HIP / x86 · logic*

**What is wrong:** run_fused_attention picks the kernel dtype tag from V's dtype alone (`_attn_dtype_tag(np.asarray(V).dtype)`, line 1694) and on the f16/bf16 path never coerces Q/K to that dtype: `Qn = np.ascontiguousarray(Qn)` preserves whatever dtype Q has, and the dispatch pointer lambda does `a.view(np.uint16)` (line 1737). A float32 Q with an f16 V (the realistic f32-query / f16-KV-cache mix) is bitwise reinterpreted as half — wrong element size, wrong indexing, garbage scores — and the result is returned tagged "metal_runtime" as if correct.

**Evidence:** Q f32 (M,D), K/V f16: tag="f16"; region._natural(Q,K,cast=False) keeps Q float32; Qn.view(np.uint16) yields an (M,2D) view of the same bytes; the kernel indexes Q as half with q_off=m*D, so it reads the low/high 16-bit halves of the first D/2 floats as half values. No shape or dtype error is raised (float32.view(uint16) is legal), so the corruption is silent. The f32 tag path is safe because it casts all three operands to f32 (lines 1699-1701); the 16-bit path casts none of them.

**Fix:** On the f16/bf16 path, np.ascontiguousarray each of Qn/Kn/Vn to the tag's dtype (or fall back to the f32 path when the three input dtypes disagree).

**Independently verified:** Traced run_fused_attention (apple_msl.py:1684-1745). `tag = _attn_dtype_tag(np.asarray(V).dtype)` keys on V alone; with V f16 and Q f32 the tag is 'f16', `region._natural(Q, K, cast=(tag=='f32'))` is called with cast=False (fusion_core._natural then does only `np.asarray(Q)` — no coercion), and the else-branch does bare `np.ascontiguousarray(Qn)`, preserving float32. The dispatch pointer is `a.view(np.uint16)` (line ~1737), which numpy accepts on a contiguous f32 array (yields (M,2D) uint16 over the same bytes), and the runtime symbol's argtypes carry no element size — so the kernel indexes Q as half at q_off=m*D, i.e. bytes [2mD,2mD+2D) of a row that actually occupies [4mD,4mD+4D). Wrong element size and wrong row stride, no exception, and the result returns tagged 'metal_runtime'. There is no dtype-agreement check anywhere in the function (only Dk==D and Nv==Nk shape checks), the f32 tag path is safe precisely because it casts all three operands (1699-1701), and the same hole hits bf16-V with f16-Q (same width, wrong interpretation). Tests only exercise uniform dtypes (test_half_precision_attention_equals_unfused_on_metal casts Q/K/V to one dtype), so nothing pins mixed input as intended. Note: the in-repo callers happen to feed uniform dtypes (paged_kv.py casts Q to f32 and gathers dense f32 K/V), so this is a latent hole in a public module API rather than a currently-firing bug.

### `python/tessera/compiler/emit/spectral_candidates.py:1203` — Framed-op verify budget scales with wrong length

*Emitter — spectral / Krylov / autotune · math*

**What is wrong:** _verify_region computes the fp32 error budget from getattr(region, 'n', 1), but for SpectralSTFTRegion 'n' is the SIGNAL length (not the transformed window length), and SpectralISTFTRegion/SpectralFilterRegion have no 'n' at all so they silently get n=1. For STFT the absolute tolerance grows linearly with signal length while output magnitudes scale only with sqrt(win), making the F4 gate vacuous for long signals; for ISTFT the budget never scales with win at all, contradicting the stated intent of _fp32_budget.

**Evidence:** SpectralSTFTRegion(n=100_000, win=64, hop=32) gives budget = max(1e-3, 1e-4*100_000) = 10.0 absolute. STFT bins of a 64-sample unit-variance windowed frame have magnitude ~sqrt(64)~8 < 10, so verify_by_reference at atol=10 passes even an all-zeros candidate output — a wrong native STFT kernel is certified F4-correct and enters arbitration. SpectralISTFTRegion(frames=..., win=4096, hop=...) gets budget max(1e-3, 1e-4*1)=1e-3 regardless of win.

**Fix:** Give each region an explicit transform-length attribute (win for STFT/ISTFT, n for (I)RFFT, bins for filter) and have _verify_region read that, e.g. a `transform_len` property, instead of getattr(region, 'n', 1). Prefer a relative-error budget (atol scaled by ref magnitude) so long signals cannot inflate the absolute tolerance.

**Independently verified:** Traced spectral_candidates.py:1191-1203. `_verify_region` passes `getattr(region,'n',1)` to `_fp32_budget`. Verified by execution: SpectralSTFTRegion(100_000,64,32) -> budget 10.0 (n is the SIGNAL length; the transformed length is win=64); SpectralISTFTRegion(100,4096,1024) and SpectralFilterRegion(129) have no `n` attribute at all -> budget 1e-3 regardless of win. The reviewer's exact example is slightly off (at n=100_000 the ref max|X| is 32.07, so all-zeros does NOT pass atol=10), but the defect is demonstrably exploitable one step out: at SpectralSTFTRegion(1_000_000,64,32) budget=100.0, 100% of bins fall inside it, and I confirmed np.allclose(zeros, ref, atol=100) -> True, plus a wrong-window candidate and a 1.5x-scaled candidate both pass. So the F4 gate is genuinely vacuous for long signals and would certify a wrong native STFT. No guard, caller contract, or test prevents this: every test in tests/unit/test_spectral_composed_lanes.py uses SpectralSTFTRegion(256,64,32) where the budget is 0.0256 and the gate has teeth. The ISTFT half of the claim is true about the code but latent-only in practice: I measured the correct composed CPU ISTFT at win=64/1024/4096/16384 with max err 4e-7..1.4e-6, all comfortably inside the fixed 1e-3, so no correct kernel is currently rejected. Central claim (budget keyed to the wrong length; STFT gate loosens with signal length) stands.

### `python/tessera/compiler/emit/spectral_candidates.py:1320` — Composed STFT/ISTFT dispatch inner FFT per frame

*Emitter — spectral / Krylov / autotune · performance*

**What is wrong:** STFTCandidate.run and ISTFTCandidate.run loop over frames in Python and call the full arbiter dispatch per frame: each frame re-runs _inner_fft, which re-enumerates candidates_for(), reconstructs a SpectralFFTRegion, re-checks applies_to/available, and executes ONE single-row hostptr transform (H2D + kernel launch + D2H per frame on ROCm) — even though the loaded library already binds a batch entry point (ts_fft_stockham_amd_hostptr_batch, configured at line 134) and run_rocm_stockham_rows demonstrates the batched path.

**Evidence:** An STFT with n=1_000_000, win=1024, hop=256 has ~3900 frames: the ROCm lane pays ~3900 candidate-discovery passes, ~3900 device availability probes (see separate finding), and ~3900 individual host-pointer transforms with two PCIe/staging transfers each, where a single ts_fft_stockham_amd_hostptr_batch call over a (3900, 1024) row matrix would do one transfer pair and one launch. Per-frame Python overhead plus per-frame transfer latency dominates by orders of magnitude over the batched kernel cost.

**Fix:** Frame the signal into a contiguous (frames, win) matrix once (windowing included), resolve the FFT lane once per call, and execute via the batch ABI (run_rocm_stockham_rows / a batched CPU loop inside one ctypes call); mirror the same structure in ISTFTCandidate.

**Independently verified:** Traced STFTCandidate.run (1308-1327) and ISTFTCandidate.run (1341-1365): both loop frames in Python and call RFFTCandidate.run/IRFFTCandidate.run per frame, each of which calls `_inner_fft` (1004-1023), which re-imports/re-enumerates `candidates_for()`, constructs a fresh SpectralFFTRegion, and re-runs `applies_to`+`available()` per frame before executing ONE single-row transform. The ROCm lane's run (954-968) uses `ts_fft_stockham_amd_hostptr` (single row), while the batch entry point `ts_fft_stockham_amd_hostptr_batch` is bound in `_configure_amd_lib` at line 134-138 and the plan-based batched path `run_rocm_stockham_rows` (611-637) exists in the same file — so the claim that a batched ABI is available and unused is exactly right. Measured the CPU lane as a proxy: SpectralSTFTRegion(65536,64,32) = 2047 frames took 402.9 ms on the composed lane vs 11.0 ms for the numpy reference and 1.6 ms for one batched np.fft.rfft over the framed matrix — i.e. the 'native' lane is 36x slower than the reference it exists to beat. Isolating the components, per-frame dispatch enumeration alone is ~5.6 us vs 4.1 us for the raw 64-pt kernel (a modest ~35% tax on CPU); the ROCm per-frame H2D/launch/D2H cost cannot be measured on this Mac, but the structural defect and the unused batch ABI are proven from source. Caveat on severity: no production caller constructs these regions today (only tests + the arbiter registry), so the impact is currently prospective.

### `src/transforms/lib/TesseraToLinalgPass.cpp:229` — "ne" lowered as ordered ONE, wrong for NaN

*C++ — Tessera→linalg lowering · math*

**What is wrong:** orderedFloatPredicate maps "ne" to arith::CmpFPredicate::ONE (ordered not-equal). IEEE-754 and numpy define != as the negation of ==, i.e. unordered not-equal (UNE): any comparison involving NaN is unequal. ONE returns false when either operand is NaN, so tessera.ne (via BinaryComparisonLowering) and tessera.compare_scalar with predicate="ne" (CompareScalarLowering, line 330) produce the wrong mask element wherever a NaN appears — the exact situation (NaN-screening masks like x != x, loss-scaler overflow checks) where ne-with-NaN is used in practice. eq/lt/le/gt/ge as ordered predicates all match numpy (NaN compares false), so "ne" is the one entry in the table that deviates from the numpy reference semantics.

**Evidence:** x = [NaN, 1.0], y = [NaN, 2.0]: numpy np.not_equal gives [True, True] (np.nan != np.nan is True); the lowered ONE compare yields [false, true]. The idiomatic NaN detector "x != x" returns all-false instead of flagging the NaNs.

**Fix:** Map "ne" to arith::CmpFPredicate::UNE in orderedFloatPredicate (and audit the ONE uses in SelectLowering line 216 and MaskedFillLowering line 433, where a NaN condition selects the false branch while numpy treats NaN as truthy).

**Independently verified:** Independently re-derived and reproduced. `orderedFloatPredicate` maps "ne"→ONE; `tessera-opt --tessera-to-linalg` on `tessera.ne` emits `arith.cmpf one`. IEEE-754 compareQuietNotEqual / C `!=` / numpy `not_equal` are all unordered-not-equal: NaN != NaN is true (verified: `np.not_equal(nan, nan)` → True), while ONE yields false. The Tessera reference for this op IS numpy — `python/tessera/__init__.py:4332` defines `ne(x,y) = np.not_equal(...)` — and the sibling backend already does it correctly: `src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateROCMCompareKernel.cpp:51` uses `arith::CmpFPredicate::UNE` for Cmp::Ne. eq/lt/le/gt/ge as ordered predicates do match numpy, so "ne" is the lone deviation, exactly as claimed. No lit fixture or unit test pins `one` for tessera.ne (the NaN-semantics tests in tests/unit/test_{x86,rocm}_compare_compiled.py assert the numpy answer, on other lanes). Note the cited SelectLowering/MaskedFillLowering ONE uses are a `!= 0` truthiness test, a weaker/secondary point.

### `src/transforms/lib/AutodiffForwardPass.cpp:325` — Loop-carried activity not fixed-pointed; tangents silently zeroed

*C++ — MLIR autodiff passes · math*

**What is wrong:** `buildFor` (line 325) and `buildWhile` (lines 384-389, 429-434) mark a region iter-arg active only if its *initial* value is active. A carried value that becomes active after iteration 1 keeps its tangent iter_arg (line 323) but every op whose operands are all such args is classified inactive, so its result tangent is left null and later reads produce a static zero (tangentOrZero, line 98) — the JVP is wrong with no diagnostic.

**Evidence:** scf.for carrying hidden state h with a constant zero init (inactive) and body `t = tanh(h); h' = t * w` with w in wrt: `tanh(h)` has only the inactive-marked operand h, so `active=false` and t's tangent is stored null even though h's tangent iter_arg holds the nonzero tangent from the previous iteration; `t*w` then reads tangentOrZero(t)=0. The dh·∂h'/∂h recurrence term is dropped every iteration, so d h_T/dw counts only the final iteration's direct contribution for T>=2. Central-difference check on 2 iterations diverges by O(1).

**Fix:** Iterate loop activity to a fixed point, or conservatively mark every tangent-carrying iter-arg active (matching the tangent mapping the comment at lines 318-324 already installs).

**Independently verified:** Independently re-derived and reproduced. I built a counterexample: scf.for with two iter args both init'd to a zero constant (inactive), body `%u = addf %h,%h` (all operands inactive-marked) and `%hn = addf %h,%w` (w in wrt, so active), yielding (%hn, %u). Running `--tessera-autodiff-forward` emits a JVP whose h-tangent slot correctly recurs (`%3 = addf %arg5, %arg1`) but whose g-tangent slot yields `%cst_2 = arith.constant dense<0.0>` — the static zero from tangentOrZero(line 98), because buildLeaf marked %u inactive (state.active only contains iter args whose *init* was active, line 325) and stored a null tangent for it. True answer: g_2 = 2w so dg = 2·dw; the emitted JVP returns 0. Silently wrong, no diagnostic. Caveat: the reviewer's own evidence sketch (`t=tanh(h); t*w`) would NOT silently zero — inputTangents[0] is a stored null and the MulF path returns failure at line 165 — so the exact example is imprecise, but the claimed mechanism (activity from init only, null result tangent, static zero on later read) is exactly what the yield path does. buildWhile (lines 384-389/429-434) is worse still: it drops the tangent block-arg mapping entirely for inactive-init state. Fix (fixed-point, or conservatively mark tangent-carrying iter args active) is right.

### `src/transforms/lib/TileToX86Pass.cpp:190` — Composed-layout wraps slowest basis leaf, aliasing addresses

*C++ — tiling & Tile IR lowering · math*

**What is wrong:** materializeX86ComposedLayouts applies RemUIOp/DivUIOp to every basis leaf including the slowest (lines 189-195). CuTe-style layouts are affine beyond their declared shape: the correct materialization retains the entire remaining quotient in the slowest basis mode. The x86 version wraps any coordinate past the declared basis extent back to 0, silently computing a wrong (aliased) linear address whenever product(basisShapes) < outerShape.

**Evidence:** The ROCm sibling (src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp:2199-2214) implements the same shared carrier with an explicit `isSlowest` guard and the comment: applying rem/div to every leaf 'wraps coordinates at the declared extent (for example column 16 back to column 0 in a [16]-basis map)'. Concrete case: basisShapes=[16], basisStrides=[1], outerShape=32, coordinate=16 — passes the `coordinate < outerShape` assert (line 169-173), then x86 computes digit = 16 % 16 = 0 → address 0, colliding with coordinate 0; ROCm correctly computes 16. Two consumers of one shared layout authority now disagree on addresses.

**Fix:** Mirror the ROCm implementation: for the last basis leaf use `remaining` directly as the digit and skip the final rem/div.

**Independently verified:** Re-derived independently. The layout authority is LayoutAlgebra.cpp:344-368 `logicalToPhysical`, which for `i+1 == shape.size()` sets `coord[i] = logical` (full remaining quotient, no rem/div) — explicitly commented 'A CuTe layout is affine beyond its declared shape'. Both sibling consumers implement that: TileToROCM.cpp:2196-2213 and NVIDIALowering.cpp:5000-5012 each guard with `isSlowest`. TileToX86Pass.cpp:189-195 applies RemUIOp/DivUIOp to every leaf including the last, so x86 is the lone outlier among three consumers of one shared carrier. Reachability: getMaterializableComposedLayout (TileOps.cpp:101-146) imposes no relation between outerShape and product(basisShapes), so outerShape > basis product is admissible — and the committed execution fixture itself already contains such a mode (tests/tessera-ir/phase2/x86_composed_layout_exec.mlir: outer shape %m=17 over a [16] basis). It passes today only because the probed coordinate is 3; coordinate 16 clears the x86 `coordinate < outerShape` assert (lines 169-173) and then wraps to digit 0, where ROCm/NVIDIA/LayoutAlgebra give 16. No test pins the wrapping behavior as intentional.

### `src/transforms/lib/TileIRLoweringPass.cpp:306` — Identical dropout mask replicated across batch and head

> **FIXED 2026-08-30 — as an operand, but on `block_dropout`, not on
> `flash_attn`; and the severity is lower than reported.** The deferral was
> right that a per-instance seed cannot be an attribute (the batch/head
> coordinates are `scf.for` induction variables). It was wrong about which op
> has to change and about the blast radius.
>
> **Correction to the finding's evidence.** "No backend can recover the
> distinction" does not hold. The only backend that executes dropout attention
> is ROCm, and it does not consume `block_dropout`'s operands at all:
> `TileToROCM.cpp:1549` records a bare `dropout = true/false` on
> `tessera_rocm.flash_attn`, and the generated kernels rebuild the mask from
> launch geometry — `counter = ((bh*Sq)+q)*Sk + k` in *both*
> `GenerateWMMAFlashAttnKernel.cpp:418` and `...BwdKernel.cpp:482`, with `bh`
> the fused batch-head block id. So the **executed** gfx1151 masks are already
> iid across batch and head, and forward/backward already agree. Apple refuses
> dropout outright (`APPLE_STREAMING_ATTN_DROPOUT_UNSUPPORTED`). The defect is
> therefore a **Tile-IR contract defect** (Decisions #29/#32 — the shared
> boundary understated what the physical kernels do), not a wrong numerical
> result on any lane that runs today.
>
> **Fix.** `tessera_attn.block_dropout` gained an `Index:$stream_offset`
> operand; `LowerFlashAttnToTileIR` derives the instance index `b*H + h` from
> the `tessera.attention_distribution` batch/query_head loops this same pass
> already annotates (Decision #30 — derive, don't ask) and passes
> `(b*H + h) * Sq * Sk_padded`, the same disjoint counter block the WMMA
> kernels use. A rank-2 attention that was never distributed passes 0 and its
> mask is bit-identical to before; a distributed instance whose annotated loops
> are unreachable **fails the match with a diagnostic** rather than silently
> emitting stream 0 (Decision #21a). `tessera.flash_attn` is untouched, so no
> `operandSegmentSizes` churn.
>
> **Why the ROCm lane cannot regress this time** (stated, not assumed — this
> Mac skips `streaming_attention_backward_rocm.mlir` via
> `REQUIRES: tessera-rocm-backend`, so it is unverified here): no fixture in
> the repo parses `tessera_attn.block_dropout` textually; `ROCMWaveLdsPipeline`
> never mentions attention or dropout ops; and `TileToROCM` tests only
> `blockDropout != nullptr`. The only change in that fixture's output is one
> more operand printed on `block_dropout`, which no CHECK line matches. **A
> lit run on the ROCm box is still owed.** Host-free coverage added at
> `tests/tessera-ir/phase3/streaming_attention_dropout_stream.mlir` (positive
> rank-4 case + `stream = %c0` rank-2 negative).


*C++ — tiling & Tile IR lowering · math*

**What is wrong:** DistributeRank4FlashAttn copies all attributes verbatim (including dropout_seed) onto every per-(batch,head) rank-2 flash_attn (line 306), and the rank-2 lowering seeds tessera_attn.block_dropout with only (seed, boundary) where boundary restarts at 0 for every slice (lines 592-603, 484-491). Every one of the B*H slices therefore draws the exact same dropout mask — dropout is fully correlated across batch and head instead of iid, and violates the repo's RNG stream-separation discipline (Decision #18).

**Evidence:** Input: rank-4 flash_attn [B=8,H=16,S,D] with dropout_p=0.1, dropout_seed=42. All 128 rank-2 instances emit block_dropout {dropout_p=0.1, seed=42} with the same boundary iter-arg sequence 0,tkv,2*tkv,... — identical (position -> dropped) masks in all heads/batches. Expected variance reduction from averaging over heads is destroyed and gradient noise is systematically correlated.

**Fix:** Fold the batch and head induction values (or a per-slice stream offset per Decision #18) into the seed/offset carried to block_dropout when distributing rank-4.

**Independently verified:** Traced and stands. TileIRLoweringPass.cpp:306 does `rank2State.addAttributes(op->getAttrs())`, copying dropout_p/dropout_seed verbatim onto every per-(batch,head) rank-2 flash_attn, and the batch/head induction values (batchIndex, queryHead) are used only for the tensor slice offsets (lines 243-266), never folded into any RNG key. The rank-2 lowering then emits `tessera_attn.block_dropout` with exactly {dropout_p, seed} plus the `boundary` iter-arg (lines 592-604), and `boundary` is a fresh loop-carried value initialized to zeroIndex at line 491 inside each slice's own kv loop, stepping 0,tkv,2*tkv,... identically for every slice. I checked the op's ABI (Attn.td:193-205 BlockDropoutOp = scores, kv_offset, dropout_p, seed) — there is no batch/head channel at all, so no backend can recover the distinction. Result: all B*H instances draw the same (position -> dropped) mask. No guard blocks this: DistributeRank4FlashAttn has no dropout check, and the rank-2 pattern only requires that a seed exist when p>0 (lines 394-397). This contradicts the per-stream RNG discipline of Decision #18.

### `src/transforms/lib/TileIRLoweringPass.cpp:1210` — Residual check misses tessera_attn.backward and schedule.prefetch

*C++ — tiling & Tile IR lowering · logic*

**What is wrong:** The Decision-#21 completeness walk (lines 1208-1221) refuses to report success over surviving tessera.flash_attn/matmul/control_* ops, but omits `tessera_attn.backward` and `schedule.prefetch`, both of which this pass registers patterns for. A backward op that fails its guard (e.g. split_count missing or < 2, dynamic shapes, missing query_block/key_block — lines 659-679) silently survives, and the pass reports the module as successfully lowered — exactly the silent partial-lowering outcome the walk's own comment says it refuses.

**Evidence:** Input: `tessera_attn.backward` with `split_count = 1` (a perfectly plausible single-split configuration). LowerAttentionBackwardToLoops returns failure at line 678 (`splitCount < 2`); no other pattern matches the op; applyPatternsGreedily returns success; the residual walk does not list the op name; the pass succeeds with an unlowered Graph-IR backward op embedded in a 'Tile-IR' module, which downstream backends will reject far from the cause or drop.

**Fix:** Add "tessera_attn.backward" and "schedule.prefetch" to the residual name list.

**Independently verified:** The defect stands, though the reviewer's evidence is wrong on both specifics. The residual walk (TileIRLoweringPass.cpp:1208-1221) lists only tessera.flash_attn/matmul/control_*, while the pass registers LowerAttentionBackwardToLoops for tessera_attn.backward (line 1180) and LowerSchedulePrefetchToTileCopy for schedule.prefetch (line 1185). Corrections: (a) split_count = 1 is impossible — BackwardOp::verify (AttnOps.cpp:526-528) rejects split_count < 2 and non-positive blocks, and query_block/key_block are required ODS attrs; (b) schedule.prefetch's guard (`numOperands != 1 || numResults != 1`, line 1026) can never fail, since Schedule_PrefetchOp is fixed 1-in/1-out (ScheduleMeshPipelineOps.td:556-566), so adding it to the list is inert. But the pattern also requires hasStaticShape() on q/k/dQ/dK/dV (lines 662-667) while the op verifier admits dynamic dims (attnDimsAgree, AttnOps.cpp:91-94). I built that input and ran it: `tessera-opt --tessera-tile-ir-lowering` on a valid dynamic-shape tessera_attn.backward printed the op unchanged and exited 0 — exactly the silently-partially-lowered module the walk's own comment says it refuses. The fix is the backward name; the prefetch half is not needed.

### `src/transforms/lib/LowerControlFlowToSCFPass.cpp:184` — Predicate extraction assumes ranked float tensor, crashes otherwise

*C++ — analysis passes · logic*

**What is wrong:** extractPredicateI1 unconditionally builds tensor.extract + getFloatAttr + arith.cmpf. lowerControlWhile only validates that @cond has one result (lines 396-399); the result type is never checked. A cond returning tensor<i1>/tensor<i32> hits b.getFloatAttr(i1, 0.0), which asserts inside FloatAttr::get (UB on the fleet's NDEBUG LLVM), and cmpf on integer operands is invalid IR; a non-tensor scalar result feeds tensor::ExtractOp a non-tensor operand. lowerControlIf has the same hole for an integer flag operand. This crashes instead of emitting a stable diagnostic or skipping, violating the unsupported-path convention.

**Evidence:** tessera.control_while whose @cond is (tensor<4xf32>) -> tensor<i1> (a boolean condition is the most natural user-written form): calleeMatches on body passes, condFn checks pass (one input, one result), predTy = tensor<i1>, then extractPredicateI1 executes getFloatAttr(i1Ty, 0.0) → assert/UB. Same for control_if with an i32 flag tensor.

**Fix:** In extractPredicateI1 (or its callers) verify the predicate is a RankedTensorType with a FloatType element; otherwise return failure and treat the op as Skipped (or emit CmpIOp sgt for integer elements).

**Independently verified:** Reproduced a hard crash on verifier-valid IR. `extractPredicateI1` (line 174) never checks that the predicate is a ranked tensor of float element type, and neither ControlIfOp::verify nor ControlWhileOp::verify (src/compiler/ir/TesseraOps.cpp:4700/4717) constrains the flag/cond type — the operands are `Variadic<AnyType>`. With a non-tensor `f32` flag operand (`tessera.control_if` with flag_arg_index=0 over an `f32`), tessera-opt segfaults (exit 139) with the stack showing `mlir::FloatAttr::get(mlir::Type,double)` called from `LowerControlFlowToSCF::extractPredicateI1` -> `lowerControlIf`; the IR passed the op verifier first, so nothing upstream blocks it. The integer-tensor variants (`cond` returning tensor<i1>, `control_if` flag tensor<1xi32>) do not crash on this NDEBUG LLVM but produce invalid IR reported as a generic `'arith.cmpf' op operand #0 must be floating-point-like` failure rather than the stable Malformed/Skipped diagnostic the pass's own convention requires. Fix direction (verify RankedTensorType + FloatType element, else failure/Skipped) is correct.

### `src/transforms/lib/SymbolicDimEqualityPass.cpp:902` — scf.while result seeding indexes out of bounds

*C++ — analysis passes · logic*

**What is wrong:** In the scf.while handler, whileOp results are seeded inside the loop over the after-region yield operands (lines 891-903). Yield-operand count equals the init/before-arg count (expectedNames.size()), but getNumResults equals the condition-arg/after-arg count — scf.while permits these to differ. When inits > results, whileOp.getResult(i) is called with i >= getNumResults(): assert in debug, silent UB on the fleet's NDEBUG LLVM. Separately, results correspond to the condition-forwarded values, not the yield operands, so even in-bounds seeding maps names by the wrong index when the before/after signatures differ.

**Evidence:** Valid IR: scf.while (%a = %x, %b = %y) : (tensor<..>, tensor<..>) -> tensor<..> — before region 2 args, condition forwards 1 value, after region 1 arg, after yield yields 2 values. expectedNames.size() = 2, loop reaches i = 1, whileOp.getResult(1) is out of range on a 1-result op. Triggered whenever the enclosing function has any dim-name flow seed (tessera.arg_dim_names).

**Fix:** Seed results in a separate loop bounded by getNumResults(), taking names from the propagated facts of condition.getArgs()[i] (which the code already validated against expectedNames).

**Independently verified:** Re-derived and reproduced. scf.while permits inits/before-args (== yield operand count) to differ from results (== condition-arg/after-arg count); the loop at lines 891-903 is bounded by `yield.getNumOperands()` (checked equal to expectedNames.size() at 889) yet indexes `whileOp.getResult(i)`. Built the reviewer's exact shape — `scf.while (%a = %x, %b = %y) : (tensor,tensor) -> tensor` with a 1-value scf.condition, 2-operand after-yield, and a `tessera.arg_dim_names` seed — and tessera-opt aborts with `Assertion failed: (resultNumber < getNumResults() && "Result number is out of range for operation"), function getOpResultImpl, Operation.h:1042`, stack frame in SymbolicDimEquality::runOnOperation. The secondary claim is also right: results correspond to the condition-forwarded values, so even in-bounds seeding maps names by the wrong index when the before/after signatures differ; the suggested fix (seed from condition.getArgs() facts, bounded by getNumResults()) is the correct shape.

### `src/transforms/lib/AdjointCollectiveInsertionPass.cpp:141` — Silent no-op drops gradient collectives entirely

*C++ — legality / verifier passes · logic*

**What is wrong:** When the function body's first block does not end in func.return (any multi-block function, e.g. after control-flow lowering), or when tessera.autodiff.arg_cotangents is out of sync with the return rewrite (origResultCount < 0, line 154), the pass silently returns without inserting any reduce_scatter/all_gather/all_reduce and without emitting a diagnostic. The repo convention (Decision #21 / 'unsupported paths must emit a stable diagnostic, never silently no-op') is violated exactly where it matters most: distributed gradients are left unsynchronized with no trace.

**Evidence:** A func.func carrying tessera.autodiff="reverse" + tessera.weight_sharding whose region has two blocks (cf-based control flow puts func.return in a later block): dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator()) is null, runOnOperation returns, zero collectives inserted, no error. Each DP rank then trains on its own unsummed gradients — silently divergent replicas. An external declaration (empty body) makes front() itself UB.

**Fix:** Locate the return by walking all blocks (or func.getBody().back()); on any bail-out path (no return op, origResultCount < 0, empty body) emit a stable diagnostic (e.g. ADJOINT_COLLECTIVE_UNSUPPORTED_SHAPE) and signalPassFailure instead of returning silently.

**Independently verified:** Reproduced with the built tessera-opt. A func carrying tessera.autodiff="reverse" + arg_cotangents + weight_sharding whose entry block ends in cf.br (return in ^bb1) passes through --tessera-adjoint-collective-insertion unchanged, exit 0, zero collectives, no diagnostic (AdjointCollectiveInsertionPass.cpp:139-141). The external-declaration case is worse: `func.getBody().front().getTerminator()` on an empty body aborted ('dyn_cast on a non-existent value', Casting.h:656). Partial mitigation, not a refutation: inside the registered tessera-autodiff-pipeline, AutodiffPass (AutodiffPass.cpp:155-160) hard-errors on a non-first-block return, so the multi-block shape cannot arrive from that producer — but this pass is registered standalone and all four phase_f5 fixtures feed it hand-written attributes, so the silent-drop input class is exactly what it accepts. Violates the repo's no-silent-no-op convention (Decision #21) on the gradient-sync path.

### `src/transforms/lib/MetadataObligationPass.cpp:250` — Module-level drop declaration falsely stale in other scopes

*C++ — legality / verifier passes · logic*

**What is wrong:** droppedFor() falls back to the module-level tessera.lowering.dropped dictionary for every scope that lacks a function-level one, but the stale-declaration check (lines 415-429) is evaluated per scope against that scope's local declarationExplainedSomething set. A module-level declaration that legitimately explains a drop in one function is therefore re-checked in every other snapshotted function, where it explains nothing, and METADATA_OBLIGATION_STALE_DECLARATION fails the whole verify on valid, correctly-declared IR.

**Evidence:** Module with @a (matmul carrying numeric_policy, dropped across the boundary, declared via module-level tessera.lowering.dropped = {numeric_policy = "consumed_by_pass"} — the documented whole-module form, line 244 comment) and @b (any function carrying only a layout attribute so it appears in the snapshot). Verify pass: @a is explained, but for scope @b the same module dict is consulted, numeric_policy was never recorded there, wasRecorded=false, and the pass emits 'declares numeric_policy dropped, but it was never present before the boundary either' and signalPassFailure — a false rejection any 2-function module hits.

**Fix:** Track which declarations were consumed globally (across all scopes) before running the stale check, or run the stale check for the module-level dictionary once against the union of all scopes rather than per scope.

**Independently verified:** Reproduced. Module with tessera.lowering.dropped = {numeric_policy = "consumed_by_pass"} at module level, snapshot scopes @a (numeric_policy, legitimately dropped) and @b (layout only): @a is explained, and @b emits 'METADATA_OBLIGATION_STALE_DECLARATION: @b declares `numeric_policy` dropped, but it was never present before the boundary either' + signalPassFailure. Cause is exactly as claimed: droppedFor() (line 244-251) hands the module dict to every scope, while declarationExplainedSomething (line 354) and the stale loop (415-429) are per scope. The accepting fixture metadata_obligation_declared.mlir pins the module-level form only in a single-function module, so no test covers the two-scope case. A module-level declaration is therefore usable only when every snapshotted scope drops that attribute.

### `src/solvers/tpp/lib/Passes/HaloInfer.cpp:66` — Halo radius formula ignores scheme; wrong for one-sided stencils

*C++ — spectral / Clifford / TPP solvers · math*

**What is wrong:** gradHalo computes radius = max(1, order/2) unconditionally. That is the reach of a CENTRAL difference only. Upwind/ENO/WENO schemes (all accepted by LegalizeSpaceTime's validSpatialScheme) use one-sided stencils whose reach is up to `order` cells, so the inferred halo under-allocates ghost cells and the distributed exchange ships too few layers.

**Evidence:** tpp.grad {scheme="upwind", order=2}: the second-order upwind stencil (3u_i - 4u_{i-1} + u_{i-2})/(2h) reads 2 cells upwind, but halo is inferred as order/2 = 1. After -tpp-distribute-halo, the cell adjacent to a partition boundary reads an unexchanged (stale/garbage) ghost value — wrong PDE results exactly at subdomain seams, on every multi-rank run with a non-central scheme. The pass never reads the `scheme` attribute at all.

**Fix:** Read `scheme` and use radius = order for upwind/eno/weno (or the scheme's true stencil reach), keeping order/2 for central; fail closed on unknown schemes.

**Independently verified:** HaloInfer.cpp:61-75 (gradHalo) reads only 'order' and 'axis'; grep confirms the file never mentions 'scheme'. radius = max(1, order/2) is the central-difference reach (the file's own header comment says 'A central scheme of accuracy order ... reaches +/- order/2 cells'). LegalizeSpaceTime.cpp:38-40 admits central|upwind|weno|eno and only requires a positive even order, so tpp.grad{scheme="upwind", order=2} is a legal, legalized program and gets halo=1 while the 2nd-order upwind stencil (3u_i-4u_{i-1}+u_{i-2})/(2h) reaches 2 cells upwind (I re-derived: Taylor-eliminating the O(h^2) term over {i,i-1,i-2} needs the i-2 tap). So the inference is mathematically wrong for a scheme the pipeline accepts, and DistributeHalo.cpp:95-103 copies that width straight into the tpp.halo.exchange plan. CAVEAT that downgrades severity below P1: the claimed wrong PDE results are not reachable today — no lowering honors 'scheme' at all (ts_stencil_grad_cpu has no scheme parameter and always computes central), and LowerTPPToTargetIR.cpp:51-54 marks ts_halo_exchange artifact_only on every backend, so there is no executing multi-rank exchange. The defect is a real latent metadata/semantic-key error, not a live miscompile.

### `src/solvers/scaling_resilience/lib/sr/passes/InsertRecomputePass.cpp:56` — isPureOp fails open: effectful ops marked recomputable

*C++ — spectral / Clifford / TPP solvers · logic*

**What is wrong:** For ops without a tessera.effect attribute, purity is decided by a name-substring blacklist (alloc/store/dealloc) plus region count, and everything else is 'assumed pure' and tagged tessera_sr.recompute_hint="recomputable". This is fail-open: RNG ops, func.call, collectives, and IO ops that lack the attribute (EffectAnnotationPass not run, or an op the annotator missed) are marked safe to recompute.

**Evidence:** A `tessera.rng.uniform` or `func.call @dropout_mask` op with no effect attr, no regions, and none of the three substrings in its name gets recompute_hint. A backward pass that honors the hint re-executes the RNG with a different result than the forward pass — silently wrong gradients (the exact aliased-RNG failure mode Decision #5/#30 records as the canonical scar; the comment even claims 'conservatively assumed pure', which is the opposite of conservative). Mirrors Decision #10a: eligibility marking must gate on analysis, with unprovable treated as unsafe.

**Fix:** Invert the default: only tag ops whose effect attribute (or a real side-effect interface query) proves purity; absence of evidence joins to not-recomputable.

**Independently verified:** InsertRecomputePass.cpp:51-60. With no tessera.effect attribute the function returns true whenever the op has zero regions and its name lacks the substrings alloc/store/dealloc — so func.call, a collective, an IO op, and any tessera.rng.* op are 'pure' and get tessera_sr.recompute_hint="recomputable" at line 124-127. The header comment ('Ops with side effects are never recomputable') and the inline comment ('conservatively assumed pure') both assert the opposite of what the code does. Nothing forces EffectAnnotationPass to run first: the pass declares no dependency, and its own fixture src/solvers/scaling_resilience/tests/sr/checkpoint.mlir runs `tessera-opt -tessera-insert-recompute` standalone, so absence of the attribute is the normal case. This is precisely the Decision #10a pattern (eligibility marking without a gating analysis) and the Decision #30 fail-open scar. Severity caveat: I grepped for a consumer of tessera_sr.recompute_hint and found none on the MLIR side (only python/tessera/compiler/checkpoint.py, an independent annotator), so the wrong-gradient outcome is latent rather than currently live — but per Decision #29/#10a an eligibility mark that a downstream pass will believe is the defect.


## P2 — real but minor (42) — CLOSED 2026-08-29

> 39 fixed in one batch; 3 needed no change. The three are, with the evidence
> that settled each:
>
> * **`vjp.py:3756` instance/group-norm affine grads** — already fixed by the P0
>   batch (`5a0a56ed` added `_affine_param_grads`). Re-checked against a
>   float64 finite-difference reference: `dW = [-1.4762, 0.4863, -0.1934,
>   3.8271]` vs FD `[-1.4760, 0.4864, -0.1936, 3.8270]`.
> * **`rocm_hip.py:282` paged attention expf** — the cooperative normalize pass
>   the finding asks for landed in the same P0 commit; the review quoted the
>   pre-fix kernel body. The residual sub-claim about page-table resolution
>   does not hold at production shapes (blockDim 256, head dim <= 256 means one
>   `d` iteration per thread).
> * **`AutodiffPass.cpp:199` f32 loss seed** — closed by the P0 fix, which
>   builds the seed from `lossValue.getType()`.
>
> Two findings surfaced work outside their own row and were routed rather than
> folded in silently: `jvp_log_cosh_loss` carried the same overflow the eager
> loss was fixed for (returned `inf` where the loss returned 399.3), and
> `vjp_cross_entropy_loss` swallowed `label_smoothing`, `ignore_index` and
> `axis` in `**_` — its gradient was 0.15 away from the finite-difference
> answer under smoothing. Both are fixed here and pinned against central
> differences of the eager loss.
>
> One finding was **escalated**: `ActivationRematerializationPass`'s peak scan
> is fixed (59.2s -> 15.0s at N=2000, and 14.6s -> 0.6s at a looser budget on
> the Mac), but profiling showed the peak scan was not the dominant cost. The
> pass emits a *quadratic number of operations* for a deep recompute chain —
> a 4,001-op function expanded to 2,001,002 ops — because its cost model prices
> only each op's own recompute, not the transitive prefix it drags along. That
> is filed as separate work, not silently folded into this batch.


### `python/tessera/compiler/profiler_rocm_native.py:255` — pre-run snapshot fails OPEN on a partial read

*Added 2026-08-29, after the review — found in the fix for the mtime clock race (PR #637). Deferred to the P2 batch by owner decision.*

**What is wrong:** `_snapshot_files` builds the pre-run baseline, and on any `OSError` it discards what it has collected and returns `{}`. `_files_written_since` then compares against an empty baseline, so every file already in the output directory looks newly written.

**Evidence:** A file disappearing between `is_file()` and `stat()` — another process cleaning or rotating a reused output directory — or a permission error mid-walk empties the baseline. If the traced application then exits successfully without producing records, the stale traces from an earlier run are parsed as this run's output and the capture reports `status: "collected"` on someone else's evidence. That is the mirror of the defect #637 fixed (a spurious `blocked`) and the worse direction of the two: a false negative wastes time, a false positive is a claim-integrity failure.

**Fix:** Distinguish "the directory was empty" from "no baseline could be established". Either keep the successfully captured entries and record the failed paths, or return a sentinel for an incomplete walk and have both collectors refuse the capture — the fail-closed form matches this file's own contract, where a blocked capture must carry a reason.


### `python/tessera/autodiff/vjp.py:3756` — instance/group norm affine params get silent None gradients

*Autodiff — reverse mode (VJP) · logic*

**What is wrong:** vjp_instance_norm (and vjp_group_norm, once its crash is fixed) applies the weight to the incoming cotangent but returns None for the weight and bias slots ('not yet wired'). Affine parameters therefore receive zero gradient with no diagnostic — they silently never train — a silent no-op on a semantic path.

**Evidence:** instance_norm forward (nn/functional.py:129) computes y*weight + bias; the correct d_weight = sum over (N,*spatial) of dout*x_hat and d_bias = sum of dout are cheap and well-defined, yet the rule returns (grad_x, None, None), so grad() reports no gradient for trainable affine parameters without any error.

**Fix:** Return d_weight = (do_raw * x_hat).sum over all axes except channel and d_bias = do_raw.sum over the same axes (using the pre-weight cotangent), or raise a stable diagnostic when weight/bias are supplied.

**Independently verified:** Verified in source: vjp_instance_norm returns (grad_x, None, None) and vjp_group_norm (grad_x, None, None, None) with only a 'weight/bias grads not yet wired' comment (vjp.py:3743, 3756), while the forwards apply y*weight+bias. No diagnostic is raised when weight/bias are supplied, so affine parameters receive no gradient silently. The claimed d_weight/d_bias formulas are the standard well-defined adjoints. Note the group_norm half is currently masked by finding 0's crash whenever weight is passed.

### `python/tessera/autodiff/vjp.py:223` — vjp_gemm recomputes A@B even for plain matmul

*Autodiff — reverse mode (VJP) · performance*

**What is wrong:** vjp_gemm unconditionally evaluates preactivation = np.matmul(A, B) (plus the activation-derivative array of ones) before computing dA/dB, even in the dominant case activation=='none' and bias is None where the preactivation is never needed. Every matmul/gemm backward — the hottest op on the tape — pays a third full GEMM, ~50% extra FLOPs over the two adjoint GEMMs actually required, plus an extra output-sized temporary.

**Evidence:** vjp_matmul delegates to vjp_gemm, so every ops.matmul backward in every model incurs the extra np.matmul(A,B); for a (4096,4096)x(4096,4096) matmul that is ~137 GFLOP of pure waste per backward call.

**Fix:** Guard: if activation == 'none', set local_dout = dout (adding bias never changes the derivative-of-identity), computing preactivation only when activation != 'none'.

**Independently verified:** Verified at vjp.py:223: preactivation = np.matmul(A, B) is computed unconditionally, plus _matmul_activation_derivative returns np.ones_like(value) for activation='none', so the dominant matmul path pays a full third GEMM plus an output-sized ones array that multiplies dout by 1. vjp_matmul delegates to vjp_gemm, so every matmul backward pays it. The proposed guard is mathematically valid: with activation='none' the post-bias derivative is identity regardless of bias, so local_dout = dout and the bias cotangent _sum_to_shape(dout, bias.shape) is unchanged. Real cost on the reference-autodiff training path, though it is numpy reference code, not a device kernel.

### `python/tessera/autodiff/operator.py:99` — matvec float64 cast silently corrupts complex operators

*Autodiff — tape / grad / transforms · math*

**What is wrong:** OperatorTangent.matvec unconditionally casts input (line 99) and output (line 105) to float64 via np.asarray(..., dtype=np.float64), which for complex vectors discards the imaginary part with only a ComplexWarning — not the fail-closed diagnostic the module's own docstring promises ('non-finite ... is an error, never a pass', Decision #21a). Any complex linear map (FFT/spectral Jacobians are in-repo consumers) returns a numerically wrong real result rather than raising. materialize() and the composed/adjoint paths inherit the same corruption.

**Evidence:** Reproduced: A = from_matvec_pair(lambda v: 1j*v, None, in_shape=(2,), out_shape=(2,)); A(np.array([1+1j, 2+0j])) returns [0., 0.] (true result is [-1+1j, 2j]); only a ComplexWarning is emitted, so the wrong value propagates into cg/gmres/certify_root silently.

**Fix:** Reject complex input with a TesseraOperatorError (or add a declared complex dtype mode); at minimum use np.asarray(v) and check np.iscomplexobj before the float64 coercion.

**Independently verified:** The mechanical defect reproduces: `OperatorTangent.from_matvec_pair(lambda v: 1j*v, None, in_shape=(2,), out_shape=(2,))` applied to [1+1j, 2+0j] returns [0., 0.] (true answer [-1+1j, 2j]) on numpy 2.5.2, emitting only two ComplexWarnings. Lines 99 and 105 both do an unconditional `np.asarray(..., dtype=np.float64)`, so the imaginary part is discarded at both the input and the output of every matvec, and `materialize()`/`__matmul__`/`.T` inherit it — no dtype check, no diagnostic. Two of the reviewer's supporting claims are wrong and I am recording the correction: the 'non-finite is an error, never a pass' promise in the module docstring is scoped to `certify_root`'s certificate evaluation, not to matvec dtype; and there is no in-repo complex consumer — every construction site (implicit.py:278/350/361/377, certify_root:321) is a real-valued finite-difference oracle, and no spectral/FFT code touches OperatorTangent. So the defect is real but latent: it requires a user-constructed complex operator, and the 'silently propagates into cg/gmres/certify_root' scenario is not reachable from current in-repo code.

### `python/tessera/autodiff/tape.py:756` — has_none_default re-parses signatures on every taped call

*Autodiff — tape / grad / transforms · performance*

**What is wrong:** promote_operand_kwargs.has_none_default calls inspect.signature(_innermost(original)) directly, bypassing the _FORWARD_SIG_CACHE that exists two functions above for exactly this cost. Every taped op call carrying kwargs whose rule declares an optional operand slot not present in the kwargs (e.g. every `ops.gemm(A, B, activation=...)` / `matmul(..., epilogue=...)` — the 'bias' slot triggers it) pays a full signature parse per call, potentially once per missing slot.

**Evidence:** Measured on this Mac: taped gemm with a kwarg costs 18.5µs/call vs 7.8µs/call without kwargs; inspect.signature on the innermost gemm alone is 6.9µs/call — ~37% of the total taped-call cost and >100% of the wrapper overhead, repeated with identical inputs across an entire training loop.

**Fix:** Cache parameter defaults alongside the names in _FORWARD_SIG_CACHE (e.g. store {name: default} in _positional_params' cache entry) and look them up in has_none_default.

**Independently verified:** Both the bypass and the cost are real. `has_none_default` (tape.py:754-761) calls `inspect.signature(_innermost(original), follow_wrapped=False)` directly, while `_positional_params` two functions above caches the same parse in `_FORWARD_SIG_CACHE` keyed on `id(fn)`. Instrumented count on this Mac: 100 taped `ops.gemm(A, B, activation='relu')` calls produce exactly 100 uncached `inspect.signature` calls; 100 taped `ops.gemm(A, B)` calls produce 0 (the `if not kwargs` early return). The path is entered because vjp_gemm's slots are `(dout, A, B, bias=None, residual=None, *, activation)`, so the loop `for k in range(len(args), len(operand_slots))` starts at k=2 on the unfilled `bias` slot. Timed: taped gemm without kwargs 11.8 us/call, with a kwarg 23.0 us/call, and `inspect.signature` on the innermost gemm alone 7.0 us/call — i.e. ~30% of the kwarg-path call cost, repeated identically every iteration of a training loop. The proposed fix (store {name: default} alongside the names in the existing cache entry) is valid; note the cache already keeps a strong ref to fn, so the id-reuse hazard is handled. Severity is arguably below P2 at realistic tensor sizes, where 7 us is dwarfed by the numpy kernel, but the claim as stated stands.

### `python/tessera/autodiff/degeneracy.py:422` — eigh_coupling missing the ill-conditioning warning band

*Autodiff — laws & algebra oracles · logic*

**What is wrong:** eigh_coupling's docstring claims 'Same declared policy, same three thresholds' as svd_coupling, but the conditioning_tolerance warning band is never wired: when a batch element has no degenerate clusters, svd_coupling calls _warn_if_ill_conditioned (line 638) while eigh_coupling just `continue`s (line 421-423). An eigenvalue gap between existence_tolerance (~k*eps) and sqrt(eps) proceeds with no TesseraDegeneracyWarning — exactly the 'silent answer that has lost most of its digits' the module says it exists to stop. Note _warn_if_ill_conditioned also hard-codes singular-value/s^2 wording, so an eigen-specific variant is needed.

**Evidence:** eigh backward on a symmetric matrix with eigenvalues [1.0, 1.0 + 1e-10, 5.0]: relative gap 1e-10 is above existence_tolerance (≈6.7e-16 for k=3) so no refusal, and below conditioning_tolerance (1.49e-8); the eigenvector coupling 1/(w_j - w_i) = 1e10 amplifies input eps-noise to ~1e-6 relative error, and nothing warns. The same spectrum passed through svd_coupling emits TesseraDegeneracyWarning.

**Fix:** Add an eigen-aware _warn_if_ill_conditioned (gaps on w itself, eigenvalue wording) and call it in eigh_coupling's `if not clusters:` branch, mirroring svd_coupling line 638.

**Independently verified:** Traced and reproduced. degeneracy.py:419-423 — eigh_coupling's `if not clusters: continue` has no warning call, while svd_coupling:637-640 calls _warn_if_ill_conditioned on the identical branch. Executed the reviewer's exact example: for spectrum [1.0, 1.0+1e-10, 5.0], existence_tolerance(3)=6.66e-16 and conditioning_tolerance=1.49e-8, so the gap sits squarely in the warning band; svd_coupling emits TesseraDegeneracyWarning ('relative gap 8.000e-12, below the sqrt(eps) half-digits threshold'), eigh_coupling emits nothing and returns a coupling matrix with max |F| = 5.0e9. The docstring at line 385 does claim 'Same declared policy, same three thresholds', and eigh_coupling is the only guard on both eigh backward (linalg_ops.py:502) and eigh forward-mode (linalg_ops.py:523), so nothing else warns. No test pins the silence as intentional — test_eigh_is_silent_on_a_well_separated_spectrum (test_factorization_degeneracy.py:245) uses diag([5,3,1]), whose gaps are far above sqrt(eps), so it would still pass with the warning wired. The note about _warn_if_ill_conditioned hard-coding s^2/singular-value wording is also correct (lines 541, 555, 557).

### `python/tessera/autodiff/implicit.py:140` — Small-norm RHS silently returns zero solution

*Autodiff — implicit / jets / remat · logic*

**What is wrong:** Both solvers use an effectively absolute convergence floor for small right-hand sides: gmres_solve uses threshold = tol * max(1.0, rhs_norm) and cg_solve (line 230) uses rs_old <= tol*tol with x initialized to zeros. Any RHS with norm below ~tol (default 1e-8) passes the initial convergence check immediately and returns x = 0 — a result with 100% relative error, since the true solution A⁻¹b scales linearly with b and can be arbitrarily large relative to b when A has small singular values.

**Evidence:** ihvp(fn, x, u) with a cotangent u of norm 1e-9 (routine for late-training gradients or small losses) returns exactly 0 instead of H⁻¹u, which for a Hessian with eigenvalue 1e-3 has true norm ~1e-6 — one thousand times the returned answer. Same for root_vjp/root_jvp: a cotangent seed of magnitude <1e-8 yields a zero implicit gradient with no diagnostic. Because VJPs are linear in the cotangent, correct behavior is scale-invariant; max(1.0, rhs_norm) breaks that invariance for ||b|| < 1.

**Fix:** Make the criterion relative: threshold = tol * rhs_norm (returning x=0 only for exactly-zero b), or tol*rhs_norm + atol with a separately declared absolute floor; apply the same normalized criterion in cg_solve.

**Independently verified:** Verified by code and execution. gmres_solve line 140 threshold = tol*max(1.0, rhs_norm) with x0=0 and matvec(0)=0 exactly for the FD oracle, so the line-145 early return fires whenever ||b|| <= tol; cg_solve line 230 has the same absolute floor (rs_old = ||b||^2 <= tol^2). Executed: gmres_solve on a well-conditioned 5x5 with ||b||=1e-9 returns exactly x=0 with converged=True, iterations=0, relative error 1.000e+00; same at 1e-12; cg_solve likewise relerr 1.000e+00. Scale-invariance check ||x(1e-9*b) - 1e-9*x(b)||/||1e-9*x(b)|| = 1.0. No caller normalizes: root_vjp/root_jvp/adjoint_state_grad/ihvp pass the raw cotangent or RHS through, so a sub-1e-8 cotangent yields a zero implicit gradient reported as converged with no diagnostic. No test in tests/unit/test_implicit_diff.py pins the absolute floor as intentional. One imprecision in the finding's wording: the invariance break is only material for ||b|| <~ tol (for tol < ||b|| < 1 the max(1.0,.) floor merely makes the tolerance stricter than relative, which is harmless), but the claimed defect at small ||b|| stands exactly as described.

### `python/tessera/autodiff/jet.py:330` — Fully-masked rows produce NaN in jet flash-attention

*Autodiff — implicit / jets / remat · logic*

**What is wrong:** jet_flash_attn (and jet_softmax at line 216) computes the order-0 shift as scores[0] - m_new. When a query row is masked in every key position — attn_bias = -inf across the whole row (the module's own 'canonical substrate' for padding masks), or k_len == 0 — m_new is -inf and the shift evaluates -inf - (-inf) = NaN, so exp propagates NaN through p, ell, and the output. The errstate guard at line 328 protects only the alpha rescale, not the shift. The final ell = 0 case (reciprocal of 0) is likewise unguarded.

**Evidence:** Q,K,V of shape (1, 4, 8) with attn_bias[0, 2, :] = -inf (a padded query row, standard varlen padding): row 2 of every output coefficient is NaN instead of a defined value, silently. Same NaN from jet_softmax on any slice that is entirely -inf. The internal causal mask cannot trigger this (col 0 is always allowed), so the bug is specifically reachable through the documented attn_bias masking path.

**Fix:** After computing m_new, guard the shift: where m_new is -inf, set the shifted score to -inf (p = 0) instead of performing the subtraction; define the all-masked-row output (zeros, per flash-attn convention) or fail closed with a diagnostic.

**Independently verified:** Reproduced, and the root cause is worse than the finding states. jet.py:325 m_new = max(m_run, blk_max); when the block max is -inf, line 330's _shift_order0(scores, m_new) evaluates -inf - (-inf) = NaN (the errstate at 328 covers only alpha), and exp/ell/out carry it. Ran it: a query row with attn_bias = -inf across all keys gives NaN in every coefficient of jet_flash_attn; jet_softmax on an all -inf slice gives NaN. The causal-only path is indeed safe (col 0 is never masked in the first block, verified empirically no NaN), so attn_bias is the reachable door, as claimed. Critical extra: a row with only the FIRST key block masked (bias[1,0:2] = -inf, i.e. ordinary left padding, NOT a fully-masked row) also NaNs, because m_run enters as -inf so m_new = -inf on that block and the later alpha=0 multiply propagates 0*NaN = NaN. That case is a genuine divergence from the canonical forward this jet lane is contractually required to match at order 0: ops.flash_attn with the same bias returns a finite, correct row. (Partial refutation of the finding's own example: for a FULLY masked row ops.flash_attn also returns NaN, so that specific case is production-consistent rather than jet-specific; and the repo's own additive-mask sentinel is -1e30, python/tessera/nn/varlen.py:60, not -inf. Neither weakens the defect - jet.py validates nothing, -inf is the repo's own vocabulary for masked scores in ops.flash_attn's causal branch and in jet.py's own jet_where_mask - and the first-block case is a clear unilateral NaN.)

### `python/tessera/autodiff/rematerialize.py:84` — Rematerialize recomputes impure fn, silently wrong gradients

*Autodiff — implicit / jets / remat · logic*

**What is wrong:** _remat_vjp re-executes fn at backward time with no purity or input-consistency check and no diagnostic. If fn draws from stateful randomness (np.random, or any non-key-functional sampler) or if a Parameter's data was mutated between forward and backward, the recomputed intermediates differ from the forward pass, and the returned gradients correspond to neither execution. This contradicts the repo's own discipline (Decision #10: only pure ops qualify for recomputation; unsupported paths must emit a stable diagnostic, never silently misbehave) — the docstring's Limitations list does not even mention purity.

**Evidence:** wrapped = rematerialize(lambda x: x * (np.random.rand(*x.shape) > 0.5)) — dropout inside a checkpointed block, a mainstream training pattern. Forward applies mask M1; backward re-runs and differentiates through a different mask M2, producing gradients that match no realized forward computation, with no warning. Same failure if an optimizer step mutates a Parameter before backward.

**Fix:** At minimum, recompute fn's output in _remat_vjp and compare against the recorded forward output (cheap: it is already recomputed), raising TesseraAutodiffError on mismatch; document the purity requirement.

**Independently verified:** Reproduced with silently wrong gradients. rematerialize.py:76-102 re-executes fn at backward with no purity check and no output-consistency check; the recorded forward output `out` (line 104) is never compared against the recomputed `sub_out` (line 84). Ran a checkpointed block containing a stateful np.random dropout mask over a Parameter: the forward output was mask M1 = [[1,1,1,0],[0,1,0,1]] while the returned gradient was M2 = [[1,0,0,0],[0,0,1,1]] - a gradient matching no realized forward computation, with no warning and no error. The pattern is reachable with the repo's own op: tessera.ops.dropout with seed=None uses np.random.default_rng(None), i.e. a fresh non-key-functional mask per call. No test in tests/unit/test_phase_e_f.py::TestRematerialize exercises an impure fn, and the module docstring's Limitations list (lines 13-22) does not mention purity. The proposed fix is valid and essentially free - sub_out is already computed at line 84, so comparing it to `out` costs one array comparison. (The Decision #10 citation is loose: that decision governs the C++ InsertRecomputePass, not this Python wrapper. The defect stands on its own.)

### `python/tessera/autodiff/implicit.py:314` — FD directional derivative not normalized by direction magnitude

*Autodiff — implicit / jets / remat · math*

**What is wrong:** _partial_jacobian_matvecs.matvec perturbs by eps * v with fixed eps = 1e-6 regardless of ||v||. Central differencing has truncation error O(eps^2 ||v||^3 F'''), i.e. relative error O((eps ||v||)^2), so the product Jv silently degrades as the direction magnitude grows. Inside GMRES the basis vectors are unit-norm (safe), but the same matvec is applied to un-normalized vectors: user tangents in root_jvp (B_op.matvec(v)) and the solution iterate in gmres_solve's apply(x)/apply(candidate).

**Evidence:** root_jvp with a tangent of norm 1e6 (e.g. tangents expressed in un-normalized physical units) gives eps*||v|| = 1: the perturbation is O(1), central differencing returns the secant of F over a unit-scale interval rather than the derivative — O(1) relative error for any nonlinear residual — with no diagnostic. At ||v|| = 1e4 the relative error is already ~1e-4, well above the 1e-8 solve tolerance being enforced around it.

**Fix:** Scale the step per call: h = eps / max(1, ||v||_inf), perturb by h*v, and divide by 2h (returning (F(a+hv)-F(a-hv))/(2h)), which keeps the truncation error O(eps^2) independent of ||v||.

**Independently verified:** Re-derived and measured. implicit.py:312-314 uses a fixed eps=1e-6 step in the direction eps*v with no normalization. Central differencing of g(t)=F(a+tv) gives (g(eps)-g(-eps))/(2 eps) = Jv + (eps^2/6) D^3F[v,v,v], so relative error scales as eps^2 ||v||^2. Measured on F(x)=2x+sin(x): relative error of matvec(v) vs the analytic Jv was 1.4e-11 at ||v||=1, 5.4e-10 at 1e2, 5.4e-8 at 1e3, 5.4e-6 at 1e4, 5.1e-2 at 1e6 - exactly quadratic growth, reaching O(1) as claimed (the finding's ~1e-4 at ||v||=1e4 is its bound eps^2||v||^2; the observed constant here is F'''/6 ~ 0.05, so the scaling law, not the exact number, is what holds). The claimed call sites are real and unguarded: root_jvp line 527 applies B_op.matvec to an arbitrary user tangent, and gmres_solve line 180 applies the same matvec to the un-normalized iterate `candidate`. root_jvp with ||v|| >= 1e4 on this residual then fails outright with 'GMRES Arnoldi breakdown' - a spurious failure on a perfectly solvable system, caused by the scale-dependent matvec. Verified the transposed action is NOT affected (rmatvec is exactly linear in r since it only ever calls matvec on unit basis vectors: relative error unchanged at 1.7e-11 for ||r|| from 1 to 1e6), which is consistent with the finding scoping this to the forward operator. The proposed fix (h = eps/max(1,||v||_inf), divide by 2h) is the standard Newton-Krylov scaling and is valid for the large-||v|| direction it targets.

### `python/tessera/rng.py:242` — truncated_normal rejection loop unbounded for tail intervals

*RNG & quantization · algorithm*

**What is wrong:** The rejection loop resamples full standard normals until [lower, upper] is filled. For a valid but tail-located interval the acceptance probability is astronomically small, so the call hangs with no diagnostic instead of completing or failing closed.

**Evidence:** truncated_normal(key, 1000, lower=6.0, upper=7.0): acceptance probability Phi(7)-Phi(6) is about 9.9e-10, so filling 1000 elements needs about 1e12 standard-normal draws - effectively an infinite loop on legal arguments. Even lower=4.0 costs ~3e4 draws per element.

**Fix:** Use the inverse-CDF construction (erfinv over Phi(lower)+u*(Phi(upper)-Phi(lower)), in standardized units) which is exact, O(n), and branch-free for every interval; or cap the loop and emit a stable diagnostic when acceptance is below a threshold.

**Independently verified:** rng.py:242-247 resamples n-filled full standard normals per iteration and keeps only entries inside [lower, upper], with no iteration cap and no diagnostic. Ran truncated_normal(RNGKey.from_seed(3), (1000,), lower=6.0, upper=7.0) on legal arguments (upper > lower passes the only guard): still looping after a 15 s alarm with no output. The acceptance probability Phi(7)-Phi(6) is ~9.9e-10, so ~1e12 draws are needed - a practical hang. The one existing test (tests/unit/test_rng_keys.py:176) uses lower=-1/upper=1 and does not pin tail behavior; the docstring only claims bounded time for the +-2 sigma default. Note for the fix: bounds are compared in unstandardized space (chunk = standard_normal*std + mean), so an inverse-CDF rewrite must standardize by (lower-mean)/std as the finding states; the alternative cap-plus-diagnostic fix is also valid and matches Decision #21's stable-diagnostic pattern.

### `python/tessera/quantization.py:231` — scale_mode is a fail-open semantic key

*RNG & quantization · logic*

**What is wrong:** quantize_fp4_packed only tests scale_mode == "mx"; every other string (including typos like "MX", "mxfp4", "nvfp4") silently takes the fp32-scale branch. scale_mode selects the scale semantics/layout (power-of-two shared exponent vs fp32), so per Decision #21a it must fail closed on an unknown value.

**Evidence:** quantize_fp4_packed(w, scale_mode="MX") returns non-power-of-two fp32 scales with no error; a downstream MXFP4 consumer that decodes the group scale as a shared exponent (e8m0) reconstructs wrong weights with no diagnostic anywhere in the chain.

**Fix:** Validate scale_mode in {"mx", "nv"} and raise ValueError naming the argument and the legal set.

**Independently verified:** quantization.py:231 is the only test of scale_mode in the function and there is no validation anywhere in the call path - any string other than the exact "mx" silently takes the fp32-scale branch, so "MX"/"mxfp4"/"nvfp4"/a typo all silently yield non-power-of-two scales while the caller believes it requested MXFP4 shared-exponent semantics. The function is exported in __all__ and re-exported through the package, so no caller contract makes the input impossible, and the parametrized test (test_apple_gpu_quantized_matmul.py:275-289) only passes the two legal values, pinning nothing. The repo's own sibling API validates the same-named parameter and has a dedicated negative test - compiler/grouped_layout.py::scale_mode_to_layout raises ValueError naming the legal set, covered by test_amd_fp8_and_gemm_dispatch.py::test_scale_mode_unknown_rejected - so fail-closed is the established convention here. One caveat on the stated evidence: today's harm is latent rather than observed, because both in-repo consumers (dequantize_fp4_packed and apple_gpu_quantized_matmul_fp4) read the returned f32 scales array and so round-trip self-consistently; the wrong-reconstruction scenario requires an e8m0-decoding consumer that does not yet exist. The fail-open defect itself stands as claimed.

### `python/tessera/rng.py:197` — uniform dtype cast can return the excluded endpoint

*RNG & quantization · math*

**What is wrong:** uniform samples in float64 on [low, high) and then casts to the requested dtype. Rounding in the downcast maps float64 values just below high onto exactly high, violating the documented half-open interval for fp32 and especially fp16 outputs.

**Evidence:** For dtype="fp16": any float64 sample in [1 - 2^-12, 1) rounds to fp16 1.0 - probability about 2.4e-4 per element, so a (1024,1024) dropout/inverse-CDF mask contains ~256 exact-1.0 values; downstream log(1-u) yields -inf. For fp32 the probability is ~3e-8 per element - rare but nonzero on large tensors.

**Fix:** After the cast, clip to np.nextafter(high, low) in the target dtype (or sample directly in the target dtype), preserving the [low, high) contract.

**Independently verified:** Re-derived the fp16 rounding boundary and reproduced it. rng.py:196-197 samples float64 in [low, high) then does np.asarray(out, dtype=...), an IEEE round-to-nearest downcast. For fp16 the ulp below 1.0 is 2^-11, so every float64 sample above the midpoint 1 - 2^-12 rounds up to exactly 1.0; measured 48 exact-1.0 values in 200,000 draws of uniform(k, dtype="fp16"), i.e. 2.4e-4, matching the predicted 2^-12, and max() == 1.0 == high. That violates the docstring's documented "Uniform[low, high)" contract. The fp32 case is the same mechanism with probability 2^-25 (~3e-8) - not observed at n=2e5, as expected, but the rounding argument is identical and the guard at line 193-194 only checks high > low, so nothing prevents it. The consequence named (log(1-u) -> -inf in inverse-CDF/dropout consumers) follows directly.

### `python/tessera/rng.py:562` — MALA/HMC recompute energy and gradient of retained state

*RNG & quantization · performance*

**What is wrong:** mala_sample.step_fn calls energy_fn(y) and grad_fn(y) every step, but y entering step t is either y_prop or the old y from step t-1, whose energy and gradient were both already computed there. hmc_sample similarly recomputes energy_fn(q) for H0 each iteration. Carrying (E(y), grad(y)) in the chain state halves the calls to the user's energy network - the dominant cost of EBM sampling.

**Evidence:** A run of n MALA steps performs 2n energy evaluations and 2n gradient evaluations where n+1 of each suffice; for an EBM whose energy_fn/grad_fn is a network forward/backward this is a straight 2x wall-clock cost on the sampler hot loop (burn_in + n_samples*thin steps).

**Fix:** Thread (y, E_y, grad_y) through _collect_chain's step state (or close over a one-element cache), updating it only when a proposal is accepted.

**Independently verified:** Instrumented both samplers with counting callbacks. mala_sample over 15 steps (burn_in=5, n_samples=10) made 30 energy_fn and 30 grad_fn calls where 16 of each suffice; hmc_sample over 15 steps made 30 energy_fn calls where 16 suffice (its 60 grad calls are genuine leapfrog work, and the finding does not claim otherwise). The caching is provably valid, not merely plausible: in mala_sample.step_fn (rng.py:555-572) both grad_fn(y)/energy_fn(y) and grad_fn(y_prop)/energy_fn(y_prop) are computed in the same step, and y_next is exactly one of those two points, so whichever survives already has both quantities available; likewise in hmc_sample.step_fn (640-654), H0 uses energy_fn(q) and H1 uses energy_fn(q_new), and q_next is one of them. The cost is real rather than cold-path: these are the exported EBM2 primitives, the callbacks are user-supplied (an EBM energy network forward/backward), and the loop runs burn_in + n_samples*thin times. The proposed fix (thread (y, E_y, grad_y) through the chain state, or close over a one-element cache) is compatible with _collect_chain's (y, key) -> (y, key, info) step signature via the closure variant.

### `python/tessera/losses.py:63` — log_cosh_loss overflows to inf for large negative errors

*Losses, optimizers, RL, nn.functional · math*

**What is wrong:** log_cosh_loss uses err + log1p(exp(-2*err)) - log(2), which is only stable for err >= 0: for negative err, exp(-2*err) = exp(2|err|) overflows float64 once |err| > ~354.9, returning inf (plus a RuntimeWarning) where the true value is ~|err| - log(2). The stable identity is symmetric in |err|.

**Evidence:** log_cosh_loss(pred=0.0, target=400.0) -> err = -400, exp(800) overflows -> loss = inf; log_cosh_loss(pred=400.0, target=0.0) correctly returns ~399.3. Same-magnitude errors give asymmetric results on unnormalized regression targets.

**Fix:** Use the |x| form: a = np.abs(err); loss = a + np.log1p(np.exp(-2.0*a)) - np.log(2.0).

**Independently verified:** losses.py:61-64 uses `err + log1p(exp(-2*err)) - log(2)`, an identity valid only for err >= 0. `_asarray` promotes to float64, so exp(-2*err) overflows once 2*|err| > ~709, i.e. |err| > ~354.9. Executed: log_cosh_loss(400.0, 0.0) -> 399.3069 (correct, err=+400) while log_cosh_loss(0.0, 400.0) -> inf with an overflow RuntimeWarning (err=-400, true value 399.3069). log_cosh is mathematically even in err, so this is a genuine asymmetry, not a modeling choice. I also confirmed the proposed fix: a=|err|; a + log1p(exp(-2a)) - log(2) reproduces 399.3069 for both signs. No caller normalizes the error, and no test pins the inf.

### `python/tessera/losses.py:111` — CE prob-target branch silently drops label_smoothing/ignore_index

*Losses, optimizers, RL, nn.functional · logic*

**What is wrong:** In cross_entropy_loss, when targets are float (probability distributions) the else-branch computes -sum(targets * log_probs) and never consults label_smoothing or ignore_index — both semantic parameters are silently ignored rather than applied or rejected, contrary to the fail-closed convention (Decision #21a). label_smoothing is not even range-validated on this path.

**Evidence:** cross_entropy_loss(logits, soft_targets, label_smoothing=0.5) returns exactly the same value as label_smoothing=0.0 with no diagnostic; even label_smoothing=7.0 (invalid) passes silently, while the integer-target path raises for it (lines 97-98).

**Fix:** Either apply smoothing to distribution targets ((1-s)*y + s/K) or raise ValueError when label_smoothing != 0 or ignore_index is meaningful with float targets.

**Independently verified:** losses.py:108-111 (the float-target else-branch) computes only `-sum(targets*log_probs, axis)`; `label_smoothing` and `ignore_index` are never read on this path, and the range check at lines 97-98 lives inside the integer branch. Executed with one-hot float targets: label_smoothing=0.0 -> 1.15607, 0.5 -> 1.15607 (bit-identical), 7.0 -> 1.15607 with no diagnostic, while the integer path with label_smoothing=7.0 correctly raises ValueError('label_smoothing must be in [0, 1)'). So a semantic parameter is silently dropped and an out-of-range value silently accepted, with the same function raising on the other path — the inconsistency is internal to this function, not an inferred convention. (The ignore_index half is weaker, since a distribution target has no single index to ignore, but the label_smoothing defect stands on its own.)

### `python/tessera/compiler/fusion_core.py:1099` — Matmul/pointwise verifiers ignore backend rtol budget

*Fusion core & emitter seams · math*

**What is wrong:** verify_synthesized_matmul (line 1099-1100) and verify_synthesized_pointwise (line 1158-1160) call np.allclose with only atol=_effective_atol(...), omitting rtol=_effective_rtol(r) that verify_synthesized_region/attention/gated all pass. A candidate that declares its precision budget via accuracy_rtol (the field candidate.py's adapter explicitly forwards, lines 137/270) is judged at numpy's strict default rtol=1e-5, so a numerically correct half-precision GEMM candidate can be rejected as a miscompile and permanently starved by the arbiter.

**Evidence:** A Tier-3 f16-accumulate GEMM candidate with accuracy_rtol=1e-2 and accuracy_atol=None: on the 32x16x32 probe (values ~N(0,0.9)), f16 accumulation error scales relatively with magnitude; allclose(a,b,atol=max(1e-3,None)) with rtol=1e-5 gives tolerance ~1e-3+1e-5*|ref|, below the candidate's declared budget — verify_candidate returns False and arbitrate drops the lead lane on every dispatch, silently degrading to a slower tier or the reference. The asymmetry with the other three verifiers (which pass rtol=_effective_rtol(r)) shows the omission is accidental, not a policy.

**Fix:** Pass rtol=_effective_rtol(r) in both allclose calls.

**Independently verified:** Code fact verified: verify_synthesized_matmul:1099-1100 and verify_synthesized_pointwise:1158-1160 call np.allclose with atol=_effective_atol(...) only, while verify_synthesized_region:1060-1062, _attention:1122-1124 and _gated:820-822 all pass rtol=_effective_rtol(r). _as_runner forwards candidate.accuracy_rtol onto the adapter (candidate.py:270) and _effective_rtol exists precisely to consume it, so for these two op-kinds a declared relative budget is silently dropped and the check runs at numpy's rtol=1e-5 (tolerance atol + 1e-5*|ref| instead of atol + rtol*|ref|). Caveat on impact: the defect is currently latent - all three OP_MATMUL candidates (nvidia_cuda.py:5106/5142/5183) and the OP_POINTWISE candidate (3830) set only accuracy_atol (_GEMM_F16_ATOL=5e-3), leaving accuracy_rtol=None where _effective_rtol returns 1e-5 anyway, so no shipped candidate is being rejected today. Nothing guards against the next one that sets it (the nvidia fused/attention/gated candidates already set accuracy_rtol=atol), and the one-line fix is safe.

### `python/tessera/compiler/emit/_fused_scalar_body.py:106` — Prologue activation recomputed N times per A element

*Fusion core & emitter seams · performance*

**What is wrong:** row_compute_body places the prologue activation inside the n-over-k double loop: `float a = A[a_index]; <prologue(a)>; v += a*B[...]`. Since a depends only on (m,k), the activation is evaluated N*K times per row instead of K — an N-fold redundancy in transcendental work (tanhf/expf for gelu/silu) that the compiler cannot hoist (the redundancy is across the outer n loop, not the inner k loop). This body is shared verbatim by the x86 C and ROCm HIP synthesized lanes.

**Evidence:** Lines 101-108: the n loop encloses the k loop, and `prologue` is emitted at the A-load site inside k. For a gelu-prologue region at M=512, N=4096, K=4096, each of the 512 rows does 4096*4096 = 16.8M gelu evaluations instead of 4096 — a 4096x redundancy on the dominant cost of the epilogue-fused GEMM (each gelu is ~1 tanhf + several mults, easily 10-30x the cost of the fma it accompanies).

**Fix:** Emit a per-row pre-pass that materializes act(A[m,0..K)) into a local buffer (or restructure to k-outer with a row accumulator), so the prologue runs K times per row.

**Independently verified:** Verified by generating the body: row_compute_body(FusedRegion(epilogue=(), prologue=('gelu',))) emits `for n { float v=0; for k { float a = A[m*K+k]; {..tanhf(..)..}; v += a*B[k*N+n]; } }`. `a` depends only on (m,k), so the activation runs N*K times per row instead of K. The redundancy is across the OUTER n loop, so eliminating it needs loop interchange, not LICM - and the compiler cannot even prove A's values are loop-invariant here: neither A/B nor the `float* row` output pointer is `restrict` (see the emitted wrapper contract in the module docstring), so stores to row[] may alias A. At M=512,N=4096,K=4096 that is 16.8M gelu evaluations per row versus 4096 (4096x), each ~1 tanhf. The body is shared verbatim by x86_c.py:114 and rocm_hip.py:93, so both synthesized lanes carry it. The suggested fix is numerically exact (the prologue is pure elementwise) though a per-row buffer needs care for dynamic K.

### `python/tessera/compiler/emit/nvidia_cuda.py:622` — split_reduced backward recomputes softmax stats Sk times

*Emitter — NVIDIA CUDA · algorithm*

**What is wrong:** The deterministic split route does asymptotically redundant work. tsr_flash_bwd_split (line 622): each thread owns one (b, hk, n) key and, for every (qh, m) query row in its partition, recomputes the row's max, z, and delta by a full O(Sk·(D+Dv)) pass — but those statistics depend only on (b, qh, m), not on n, so the same stats are recomputed once per key, an Sk-fold blowup: O(B·Hq·Sq·Sk²·(D+Dv)) total instead of O(B·Hq·Sq·Sk·(D+Dv)). Separately, tsr_flash_bwd_dq (line 619) nests the n loop inside the d loop, so per output row it recomputes every score (O(D) each) plus dp (O(Dv)) D times: O(Sk·D·(D+Dv)) per row where O(Sk·(D+Dv)+Sk·D) suffices (the atomic kernel at line 613 already does it the cheap way with an aq[] accumulator).

**Evidence:** For B=1, Hq=8, Sq=Sk=1024, D=Dv=128: the split kernel performs ~1024x the necessary stat-pass work and bwd_dq ~128x the necessary score work, so run_flash_attention_backward(..., deterministic=True) costs orders of magnitude more device time than the atomic route for reasons unrelated to determinism, and measure_flash_attention_backward_device feeds that inflated number to the route arbiter.

**Fix:** Add a small precompute kernel writing per-(b,qh,m) row stats (max, z, delta) to a [B,Hq,Sq]x3 workspace (the route already accepts a workspace budget), then have bwd_split and bwd_dq read them; restructure bwd_dq with the n loop outermost accumulating into a per-thread aq[D] array as tsr_flash_bwd already does.

**Independently verified:** Re-derived both complexities independently. tsr_flash_bwd_split (line 620-623): one thread owns (b,hk,n); the body iterates qh over `ratio=Hq/Hkv` and m over [Sq*split/2, Sq*(split+1)/2), and for EACH (qh,m) it runs a full max pass over j<Sk (tsr_score is O(D) each) plus a full z/delta_num pass over j<Sk (O(D+Dv) each). Those statistics depend only on (b,qh,m), not on n, yet the kernel is n-parallel — total = B*Hkv*Sk * ratio * (Sq/2) * Sk*(D+Dv) = B*Hq*Sq*Sk^2*(D+Dv)/2, an Sk-fold blowup over the O(B*Hq*Sq*Sk*(D+Dv)) the row-parallel atomic kernel achieves. tsr_flash_bwd_dq (line 619) confirms the second half: `for(int d=0;d<D;++d){ for(long n=0;n<Sk;++n){ tsr_score(...) /*O(D)*/ ; for(j<Dv) dp+=... } }` — every score and every dp is recomputed D times, O(Sk*D*(D+Dv)) per row, while the atomic kernel at line 613 already accumulates into aq[D] with the n loop outermost at O(Sk*(D+Dv+D)). The route is reachable (run_flash_attention_backward forces split_reduced on deterministic=True, line 718-721) and is not a dead/cold path. The proposed fixes are valid: a per-(b,qh,m) stats workspace is compatible with the deterministic contract (the route already carries a workspace budget), and restructuring bwd_dq with n outermost into aq[d] preserves the per-d accumulation order exactly, so numerics are unchanged.

### `python/tessera/compiler/emit/nvidia_cuda.py:879` — Decay product recomputed O(S) per key, O(S^3) total

*Emitter — NVIDIA CUDA · algorithm*

**What is wrong:** In tsr_lav (line 879) the decay factor for pair (n,m) is computed by the inner loop 'for(long u=n+1;u<=m;u++)fac*=dec[u]' for every n — O(S) work per key inside a loop that already runs O(S) keys per output element, making the decayed forward O(B·H·S³·...) where O(S²) suffices. The backward kernel kb (line 907) has the same loop. Iterating n from m downward with a running product (fac(m,m)=1; fac(n-1,m)=fac(n,m)*dec[n]) computes every factor in O(1) per key with identical fixed-order f32 arithmetic per (m) row.

**Evidence:** run_linear_attention_variant with decay, S=2048: each of the B·H·S·Dv threads executes ~S²/2 ≈ 2M extra multiplies purely for decay factors — roughly a 1000x-scale inflation of the decay path's work versus the descending-n formulation, on a lane that is also the base for future decay variants (GDN/Mamba-style mixers where decay is the common case).

**Fix:** Reverse the key loop to n=m..0 and maintain the running decay product; same change in kb. Numerics stay a fixed-order product of the same factors, so the F4 oracle tolerance is unaffected.

**Independently verified:** Verified at nvidia_cuda.py:879 (tsr_lav): inside `for(long n=0;n<=m;n++)` the decay factor is built by `if(dec)for(long u=n+1;u<=m;u++)fac*=dec[...]`, i.e. O(m-n) per key. Summing over n gives ~m^2/2 multiplies per thread on top of the m*D score work, and with B*H*S*Dv threads the decay term contributes an S^3 factor (B*H*Dv*S^3/6) where the non-decay kernel is S^2*D. At S=2048, D=128 that is ~2M decay multiplies vs ~262k score multiplies per thread — the decay path alone is ~S/2 (~1000x) more work than necessary, and the whole kernel is ~8x inflated. The backward kernel kb (line 907) contains the identical `if(de)for(long u=n+1;u<=m;u++)fac*=de[...]` loop. The path is live (runtime.py:10686/10731 route decay through both). The suggested O(1)-per-key running product is algebraically right (fac(m,m)=1, fac(n-1,m)=fac(n,m)*dec[n]), but one detail of the finding is wrong: reversing the key loop does NOT keep 'identical fixed-order f32 arithmetic' — both the decay product's association order and the y/a[] accumulation order over n flip, so f32 results move by rounding. That affects the fix's framing, not the defect, which stands.

### `python/tessera/compiler/emit/nvidia_cuda.py:2022` — Unknown optimizer kind silently executes Adam

*Emitter — NVIDIA CUDA · logic*

**What is wrong:** The opt_k kernel dispatches kind 0 (SGD), 1/5 (momentum/Nesterov), 4 (Lion), and routes everything else — including any invalid code — into the trailing else branch, which is the Adam/AdamW update. Neither the kernel entry tessera_nvidia_optimizer_f32 (line 2024) nor the Python wrapper run_optimizer_f32 (line 2028) validates kind, so a mis-mapped or out-of-range optimizer code silently produces Adam parameter updates tagged as a successful launch (rc 1). This is exactly the silently-defaulted-semantics failure mode Decision #21a forbids: a semantic selector falling open to a default instead of failing closed with a diagnostic.

**Evidence:** run_optimizer_f32(kind=6, ...) (e.g. a new optimizer code added to the name->code map upstream but not to this kernel) returns Adam-updated parameters with no error; training proceeds with the wrong optimizer and nothing downstream can detect it. Every sibling entry in this file (binary kind 0..8, solver unary 0..7, reduce 0..3) range-checks its enum and returns rc 2 — the optimizer entry is the one that doesn't.

**Fix:** Add 'kind<0||kind>5' (rejecting the unused gap too, i.e. validate against the exact supported set {0,1,2,3,4,5}) to the entry's rc-2 guard and mirror the check in run_optimizer_f32 with a ValueError naming the code.

**Independently verified:** Code fact verified: opt_k (line 2022) dispatches kind==0, kind==1||kind==5, kind==4, and routes everything else — including out-of-range codes — into the trailing Adam/AdamW `else`. Neither tessera_nvidia_optimizer_f32 (line 2024, which validates nothing at all, not even pointers or n) nor run_optimizer_f32 (line 2028, which only checks buffer sizes) range-checks kind, so a direct call with kind=6 returns Adam-updated parameters and rc 1. The sibling-entry claim also checks out: the reduce entry guards `kind<0||kind>3` (lines 1372/1378) and tessera_nvidia_moe_timed guards `kind<0||kind>2` (line 2078), while the optimizer entry does not. No test pins the fallthrough as intentional. Mitigating context the finding omits, which lowers severity to latent-hardening rather than a live wrong-optimizer bug: the only in-tree caller is runtime.py's _optimizer_compute, whose kind comes from the closed _OPTIMIZER_OPS dict {0,1,2,3,4,5} behind an explicit `if name not in _OPTIMIZER_OPS: raise ValueError` (runtime.py:9747, 15553-15560), so today no out-of-range code can reach the kernel through the runtime path. The fail-open semantic selector and the proposed guard are nonetheless real and match Decision #21a and this file's own conventions.

### `python/tessera/compiler/emit/rocm_hip.py:104` — Unchecked hipMemcpy/hipMalloc can report garbage as native

*Emitter — Apple MSL / ROCm HIP / x86 · logic*

**What is wrong:** The generated fused-region host wrapper checks hipMalloc for A/B/O but ignores the return values of hipMemcpy(dA/dB) and of the bias/residual hipMalloc+hipMemcpy. If an H2D copy fails the kernel computes on uninitialized device memory, hipDeviceSynchronize still succeeds, and the wrapper returns 1 — the runner then reports a wrong result under the "rocm_hip" real-execution tag (a claim-integrity violation). If the bias hipMalloc fails, dbias stays NULL and the kernel dereferences it, faulting the HIP context.

**Evidence:** Lines 104-109 of the emitted source: `hipMemcpy(dA,hA,szA,...);` return ignored; `if (hbias) { hipMalloc(&dbias,(size_t)N*sizeof(float)); hipMemcpy(dbias,hbias,...); }` — both ignored. A failed copy (e.g. device OOM after the mallocs, or a transient WSL2 /dev/dxg error) leaves dA/dbias invalid yet `ok` is computed solely from hipDeviceSynchronize; rc==1 is the runner's only success criterion (line 619). The repo's own paged-KV wrapper in the same file checks every hipMemcpy (line 224), so this wrapper is the outlier.

**Fix:** Check every hipMalloc/hipMemcpy in _synthesize_fused_hip and return the error code (2/3) so the runner falls back to the reference.

**Independently verified:** Verified the emitted wrapper in _synthesize_fused_hip (rocm_hip.py:95-119): only the three hipMalloc calls for dA/dB/dO are checked; `hipMemcpy(dA,...)`, `hipMemcpy(dB,...)`, the conditional `hipMalloc(&dbias,...)`/`hipMemcpy(dbias,...)`, the residual pair, AND the final D2H `hipMemcpy(hout,dO,szO,...)` all ignore their return values. `ok` is derived solely from hipDeviceSynchronize. A failed H2D leaves device memory uninitialized; a non-sticky hipMemcpy error is not resurfaced by hipDeviceSynchronize (it is retrieved via hipGetLastError), so the wrapper returns 1 and RocmHipRunner.run_fused_region (line 617-620) returns that buffer under the `rocm_hip` real-execution tag — a claim-integrity violation. The unchecked D2H is an even cleaner instance: on failure `out` stays the caller's np.zeros and is still tagged rocm_hip. The bias-malloc case does dereference NULL in the kernel, which faults the context (it happens to be caught by the sync check, but only by accident). The sibling wrapper in the same file (_synthesize_paged_kv_read_hip, line 224) checks every hipMalloc and every hipMemcpy, so this wrapper is the outlier, and the repo's standing rules (Decision #21, no silent no-op; never report green over a red lane) put it on the wrong side.

### `python/tessera/compiler/emit/rocm_hip.py:282` — Paged attention recomputes expf per output dim

*Emitter — Apple MSL / ROCm HIP / x86 · performance*

**What is wrong:** The paged_attn output loop computes `acc += expf(scores[j]-m)/z*v[d]` inside the d loop, so the softmax weight (one expf + one division) and the page-table address (`idx[j]`, `table[tok/L]`) are recomputed for every (j,d) pair — D-fold redundant transcendental work in the hot decode kernel, when one cooperative pass normalizing scores[] in place would compute each weight exactly once per block.

**Evidence:** Kernel text: `for(int d=t;d<D;d+=256){float acc=0.f;for(int j=0;j<T;j++){long long tok=idx[j];int pp=table[tok/L],off=tok%L;...acc+=expf(scores[j]-m)/z*v[d];}...}`. Per block this is T*D expf calls and T*D page-table resolutions instead of T (e.g. T=4096, D=128: 524k expf vs 4k, a 128x reduction). The Apple materialized attention kernel in this same review scope already does it the cheap way: it overwrites scores[n] with exp(scores[n]-mx) once, then the P·V loop only multiplies (apple_msl.py:1559-1571).

**Fix:** Before the output loop, add a cooperative pass `for(int j=t;j<T;j+=256) scores[j]=expf(scores[j]-m)/z;` followed by __syncthreads(), and have the d loop use scores[j] directly (also hoisting the v-page address computation out of the d dimension by iterating j outer / d inner per thread).

**Independently verified:** Re-derived from the emitted output loop: `for(int d=t;d<D;d+=256){float acc=0.f;for(int j=0;j<T;j++){long long tok=idx[j];int pp=table[tok/L],off=tok%L;const float*v=...;acc+=expf(scores[j]-m)/z*v[d];}...}`. The softmax weight depends only on j, yet it is recomputed inside the j loop for every d-slice a thread owns, and the page-table address resolution (idx[j], table[tok/L], the v base pointer) is likewise j-only work recomputed per active lane. Per block that is one expf plus one full-precision float division (hipcc -O3 without fast-math cannot strength-reduce `/z` to a reciprocal multiply, so it expands to a ~10-instruction div sequence) executed once per wave per j across all active waves, versus T/256 iterations per thread for the proposed cooperative pass — the ALU work in the inner loop is dominated by this redundant transcendental+divide, while the necessary work is one coalesced v[d] load and one FMA. The proposed fix (`for(int j=t;j<T;j+=256) scores[j]=expf(scores[j]-m)/z;` + __syncthreads(), then use scores[j] directly) is numerically identical (same values, computed once) and fits the existing extern __shared__ scores[T] allocation; the Apple materialized attention kernel in the same review scope already uses exactly this shape (synthesize_attention_msl overwrites scores[n]=exp(...) before the P·V loop). Not a cold path: run_paged_attention_direct_f32 is the production route in paged_kv.py::_paged_attention_rocm and is the kernel whose device_ms is measured with reps=20.

### `python/tessera/compiler/emit/spectral_candidates.py:949` — ROCm availability probe launches device FFT every call

> **Measured on gfx1151, 2026-08-29 — confirmed, with numbers.**
> `RocmStockhamFFTCandidate.available()` costs **287.7 ms** on the first call
> (library load + device probe) and **0.371 ms on every call after**, over 200
> calls. It is NOT memoized: a cached boolean would be ~1e-4 ms, so each call
> pays roughly **3700x** what the answer is worth — and the answer cannot change
> within a process. At the scale this finding cites (an STFT with n=1M, win=1024,
> hop=256 → ~3900 frames, one probe per frame) that is **~1.45 s of pure
> probing** before any real work. This is a cheap, self-contained fix (memoize
> per process) and should lead the P2 batch.


*Emitter — spectral / Krylov / autotune · performance*

**What is wrong:** RocmStockhamFFTCandidate.available() runs a real 4-point device transform (ts_fft_stockham_amd_hostptr) on every invocation with no memoization. available() is called on every arbitration (_inner_fft per composed frame, _ComposedSpectralCandidate.available, measured_arbitrate cache-hit revalidation, corpus_winner), so each dispatch pays an extra kernel launch plus two host-device transfers purely to answer a question whose answer cannot change within a process.

**Evidence:** For target='rocm', each _inner_fft call (once per STFT frame, see per-frame finding) executes the probe transform before the real one — doubling launch count. Even in the cached-winner fast path of measured_arbitrate (autotune.py:416), every hit re-runs the probe, so the 'no re-timing' cache still costs a device round-trip per lookup (~tens of microseconds launch + transfer latency each).

**Fix:** Memoize the probe result (e.g. in _libs or a module-level tri-state) after the first successful/failed probe, exactly as the library handle itself is cached.

**Independently verified:** Traced RocmStockhamFFTCandidate.available() (940-952): `_amd_candidate_lib()` is memoized via `_libs`, but the 4-element `ts_fft_stockham_amd_hostptr` probe transform is re-executed on every call with no caching. Confirmed the call sites the finding names: candidate.py:351-352 (`arbitrate` calls available() per candidate), autotune.py:416 (`measured_arbitrate` cache-HIT path re-runs `c.available()` before returning the cached winner, so the 'no re-timing' fast path still pays a device round-trip), spectral_candidates.py:1016 (`_inner_fft` calls available() per composed frame) and 1245 (`_ComposedSpectralCandidate.available` iterates FFT candidates calling available()). No memoization exists and no test pins re-probing as intentional. Off-silicon there is no cost (`_amd_candidate_lib()` returns None and available() short-circuits), so this is a cost paid only on the ROCm box — which is exactly where it matters. Magnitude (tens of us per probe) is not measurable here, but the redundant per-dispatch device round-trip is provable from source and the fix (memoize alongside the already-cached library handle) is trivially valid.

### `python/tessera/compiler/emit/spectral_candidates.py:118` — Temp directory created before compile-cache hit check

*Emitter — spectral / Krylov / autotune · logic*

**What is wrong:** _cpu_lib() calls tempfile.mkdtemp() unconditionally before _compile() consults the _libs cache, so every call after the first creates a fresh empty directory that is never used and never removed (no cleanup/atexit). _amd_source_lib (line 482) has the same order bug. Since _cpu_lib() is invoked from both available() and run() of the CPU candidate — per frame in composed STFT/ISTFT — directories accumulate without bound in TMPDIR for the life of the machine, and the compiled .so directories themselves are also never cleaned.

**Evidence:** One CPU-target STFT call with 1000 frames executes available()+run() per frame via _inner_fft, creating ~2000 empty tessera_spectral_cpu_* directories per STFT call; a long-running process on a tmpfs /tmp steadily consumes inodes/memory. Contrast tpp_candidates.py:_cpu_lib (line 129), which creates the tempdir only inside the cache-miss branch.

**Fix:** Check the _libs cache (early-return) before calling tempfile.mkdtemp, and register cleanup of the created directory.

**Independently verified:** Confirmed and materially WORSE than described. spectral_candidates.py:113-126: `_cpu_lib` calls `tempfile.mkdtemp` at line 118 unconditionally, then `_compile` (99-110) early-returns from the `_libs` cache — so the directory is created and abandoned on every call after the first, with no cleanup or atexit. `_amd_source_lib` (477-487) has the identical ordering. Reproduced: 5 calls to `_cpu_lib()` created 5 directories, 4 of them empty. The contrast the reviewer cites is real — tpp_candidates.py:123-124 does `if _lib: return _lib[0]` BEFORE mkdtemp. Beyond the inode leak, I measured the runtime cost: `_inner_fft` on the cpu target costs 194 us/call, of which 188 us is the mkdtemp; memoizing `_cpu_lib` drops it to 5.6 us — a 35x per-frame speedup. So this is the dominant cost of the composed CPU spectral lane, not merely a housekeeping issue.

### `python/tessera/compiler/emit/spectral_candidates.py:1113` — STFT crashes when signal shorter than window

*Emitter — spectral / Krylov / autotune · logic*

**What is wrong:** SpectralSTFTRegion.frames advertises max(1, ...) — i.e. one frame even when n < win — but reference() slices x[s:s+win] which yields only n samples and multiplies it by the length-win window, raising a broadcast ValueError. STFTCandidate.run's except-handler then calls the same region.reference and the exception escapes uncaught, so the 'decline to reference' contract (Decision #21) turns into a crash instead of a diagnostic or a defined semantics.

**Evidence:** SpectralSTFTRegion(n=32, win=64, hop=16).reference(x, w): range(0, max(1, 32-64+1), 16) = [0]; x[0:64] has length 32; (32,)*(64,) → 'operands could not be broadcast together'. frames property simultaneously reports 1. The same mismatch makes _verify_region on such a region raise instead of returning False.

**Fix:** Either fail closed in __init__ with a stable diagnostic requiring n >= win, or zero-pad the trailing partial frame so reference, candidate, and the frames property agree.

**Independently verified:** Re-derived and reproduced. SpectralSTFTRegion(32,64,16): `frames` = max(1,(32-64)//16+1) = max(1,-1) = 1, but `reference` iterates range(0, max(1,32-64+1), 16) = [0] and evaluates x[0:64] (length 32) * w (length 64) -> ValueError 'operands could not be broadcast together with shapes (32,) (64,)'. Executed: reference raises; STFTCandidate.run raises the SAME ValueError out of its except-handler (the handler calls region.reference, which re-raises), so the exception ESCAPES and the Decision #21 decline-to-reference contract becomes a crash; `_verify_region` also raises rather than returning False. No __init__ validation, no caller guard, and no test covers n<win. Notably the production op `tessera.ops.stft` (python/tessera/__init__.py ~2903) explicitly zero-pads when `moved.shape[-1] < fft_length`, i.e. it DEFINES semantics for this case — so the region's reference both diverges from the shipped op and crashes, while its own `frames` property advertises one frame. Reachability caveat: no production code constructs SpectralSTFTRegion today (tests only), so this is latent.

### `src/transforms/lib/TesseraToLinalgPass.cpp:536` — Matmul creates transpose IR before match checks

*C++ — Tessera→linalg lowering · logic*

**What is wrong:** MatmulLowering emits emitTranspose2d (tensor.empty + linalg.transpose) at lines 536-538 for transposeA/transposeB, but the pattern can still fail afterwards: the K/M/N shape-consistency check at line 544 and the float-only element check at line 548-549 both return notifyMatchFailure. A RewritePattern must not mutate the IR before committing to success; returning failure after creating ops violates the greedy-driver contract (an abort under MLIR's expensive-pattern-API checks, and leaked ops plus potentially corrupted worklist state on the NDEBUG builds the fleet runs). The reachable trigger is real: an integer-element tessera.matmul with transposeA=true passes every check up to line 536, gets its transpose materialized, then fails float-only at 549.

**Evidence:** "tessera.matmul"(%a, %b) {transposeA = true} : (tensor<8x4xi32>, tensor<8x16xi32>) -> tensor<4x16xi32> — rank-2, static shapes, so the pattern reaches line 536, creates tensor.empty + linalg.transpose, then returns failure at line 549 ("Phase 1 matmul is float-only"), leaving the created ops behind on a "not matched" path. Same leak for any post-transpose shape mismatch at line 544-545.

**Fix:** Perform all rejection checks (element type, post-transpose shape agreement computed from the types, not from materialized ops) before the first rewriter.create; only then emit the transposes.

**Independently verified:** Traced and then reproduced with a WORSE symptom than claimed. The trigger is real: MatmulOp::verify (src/compiler/ir/TesseraOps.cpp:163) checks rank, post-transpose K, and result M/N but says nothing about element type, so `"tessera.matmul"(%a,%b) {transposeA = true} : (tensor<8x4xi32>, tensor<8x16xi32>) -> tensor<4x16xi32>` verifies, reaches line 536, materializes tensor.empty + linalg.transpose, then returns notifyMatchFailure at 549 (float-only). Running it: `tessera-opt mm.mlir --tessera-to-linalg` exits 1 with EMPTY stdout and EMPTY stderr — the whole pass fails silently with no diagnostic, on a stock NDEBUG build (the greedy driver keeps seeing IR change on a failing pattern and never converges). Controls: the same i32 matmul WITHOUT transposeA is left alone and the pass exits 0; the same shape in f32 with transposeA lowers correctly. The post-transpose shape check at 544 is indeed unreachable (the ODS verifier covers it for the static shapes this pattern requires), but the element-type path alone confirms the finding, and the fix as stated (check element type before the first create) is correct.

### `src/transforms/lib/AutodiffPairedPass.cpp:961` — While tape sizes dynamic dims from init; grows out of bounds

*C++ — MLIR autodiff passes · logic*

**What is wrong:** materializeWhileResiduals builds each state tape with `tensor.empty` whose dynamic extents come from the loop *init* (getDynamicTensorSizes on `init`, lines 961-964), and insertState (lines 1042-1047) slices with the current iteration's `tensor.dim` — with no shape-envelope requirement and no runtime guard. A ranked state whose dynamic extent grows across iterations (legal in scf.while: `tensor<?xf32>` yields any extent) produces an out-of-bounds tensor.insert_slice — UB — instead of a diagnostic.

**Evidence:** State `tensor<?xf32>` starting at extent 4 and concatenated to extent 8 in iteration 1: tape is `tensor<maxIters x 4 x f32>` at runtime but iteration 1 inserts a size-8 slice at offset [1,0]. The sibling paths already treat this as fail-closed: materializeGenericForResiduals requires shape envelopes for every dynamic slot (requireEveryDynamic=true, line 1188) and emits the 'saved dynamic state exceeds its slot envelope' cf.assert (line 1310); the while path silently proceeds, violating the fail-closed convention (Decision #21a).

**Fix:** Require readSavedSlotShapeEnvelopes for dynamic while state (as generic_for does) or emit a diagnostic rejecting dynamic-extent differentiable while state.

**Independently verified:** Reproduced. A saved `scf.while` carrying `tensor<?xf32>` state is accepted with no diagnostic: the pass emits `%dim = tensor.dim %arg0, %c0` (the *init*) → `tensor.empty(%dim) : tensor<3x?xf32>` and, inside the body, `tensor.insert_slice %arg3 into %arg4[%arg2, 0] [1, %dim_2]` where `%dim_2 = tensor.dim %arg3` is the *current* iteration extent. Nothing constrains those to agree: `getResidualTapeType` (line 154) propagates dynamic dims, and the admission check at 949-972 requires only float element type and ranked tensor. A legal body such as `%next = tessera.mul(%big,%big)` (a larger loop-invariant `tensor<?xf32>`) makes the extent grow and the insert_slice go out of bounds. The sibling path is provably fail-closed by contrast: `materializeGenericForResiduals` calls readSavedSlotShapeEnvelopes with requireEveryDynamic=true (1187-1194), builds a separate shape tape, and emits the `cf.assert "saved dynamic state exceeds its slot envelope"` (1306-1314). Related second hole: the while backward extracts with size `dim(tape,1)` (init extent) rather than a per-slot saved extent, so any varying extent restores a wrong-sized state. Decision #21a fail-closed convention is violated.

### `src/transforms/lib/RegionAdjointInterface.cpp:378` — SAVE tape lookup is linear select chain, O(trip^2)

*C++ — MLIR autodiff passes · algorithm*

**What is wrong:** buildForAdjoint recovers the checkpointed state by iterating all checkpointIndices and emitting one full-tensor scf.if select per slot per state (lines 378-431, plus the checkpointOrdinal chain at 328-336). Under the SAVE policy checkpointIndices has trip-1 entries, so the emitted backward body is O(trip) IR and the backward loop executes O(trip) tensor selects per iteration — O(trip^2) total — when the slot is directly computable as `ordinal - 1` with one dynamic-index tensor.extract_slice.

**Evidence:** trip=1024, state tensor 1024xf32: ~1M conditional tensor-yield evaluations (each an scf.if yielding the full state, likely a copy after bufferization) versus 1023 O(1) extracts. The O(1) alternative already exists in this file: the non-hybrid while path does `extractState(stateIndex, reverseOrdinal)` with a dynamic slot (lines 697-707).

**Fix:** For save policy compute the slot as `ordinal - 1` (dense interior checkpoints) and emit a single dynamic extract_slice; keep the select chain only for sparse hybrid checkpoints.

**Independently verified:** Structure confirmed against emitted IR. Running the existing generic/control_scan SAVE fixture (`autodiff_saved_region_ssa.mlir`, checkpoints [1,2], trip 3) the backward reverse loop contains one `scf.if` per checkpoint slot per state, each yielding the whole state tensor (`%18`, `%20` in the output), plus the `checkpointOrdinal` arith.select chain (%14, %16 — dead under the save branch at line 434, a minor overcount in the finding). The paired pass enforces `indices.size() == trip-1` for policy "save" (AutodiffPairedPass.cpp:1168-1169), so the chain length is trip-1: O(trip) emitted IR and O(trip) full-tensor selects per backward iteration = O(trip^2). LICM can hoist the extract_slices (constant slot, loop-invariant) but not the selects, whose condition depends on the reverse ordinal. The proposed O(1) fix is arithmetically valid: for dense interior checkpoints 1..trip-1, the chain's 'largest checkpoint <= ordinal' is exactly slot = ordinal-1 (with init at ordinal 0), and the O(1) form already exists in this file for the non-hybrid while path (`extractState(stateIndex, reverseOrdinal)`, lines 697-707). Contingency worth noting: cost is negligible at fixture trips (3-7) and only bites when the residual evaluator selects SAVE for a large step count — which it does whenever memory budget allows (residual_evaluator.py:171-177), so realistic for a long scan with a small state.

### `src/transforms/lib/AutodiffPass.cpp:199` — Scalar loss seed hardcoded to f32

*C++ — MLIR autodiff passes · logic*

**What is wrong:** The non-shaped scalar seed branch builds `getF32FloatAttr(1.0f)` regardless of the loss's actual type, so an f64/f16/bf16 scalar loss gets an f32 cotangent seed that mismatches every downstream addf/adjoint operand type. Currently masked by the P0 crash at line 187, it becomes live the moment that guard is fixed.

**Evidence:** Loss of type f64: seed is `arith.constant 1.0 : f32`; the first accumulateCotangent or adjoint consuming it feeds addf mixed f32/f64 operands, failing the arith verifier (or, for an integer scalar loss, producing a float seed for an integer path).

**Fix:** Build the seed from the loss's own type (FloatAttr/IntegerAttr on `lossValue.getType()`), mirroring the shaped branch at lines 190-194.

**Independently verified:** Source is unambiguous: line 199 builds `builder.getF32FloatAttr(1.0f)` from no type at all, while the shaped branch (190-194) correctly derives FloatAttr/IntegerAttr from the element type. Reachability after the index-0 guard fix is real and simple: `func @f(%a: f64) -> f64 attributes {tessera.autodiff="reverse"} { return %a : f64 }` seeds cotan[%a] with an `arith.constant 1.0 : f32`, and the Step-5 rewrite (326-360) appends `cotanV.getType()` to the signature — yielding an f32 gradient for an f64 argument, i.e. a type-incorrect result rather than a diagnostic; an integer scalar loss gets a float seed. Note the mismatch does not always reach an arith verifier failure as the evidence states — arith ops implement neither AdjointInterface nor LinearTransposeInterface, so many scalar programs fail earlier at [AUTODIFF_OP_NOT_DIFFERENTIABLE] (275-283) — but the defect itself stands and is currently masked only by the index-0 crash. Fix (build the attr from lossValue.getType()) is correct.

### `src/transforms/lib/AutodiffPairedPass.cpp:1774` — Saved while clone erased while zeros_like still uses it

*C++ — MLIR autodiff passes · logic*

**What is wrong:** `eraseSavedPrimal` is decided (line 1767) from `use_empty()` *before* differentiateOperation runs, but buildWhileAdjoint can create a new use of the clone: for a differentiable state with a null cotangent seed and a dynamic/unranked shape, initialCotangents calls buildZeroLike on `whileOp.getResult(index)` (RegionAdjointInterface.cpp line 662), which emits `tessera.custom_adjoint_call` taking the clone's result as an operand. The subsequent `op->erase()` destroys an operation that still has uses — an assert on an assertions build, dangling-use IR corruption on the fleet's NDEBUG builds.

**Evidence:** A saved scf.while with two float states where only one is seeded from the forward return and the unseeded one is dynamically shaped: use_empty() is true at line 1767 (results are only referenced via the cotangent map, not IR uses), buildWhileAdjoint then materializes zeros_like(clone.result), and line 1775 erases the clone under that live use.

**Fix:** Re-check use_empty() after differentiateOperation before erasing, or have buildZeroLike for while-state seeds derive extents from the loop inits/residual tape rather than the primal result.

**Independently verified:** Reproduced and isolated. A saved `scf.while` with two dynamically-shaped float states where only the first is returned (so the second's outputCotangent is null) crashes `--tessera-autodiff-paired` in the MLIR verifier/printer (`OperationVerifier::verifyOpAndDominance`, also on `--verify-each=false` via AsmState), the classic NDEBUG signature of a freed operation still referenced. Two controls isolate the cause: (a) identical program with `tensor<4xf32>` instead of `tensor<?xf32>` prints clean IR — buildZeroLike takes the static branch (RegionAdjointInterface.cpp:32-35) and creates no use; (b) identical dynamic program with BOTH states returned (both cotangents seeded) also prints clean IR, with the while clone erased and no `zeros_like` on it. Only the dynamic + unseeded-state combination hits line 662's `buildZeroLike(builder, whileOp.getResult(index))` → `tessera.custom_adjoint_call "zeros_like"(clone.result)` (lines 38-42), which creates a use of the clone AFTER `eraseSavedPrimal` was latched from `use_empty()` at AutodiffPairedPass.cpp:1767, and `op->erase()` at 1775 then destroys it under that live use. Both proposed fixes are sound.

### `src/transforms/lib/SymbolicDimEqualityPass.cpp:574` — Matmul contract check ignores transposeA/transposeB

*C++ — analysis passes · logic*

**What is wrong:** checkMatmul always compares lhs->back() against rhs->front() and errors on mismatch. The comment (lines 572-573) acknowledges transposed variants shift the contracting positions, but the code neither reads transposeA/transposeB nor skips when they are set, so a correctly annotated transposed matmul is reported as SYMDIM_MATMUL_CONTRACT_VIOLATION — a false verifier failure on correct IR. The V2 flow-propagation rule (lines 736-741) has the same blind spot when computing out-names.

**Evidence:** tessera.matmul {transposeB = true, tessera.dim_names_lhs = ["M","K"], tessera.dim_names_rhs = ["N","K"]} — semantically correct (contraction on K), but kL="K" vs kR="N" → the pass fails the module.

**Fix:** Read transposeA/transposeB and pick lhs position (back vs front) and rhs position (front vs back) accordingly, or skip the check with no diagnostic when either transpose attr is present (best-effort contract).

**Independently verified:** Reproduced a false verifier failure on correct IR, and the file's own header states the intended contract the code does not implement. `tessera.matmul` really does carry transposeA/transposeB (TesseraOps.td:216-217, and a canonicalizer that folds a transpose into those flags). Running the pass on `"tessera.matmul"(%a,%b) {transposeB = true, tessera.dim_names_lhs=["M","K"], tessera.dim_names_rhs=["N","K"]} : (tensor<4x8xf32>, tensor<16x8xf32>) -> tensor<4x16xf32>` — shape-consistent and verifier-accepted — emits `SYMDIM_MATMUL_CONTRACT_VIOLATION: lhs contracts on 'K' but rhs contracts on 'N'`. The pass header (line 26-27) documents the rule as "K dim name on lhs (last) must equal K dim name on rhs (first, modulo transposeA/B)" — the "modulo transposeA/B" is not implemented. The flow rule at 736-741 has the same blind spot (out = lhs[:-1] + rhs.back() gives [M,K] instead of [M,N] under transposeB), which can then cascade into SYMDIM_FLOW_INCONSISTENCY / SYMDIM_CALL_ARG_MISMATCH. Mitigating context: no in-tree producer emits dim_names_lhs/rhs today (tests only), so it is currently a latent false-positive.

### `src/transforms/lib/ActivationRematerializationPass.cpp:475` — Budget selection loop is O(N^3) via repeated peak scan

*C++ — analysis passes · algorithm*

**What is wrong:** estimatedPeak (lines 454-470) is O(points x active) = O(N x C) per call, and the selection loop calls it once per removal plus once per while-condition check, giving O(K x N x C) with C ~ N. It recomputes the full interval-overlap sum from scratch at every point instead of using an O(N) difference-array sweep (add bytes at begin, subtract after end, prefix-sum, max), and could be updated incrementally per removal.

**Evidence:** A realistic backward training graph of N = 5000 result-bearing ops with a tight budget forcing K = several hundred removals: ~K x N^2 = 10^10 inner interval tests in a single pass invocation — minutes of compile time for what an O(N log N) or incremental O(N) sweep does in milliseconds.

**Fix:** Compute the peak once with a difference array over ordinals; after removing a candidate, subtract its interval from the difference array and recompute the running max in O(N) — total O(K x N) worst case.

**Independently verified:** Complexity re-derived and cost measured. estimatedPeak (454-470) is O(|ordered| x |active|) and is called once per while-iteration (475) plus twice for the before/after attributes, giving O(K x N x C) with C ~ N. Measured with a synthetic function of N result-bearing pure ops all live to a final sink, `tessera.remat_budget_mb = 1` forcing K ~ N removals: N=500 -> 0.54s, N=1000 -> 3.07s, N=2000 -> 18.82s (growth factor ~6x per doubling, i.e. ~N^2.6), against a 0.08s baseline for the same file with no pass — so essentially all of it is this loop, and extrapolation to N=5000 is ~5 minutes, matching the reviewer's estimate. The proposed fix is valid: intervals are inclusive [begin,end] over dense ordinals, so a difference array (+bytes at begin, -bytes at end+1, prefix-sum max) gives the peak in O(N+C), and subtracting a removed candidate's interval keeps each iteration O(N) -> O(K x N) total.

### `src/transforms/lib/SymbolicDimEqualityPass.cpp:196` — Malformed dim_bindings entries silently dropped

*C++ — analysis passes · logic*

**What is wrong:** readBindings skips any tessera.dim_bindings entry that is not a StringAttr or fails parseBinding (lines 194-197) with no diagnostic; readDimSizes (line 182) likewise drops non-integer dim_sizes entries. A typo in a binding string disables that equation's verification entirely while the pass still reports success — a fail-open on a semantic attribute in a pass whose whole purpose is fail-closed verification.

**Evidence:** tessera.dim_bindings = ["D = H ** Dh"] (double star -> empty product token -> parseProductTerm returns nullopt): the binding is dropped, dim_sizes {D=512, H=8, Dh=32} (a real 512 != 256 violation) passes the verifier silently.

**Fix:** Emit a stable SYMDIM_BINDING_MALFORMED error (and fail the pass) when a dim_bindings element is not a string or does not parse, and when a dim_sizes value is not an IntegerAttr.

**Independently verified:** Reproduced end-to-end. readBindings (190-200) drops any element that is not a StringAttr or that parseBinding rejects, and readDimSizes (177-186) drops non-IntegerAttr values, both with no diagnostic. Ran three functions carrying the same真 violation (D=512, H=8, Dh=32): the well-formed `"D = H * Dh"` correctly fires `SYMDIM_BINDING_VIOLATION ... product of RHS = 256`; the typo'd `"D = H ** Dh"` (empty product token -> parseProductTerm nullopt) and the variant with `D = "512"` in dim_sizes both pass silently with exit 0. This is a fail-open on a semantic annotation inside a pass whose purpose is fail-closed verification, and it is inconsistent with the newer typed carrier in the same file, which fails closed loudly (SYMDIM_PRESBURGER_MALFORMED, line 213-222). Checked the counter-argument that strictness would reject the documented out-of-scope grammar (constants, subtraction, parens, lines 97-101): it would not — those forms parse into unresolvable symbols and are skipped by resolution, not by parseBinding, so only genuinely malformed strings would newly error. No producer emits dim_bindings (Python only declares it in pass_metadata/diagnostic_codes), so these are hand-authored attributes — precisely where a typo happens.

### `src/transforms/lib/IRContractLegalityPass.cpp:417` — Narrowing-accum rule rejects legal int-storage/float-accum pairs

*C++ — legality / verifier passes · math*

**What is wrong:** The narrowing check compares numericPolicyMantissaBits across the int/float domain boundary, so integer storage with a float accumulator — which the comment on lines 407-409 explicitly declares 'the ordinary dequantized-weight path and stays legal' — is refused whenever the integer's representable width exceeds the float's mantissa: int16 (16) into fp16 (11) or bf16 (8), and int32 (32) into fp32 (24), all emit NUMERIC_POLICY_NARROWING_ACCUM. Only the int8 case slips through (8 <= 11). The 25.8x-dominance measurement justifying refusal was derived for float-into-float and does not transfer to dequantization, where storage bits are integer codes, not a running-sum precision.

**Evidence:** numeric_policy = {storage = "int16", accum = "fp16"} (a standard int16-quantized weight dequantized and accumulated in fp16) → storageBits=16 > accumBits=11, both bits>0, storage not float so the int-accum early-exit at line 410 is skipped → hard failure with a diagnostic asserting a claim ('bit-identical to narrowing the storage') measured only for float pairs.

**Fix:** Restrict the accumBits < storageBits comparison to same-domain pairs (float/float and int/int); for int storage with float accum either accept unconditionally per the stated dequant contract, or gate on a genuinely derived exact-representability rule with its own diagnostic code.

**Independently verified:** Reproduced: numeric_policy = {storage="int16", accum="fp16"} and {storage="int32", accum="fp32"} both fail with NUMERIC_POLICY_NARROWING_ACCUM; {storage="int8", accum="fp16"} passes (8<=11), exactly the boundary the finding predicts. int16/int32 are in knownStorageDtypes and are not lowPrecisionStorages, so no earlier rule intercepts. The comparison at line 417 is cross-domain: numericPolicyMantissaBits returns float significand bits for floats and representable width for ints (line 204-216), and the justification attached to the diagnostic — the 25.8x dominance and the BIT-IDENTICAL claim (header lines 118-137) — was measured only for fp32/fp16 and fp32/bf16 storage pairs. The remedy the note prints ('narrow storage to "fp16"') is incoherent for an integer-quantized storage, and it contradicts this same function's comment at 407-409 declaring int-storage/float-accum 'the ordinary dequantized-weight path [that] stays legal'. Impact is latent: no in-tree producer emits int16/int32 storage with a narrower float accum (all in-tree int policies are int4/int8 with int32/fp32), but such a policy is expressible from the canonical dtype set.

### `src/transforms/lib/MaterializeControlPayloadPass.cpp:177` — Shared stub across control ops silently overwritten

*C++ — legality / verifier passes · logic*

**What is wrong:** emitOpList clears and rewrites the stub's body in place (stub.getBody().getBlocks().clear()). materializeControlIf guards the intra-op aliasing case (thenStub == elseStub, line 293), but nothing guards two different control ops in the ctrl worklist resolving to the same func.func symbol: the second materialization overwrites the first op's already-emitted body, so both loops silently execute the second payload. This is exactly the 'silently changing the branch semantics' hazard the intra-op guard exists for, un-handled in the cross-op case.

**Evidence:** Two tessera.control_for ops both carrying body = @loop_body (legal IR — one symbol, two call sites) with different body_opcodes payloads: op 1 materializes @loop_body from payload 1 and strips its attrs; op 2 clears @loop_body and re-emits payload 2. After the pass, loop 1's CF2 lowering calls @loop_body and computes loop 2's body — wrong results with no diagnostic and no verifier failure (both signatures can match).

**Fix:** Track materialized stubs across the worklist (DenseSet<Operation*>); on a second control op resolving to an already-materialized stub, leave its payload intact for the guard (return false) unless the payload is byte-identical.

**Independently verified:** Reproduced. Two tessera.control_for ops both naming body = @loop_body with different payloads (rmsnorm vs relu): after --tessera-materialize-control-payload, @loop_body contains only tessera.relu and BOTH loops call it; the first op's payload attributes were already stripped, so the wrong body is now the only body. No diagnostic, no verifier failure. The intra-op guard (thenStub == elseStub, line 293; bodyStub == condStub, line 369) shows the author recognized the hazard; emitOpList's stub.getBody().getBlocks().clear() (line 177) has no cross-worklist protection. Latent today only because the Python emitter (_jit_boundary.py:1739-1747) emits one control_for per module with a hardcoded @loop_body — which is also why a second loop in one module would collide.

### `src/transforms/lib/WarpSpecLegalityPass.cpp:279` — MMA token-sync check fails open through intermediate ops

*C++ — legality / verifier passes · logic*

**What is wrong:** The WARPSPEC_MMA_NOT_TOKEN_SYNCED presence check attributes an mma operand to an async producer only when provenance resolution terminates directly on a tile.async_copy / tile.tma.copy_async / producer warp region. resolveThroughLoopCarry stops at the first non-block-argument defining op, so any intervening op between the copy and the mma (a cast, slice, transpose, or an scf.for result feeding the mma from outside the loop) becomes the root, isAsyncDataProducer returns false, the operand is silently dropped from fromProducer, and the missing-completion-token race is never reported — a fail-open in a check the file documents as closing the measured §5.1.1 fail-open.

**Evidence:** %staged = tile.tma.copy_async ... ; %t = tile.transpose(%staged) ; tile.mma(%t, %w) with no !tile.async_token operand: root of %t is the transpose, line 279 'continue' skips it, fromProducer stays empty, no diagnostic — the mma runs against an in-flight copy exactly as in the deadlock/race signature the pass claims to reject.

**Fix:** When a resolved root is not a recognized producer, either transitively chase its operands to a fixed point, or fail closed with an explicit 'cannot prove staged-data provenance' diagnostic mirroring TILE_BARRIER_ORIGIN_UNRESOLVED, rather than silently excluding the operand from the check.

**Independently verified:** Reproduced. %tA,%tok = tile.async_copy ; %t = tile.transpose(%tA) ; tile.mma(%t, %B) with no async_token operand passes --tessera-warpspec-legality silently (exit 0), while the identical kernel without the transpose is flagged WARPSPEC_MMA_NOT_TOKEN_SYNCED by the committed fixture. Mechanism is as claimed: resolveThroughLoopCarry (TileValueProvenance.h) inserts the first non-block-argument value as a root and stops, so the transpose result becomes the root, isAsyncDataProducer returns false, line 279 'continue' drops the operand and fromProducer stays empty. Aggravating: the caller never consults res.complete, although the provenance header states callers must treat incomplete resolution as unproven, not permission (Decision #30). This is a missed detection, not a miscompile by this pass — but it is the fail-open class the pass documents itself as closing.

### `src/transforms/lib/AdjointCollectiveInsertionPass.cpp:169` — Unchecked ArrayAttr index; OOB on short cotangent array

*C++ — legality / verifier passes · logic*

**What is wrong:** cotanArrayAttr[arg.getArgNumber()] is indexed for every function argument without checking cotanArrayAttr.size() against the argument count. ArrayAttr::operator[] indexes the underlying ArrayRef; on the fleet's NDEBUG LLVM builds (per CLAUDE.md, all boxes are assertions-OFF) an array shorter than the argument list is a silent out-of-bounds read into adjacent attribute storage rather than a trapped assert — the pass may then treat garbage as a populated slot and rewrite the wrong return operand.

**Evidence:** Hand-written or drifted IR (the class of input the origResultCount<0 defensive bail at line 154 already anticipates): func with 4 args and tessera.autodiff.arg_cotangents = ["d_a", "d_b"] — args 2 and 3 index past the 2-element array. On NDEBUG this reads out of bounds; whatever Attribute bits come back can dyn_cast to a non-empty StringAttr and desynchronize cotanIndex, splicing a collective onto an unrelated return operand.

**Fix:** Bail with a diagnostic when cotanArrayAttr.size() != func.getNumArguments(), the same way the origResultCount<0 mismatch is already treated — but with an error rather than a silent return (see the P1 finding on silent bail-outs).

**Independently verified:** Reproduced: a 4-arg func with tessera.autodiff.arg_cotangents = ["%cotan_arg_0", "%cotan_arg_1"] aborts inside mlir::ArrayAttr::operator[] called from AdjointCollectiveInsertionPass::runOnOperation (line 169) — the index is unchecked against cotanArrayAttr.size(). One correction to the evidence: this build has header asserts enabled, so it traps ('index out of bounds', BuiltinAttributes.h.inc:466) rather than silently reading adjacent storage; on a true NDEBUG build it is the out-of-bounds read the finding describes. Mitigating: AutodiffPass always writes exactly one entry per argument (AutodiffPass.cpp:326-339), so the short array requires hand-written or drifted IR — the same input class the origResultCount<0 bail at line 154 already anticipates, and the class every phase_f5 fixture belongs to.

### `src/transforms/lib/TileBufferArenaPass.cpp:60` — getIntOrFloatBitWidth on non-scalar element type crashes

*C++ — legality / verifier passes · logic*

**What is wrong:** staticByteSize (line 60) and elementAlign (line 73) call getIntOrFloatBitWidth() unconditionally on the memref element type. That method asserts the type is an integer or float; for a memref of vector, complex, or index elements it is an assertion failure on an assertions build and undefined behavior on the fleet's NDEBUG builds — while every neighboring unknown (dynamic shape, missing size) is handled by the deliberate poison/-1 skip path. The same unguarded call recurs in the dynamic path at line 257.

**Evidence:** tile.alloc_shared over memref<64x8xvector<4xf16>, 3> (vectorized staging, a natural Tile IR shape) with a tile.buffer_group attribute: layoutSpace calls staticByteSize → getIntOrFloatBitWidth on vector<4xf16> → assert/UB, taking down tessera-opt instead of leaving the group unplaced the way an unknown static size does.

**Fix:** Guard with isIntOrFloat() (or use DataLayout type size) and return -1 / alignment 1 on anything else, so non-scalar element types flow into the existing unplaceable-group path instead of crashing.

**Independently verified:** Reproduced: tile.alloc_shared over memref<64x8xvector<4xf16>, 3> with tile.buffer_group crashes tessera-opt in staticByteSize -> mlir::Type::getIntOrFloatBitWidth (TileBufferArenaPass.cpp:60), stack trace confirmed. tile.alloc_shared has no ODS definition (no .td declares it; fixtures run with --allow-unregistered-dialect), so no operand-type constraint prevents vector/index/complex element types, and the static-shape guard at line 55 does not cover element type. elementAlign (line 73) and the dynamic path (line 257) carry the same unguarded call. This contradicts the pass's own discipline for unknowns — an unknown static size yields -1 and leaves the group unplaced. Impact is latent: every in-tree producer stages scalar bf16/f16 tiles, so no current pipeline reaches it.

### `src/solvers/scaling_resilience/lib/sr/passes/InsertRecomputePass.cpp:100` — Live-set is a cumulative sum; deaths never subtracted

*C++ — spectral / Clifford / TPP solvers · math*

**What is wrong:** liveBytes only ever accumulates result bytes and is reset at checkpoints; a value's bytes are never removed when its last use passes. The quantity compared to --memory-budget-mb is therefore 'total bytes produced since last checkpoint', not live-set size, so the pass checkpoints far more often than the budget requires (Decision #10 specifies a live-set scan).

**Evidence:** A chain of 64 elementwise ops each producing a 1 GiB tensor consumed immediately by the next op has true peak liveness ~2 GiB, well under the 4 GiB default budget — yet liveBytes crosses 4 GiB after 4 ops and the pass inserts ~16 checkpoints on a program that needed zero, forcing unnecessary recompute segments. Also note estimateTensorBytes prices any dynamic-shape tensor at 4096 bytes, so a dynamically-shaped activation-heavy graph never triggers a checkpoint at all.

**Fix:** Track per-value last uses (walk uses once, O(ops)) and subtract a value's bytes at its final consumer before comparing against the budget.

**Independently verified:** InsertRecomputePass.cpp:89-128. liveBytes is only ever incremented (line 100) and zeroed at a checkpoint (107/118); there is no use-walk and nothing is ever subtracted when a value dies, so the quantity compared to the budget at line 113 is bytes-produced-since-last-checkpoint, contradicting both the pass header ('live-set size') and Decision #10's 'greedy live-set scan'. Re-derived the example: 64 chained 1 GiB elementwise ops have true peak liveness ~2 GiB, but liveBytes crosses the 4 GiB default on the 5th op, yielding ~12 spurious checkpoints (the finding says ~16 — arithmetic slack, substance holds). Two aggravators I verified independently: estimateTensorBytes returns 4096 for any non-static-shaped type (line 37-38), and because the dyn_cast<ShapedType> also fails for scalars, every index/i1/i64 result is likewise priced at 4096 — so dynamic-shape graphs under-trigger while scalar-heavy ones over-trigger. Proposed fix (single use-walk to find last uses, subtract at the final consumer) is valid and O(ops).

### `src/solvers/tpp/lib/TargetHooks/CPU/Stencil.cpp:36` — order>=6 silently executes the order-4 stencil

*C++ — spectral / Clifford / TPP solvers · logic*

**What is wrong:** ts_stencil_grad_cpu branches `if (order >= 4)` into the 4th-order formula, so any higher declared accuracy (order 6, 8 — all accepted by LegalizeSpaceTime's 'positive even order' check, and for which HaloInfer allocates halo order/2 = 3, 4) is silently computed at 4th order. The result has the wrong truncation error versus the operator the IR declared, with no diagnostic anywhere on the path — a silently defaulted semantic attribute.

**Evidence:** tpp.grad {scheme="central", order=6} legalizes cleanly, gets halo=3, and lowers to ts_stencil_grad_cpu, which returns the order-4 result: for f(x)=x^5 on a periodic grid the order-6 stencil is exact while the returned order-4 value carries an O(h^4) error — a numerically wrong answer relative to the declared scheme that a convergence study will attribute to the PDE code. Also the leading `if (!in || ...) return;` silently no-ops on invalid arguments with no status return.

**Fix:** Either implement the requested order or reject it: have the lowering pass gate order to {2,4} for this symbol (stable diagnostic), and give the C hook an error return instead of silent early-outs.

**Independently verified:** Stencil.cpp:36-40: `if (order >= 4)` selects the 4th-order tap set (-f[i+2]+8f[i+1]-8f[i-1]+f[i-2])/(12h) — which I verified is the correct 4th-order central first derivative — for order 4, 6, 8 alike; there is no order-6 branch and no rejection. Nothing upstream gates it: LegalizeSpaceTime.cpp:91-97 accepts any positive even order, HaloInfer gives order 6 halo=3, and LowerTPPToTargetIR.cpp:51-54,94-99 marks tpp.grad on cpu 'executable' with call ts_stencil_grad_cpu without ever reading `order`. So a declared order-6 operator is executed at 4th order with no diagnostic on any path — a silently defaulted semantic attribute (Decision #21a). The C entry point also returns void and early-returns on invalid nx/ny/axis/spacing (lines 21-23), as claimed. Partial mitigation, not a refutation: the Python arbiter's reference (compiler/emit/tpp_candidates.py, central_difference_taps) raises ValueError for (1,6), so an arbiter verification of order 6 throws rather than silently passing — but CpuStencilGradCandidate.run still forwards region.order to the hook and returns the order-4 answer, and the C ABI symbol itself is unguarded.

### `src/solvers/tpp/lib/Passes/AsyncPrefetch.cpp:81` — Prefetch hoist checks SSA deps only, not memory writes

*C++ — spectral / Clifford / TPP solvers · logic*

**What is wrong:** The hoist legality check producesOperandOf() only tests whether prev's SSA results feed the prefetch. A preceding op that writes THROUGH a memref the prefetch reads (memref.store, linalg on memrefs, a collective writing a buffer — all zero-result ops) passes the check, and the prefetch is moved above the write, prefetching stale data.

**Evidence:** Block: `memref.store %v, %buf[...]` followed by `schedule.prefetch %buf {overlap="compute"}`. The store has no results, so producesOperandOf returns false and the prefetch is hoisted above it; the double-buffer stage now holds pre-store contents and the consuming compute reads stale values. The header comment claims 'dependency-safe', but the check derives only SSA facts — a told-not-derived memory fact that fails open (Decision #30).

**Fix:** Also refuse to hoist when prev has memory write effects (MemoryEffectOpInterface, or conservatively any zero-result op with shared memref operands) on any buffer the prefetch touches.

**Independently verified:** AsyncPrefetch.cpp:33-40 + 77-84: producesOperandOf compares prev->getResults() against op->getOperands() only, and the hoist at line 81 fires whenever prev is non-terminator and that SSA test is false. A zero-result writer — memref.store, a linalg op on memrefs, a collective writing a buffer — always fails the test, so the prefetch moves above the write. Nothing restricts the prefetch source to value semantics: Schedule_PrefetchOp takes AnyType:$source (ScheduleMeshPipelineOps.td:556-566) and its verifier (ScheduleDialect.cpp:748-757) checks only type preservation, non-empty `into`, and the overlap enum — a memref source is legal. The pass also never consults MemoryEffectOpInterface for prev or for the prefetch, and the pass runs on a plain module walk with no bufferization ordering constraint, so the header's 'dependency-safe' claim rests on an SSA-only fact (Decision #30, told-not-derived, failing open). The existing fixture (async_prefetch_overlap.mlir) uses only tensor values and comp.matmul, so it does not pin the memref case as intentional. Proposed fix (refuse the hoist when prev has write effects on a buffer the prefetch touches) is the right shape.


## P3 — improvement opportunities (16) — FIXED IN SOURCE 2026-08-30, 3 rows device-unverified

> All 16 have a fix committed, together with the 3 P1 rows that had been left
> open as performance/scope items. **"Fixed" is not "closed" for three of
> them.** `nvidia_cuda.py:626` (flash-backward cleanup),
> `nvidia_solver_krylov.py:79` (warp-per-row matvec) and the ROCm half of the
> spectral per-frame row are verified only as generated text plus a clean
> `clang++ -fsyntax-only` parse — no GPU has executed any of them, and this
> repo does not count a source-level fix as a device claim. The obligations are
> recorded in `../backend/nvidia/todo.md` and `../backend/rocm/todo.md`; until
> those runs happen, treat those three rows as open on their own hardware.
>
> The other thirteen are host-measured on this Mac, and `apple_msl.py:1327`
> additionally executed on the Apple GPU.
>
> Two of the sixteen turned out not to be mere improvements:
>
> * **`laws.py:749`** was a correctness defect wearing a performance row's
>   clothes. A forward-mode kink check that could not be evaluated returned
>   `status='pass'` with a detail claiming both modes agree — a law oracle
>   reporting agreement it never verified. Reproduced with a rigged `maximum`
>   JVP; every unevaluable path now returns a named `rule_error`, and three
>   further empty-mask/no-probe-ran holes were closed in the process.
> * **`mixed_precision.py:189`** made a persistently-overflowing run stall
>   forever with `step()` always False. The floor is now `2**-14` raising
>   `E_GRAD_SCALE_EXHAUSTED`; a zero floor would have been its own
>   silent-wrong-answer, since zero gradients never overflow.
>
> Largest measured wins: conv1d 6430 ms -> 0.94 ms, `svd_coupling` on
> (4096,8,8) 63.8 -> 1.2 ms, `vjp_moe` 171.6 -> 9.0 ms, the pointwise exit
> scan 606 -> 10.9 ms at 7000 ops, and GMRES's FD adjoint from 2964 residual
> evaluations down to the dense budget of 84.
>
> Two claims were deliberately NOT made. The CUDA flash-backward cleanup and
> the Krylov warp-per-row rewrite are verified as generated text and a clean
> `clang++ -fsyntax-only` parse only — no GPU ran them, and the coalescing
> gain is stated as a property of the emitted access pattern (32 transactions
> per load down to 4 for f32), never as a speedup. The Krylov change also
> alters per-row summation order, so its results are no longer bit-identical
> to a sequential sum; `tests/performance/nvidia/test_solver_krylov_ratchet.py`
> compares against a baseline recorded with the old matvec and likely needs
> re-recording on the CUDA box.


### `python/tessera/autodiff/vjp.py:703` — vjp_moe per-token Python loop is vectorizable

*Autodiff — reverse mode (VJP) · performance*

**What is wrong:** The MoE backward iterates tokens one-by-one in Python, doing a (D,)@(D,E) matvec and an outer-product accumulation per token. Grouping tokens by expert (argsort of route_arr) turns this into one batched GEMM pair per expert — the same structure vjp_grouped_gemm already uses — removing T Python iterations and T small-BLAS dispatches.

**Evidence:** For a realistic T=4096 tokens, D=1024, 8 experts, the loop performs 8192 tiny BLAS calls plus 4096 np.outer allocations versus 16 large GEMMs when segmented; Python/dispatch overhead dominates, typically a 10-100x wall-clock difference at these sizes.

**Fix:** order = np.argsort(route_arr); split tokens/dout by np.bincount(route_arr) boundaries; per expert e: dx[rows] = dout[rows] @ E[e].T and dE[e] = tokens[rows].T @ dout[rows].

**Independently verified:** Verified at vjp.py:703-708: a Python for-loop over all tokens doing a (D,)@(D,E) matvec and np.outer accumulation per token. The proposed expert-segmented batched-GEMM restructuring is mathematically equivalent (routing is a fixed integer partition) and is the same structure vjp_grouped_gemm (vjp.py:5681) already uses. At the claimed sizes (T=4096, D=1024) per-token Python/BLAS-dispatch overhead genuinely dominates; the exact 10-100x figure is an estimate but the order-of-magnitude cost and the validity of the fix stand.

### `python/tessera/autodiff/transforms.py:288` — jacfwd runs one redundant full forward pass

*Autodiff — tape / grad / transforms · performance*

**What is wrong:** jacfwd calls fn(*args) once purely to learn out_shape, then discards the result; the first jvp call already evaluates the same primal and returns it. For an expensive fn with a small differentiated input this is a fixed extra forward — e.g. in_size=1 does 2 forward evaluations instead of 1 (2x), in_size=4 does 5 instead of 4 (+25%).

**Evidence:** Line 288: `sample = np.asarray(fn(*args, **kwargs))` used only for `out_shape`; each subsequent `jvp(fn_of_one, ai, tangent)` (line 310) re-evaluates the full primal anyway and returns it as the first tuple element, which is thrown away (`_, dy = ...`).

**Fix:** Defer buffer allocation until the first jvp call per arg and take out_shape from that call's primal output, dropping the sample run.

**Independently verified:** Verified by reading both sides. transforms.py:288 computes `sample = np.asarray(fn(*args, **kwargs))` and uses it only for `out_shape` (line 289); the value is discarded. `jvp` (jvp.py:1775-1808) genuinely evaluates the primal inside the trace and returns it as the first tuple element, which jacfwd throws away at line 310 (`_, dy = ...`). So for in_size >= 1 the sample run is a strictly redundant full forward, and the multiplier is exactly (n+1)/n as claimed. This is corroborated, not refuted, by tests/unit/test_derivative_methodology.py:175-188, which asserts `fwd[0] == per_pass * (n + 1)` — it measures the extra pass rather than justifying it, and the docstring there is descriptive ('one sample pass plus one per input'), not a design contract. The proposed fix is valid but that test would need updating to `per_pass * n`, and the deferred-allocation version must handle the zero-size-input / empty-argnums edge where no jvp call ever runs and out_shape would stay unknown.

### `python/tessera/autodiff/laws.py:749` — Kink forward-mode check silently skipped, reported as agreement

*Autodiff — laws & algebra oracles · logic*

**What is wrong:** _check_split_forward_mode returns None both when forward mode conforms AND when it could not be checked at all — `if out.shape != m.shape: return None` (line 749-750) and `if len(idx) < 2: return None` (line 710-711). kink_check treats any None as success and appends the detail 'both modes agree' (line 679), so a JVP whose output shape doesn't match the mask (e.g. a keepdims mismatch, or a scalar where an array is expected) gets a passing kink row whose detail affirmatively asserts a conformance that was never evaluated — a claim-integrity violation in the very harness built to enforce claim integrity.

**Evidence:** Register a SUBGRAD_SPLIT op whose JVP returns a keepdims-reduced (1,1) tangent while masks[i] is (1,3): out.shape != m.shape on every operand, every probe returns None, and the LawResult is status='pass' with detail 'both modes agree' although forward-mode selection was never inspected.

**Fix:** Distinguish 'conforms' from 'not checkable': return a sentinel (or a not_applicable/rule_error LawResult) on the shape-mismatch and <2-ties paths, and only append 'both modes agree' when a discriminating probe actually ran.

**Independently verified:** Reproduced end-to-end. laws.py:749-750 (`if out.shape != m.shape: return None`) and :710-711 (`if len(idx) < 2: return None`) return the same sentinel as the conformance path, and kink_check:656-658 treats any None as success then unconditionally appends 'both modes agree' at :679. I constructed the claimed case directly: reusing the real `maximum` KINK_SPEC and VJP with a JVP that is a hard first-select AND returns a keepdims-reduced (1,1) tangent, kink_check returns status='pass', detail='2 tied elements share 0.5; both modes agree' — an affirmative conformance claim for a check that never executed, and the wrong-selection JVP the probe was written (per its own docstring, PR #588) to catch goes undetected. The same silent-skip shape guard exists on the non-split branch at :671-673 (`if np.shape(t_out) == np.shape(m) else np.array([])` then `if sel.size`). All 12 currently registered kink ops pass with real probes, so this is latent harness fragility, not an active false green — consistent with the reported P3.

### `python/tessera/autodiff/law_inputs.py:159` — atan2 spec guards a==b, not its real singularity

*Autodiff — laws & algebra oracles · logic*

**What is wrong:** The atan2 spec uses _binary(tie_gap=0.3), which displaces b away from a==b ties — but a==b is not a nonsmooth point of atan2 at all. atan2(y, x)'s actual bad set is the origin (derivative singularity, both partials ~1/(x^2+y^2)) and the branch cut y=0 with x<0 (a 2π value jump that the chain law's central finite difference straddles). Nothing in the spec bounds |a| (=y) away from 0 or keeps b (=x) positive, so the file's own contract — 'kink avoidance is part of the spec's contract' — is unfulfilled; the guard that is present protects a nonexistent kink.

**Evidence:** If any sampled element has y within ~eps_fd·|tangent| ≈ 1e-5 of 0 with x < 0, shifted(+1) and shifted(-1) in chain_check evaluate atan2 on opposite sides of the cut and the FD contains a 2π/2e-5 = O(1e5) spurious term → chain-law fail on a correct rule. With the fixed op_rng seed this is a latent landmine rather than a flake: any reseeding/regeneration of the probe (e.g. renaming the op or changing the salt string) can permanently flip the row.

**Fix:** Give atan2 its own make: e.g. y = _away_from(rng.standard_normal(shape), 0.0, margin=0.25), x = 0.5 + np.abs(rng.standard_normal(shape)) (or any construction keeping (x, y) bounded away from the cut and origin).

**Independently verified:** Verified. python/tessera/__init__.py:4230 defines `atan2(y, x)`, so the reviewer's a=y / b=x mapping is right, and _binary(tie_gap=0.3) (law_inputs.py:98-105) only displaces b when |a-b| < 0.3 — a `maximum`-style tie guard. atan2 is perfectly smooth on the line y==x (away from the origin), so that guard protects a nonexistent kink, while nothing bounds y away from 0 or keeps x > 0. The blow-up is real: chain_check's `shifted(±1)` (laws.py:452-459) evaluates the canonical forward at p ± 1e-5·t and takes a central difference, and the branch cut is a 2π value jump. I ran chain_check on a rigged near-cut spec (y[0,0]=1e-7, x = -0.5 - |N(0,1)|) with the unmodified, correct atan2 JVP: status='fail', max_rel_residual=0.99999734. With the committed spec and the fixed op_rng seed the row currently passes (max_rel_residual 6.66e-10; the sample's min|y| happens to be 0.0143), so this is exactly the latent seed-dependent landmine described, not a live failure. The file's own header contract ('kink avoidance is part of the spec's contract') is therefore unmet for atan2, and the proposed replacement make() addresses the actual bad set.

### `python/tessera/autodiff/degeneracy.py:743` — check_full_rank judges batched spectra globally, not per element

*Autodiff — laws & algebra oracles · logic*

**What is wrong:** check_full_rank computes smallest = min(|s|) and scale = max(|s|) over the ENTIRE array including batch dimensions, so a batched call is judged on a cross-batch ratio. A batch element that is perfectly full-rank at its own scale is refused when it coexists with a much larger-scaled element. This is inconsistent with check_factor_rank in the same file, which explicitly flattens to per-row diagonals and judges each row's own min/max ratio (lines 791-795), and with svd_coupling, which judges 'each batch element on its own spectrum'.

**Evidence:** Batched nuclear-norm gradient with spectra [[1.0, 0.5], [1e-16, 5e-17]]: element 1 has own ratio 0.5 (fully regular; U·Vᵀ well-defined), but globally smallest/scale = 5e-17 <= existence_tolerance(2) ≈ 4.4e-16 → TesseraDegeneracyError refuses the whole batch under FAIL_CLOSED. (The reverse direction is safe: the global ratio lower-bounds every per-row ratio, so no rank-deficient element is falsely accepted — the bug is spurious refusal only.)

**Fix:** Mirror check_factor_rank: reshape to (-1, k), compute per-row smallest/scale, and judge/report the worst row.

**Independently verified:** Traced and reproduced. degeneracy.py:743-745 computes `scale = max(|s_arr|)` and `smallest = min(|s_arr|)` over the whole array, including leading batch dims, while `tol` on line 742 is derived from s_arr.shape[-1] (per-row k) — the two are inconsistent by construction. _norm_vjp (linalg_ops.py:436-441) is fully batched and passes the batched sv straight in. I built the reviewer's exact case: two 2x2 blocks with spectra [1.0, 0.5] and [1e-16, 5e-17] — per-element ratios 0.5 and 0.5, both fully regular at their own scale. Each element alone returns a gradient; the stacked batch raises TesseraDegeneracyError ('smallest singular value 5.000000e-17, relative 5.000e-17 <= tol 4.441e-16'). The inconsistency with siblings is real: check_factor_rank (lines 791-795) explicitly reshapes to (-1, k) and picks the worst per-row ratio, and svd_coupling (line 625, docstring line 581) judges each batch element on its own spectrum. The direction of the bug is as stated — the global ratio lower-bounds every per-row ratio, so only spurious refusal is possible. No test pins the global behavior.

### `python/tessera/autodiff/degeneracy.py:634` — Per-batch-element Python loop on the factorization backward path

*Autodiff — laws & algebra oracles · performance*

**What is wrong:** svd_coupling and eigh_coupling (line 419) run a Python for-loop over every flattened batch element under FAIL_CLOSED/GENERALIZED — the default policy — each iteration doing argsort, Python-list cluster building, and np.ix_ mask writes. Since vjp.py routes every svd/eigh backward through these helpers, a batched factorization backward pays O(B) interpreter overhead that dominates the O(k^3) numpy work for the small per-element k typical of batched use.

**Evidence:** A backward through svd on a (4096, 8, 8) batch runs 4096 iterations of singular_value_clusters (each: asarray, argsort, zip loop over k, list-of-tuples) at roughly 20-50 µs per iteration → ~0.1-0.2 s of pure guard overhead per backward call, versus microseconds for a vectorized gap test. The common no-degeneracy case needs only a vectorized screen: sort s^2 per row, diff, compare against atol — falling back to the per-element diagnostic path only for the (rare) rows that flag.

**Fix:** Vectorize the fast path: compute per-row sorted-gap minima in one numpy pass; enter the existing per-element cluster/diagnostic loop only for rows whose minimum gap is <= atol (and for the warning band).

**Independently verified:** Measured, and the cost is real. Confirmed the loops exist and are the default path: active_policy('svd') and ('eigh') both resolve to fail_closed, so svd_coupling:634-657 and eigh_coupling:419-439 iterate every flattened batch element, each doing singular_value_clusters/eigenvalue_clusters -> _cluster_indices (asarray, argsort, Python zip loop over k, list-of-tuples) plus np.ix_ mask writes. Benchmarked on this M1 Max: (4096,8,8) -> svd_coupling 58.4 ms vs 37.4 ms for the np.linalg.svd itself (14.3 us/element); (1024,8,8) -> 14.9 ms vs 9.7 ms. So at the small-k batched shapes the finding scopes to, the guard costs ~1.5x the factorization it guards. The reviewer's per-iteration estimate (20-50 us) is ~2-3x high and the absolute figure (0.1-0.2 s) ~3x high, and the ratio inverts at large k (64x(64,64) -> 3.2 ms guard vs 72 ms svd, ~4%), but the claim is explicitly scoped to small per-element k and holds there. The proposed vectorized screen is valid: _cluster_indices (lines 337-356) declares a cluster exactly when some adjacent sorted gap <= tol*max|v|, so a per-row sorted-diff minimum compared against the row's atol is an exact existence test, with the per-element loop retained for rows that flag and for the warning band.

### `python/tessera/autodiff/mixed_precision.py:189` — GradScaler backoff floored at 1.0 blocks sub-unity scale

*Autodiff — implicit / jets / remat · logic*

**What is wrong:** On overflow, the scale updates as max(self._scale * self._backoff, 1.0), so the loss scale can never fall below 1. When unscaled fp16 gradients are themselves near the fp16 max (~65504) — large-magnitude losses where the correct loss scale is < 1 — every step detects inf, is skipped, and the scale stays pinned at 1.0: training silently stalls with step() returning False forever. torch's GradScaler deliberately allows the scale to decay below 1 for exactly this case.

**Evidence:** fp16 autocast with raw gradient magnitudes ~1e5: at scale 1.0 the scaled grads overflow fp16, _has_inf_nan fires, backoff computes max(0.5, 1.0) = 1.0, and the loop repeats indefinitely — no parameter update is ever taken and no error is raised.

**Fix:** Remove the 1.0 floor (or make the floor a tiny positive value like 2**-14) so backoff can reach sub-unity scales.

**Independently verified:** The mechanism is confirmed by code and demonstrated by execution, though the finding's specific fp16 evidence does not hold in this lane. mixed_precision.py:189 self._scale = max(self._scale * self._backoff, 1.0) makes 1.0 an absolute floor, and step() returns False with no diagnostic on every overflow, so a workload whose correct loss scale is sub-unity is a permanent silent stall. Demonstrated: with GradScaler(init_scale=1.0) and a raw cotangent whose backward overflows the gradient dtype, step() returned False on all 5 iterations with the scale pinned at 1.0, while the same backward at scale 0.5 and 0.25 produced finite gradients - i.e. the recovery the class exists to provide is unreachable. The claimed torch divergence is accurate (torch's _amp_update_scale_ has no floor). Correction to the finding's evidence: in this numpy reference lane p.grad is stored fp32 and the VJP arithmetic runs fp32 even inside autocast('fp16') (measured: a gradient of 640000.0 - well past the fp16 max - came back finite), so 'unscaled fp16 gradients near 65504' is not the realizable trigger; the realizable trigger is fp32 overflow, which needs extreme magnitudes and matches the finding's own P3 severity. The defect itself - no sub-unity scale possible, permanent silent stall - is real.

### `python/tessera/rng.py:53` — repr-based hashing forks streams on numpy scalar inputs

*RNG & quantization · logic*

**What is wrong:** _hash_to_u64 hashes repr(p), so fold_in(np.int64(3)) hashes the string "np.int64(3)" (numpy >= 2) while fold_in(3) hashes "3" - equal values silently produce different streams, and the same numpy-scalar input hashes differently across numpy 1.x ("3") and 2.x ("np.int64(3)"), undermining checkpoint replay of fold_in(epoch)/fold_in(rank) chains where the value came from a numpy computation.

**Evidence:** RNGKey.from_seed(0).fold_in(np.int64(3)) != RNGKey.from_seed(0).fold_in(3): blake2b inputs differ (b"np.int64(3)" vs b"3"), yielding different seed_low and a fully divergent sample stream with no warning.

**Fix:** Normalize supported types before hashing (int(p) for integral types including np.integer, explicit encodings for str/bytes) and raise on anything else instead of falling back to repr.

**Independently verified:** Reproduced on numpy 2.5.2: repr(np.int64(3)) == "np.int64(3)", and RNGKey.from_seed(0).fold_in(np.int64(3)).seed_low == 5425579255823077476 while fold_in(3).seed_low == 15111861101556577111 - equal values, fully divergent streams, no warning. _hash_to_u64 (rng.py:49-55) hashes repr(p) with no normalization, and fold_in (122-136) passes `data` through unmodified. The `data: int | str | bytes` annotation is not enforced at runtime and isinstance(np.int64(3), int) is False, so nothing rejects the input. This also contradicts the function's own docstring claim of a "machine-independent hash" and the module's determinism/replay contract: the same numpy scalar hashes as "3" under numpy 1.x and "np.int64(3)" under 2.x, so a checkpoint replaying fold_in(epoch)/fold_in(rank) where the value came from a numpy computation is not reproducible across numpy major versions. Existing tests (test_rng_keys.py:110-124, 266-289) only pass Python ints and strs, so nothing pins the current behavior as intentional. Correctly rated P3 - it needs a numpy-typed argument to trigger - but the defect is real and the proposed normalize-or-raise fix is the right shape.

### `python/tessera/nn/functional.py:185` — conv1d/conv_transpose use six-deep Python scalar loops

*Losses, optimizers, RL, nn.functional · performance*

**What is wrong:** conv1d accumulates every output scalar with a Python loop nest over (batch, group, out-channel, position, in-channel, kernel), ~N*C_out*L_out*C_in/g*K interpreter-level float ops per call; conv_transpose (line 223) has the same structure. Both are fully vectorizable with numpy (sliding_window_view + einsum) for a 100-1000x wall-clock win at identical reference semantics.

**Evidence:** A modest N=8, C_in=C_out=64, L=1024, K=3 call executes ~1.6e8 Python-level multiply-adds — tens of seconds to minutes per forward — versus milliseconds vectorized; every test or model using the functional conv surface pays this.

**Fix:** Build windows with np.lib.stride_tricks.sliding_window_view(padded, kernel_span, axis=-1)[..., ::stride, ::dilation] and contract with np.einsum('ngilk,goik->ngol', ...), keeping the loop version only as an inner comment-documented oracle if desired.

**Independently verified:** functional.py:185-196 is a six-deep Python scalar loop (batch, group, out-channel, position, in-channel, kernel) with a scalar `acc +=`; conv_transpose at 219-230 has the same structure (five loops with an inner vector slice). Measured actual wall-clock for F.conv1d: (2,16,16,L=64,K=3) 0.024 s, (4,32,32,128,3) 0.391 s, (8,64,64,256,3) 6.29 s — extrapolating the cited (8,64,64,1024,3) gives ~25 s, matching the finding's 'tens of seconds'. I implemented the proposed fix (sliding_window_view(padded, span, axis=-1)[..., ::stride, ::dilation], reshape to groups, einsum 'ngilk,goik->ngol') and verified it against F.conv1d on three configs covering stride=2/padding=2/dilation=2/groups=2 and stride=3/groups=3: max abs diff 9.5e-07 (float32 accumulation-order noise), and 6.29 s -> 0.044 s, a 144x speedup. Not a cold path: F.conv1d backs nn.layers.Conv1d.__call__ (layers.py:213), F.conv_transpose backs layers.py:260, and runtime.py:26064-26067 dispatches 'tessera.conv1d'/'tessera.conv_transpose' to them. Note the proposed fix must keep the existing shape/group validation at lines 170-181, which the loop currently performs before entering the nest.

### `python/tessera/compiler/fusion_core.py:596` — Pointwise exit scan is quadratic in graph size

*Fusion core & emitter seams · algorithm*

**What is wrong:** discover_pointwise_graph computes region exits by calling _consumers(ops, value) (line 611-612: a full linear scan of every op's inputs) once per member of every candidate component, making the exit computation O(candidate_ops * total_ops * avg_arity). Every other discoverer in this file builds a by_input index once and does O(1) lookups; this one does not, and it runs in the runtime fusion prepass on each graph execution (runtime.py:33863).

**Evidence:** For a traced model graph with ~5,000 ops of which ~2,000 are pointwise candidates spread over many components, the exit scan performs on the order of 2,000 * 5,000 * 2 = 20M membership probes per execution prepass, versus ~10K operations with a precomputed use map — pure Python overhead of tens of milliseconds repeated per dispatch, dwarfing the discovery it serves.

**Fix:** Build the by_input consumer index once (as discover_fusable_regions already does at lines 1221-1226) and replace _consumers with a dict lookup.

**Independently verified:** Algorithm claim verified. fusion_core.py:595-597 calls _consumers(ops, ops[i].output) once per member, and _consumers (611-612) is a full linear scan over every op's inputs - no index. Total work is O(sum(members) * len(ops) * arity) = O(candidates * ops) per prepass; the `len(members) > _PW_MAX_INPUTS*2` (32) guard at line 581 bounds a single component but not the sum over components, so the worst case stands. The contrast is accurate: discover_fusable_regions builds by_input at 1221-1226 and discover_attention_regions at 1424-1427, both O(1) lookups. It is on a per-execution path: runtime.py:34164 calls _apple_gpu_try_synthesized_fusion on every Apple GPU artifact invocation, reaching discover_pointwise_graph at 33863. The proposed fix (reuse the by_input map) is exactly equivalent to _consumers. Magnitude scales with real artifact op-count, which I did not measure, so P3 is the right severity - the asymptotic defect and the fix are both valid.

### `python/tessera/compiler/emit/nvidia_cuda.py:626` — Flash-backward entries leak device memory on alloc failure

*Emitter — NVIDIA CUDA · logic*

**What is wrong:** TSR_ATOMIC_ENTRY (line 626) returns 2 from its cudaMalloc chain and from the 'if(hb&&cudaMalloc(&bi,nb))return 2;' guard without freeing the up-to-7 buffers already allocated; the f16 backward wrapper (lines 659-660) has the same pattern across 14 device allocations plus its memcpy-failure 'return 3'. In the long-lived arbiter/autotuner process, each failed attempt under memory pressure permanently strands hundreds of MB (nq+nk+no+nv can be GB-scale for Sq=Sk=4k), making subsequent OOM more likely — the exact leak class PR #297 fixed for the pointwise wrapper (line 993 comment) but that fix was not applied here.

**Evidence:** run_flash_attention_backward on a device near capacity: cudaMalloc(&dv,nv) fails after go/q/k/v/dq/dk succeeded; the entry returns 2, the Python wrapper raises, the caller retries with a smaller batch — but the ~6 stranded buffers are never freed (the .so's pointers are locals), so retries that should fit now also fail. The split/timed siblings in the same source use goto fail cleanup correctly, showing the intended pattern.

**Fix:** Route the atomic entry's and the f16 wrapper's allocation/memcpy failures through the same goto-fail cleanup block their _timed and SPLIT siblings already use.

**Independently verified:** Verified verbatim. TSR_ATOMIC_ENTRY (line 626) does `if(cudaMalloc(&go,no)||cudaMalloc(&q,nq)||cudaMalloc(&k,nk)||cudaMalloc(&v,nv)||cudaMalloc(&dq,nq)||cudaMalloc(&dk,nk)||cudaMalloc(&dv,nv))return 2;` followed by `if(hb&&cudaMalloc(&bi,nb))return 2;` — both early returns strand every allocation that already succeeded (up to 7 buffers, nq+nk+no+nv is GB-scale at Sq=Sk=4k), since the pointers are locals in the loaded .so and the CUDA context outlives the call in the long-lived arbiter process. The f16 wrapper (lines 659-660) repeats it across 14 cudaMallocs plus the bias alloc, and its memcpy-failure `return 3` leaks all of them. The contrast the finding draws is accurate: TSR_ATOMIC_ENTRY_timed (627), TSR_SPLIT_ENTRY (629) and TSR_SPLIT_ENTRY_timed (631) all route failures through a `goto fail` block that frees every pointer, and the pointwise wrapper carries an explicit comment at lines 993-995 citing the PR #297 review for exactly this cleanup pattern — so the intended convention exists in this file and these two entries are the ones that skip it. P3 severity is right (only reachable under allocation/memcpy failure).

### `python/tessera/compiler/emit/apple_msl.py:1327` — bf16 pointwise DAG silently returns float32

*Emitter — Apple MSL / ROCm HIP / x86 · logic*

**What is wrong:** pointwise_operand_plan classifies element type as only f16-or-f32 (`elem = "f16" if in_dtype == np.float16 else "f32"`), so a bf16 input to run_pointwise_graph/run_pointwise_reduce is computed in f32 and returned as float32 — the storage dtype is not round-tripped, unlike every other lane in this module (run_fused_region, run_gated_matmul_region, run_norm_chain_region all convert bf16 back via `.astype(bf16)`).

**Evidence:** run_fused_region lines 989-1000 explicitly emulate bf16 in f32 and cast back (`return out32.astype(bf16), ex32`); run_pointwise_graph line 1373/1402 returns `...astype(npdt)` where npdt is float32 for bf16 input. A bf16 tensor chained through a fused pointwise DAG therefore comes back float32 (2x memory, dtype mismatch for any consumer that checks), while the same tensor through a fused matmul epilogue comes back bf16 — an inconsistent dtype contract across ops in one backend.

**Fix:** Detect ml_dtypes.bfloat16 in pointwise_operand_plan (as _bf16_dtype() does elsewhere), compute in f32, and cast the returned array back to bf16 to match the module's other bf16 lanes.

**Independently verified:** Verified in source and by execution. pointwise_operand_plan (apple_msl.py ~1325-1330) does `elem = 'f16' if in_dtype == np.float16 else 'f32'` with `npdt` following, and I ran it with an ml_dtypes.bfloat16 input: it returns elem='f32', npdt=numpy.float32, arrays cast to float32 — so run_pointwise_graph's returns (`out` allocated npdt, and both reference fallbacks `.astype(npdt)`) hand back float32 for a bf16 input. run_pointwise_reduce (1478-1503) shares the identical classification and the identical consequence. The contrast the finding draws is accurate: run_fused_region f32-emulates bf16 and returns `out32.astype(bf16)` (989-1000), run_gated_matmul_region does the same (1836-1840), and run_norm_chain_region has a native bf16 element type (1160-1172, elem='bf16'). _bf16_dtype() already exists in this module, so the detection helper is right there. Consequence is real, not cosmetic: the runtime graph-fusion orchestrator writes the result straight back into `values[out_v]`, so a bf16 tensor through a fused pointwise DAG silently becomes float32 for every downstream consumer. Low severity (numbers are correct, only the storage dtype contract breaks), matching the reported P3.

### `python/tessera/compiler/emit/nvidia_solver_krylov.py:79` — Krylov matvec is one-thread-per-row, fully uncoalesced

*Emitter — spectral / Krylov / autotune · performance*

**What is wrong:** tsr_matvec assigns each matrix row to a single thread that serially reduces n columns: at any instant, adjacent threads in a warp read a[row*n + col] for consecutive rows, i.e. addresses n*sizeof(T) apart — every global load is a separate memory transaction. The matvec is the O(n^2) dominant cost of every CG/GMRES iteration (called 2-3x per iteration including true-residual checks), so the whole solver runs at a small fraction of achievable DRAM bandwidth.

**Evidence:** For f32 n=4096, a warp issues 32 loads spanning 32*16KB — 32 transactions where a coalesced layout needs 4; effective bandwidth drops ~8-32x on the kernel that is >95% of the solve time. A warp-per-row reduction (warp shuffle over columns) or a blocked shared-memory x with coalesced column-major access is the standard fix and keeps the deterministic reduction order per row.

**Fix:** Assign a warp (or block) per row with a shuffle-based intra-warp reduction over columns; column accesses become coalesced across lanes while the per-row sum order stays fixed and deterministic.

**Independently verified:** Verified the access pattern at nvidia_solver_krylov.py:71-83. `tsr_matvec` is a grid-stride loop over ROWS; each thread walks its whole row (`arow = a + row*n`, inner loop over col), so at a fixed inner iteration adjacent lanes touch addresses n*sizeof(T) apart — provably strided, not coalesced. Host side confirms one-thread-per-row is the actual geometry, not a degenerate case: threads=256 and `useful = ceil(n/256)`, so for n=4096 the launch is 16x256 = exactly n threads and the grid-stride loop makes one pass. Dominance is real: matvec is O(n^2) while every other per-iteration op is O(n), and it is invoked 1-2x per CG iteration (line 111 plus the true-residual check at 139) and again at 171/199/222/301. Fix is valid: warp-per-row with lane l striding columns by 32 makes the column reads coalesced, and a shuffle reduction stays deterministic for a fixed launch geometry — consistent with this file's own stated contract (line 62-63). One correction to the reviewer's magnitude: because each thread reads its row sequentially, a fetched 32B/128B sector is reused on the next 7/31 inner iterations and easily fits in L1 (32 rows x 128B = 4KB), so DRAM byte traffic is NOT 8-32x inflated — the real penalty is memory-request/LSU throughput (32 sectors per warp load instruction instead of 4). Substantial and worth fixing, but likely below the claimed 8-32x bandwidth figure. Cannot be measured here (no CUDA on this host).

### `python/tessera/compiler/emit/spectral_candidates.py:1292` — Hermitian mirror built by interpreted per-bin loop

*Emitter — spectral / Krylov / autotune · performance*

**What is wrong:** IRFFTCandidate.run reconstructs the full spectrum with a Python for-loop assigning one conjugated bin per iteration (O(n) interpreter dispatches plus a scalar np.conj object round-trip per bin). This sits on the ISTFT hot path — it runs once per frame — and is replaceable by one vectorized slice assignment.

**Evidence:** For win=4096 and 1000 frames, ISTFTCandidate executes ~2M interpreted loop iterations (~hundreds of ms of pure Python) versus a single full[bins:] = np.conj(half[1:n-bins+1])[::-1] per frame (~microseconds). The loop cost can exceed the native FFT time the composed lane exists to exploit.

**Fix:** Replace the loop with the vectorized mirror: full[region.bins:] = np.conj(half[1:n - region.bins + 1])[::-1] (handles both even n, where Nyquist sits at bins-1 and is not mirrored, and odd n).

**Independently verified:** Traced IRFFTCandidate.run:1290-1293 — the Hermitian mirror is an interpreted `for k_idx in range(1,(n+1)//2)` with a scalar `np.conj` per bin, and ISTFTCandidate.run calls it once per frame (line 1355). Independently validated the proposed vectorized replacement `full[bins:] = np.conj(half[1:n-bins+1])[::-1]`: I re-derived the index arithmetic for both parities (even n: LHS n/2+1..n-1, RHS half[1..n/2-1] reversed, Nyquist at bins-1 correctly not mirrored; odd n: LHS bins..n-1, RHS half[1..(n-1)/2] reversed) and confirmed bit-exact equality against the loop for n = 63, 64, 127, 128, 4096, 4097. Measured at n=4096: loop 0.407 ms/call vs vectorized 0.0028 ms/call (146x), and the loop is 10x the cost of the entire np.fft.ifft (0.039 ms) it feeds — so on the ISTFT path the interpreted mirror dominates the transform it exists to accelerate. For 1000 frames that is ~0.4 s of pure Python, matching the finding's estimate.

### `src/transforms/lib/TesseraToLinalgPass.cpp:2037` — SGD backward materializes redundant identity-copy tensor

*C++ — Tessera→linalg lowering · performance*

**What is wrong:** SGDBackwardLowering builds paramGrad as a full linalg.generic whose body just yields the incoming cotangent unchanged (fn returns its argument), materializing a fresh tensor.empty and a whole-tensor copy loop for a value that is bit-identical to operand 0. StopGradientLowering (line 2951) shows the cheap alternative: forward the SSA value directly in replaceOp. Unless a later fusion/copy-elision pass happens to fold it, this is one extra O(n) read+write of a parameter-sized tensor per parameter per optimizer-backward step — pure memory traffic with zero computation, on models where parameter tensors are the largest objects in the program.

**Evidence:** For a 4096x4096 f32 parameter, the emitted generic writes 64 MiB and reads 64 MiB per sgd_backward solely to reproduce its input; a training graph differentiating through k SGD steps emits k such copies.

**Fix:** Replace the identity generic with the operand itself: rewriter.replaceOp(op, ValueRange{op->getOperand(0), gradGrad}).

**Independently verified:** The identity generic is real and it survives the production pipeline. Line 2037-2039 builds paramGrad via emitUnaryElementwise with `[](OpBuilder&, Location, Value c){ return c; }` — verified in the emitted IR: `linalg.generic ... ins(%arg0) outs(%0) { ^bb0(%in, %out): linalg.yield %in }`. I then checked whether anything folds it. `--canonicalize` DOES fold it (upstream EraseIdentityLinalgOp; the pass output becomes `return %arg0`), but the production JIT does not run canonicalize after this pass: tools/tessera-jit/tessera_jit.cpp:668-676 runs Canonicalizer+CSE BEFORE `createTesseraToLinalgPass`, then goes straight to ConvertElementwiseToLinalg → EmptyTensorToAllocTensor → one-shot-bufferize, and the tessera-opt emit spine (tools/tessera-opt/tessera-opt.cpp:250) has no canonicalize either. Running the real pipeline shape confirms the cost lands: after one-shot-bufferize the identity becomes `memref.alloc()` + a full identity `linalg.generic` copy loop. The proposed fix is valid — result type is already checked equal to the operand type at line 2031, and StopGradientLowering (2951) does exactly `rewriter.replaceOp(op, op->getOperand(0))`. P3 sizing is fair (sgd_backward only arises when differentiating through an optimizer step), but the cost and the fix are both real.

### `src/solvers/spectral/lib/TargetHooks/CPU/StockhamRadix4.cpp:137` — Per-j heap allocation in generic radix stage

*C++ — spectral / Clifford / TPP solvers · performance*

**What is wrong:** ts_stockham_rn_cpu constructs `std::vector<cf> stage(r)` inside the j-loop, performing one heap allocation+free per j (L iterations) per stage, while the scratch vector `c` is correctly hoisted just below it.

**Evidence:** The odd-prime stage runs LAST in the plan order (4s, then 2, then odd primes), so L is maximal when it executes: for N = 3*2^20 the radix-3 stage runs with L = 2^20, m = 1 — about one million malloc/free pairs (~20-50ns each, tens of ms) against a stage doing only ~9M flops (~few ms). The allocation dominates the stage cost on any large N with a residual odd factor.

**Fix:** Hoist `stage` above the j-loop (it is overwritten fully each iteration), like `c` already is.

**Independently verified:** StockhamRadix4.cpp:135-142: `std::vector<cf> stage(r)` is declared inside the `for (int j = 0; j < L; ++j)` loop, one heap allocation+free per j, while `std::vector<cf> c(r)` at line 134 is correctly hoisted directly above it. The cost claim checks out: FFTPlan.h fft_plan drains 4s, then 2s, then odd primes ascending (lines ~103-113), so the generic radix-r stage always runs LAST, where L = N/r is maximal and m = N/(r*L) = 1. For N = 3*2^20 that is L = 1,048,576 malloc/free pairs against a stage doing only m*L*(r^2+r) ~ 1.2e7 complex ops — allocation dominates. The suggested fix is valid: `stage` is fully overwritten every iteration by the q-loop at 138-142 (no carry across j), and stage[0] is never even read (line 146 special-cases q==0), so hoisting it above the j-loop is semantics-preserving. Cold-path objection does not hold — this is the shipped execution kernel the correctness sentinel and the tpp/spectral CPU candidate dlopen and run.


## Refuted / uncertain (19) — reported by a reviewer, did not survive verification

- **REFUTED** `jvp.py:4026` (P1) — int8/int4 quantize JVP primal contradicts canonical forward. The cited failure path does not exist. `quantize_int8`/`quantize_int4`/`dequantize_int8` are NOT in `ops.registry._entries` (verified: 385 entries, none of these), so `install_op_wrappers` (tape.py:1017-1043, which wraps only `tessera.ops.*`) never wraps them and `record_op` never dispatches them — the JVP's primal_out can never replace the forward's. Verified directly: under `jvp(...)`, `q,s,zp = ts.quantize_int8(x)` unpacks fine and `ts.dequantize_int8(q,s,zp)` returns the correctly scaled values (array([0.992, 2.008, 3.0])), i.e. the real forward ran untouched. Separately, the primal deviation is an explicitly declared convention, not an oversight: law_inputs.py:390-397 marks the quantize family `chain=False` with the note 'an STE primal is deliberately NOT the canonical forward's output (the forward returns a (q, scale, zero_point) tuple), so the chain law's primal anchor does not apply — this is a declared convention, not a defect', and `vjp_dequantize_int8` (vjp.py:5653-5658) declares the matching STE cotangent pass-through, so the JVP/VJP pair is adjoint-consistent (`dequantize_int8 adjoint pass`). The only true residue is a stale docstring on jvp_quantize_int8 ('Returns the (q_int8, scale) tuple').

- **REFUTED** `jvp.py:3149` (P1) — bidirectional_scan JVP silently replaces primal with zeros. Unreachable as described. There is no `ops.bidirectional_scan` — `hasattr(ts.ops, 'bidirectional_scan')` is False and the name is absent from `ops.registry._entries`, so the literal scenario `y = ops.bidirectional_scan(...)` raises AttributeError rather than producing a garbage primal. The public spelling is `tessera.nn.functional.bidirectional_scan`, which is not tape/JVP-wrapped; verified under a live trace that `jvp(f, (xs,), (ones,))` over `F.bidirectional_scan` returns the correct (3,2) primal and a (3,2) tangent — the rule is never invoked. The rule also has no law spec (`bidirectional_scan adjoint no_spec`), so nothing else calls it either. The `np.zeros(1), np.zeros(1)` body is dead placeholder code (arguably a Decision #29 'declaration with no consumer' issue), not a silent-wrong-primal defect on any live path.

- **REFUTED** `jvp.py:2641` (P2) — logsumexp axis=None primal keeps full rank; no -inf guard. jvp_logsumexp is no longer the production rule. `jet.register_jet_derived_structured_rules` (jet.py:539-570, STRUCTURED_RETIREES = softmax/logsumexp/rmsnorm) runs at import and replaces the registry entry: `get_jvp('logsumexp')` returns `jvp_logsumexp__jet`, and the hand rule is parked in `RETIRED_HAND_RULES` as a #31 oracle. Verified on the live trace: `jvp(lambda a: ts.ops.logsumexp(a), (x,), (ones,))` returns a 0-d primal, exactly matching the untraced `ops.logsumexp` — so the claimed 'traced result's shape disagrees with the untraced forward' is false. Residual (not the reported defect, and much lower severity): the retired oracle does return (1,1) for axis=None while production returns (), and test_retired_structured.py's envelope includes {'axis': None} but compares with `np.testing.assert_allclose`, which broadcasts () against (1,1) and so cannot see it; the missing -inf clamp likewise only affects the oracle.

- **REFUTED** `jvp.py:1694` (P2) — max_pool tangent gather is a per-pixel Python double loop. The loop is real and the proposed rewrite is valid (I implemented `np.take_along_axis(dx_flat, argmax[...,None], axis=-1)[...,0]` and it matches the current output bit-for-bit), but the claimed cost is not realized anywhere. `max_pool` is not a registered op (`hasattr(ts.ops,'max_pool')` is False; absent from `ops.registry._entries`), so it is never wrapped and neither `jvp` nor `jacfwd` can reach the rule — the stated 'under jacfwd this multiplies by the input size' amplification is impossible. The only in-repo caller is the law sweep, whose spec is `law_inputs.py:1081` = shape (1,2,4,4), kernel 2, i.e. 8 inner iterations: measured 100 calls in 2.0 ms (~20 µs/call). At the reviewer's hypothetical (8,64,56,56) the call costs 0.12 s, but nothing in the tree calls it at that size. Valid cleanup; the asserted performance defect does not stand.

- **UNCERTAIN** `fusion_core.py:1042` (P1) — F4 verdict cache keys omit semantic region fields. The key omissions are factually true (FusedRegion carries storage_dtype/a_layout/b_layout, AttentionRegion carries q_transposed/k_transposed/storage_dtype, none appear in the keys at 1042/1111/807), and I did demonstrate one real skipped probe (the attention warm-cache experiment in finding 1). But both concrete harm scenarios are blocked: (a) every candidate probe goes through verify_candidate, which passes force=True (candidate.py:327), so the cache is never *read* on the candidate-adapter path - the 'fp8 reuses the f32 verdict' sequence cannot occur; (b) the only production cache-reading callers are the Apple prepass (runtime.py:33771/33846/33871) whose AppleMSLRunner never mentions storage_dtype, a_layout or b_layout (grep of apple_msl.py returns nothing) - it derives dtype from the array's dtype - so those fields select nothing there. For attention, the transposed variant runs the *same* kernel source (orientation is applied host-side by _natural), so the skipped probe admits no different kernel. Real latent hazard / Decision #29 gap, but I could not trace a path where a semantically different kernel passes ungated.

- **UNCERTAIN** `candidate.py:327` (P1) — verify_candidate re-probes every candidate on every dispatch. The mechanism is exactly as described and independently confirmed: verify_candidate always passes force=True (candidate.py:327), arbitrate re-gates every applicable candidate per call (361-362), and run_arbitrated calls arbitrate on every execution (444) - even a corpus/force hit still re-verifies the single forced candidate. The reviewer's explanation of *why* force=True is needed (adapter.last_execution would be None on a cache hit, wrongly passing the line-332 ran-a-real-kernel check) is correct. What I could not confirm is the claimed live cost: the only production caller of run_arbitrated is runtime.py:30737 (NVIDIA matmul_relu), which passes verify=False; the other verify=True path, autotune.measured_arbitrate:435, returns early on a MeasureCache hit (line 414) so its probes are amortized per (device,target,op,bucket,dtype). Every remaining caller is a unit test. So the per-dispatch probe overhead is a property of the default API but is not paid on any shipped dispatch path today; settling the impact needs a caller that takes the default.

- **REFUTED** `autotune.py:373` (P3) — corpus_winner scans entire cache per dispatch. The structural observation is true (autotune.py:373-381 linearly scans `cache._store.items()` where a keyed lookup would do when dtype is known), but the finding's cost premise is false at real sizes. The committed fleet corpus at benchmarks/baselines/autotune_corpus.json holds 87 records, not 'thousands' — I loaded it and measured `corpus_winner` at 4.5 us/call, versus 0.048 us for a raw dict get. That is negligible, and it is dwarfed by what corpus_winner does immediately afterwards anyway (lines 386-389 iterate `candidates_for(target,op)` calling `applies_to`+`available()`, and for the ROCm FFT lane `available()` is a device round-trip — see finding index 2). The hot path is also much narrower than claimed: `corpus_winner` returns None at line 364-365 whenever `_infer_dims` yields None, which it does for every op except matmul/fused_region/attention with 2-D operands, so it never scans for spectral/TPP/most dispatches. I did reproduce the scaling concern by synthetically padding the store to 5000 rows (129 us/call), so the fix is a legitimate future-proofing nit — but the claimed present-day cost ('~ms scale in aggregate', 'thousands of rows') does not exist. Separately, the proposed O(1) fix only covers the dtype-known case; `run_arbitrated` passes dtype=None by default and corpus_winner falls back to `getattr(region,'dtype',None)`, so the wildcard path (which needs the secondary index) is common, not rare.

- **REFUTED** `TesseraToLinalgPass.cpp:1266` (P1) — Unchecked attribute derefs crash on missing attrs. No cited attribute can be null when the pattern runs. Every one is either (a) ODS-REQUIRED — `F64Attr:$lr` on tessera.sgd/sgd_backward/momentum/momentum_backward and `Tessera_RegressionBackwardKindAttr:$kind` (a StringBasedAttr, not an EnumAttr class) on loss.regression_backward — so absence is a parse/verify error and the pass never sees the op; or (b) `DefaultValuedAttr<...>`, whose default is materialized into the op's properties by the ODS-generated `populateDefaultProperties` (called by both the parser and `Operation::create`), and `Operation::getAttr` consults inherent/property attributes, so `getAttrOfType` returns a valid attribute. Empirically reproduced the reviewer's own evidence: `"tessera.loss.binary_cross_entropy"(%x,%t)` with NO reduction attr lowers cleanly through `--tessera-to-linalg` (emits the mean 0.25 reciprocal, exit 0), no crash; likewise loss.huber with no delta and loss.regression_backward with only `kind`. `"tessera.sgd"` with no lr fails at parse: "'tessera.sgd' op requires attribute 'lr'". `tessera.adam_backward` with no attrs prints `<{adamw = false, ..., step = 1 : i64}>` — the defaults are physically present. No segfault exists on any cited line.

- **REFUTED** `TesseraToLinalgPass.cpp:367` (P2) — broadcast_dimensions unvalidated: OOB index and dim range. The guard exists one level up, in the op's own verifier, and I reproduced both of the reviewer's evidence cases as clean parse-time errors. BroadcastInDimOp::verify (src/compiler/ir/TesseraOps.cpp:3229-3269) checks exactly what the finding says is unvalidated: `dimensions.size() != inTy.getRank()` → "broadcast_dimensions length must equal input rank"; each entry `dyn_cast<IntegerAttr>` non-null; and `outputDim <= previous || outputDim >= outTy.getRank()` with previous starting at -1 → strictly increasing AND in [0, outRank), which also excludes negatives. Ran it: `broadcast_dimensions = [0]` on tensor<4x5xf32>→tensor<4x5x6xf32> errors with the length message; `[0, 7]` errors with the range message. Neither reaches the pattern. The "matches by op-name string so no verifier screened it" argument is wrong: `--allow-unregistered-dialect` only relaxes UNregistered ops, and tessera.broadcast_in_dim is registered wherever this pass is available, so the parser verifies it (and the pass manager re-verifies after every pass).

- **REFUTED** `TesseraToLinalgPass.cpp:90` (P2) — Integer div silently assumes signed semantics. The claimed silent-wrong-value scenario is not expressible in a legal Tessera program. Canonical integer dtypes are signed only — python/tessera/dtype.py `_CANONICAL_INTS = {int4,int8,int16,int32,int64}`; uint8/16/32/64 live in `_PLANNED_GATED_DTYPES` and `canonicalize_dtype` rejects them unless `allow_planned_gated=True` (Decision #15a). So a signless `iN` element type in Graph IR means signed intN, and DivSIOp is the correct lowering for every first-class integer dtype. For a self-describing unsigned type the failure is loud, not silent: I ran `tessera.div` on tensor<4xui32> and got a hard verifier error ("'arith.divsi' op operand #0 must be signless-non-zero-bitwidth-integer-like, but got 'ui32'"), never a wrong result; `tensor<4xi32>` emits `arith.divsi` as intended. The observation that BinaryComparisonLowering fails closed on signless integers is a fair consistency point (that lane genuinely supports unsigned data via the hand-written x86 kernel path), but the reported defect — an unsigned-intent division silently computed as signed — has no reachable trigger.

- **REFUTED** `TesseraToLinalgPass.cpp:2546` (P3) — FuncOp mutated in-place inside patterns without rewriter. The cited failure mode cannot occur here, for two independent reasons I checked against upstream MLIR source (mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp). (1) `ExpensiveChecks::notifyRewriteFailure` (the "pattern failure must not modify the IR" abort) only runs when a pattern returns failure; in all three cited patterns the raw setAttr sits in the AccumRealizability::Widen branch and every path after it reaches `rewriter.replaceOp(...); return success()` — no failure follows the mutation. (2) `notifyRewriteSuccess`'s per-op "operation finger print changed" check explicitly skips the top-level op, and this pass is `OperationPass<func::FuncOp>` calling `applyPatternsGreedily(getOperation(), ...)`, so the mutated func::FuncOp IS the top-level op and is never fingerprint-compared on success. (Even if it were nested under a module, `notifyOperationInserted` calls `invalidateFingerPrint(op->getParentOp())`, and each of these patterns inserts ops directly into the func's block right after the setAttr, erasing the func's fingerprint before the check.) Driver bookkeeping is likewise unaffected: no pattern in this pass matches func.func, so a missed worklist re-visit costs nothing, and the pass reads `tessera.numeric_policy.consumed` in its own post-driver walk. Also inaccurate: MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS is a separate CMake option, off by default even in assertions builds, so rebuilding an assertions toolchain would not surface this. Using `rewriter.modifyOpInPlace` remains better hygiene, but no defect stands.

- **REFUTED** `TileToX86Pass.cpp:837` (P1) — norm_kernel kind unvalidated: crash or silent layernorm. The dialect verifier makes both claimed inputs impossible, and I confirmed it by execution. NormKernelOp::verify (src/compiler/ir/TileOps.cpp:1902-1926) requires exactly 5 inputs with types (ptr,ptr,i64,i64,f32) and `kind` non-null and in {rmsnorm, layernorm}; ArgReduceKernelOp::verify (:1869-1884) and ScanKernelOp::verify (:1886-1899) likewise require a non-null kind in their enumerated sets plus operand-count/type checks via verifyPointerAndI64Tail. Running tessera-opt with no passes at all: `kind = "groupnorm"` → "error: 'tile.norm_kernel' op requires kind=rmsnorm|layernorm"; kind attribute omitted → the same error. Since tile is a registered dialect, an op spelled tile.norm_kernel is always parsed as NormKernelOp and verified (even under --allow-unregistered-dialect, which only affects unknown op names), so `kind.getValue()` at line 837 can never see a null attr and the StringSwitch at 812/816 can never see an unenumerated value. The operand-index reads are likewise pre-validated. No crash, no defaulted semantic key.

- **REFUTED** `TilingPass.cpp:204` (P2) — TileMatmul never validates result shape against M,N. The claimed input is rejected before it can reach the pattern. MatmulOp::verify (src/compiler/ir/TesseraOps.cpp:179-192) explicitly enforces result rows == lhs M and result cols == rhs N (honoring transposeA/B), and its comment names this exact scenario: 'a malformed (4x8)@(8x16)->(5x5) would pass rank+K and could be lowered to an executable value call that silently produces a wrong-shaped output.' I ran the reviewer's literal input through tessera-opt --tessera-tiling: "error: 'tessera.matmul' op result column dimension must equal rhs N (16 vs 4)" — rejected at parse/verify, so TileMatmul is never reached. tessera.matmul is a registered op, so the 'unverified/unregistered producer' escape does not exist, and TileMatmul's own inner matmuls are constructed with consistent {tm,tn} types. The TileMatmulValue re-check the finding cites as proof is documented at TilingPass.cpp:499-500 as deliberate belt-and-braces: 'The registered MatmulOp verifier also enforces this; we re-check so the value tile op is never created for a shape-inconsistent matmul.'

- **REFUTED** `TilingPass.cpp:101` (P3) — hasOptionalOperand tautology defeats segment check. The tautology is real but it is not a defect, and the proposed fix is a regression. tessera.ebm.langevin_step (TesseraOps.td:726-739) declares `Optional<TensorType>:$noise` with a single optional operand and no AttrSizedOperandSegments trait, so it never carries operandSegmentSizes — I verified this by round-tripping the op through tessera-opt (no segment attr printed). Therefore hasOperandSegment(op,2) is always false for the sole caller, and the operand count is the correct and only available signal: with grad non-variadic, exactly 3 operands means noise is present, and the claimed ambiguous [1,2,0] segment shape cannot exist for this op. Applying the recommended fix (dropping the `op->getNumOperands() > index` disjunct) would make hasOptionalOperand return false for every langevin_step, killing the lowering — I confirmed it currently fires: `tessera-opt --tessera-tiling=value-mode=true` rewrites the 3-operand op to `tile.ebm_langevin_step ... has_noise = true`. The predicate is redundant code worth simplifying, not a guard that fails to verify something reachable.

- **REFUTED** `ActivationRematerializationPass.cpp:559` (P1) — Remat purity gate ignores registered Random semantic effect. The claimed input class — an op that is MLIR-memory-effect-free but semantically Random — does not exist in the Tessera dialect, and the dialect encodes exactly this distinction on purpose. src/compiler/ir/TesseraOps.td: `Tessera_DropoutOp` (line 2343) is `MemoryEffects<[MemWrite]>`; the implicit-stream `Tessera_StatefulRNGOp` class (rng_uniform/rng_normal, line 2413) deliberately carries NO Pure trait and no MemoryEffectOpInterface (comment: "they therefore carry no Pure trait and remain conservatively side-effecting so they are never CSE'd, cloned, or rematerialized"), and `mlir::isMemoryEffectFree` returns false for an op with neither the interface nor recursive effects — so gate 2 rejects both. I ran both through the pass: each emits `REMAT_EFFECTFUL` (verified with tessera-opt on tessera.dropout and tessera.rng_uniform tagged tessera.recompute + tessera.effect_kind="random"). The only Pure ops with a random flavour are the explicit-key Philox ops (line 2370) and spec_accept_*_sample (line 3239/3277), whose randomness is an SSA operand — the ODS comment states they "are deterministic functions of their integer key/counter operands and may therefore be cloned or rematerialized without changing sample identity", and op_catalog.py registers them effect="pure". The proposed fix would actively break that contract: those ops carry tessera.stochastic_identity="key_counter" (!= "none"), so rejecting on stochastic_identity would refuse the one RNG form that is provably remat-safe. GraphDataflow's hasStochasticIdentity is stricter for a different reason (reordering waits across a sample identity), not because remat of a keyed sample is unsound.

- **REFUTED** `DtypeLegalizePass.cpp:314` (P2) — StoragePackConsume silently skips malformed packing markers. Two disproofs. (1) The finding misreads the file's line-27 statement: it says "never silently skip — an op with a reduced-precision storage and no recognizable policy is left untouched for IRContractLegalityPass to flag, not quietly 'legalized'". Leaving an unrecognizable marker untouched for a downstream contract gate IS the stated design; what is forbidden is fabricating a descriptor. (2) The claimed harm ("backend packed load/store later finds no #tile.packed_format and misbehaves or fails far from the root cause") does not occur: the consumers fail closed on the same op. src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp:280-293 `requireStoragePackDescriptor` errors with "NVIDIA packed matmul requires terminal storage legalization with an int8 tessera.storage_pack descriptor" whenever packed/container/descriptor is missing, and the ROCm suite has a matching storage_pack_reject_tile_lowering fixture. Decision #21a is about never silently defaulting a semantic attribute; this pass defaults nothing. I did find one in-tree way to manufacture the malformed marker (LayoutAssignmentPass::propagatePhysicalStorage, line 121-141, resolves the three attrs independently, so conflicting containers can drop storage_container while storage_packed still propagates) — but even then the outcome is a loud consumer-side error on the same op, so this is at most a diagnostic-locality nit, not the fail-open the finding asserts.

- **REFUTED** `LowerControlFlowToSCFPass.cpp:402` (P2) — Loop bound attributes lowered without positivity validation. Both cited failure paths are blocked by the op verifiers, which run on the parsed module before the pass. src/compiler/ir/TesseraOps.cpp:4717 `ControlWhileOp::verify()` begins `if (getMaxItersAttr().getInt() <= 0) return emitOpError("max_iters must be positive")`, and 4666 `ControlForOp::verify()` begins `if (getStep() == 0) return emitOpError("step must be non-zero")`. Confirmed with tessera-opt: `max_iters = -1` is rejected with `'tessera.control_while' op max_iters must be positive` and `step = 0` with `'tessera.control_for' op step must be non-zero` — both name the source op and the reason, contrary to the claim of a generic downstream scf error, and the ~2^64-iteration unbounded-loop scenario is unreachable. One residual the finding did not claim: a NEGATIVE step (step = -2 with start<stop) passes both the Tessera verifier and MLIR's scf.for verifier and lowers silently to `scf.for %arg1 = %c0 to %c10 step %c-2` (exit 0). That is a real, separate gap worth a follow-up, but it is not the defect as claimed, and the finding's stated mechanisms and evidence are disproved.

- **REFUTED** `TileBufferArenaPass.cpp:304` (P1) — Dynamic shared arena materialized as per-thread alloca. The finding did not check the downstream consumers. The address-space-3 memref.alloca is a deliberate intermediate form, not the final allocation. src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/ROCMDynamicLDS.cpp opens with the exact premise the reviewer used — 'memref.alloca with memory space 3 lowers to llvm.alloca addrspace(3), which is private stack allocation ... not HIP dynamic LDS' — and then replaces those runtime-sized arenas with slices of one AMDGPU external zero-length workgroup symbol sized by hipModuleLaunchKernel's sharedMemBytes, with liveness-based slot coloring; it names TileBufferArenaPass as its input producer (line 20). It is in the production ROCm pipeline (Passes.cpp:387, after ConvertGpuOpsToROCDL). NVIDIA has the mirror pass, NVIDIADynamicShared.cpp ('nvidia-materialize-dynamic-shared'), registered with its own sm120 fixture. So the proposed fix ('lower to the launch-configured dynamic-LDS mechanism') is the shipped design; the static path's memref.global comment explains why the STATIC arena cannot use alloca (fixed size, statically reserved), which does not apply to a runtime-sized arena that must come from sharedMemBytes.

- **REFUTED** `ExpandProductTable.cpp:87` (P2) — Missing algebra attr silently skips expansion. The claimed input is impossible. CL_GeoProductOp declares `I64ArrayAttr:$algebra` in `let arguments` (src/solvers/clifford/lib/Dialect/Clifford/CliffordOps.td:72-84) — a required, non-Optional ODS attribute — and ts-clifford-opt.cpp:35 registers the dialect, so MLIR's generated verifyInvariants rejects a geo_product with no `algebra` at parse time and after every pass. No producer can create one either: RotorSandwichFold.cpp:73-91 refuses to fold unless all three ops carry `algebra` and copies it onto the op it builds. The composition scenario is additionally blocked: the canonical --tessera-clifford-pipeline runs AnnotateAlgebra first, which emits a hard error and signalPassFailure on a missing or size!=3 `algebra` (AnnotateAlgebra.cpp:74-88), aborting before GradeFusion ever deletes a grade op. The only residual is a hand-written, never-annotated module with a well-typed but wrong-length algebra (e.g. [3,0]) run through expand standalone, where line 87 does skip silently — a narrow ill-formed-IR corner, not the 'producer forgot the attr' path the finding rests on, and not a silent wrong value in any pipeline.
