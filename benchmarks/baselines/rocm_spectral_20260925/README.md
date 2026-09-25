# ROCm spectral STFT/ISTFT: before/after packets — 2026-09-25

Owner `TSOL-POLICY-PHYS-1`; sync `ROCM-SPECTRAL-FFT-POLICY-2026-09-25`. This
closes the follow-up recorded under `NVIDIA-SPECTRAL-DEEPEN-2026-09-25`.

Produced by `benchmarks/spectral/benchmark_rocm_spectral.py`, schema
`tessera.rocm_spectral_benchmark.v1`, on one shape: 8×16000, `n_fft=512`,
`hop=128`, one-sided. Latencies are warm `host_wall_synchronized` medians
through the public `@tessera.jit` entry points. Forward rows are checked
against NumPy before timing; reverse and tangent correctness is covered by
`tests/unit/test_autodiff_spectral_target_binding.py`. WSL2 wall clock; not a
promotion claim. **The two chips are separate proofs and no number transfers
between them.**

| File | Host | Revision |
|---|---|---|
| `gfx1151_before.json` | Princess-Luna (Radeon 8060S, gfx1151) | `0a4d77a9` (main), 3 repeats |
| `gfx1151_after.json` | Princess-Luna | `017a54d8`, 21 repeats |
| `gfx1201_before.json` | Tajasarus (RX 9070 XT, gfx1201) | `0a4d77a9` (main), 3 repeats |
| `gfx1201_after.json` | Tajasarus | `017a54d8`, 21 repeats |

## Result

| Case | gfx1151 before | gfx1151 after | gfx1201 before | gfx1201 after |
|---|---:|---:|---:|---:|
| STFT forward (public) | 1.47 | 1.46 | 1.40 | 1.33 |
| ISTFT forward (public) | 2.00 | 2.05 | 1.85 | 1.83 |
| STFT JVP | 1047.1 | 35.2 | refused | refused |
| STFT VJP | 11059.0 | 15.6 | 2203.7 | 3.96 |
| ISTFT JVP | 18.2 | 16.9 | refused | refused |
| ISTFT VJP | 938.3 | 24.5 | 221.3 | 4.75 |

All values are in ms.

The public forward rows go through a different route and are unchanged. The
composite broadcast-layout STFT used by the JVP went from ~347 ms to ~12 ms per
call on gfx1151. On gfx1201 the native JVP is refused by an existing arch gate
("native ROCm JVP requires exact gfx1151"); that gap is recorded in the ROCm
queue, not changed here.

## Attribution

- **Direct DFTs** (cProfile, native call time vs. kernel math): the
  broadcast-layout STFT/ISTFT, streaming STFT and STFT/ISTFT reverse
  entry points computed an fp64 cos/sin sum over `n_fft` for every output
  element. The STFT reverse input kernel was worse: it summed over every frame,
  position and bin for each sample. All of these now run batched forward C2C
  child plans.
- **Per-call compile** (subprocess trace): every warm reverse call re-ran the
  five-process `tessera-opt` chain to produce a package image, about 160 ms of a
  ~190 ms gfx1151 STFT reverse. The image is now cached by (compiler digest,
  arch, carrier IR).
- **Host staging**: per-element div/mod and one-byte-size `memcpy` calls, now
  bulk copies where the layout allows.
- **Build type**: Princess-Luna's `build/` has an empty `CMAKE_BUILD_TYPE`, so
  the composite library compiles without `-O`. The same sources rebuilt with
  `-O2` and loaded through `TESSERA_ROCM_SPECTRAL_LIB` gave STFT VJP 3.94 ms,
  ISTFT VJP 4.46 ms, ISTFT JVP 4.50 ms and STFT JVP 10.7 ms on gfx1151 (11
  repeats, not a recorded packet). That is the same level as Tajasarus, whose
  tree is `Release`. The remaining gfx1151 gap is that build setting, not this
  code.

## Semantics fix found on the way

`pad_mode` is a centered-framing policy. A non-centered frame that runs past
the signal is zero-filled by `tessera.ops.stft` and the reference VJP. The
former ROCm kernels reflected it. The FFT paths now zero-fill it, and
`test_rocm_fft_policy_envelopes_match_independent_references` fails against
main's library on that forward assertion. CUDA and x86 carry the same
unconditional-reflect pattern; see their queues. They were not run here.

## Validation

At `017a54d8`, the ROCm spectral gate (`test_native_vjp_execution_certificates`,
`test_autodiff_spectral_target_binding`, `test_native_jvp_compiled`,
`test_rocm_spectral_compiled`, `test_scheduled_spectral`,
`test_spectral_{candidates,complex_contract,composed_lanes,inferred_dag,streaming}`,
`test_stft_adjoint_contract`, `test_autodiff_physical_products_evidence`,
`test_backend_capability_extension`) gave:

- **gfx1151:** 362 passed, 3 skipped, 1 failed.
- **gfx1201:** 355 passed, 10 skipped, 1 failed. The extra skips are existing
  "exact gfx1151 required" gates.

On both chips the one failure is the pre-existing
`test_every_declared_rocm_vjp_family_records_an_exact_certificate`. It calls
`test_public_gfx1151_attention_vjp_consumes_prebuilt_program`, which was
renamed in `b59da796` (2026-09-13), and it stops with an `AttributeError`
before reaching its spectral calls. Those spectral tests run directly in the
same gate and pass.

## Correction: `cold_ms` in these packets is a second call (2026-09-25)

Each case was invoked once, untimed, for its correctness check before the
timer recorded `cold_ms`. So `cold_ms` in the JSON files here measured the
**second** call, after compilation, package images and plans were already
populated. Do not read it as cold-start cost. The warm medians (`latency_ms`,
`p10_ms`, `p90_ms`) are unaffected. The benchmark now times the first
invocation as `cold_ms`. These packets are left as recorded rather than
relabelled; a cold figure needs a new measurement. It probes the composite-package `image_arch` only
after every case is timed, so the first case that needs that package pays for
loading it.
