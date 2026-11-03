# Qwen3-8B perplexity with the native quantized MLP in every layer, gfx1151 (stage 2)

Owner ROCM-NVFP4-INGEST-1. Synchronization key QWEN3-NATIVE-MLP-PERPLEXITY-GFX1151-20261102.
Host: Princess-Luna, Radeon 8060S (gfx1151, Strix Halo, WSL2), source
`3a96c891d`. **gfx1151 evidence only.** It neither borrows from nor stands for
the gfx1201 packet (`../qwen3_native_mlp_perplexity_20261101/`).

## Acceptance rule (fixed before measuring)

Committed in `3a96c891d` before this run started: the stage 2 rule, unchanged,
applied on gfx1151. Let Δ(arm) = ppl(arm) − ppl(BF16). The gate passes only
if all of these hold:
- Δ(shipped) > 0;
- Δ(native) ≤ 1.5 × Δ(shipped);
- every native output is finite;
- the staged certificate passes on each layer's first native chunk.

## What was measured

The same harness, model, corpus windows and arms as the gfx1201 packet; the
token-id digest is identical. The native arm is the public seven-member
quantized MLP (`canonical_rocm_quantized_mlp_program`):
- built from this commit's own `tessera-opt` and ROCm runtime;
- compiled for `rocm_gfx1151` from the live device (`quantized_mlp_arch`);
- run in 128-row chunks.

The run held the shared device lock: `flock /tmp/tessera-gfx1151-device.lock`
wrapped the whole run. The Torch reference venv matches Tajasarus's package
list exactly, and the checkpoint cache is a byte-identical copy of
Tajasarus's.

**The host arms reproduce the gfx1201 run.** The `bf16` and `shipped`
per-token NLL arrays equal the gfx1201 packet's bitwise, on a different host.
That leaves the native arm as the only thing that differs between the two
packets.

## Correctness

- **Finite outputs:** 2,304 native chunk calls and 2,304 shipped chunk calls,
  all finite.
- **Certificates:** the staged tolerance certificate ran and passed on the
  first native chunk of every one of the 36 layers. The resident and staged
  outputs were bitwise equal each time, and the propagated bound passed.
- **Largest independent error:** the biggest full independent error is 1.0, at
  layer 34, in that layer's large-magnitude BF16 outputs. It is inside that
  layer's propagated per-stage bound. The gfx1201 run's largest was 0.5, at
  layer 35.
- **Device identity:** rocminfo reports the iGPU's UUID as
  `GPU-ffffffffffffffff`. The packet identifies the device by architecture,
  marketing name and host.

## Result

| Arm | Perplexity | Δ vs BF16 |
| --- | ---: | ---: |
| BF16 | 12.3763 | — |
| Shipped NVFP4 | 12.6881 | +0.3118 |
| Native (gfx1151) | 12.7076 | +0.3313 |

**Δ(native) / Δ(shipped) = 1.063 ≤ 1.5: the gate passes on gfx1151.**

- **Noise.** A window bootstrap of the ratio gives a 95% interval of
  0.98–1.18 (10,000 resamples, seed 20261102), entirely below 1.5. The paired
  native − shipped mean token NLL difference is +0.0015 (window SE 0.0010).
- **Against gfx1201.** The gfx1151 native NLL is not bitwise equal to the
  gfx1201 native NLL; the two are different programs on different ISAs. The
  mean per-token difference, gfx1151 − gfx1201, is +0.0013 (window SE 0.0014),
  so the two architectures cannot be told apart on this corpus. Neither
  result stands for the other.
- **Error growth through depth.** The native arm's MLP-input drift from BF16
  is 1.003–1.017× the shipped arm's at every layer after the first. It peaks
  at layer 18 (shipped 21.8%, native 22.2%) and is 15.8% and 16.0% at layer 35.

Per-window perplexity (`per_window_perplexity`):

| Window | BF16 | Shipped | Native | Native − shipped |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8.79 | 8.97 | 9.00 | +0.03 |
| 1 | 17.19 | 17.56 | 17.70 | +0.14 |
| 2 | 14.50 | 14.90 | 14.94 | +0.04 |
| 3 | 26.46 | 27.07 | 27.07 | +0.01 |
| 4 | 6.68 | 6.91 | 6.92 | +0.01 |
| 5 | 10.81 | 11.36 | 11.41 | +0.05 |
| 6 | 8.83 | 9.33 | 9.36 | +0.03 |
| 7 | 15.69 | 16.34 | 16.34 | −0.00 |
| 8 | 14.19 | 13.91 | 13.92 | +0.01 |
| 9 | 10.92 | 10.81 | 10.88 | +0.07 |
| 10 | 35.13 | 36.09 | 36.05 | −0.03 |
| 11 | 8.55 | 8.57 | 8.61 | +0.04 |
| 12 | 8.25 | 8.73 | 8.76 | +0.03 |
| 13 | 15.72 | 16.31 | 16.32 | +0.01 |
| 14 | 10.69 | 11.26 | 11.18 | −0.08 |
| 15 | 8.10 | 7.95 | 7.90 | −0.05 |

## Does not close

- **Scope:** one corpus (wikitext-2, 8,176 tokens) and one model, with only the
  MLP substituted and everything else BF16.
- **Not covered:**
  - downstream task accuracy;
  - quantized attention or `lm_head`;
  - the checkpoint's W4A4 deployment recipe;
  - the MXFP4 arms (`../qwen3_mxfp4_perplexity_20261102/`);
  - any default or selector promotion.
- **No performance claim.**

## Files

- `perplexity.json`: identities, per-arm and per-window perplexity, the
  verdict, drift, per-layer weight error, and all 36 certificates.
- `token_nll.npz`: per-token NLL for each arm, shape 16 × 511, plus the token
  ids.
- `run.txt`: the progress log and `time -v` output (1 h 26 min wall, 16.3 GiB
  peak RSS), with two overflow warnings from the FP64 SiLU references
  removed.
- `harness-tests.txt`: the host-free harness tests on Princess-Luna.

Scripts: `benchmarks/rocm/record_qwen3_native_mlp_perplexity.py --arms native`
and `benchmarks/rocm/qwen3_native_mlp_worker.py`.
