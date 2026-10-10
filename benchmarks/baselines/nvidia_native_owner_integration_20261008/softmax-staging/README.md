# Isolated native softmax staging candidate

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync NVIDIA-SOFTMAX-STAGING-2026-10-08.

The diagnostic cProfile identifies native submission as the dominant warm cost. Source inspection finds two cuMemAlloc/cuMemFree pairs per invocation. The candidate reuses the existing aligned retained staging arena while invokeImpl holds g_mu through synchronous completion; arithmetic, PTX image, ABI and stream policy are unchanged.

Owning RTX 5070: 32 native/public/portable/changed-input checks pass. Nine growing/shrinking profiles yield 18 verified softmax outputs, with nine native matmul calls interleaved through the same arena. Every retained host result stays bit-identical after subsequent calls.

Control runtime SHA: e2fd477fbdfc6ae2e91d6e651daa3e55678e21128a8e18ca615dc509753fd290.
Candidate runtime SHA: 3e7ecd6be869d37cd074e3abc5a422e2598880058ba2247b3a78fc3895c44a88.
Source diff: candidate.patch. The authoritative checkout and runtime remain unchanged while its aggregate suite runs.

Matched fresh-process A/B is prepared but must wait for the live aggregate/test process to finish. No performance promotion, asynchronous or concurrent-owner claim. The diagnostic profile ran alongside CPU unit work and is not a timing comparison.

This scratch evidence has not yet been copied into the authoritative branch or published.

## Normalization regression attribution

The shared launch helper candidate passes 51 numerical/contract checks. Four long BF16 norm-to-matmul cases (K4096/K8192, serial/cooperative) fail identically with the unchanged control runtime, before native execution: prepared tensor edge requires synchronous typed matmul. This identifies an existing consumer admission issue, not a candidate numerical failure. The composed envelope remains unproved and must be repaired; no test or gate was weakened. Both terminal logs are preserved here.

## Matched idle-host A/B completed

Five alternating fresh-process windows cover 18 matched public softmax/safe profiles. Native package and descriptor identity match between arms, and each pair passes numerical checks before and after timing. Summary: {"comparisons": 18, "windows": 5, "public_wall_median": 2.269866023365104, "public_wall_min": 2.1689253209819, "public_wall_max": 2.3526860995545174, "resident_device_median": 0.9967282737886709, "resident_device_min": 0.9270367146752672, "resident_device_max": 1.0310244272256999}. The public-wall ratio measures native host staging allocation reuse, not kernel speedup. Resident event timings are separate and traverse unchanged arithmetic. No asynchronous ownership or general compiler closure is claimed. Packet: ab_packet.json.
